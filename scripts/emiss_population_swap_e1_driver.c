#define _GNU_SOURCE
#include "../src/lumina.h"
#include "../src/lumina_cmfgen.h"

#include <errno.h>
#include <stdint.h>
#include <sys/stat.h>

int nlte_build_projection(NLTEConfig *, AtomicData *, OpacityState *, int,
                          const int *, const int *, int, int, int *);

enum { E1_NS = 50, E1_NB = 1000 };

typedef struct {
    int ns, nb;
    double texp;
    double *r_edge, *nu, *dnu, *chi, *chies, *etaf, *etac, *eta, *J;
} Frozen;

typedef struct {
    int ns;
    uint64_t nl;
    unsigned char *covered;
    double *pop;
} BPop;

static int load_env(const char *run) {
    char path[4096], line[8192];
    snprintf(path, sizeof(path), "%s/stdout.log", run);
    FILE *fp = fopen(path, "r"); if (!fp) return -1;
    int in = 0, n = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "=== RESOLVED CONFIG")) { in = 1; continue; }
        if (!in) continue;
        if (strstr(line, "argv:")) break;
        char key[256], val[7680];
        if (sscanf(line, " %255[^=]=%7679[^\n]", key, val) == 2) {
            if (!strcmp(key, "LUMINA_BIN") || !strcmp(key, "OMP_NUM_THREADS") ||
                !strcmp(key, "LUMINA_CMF_SOLVE_GPU")) continue;
            setenv(key, val, 1); n++;
        }
    }
    fclose(fp);
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("LUMINA_CMF_SOLVE_GPU", "0", 1);
    return n;
}

static int rd(FILE *fp, void *p, size_t z, size_t n) {
    return fread(p, z, n, fp) == n ? 0 : -1;
}

static int load_frozen(const char *path, Frozen *x) {
    memset(x, 0, sizeof(*x));
    FILE *fp = fopen(path, "rb"); if (!fp) return -1;
    unsigned char magic[8]; uint32_t endian, ver, flags, reserved;
    uint64_t ns, nb, iter, gen;
    if (rd(fp, magic, 1, 8) || memcmp(magic, "LCMFCE01", 8) ||
        rd(fp, &endian, 4, 1) || rd(fp, &ver, 4, 1) ||
        rd(fp, &ns, 8, 1) || rd(fp, &nb, 8, 1) ||
        rd(fp, &iter, 8, 1) || rd(fp, &gen, 8, 1) ||
        rd(fp, &flags, 4, 1) || rd(fp, &reserved, 4, 1) ||
        rd(fp, &x->texp, 8, 1) || endian != 0x01020304 || ver != 1 ||
        ns != E1_NS || nb != E1_NB) { fclose(fp); return -1; }
    (void)iter; (void)gen; (void)flags; (void)reserved;
    x->ns = (int)ns; x->nb = (int)nb;
    size_t cells = (size_t)x->ns * x->nb;
    x->r_edge = malloc((ns + 1) * 8); x->nu = malloc(nb * 8); x->dnu = malloc(nb * 8);
    x->chi = malloc(cells * 8); x->chies = malloc(cells * 8);
    x->etaf = malloc(cells * 8); x->etac = malloc(cells * 8);
    x->eta = malloc(cells * 8); x->J = malloc(cells * 8);
    if (!x->r_edge || !x->nu || !x->dnu || !x->chi || !x->chies ||
        !x->etaf || !x->etac || !x->eta || !x->J) { fclose(fp); return -1; }
    int fail = rd(fp,x->r_edge,8,ns+1) || rd(fp,x->nu,8,nb) || rd(fp,x->dnu,8,nb) ||
        rd(fp,x->chi,8,cells) || rd(fp,x->chies,8,cells) ||
        rd(fp,x->etaf,8,cells) || rd(fp,x->etac,8,cells) ||
        rd(fp,x->eta,8,cells) || rd(fp,x->J,8,cells);
    int extra = fgetc(fp); fclose(fp);
    return (fail || extra != EOF) ? -1 : 0;
}

static int load_bpop(const char *path, BPop *b, uint64_t expected_levels) {
    memset(b, 0, sizeof(*b)); FILE *fp = fopen(path, "rb"); if (!fp) return -1;
    char magic[8]; uint32_t ver, ns; uint64_t nl;
    if (rd(fp,magic,1,8) || memcmp(magic,"E1POP001",8) || rd(fp,&ver,4,1) ||
        rd(fp,&ns,4,1) || rd(fp,&nl,8,1) || ver != 1 || ns != E1_NS ||
        nl != expected_levels) { fclose(fp); return -1; }
    b->ns=(int)ns; b->nl=nl; b->covered=malloc(nl); b->pop=malloc(nl*ns*8);
    if (!b->covered || !b->pop || rd(fp,b->covered,1,nl) ||
        rd(fp,b->pop,8,nl*ns) || fgetc(fp)!=EOF) { fclose(fp); return -1; }
    fclose(fp); return 0;
}

static int find_ip(const AtomicData *a, int Z, int ion) {
    for (int i=0;i<a->n_ion_pops;i++)
        if (a->ion_pop_Z[i]==Z && a->ion_pop_stage[i]==ion) return i;
    return -1;
}

static int find_gl(const AtomicData *a, int Z, int ion, int lev) {
    int ip=find_ip(a,Z,ion); if(ip<0)return -1;
    int g=a->level_offset[ip]+lev;
    if(g<a->level_offset[ip+1] && a->level_num[g]==lev)return g;
    for(g=a->level_offset[ip];g<a->level_offset[ip+1];g++)if(a->level_num[g]==lev)return g;
    return -1;
}

static int load_state(const char *run, PlasmaState *p, OpacityState *o) {
    char path[4096], line[1024]; snprintf(path,sizeof(path),"%s/lumina_plasma_state.csv",run);
    FILE *fp=fopen(path,"r"); if(!fp)return -1; fgets(line,sizeof(line),fp);
    int seen=0,s; double W,T,n,Te;
    while(fgets(line,sizeof(line),fp)) if(sscanf(line,"%d,%lf,%lf,%lf,%lf",&s,&W,&T,&n,&Te)==5 && s>=0&&s<p->n_shells){
        p->W[s]=W;p->T_rad[s]=T;p->n_electron[s]=n;p->T_e[s]=Te;o->electron_density[s]=n;seen++;}
    fclose(fp); return seen==p->n_shells?0:-1;
}

static int load_ions(const char *run, AtomicData *a, int ns) {
    char path[4096],line[1024];snprintf(path,sizeof(path),"%s/lumina_ion_pops.csv",run);
    FILE *fp=fopen(path,"r");if(!fp)return -1;fgets(line,sizeof(line),fp);int nrow=0,s,Z,ion;double n;
    while(fgets(line,sizeof(line),fp))if(sscanf(line,"%d,%d,%d,%lf",&s,&Z,&ion,&n)==4&&s>=0&&s<ns){int ip=find_ip(a,Z,ion);if(ip>=0){a->ion_number_density[(size_t)ip*ns+s]=n;nrow++;}}
    fclose(fp);return nrow>0?0:-1;
}

static int load_levels_A(const char *run, AtomicData *a, NLTEConfig *n, int ns) {
    char path[4096],line[2048];snprintf(path,sizeof(path),"%s/lumina_levelpop.csv",run);
    FILE *fp=fopen(path,"r");if(!fp)return -1;fgets(line,sizeof(line),fp);
    int row=0,s,Z,ion,lev,g,hs,nsp;double E,nk,ng,bk;
    while(fgets(line,sizeof(line),fp))if(sscanf(line,"%d,%d,%d,%d,%lf,%d,%lf,%lf,%lf,%d,%d",&s,&Z,&ion,&lev,&E,&g,&nk,&ng,&bk,&hs,&nsp)==11&&s>=0&&s<ns){
        int gl=find_gl(a,Z,ion,lev);if(gl>=0){int nl=n->global_to_nlte_level[gl];if(nl>=0){n->nlte_level_populations[(size_t)nl*ns+s]=nk;row++;}}}
    fclose(fp);return row>0?0:-1;
}

static int load_dep(const char *path, double *h, int ns) {
    FILE *fp=fopen(path,"r");if(!fp)return -1;char line[512];fgets(line,sizeof(line),fp);int s,n=0;double v;
    while(fgets(line,sizeof(line),fp))if(sscanf(line,"%d,%lf",&s,&v)==2&&s>=0&&s<ns){h[s]=v;n++;}
    fclose(fp);return n==ns?0:-1;
}

static void frozen_to_ascending(const Frozen *f, const double *src, double *dst) {
    for(int s=0;s<f->ns;s++)for(int b=0;b<f->nb;b++)dst[(size_t)s*f->nb+b]=src[(size_t)s*f->nb+(f->nb-1-b)];
}

static void set_audit(CMFGENState *c) {
    size_t n=(size_t)c->n_shells*c->n_bins;
    for(size_t q=0;q<n;q++)c->eta_total_audit[q]=c->chi_tot[q]*c->S_fixed[q]+c->chi_es[q]*c->J[q];
}

static double rel_l1(const double *a,const double *b,size_t n) {
    long double x=0,y=0;for(size_t i=0;i<n;i++){x+=fabsl((long double)a[i]-b[i]);y+=fabsl((long double)b[i]);}return y>0?(double)(x/y):NAN;
}

int main(int argc,char **argv){
    const char *run=argc>1?argv[1]:"/gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605";
    const char *model=argc>2?argv[2]:"data/tardis_reference_toy06_19p48d_sivcaiv";
    const char *bpath=argc>3?argv[3]:"validation/emiss_e1/cmfgen_b_populations.bin";
    const char *out=argc>4?argv[4]:"validation/emiss_e1";
    if(load_env(run)<=0){fprintf(stderr,"environment load failed\n");return 1;}
    mkdir(out,0775);
    char path[4096];snprintf(path,sizeof(path),"%s/chieta_iter10",run);Frozen fr;
    if(load_frozen(path,&fr)){fprintf(stderr,"frozen capture load failed\n");return 1;}
    Geometry geo={0};OpacityState op={0};PlasmaState plasma={0};MCConfig cfg={0};AtomicData atom={0};
    if(load_tardis_reference_data(model,&geo,&op,&plasma,&cfg)||load_atomic_data(&atom,model,geo.n_shells))return 1;
    const char *sig=getenv("LUMINA_CMFGEN_SIGMA_BF");if(sig&&*sig)if(load_cmfgen_sigma_bf(&atom,sig))return 1;
    if(artis_parity_enabled()){char p3[4096];snprintf(p3,sizeof(p3),"%s/feiii_col_zhang.bin",model);load_feiii_coldata(&atom,p3);if(load_ion_coldata_manifest(&atom,model)<=0)return 1;}
    omega_cmfgen_arm(&atom);inject_topstage_continuum_levels(&atom,&op);
    int ns=geo.n_shells;plasma.n_electron=calloc(ns,8);plasma.T_e=calloc(ns,8);if(!plasma.n_electron||!plasma.T_e)return 1;
    if(load_state(run,&plasma,&op)||load_ions(run,&atom,ns))return 1;
    NLTEConfig nlte={0};if(nlte_init(&nlte,&atom,&op,ns))return 1;
    if(load_levels_A(run,&atom,&nlte,ns))return 1;
    lumina_oracle_prepare_partitions(&atom,&plasma,ns);
    lumina_oracle_compute_tau_sobolev(&atom,&plasma,&op,geo.time_explosion);
    nlte_update_tau_sobolev(&nlte,&atom,&op,geo.time_explosion,ns);
    lumina_oracle_prepare_line_eps(&nlte,&atom,&op);
    size_t nls=(size_t)atom.n_lines*ns;double *tauA=malloc(nls*8);memcpy(tauA,op.tau_sobolev,nls*8);
    BPop bp;if(load_bpop(bpath,&bp,atom.n_levels))return 1;
    BFOpacity bf={0};bf_opacity_init(&bf,ns);bf_set_nlte_pops(&nlte);compute_bf_opacity(&bf,&atom,&plasma,ns);
    double *dep=calloc(ns,8);snprintf(path,sizeof(path),"%s/deposition_cmfgen.csv",model);if(load_dep(path,dep,ns))return 1;cmfgen_set_deposition(dep,ns);
    CMFGENState ca;if(cmfgen_init(&ca,&geo))return 1;frozen_to_ascending(&fr,fr.J,ca.J);cmfgen_assemble(&ca,&geo,&op,&bf,&plasma);set_audit(&ca);
    snprintf(path,sizeof(path),"%s/chieta_A_replay",out);if(cmfgen_dump_frozen_chieta(&ca,&geo,10,10,1,path))return 1;
    size_t cells=(size_t)ns*E1_NB;double *fchi=malloc(cells*8),*feta=malloc(cells*8);frozen_to_ascending(&fr,fr.chi,fchi);frozen_to_ascending(&fr,fr.etaf,feta);
    double A_chi_rel=rel_l1(ca.chi_tot,fchi,cells);double *aeta=malloc(cells*8);for(size_t q=0;q<cells;q++)aeta[q]=ca.chi_tot[q]*ca.S_fixed[q];double A_eta_rel=rel_l1(aeta,feta,cells);

    int tz[NLTE_MAX_IONS],ti[NLTE_MAX_IONS],nt=0;unsigned char *full=calloc(atom.n_ion_pops,1);
    for(int ip=0;ip<atom.n_ion_pops;ip++){int ok=atom.level_offset[ip+1]>atom.level_offset[ip];for(int gl=atom.level_offset[ip];gl<atom.level_offset[ip+1];gl++)if(!bp.covered[gl]){ok=0;break;}if(ok){if(nt>=NLTE_MAX_IONS){fprintf(stderr,"too many fully covered ions\n");return 1;}full[ip]=1;tz[nt]=atom.ion_pop_Z[ip];ti[nt]=atom.ion_pop_stage[ip];nt++;}}
    NLTEConfig bn={0};int mapped=0;if(nlte_build_projection(&bn,&atom,&op,ns,tz,ti,nt,0,&mapped))return 1;
    long b_level_cells=0,b_ion_cells=0,b_line_cells=0,b_line_uv=0;
    for(int j=0;j<bn.n_nlte_levels_total;j++){int gl=bn.nlte_to_global_level[j];for(int s=0;s<ns;s++){double v=bp.pop[(size_t)gl*ns+s];if(isfinite(v)){bn.nlte_level_populations[(size_t)j*ns+s]=v;b_level_cells++;}}}
    for(int ip=0;ip<atom.n_ion_pops;ip++)if(full[ip])for(int s=0;s<ns;s++){double sum=0;int ok=1;for(int gl=atom.level_offset[ip];gl<atom.level_offset[ip+1];gl++){double v=bp.pop[(size_t)gl*ns+s];if(!isfinite(v)){ok=0;break;}sum+=v;}if(ok){atom.ion_number_density[(size_t)ip*ns+s]=sum;b_ion_cells++;}}
    memcpy(op.tau_sobolev,tauA,nls*8);
    for(int l=0;l<atom.n_lines;l++){int lo=find_gl(&atom,atom.line_atomic_number[l],atom.line_ion_number[l],atom.line_level_lower[l]);int up=find_gl(&atom,atom.line_atomic_number[l],atom.line_ion_number[l],atom.line_level_upper[l]);if(lo<0||up<0||!bp.covered[lo]||!bp.covered[up])continue;double pref=SOBOLEV_COEFF*atom.line_f_lu[l]*atom.line_wavelength_cm[l]*geo.time_explosion;for(int s=0;s<ns;s++){double nl=bp.pop[(size_t)lo*ns+s],nu=bp.pop[(size_t)up*ns+s];if(!isfinite(nl)||!isfinite(nu))continue;double stim=1.0;if(nl>0&&nu>0&&atom.level_g[lo]>0&&atom.level_g[up]>0){stim=1.0-(double)atom.level_g[lo]*nu/((double)atom.level_g[up]*nl);if(stim<0)stim=0;}double t=pref*nl*stim;if(!(t>1e-100))t=1e-100;op.tau_sobolev[(size_t)l*ns+s]=t;b_line_cells++;if(atom.line_wavelength_cm[l]>=600e-8&&atom.line_wavelength_cm[l]<3000e-8)b_line_uv++;}}
    bf_set_nlte_pops(&bn);compute_bf_opacity(&bf,&atom,&plasma,ns);
    CMFGENState cb;if(cmfgen_init(&cb,&geo))return 1;frozen_to_ascending(&fr,fr.J,cb.J);cmfgen_assemble(&cb,&geo,&op,&bf,&plasma);set_audit(&cb);
    snprintf(path,sizeof(path),"%s/chieta_B",out);if(cmfgen_dump_frozen_chieta(&cb,&geo,10,10,1,path))return 1;
    snprintf(path,sizeof(path),"%s/assembly_audit.json",out);FILE *js=fopen(path,"w");if(!js)return 1;
    fprintf(js,"{\n  \"A_replay_chi_rel_l1\": %.17g,\n  \"A_replay_eta_fixed_rel_l1\": %.17g,\n  \"B_full_ions\": %d,\n  \"B_projection_lines\": %d,\n  \"B_level_shell_cells\": %ld,\n  \"B_ion_shell_cells\": %ld,\n  \"B_line_shell_cells\": %ld,\n  \"B_line_shell_cells_600_3000A\": %ld,\n  \"total_line_shell_cells\": %zu,\n  \"no_new_clamp\": true\n}\n",A_chi_rel,A_eta_rel,nt,mapped,b_level_cells,b_ion_cells,b_line_cells,b_line_uv,nls);fclose(js);
    printf("[E1] A replay relL1 chi=%.6e eta_fixed=%.6e; B ions=%d line-cells=%ld/%zu\n",A_chi_rel,A_eta_rel,nt,b_line_cells,nls);
    return 0;
}
