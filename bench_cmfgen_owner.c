/* CMFGEN frozen-state adapter: real production Si/S owner, no SE substitute.
 * Input is produced by scripts/prepare_cmfgen_owner_response.py.  The native
 * J_nu is piecewise linear; registered Gaussian line integrals and canonical
 * continuum publication both consume that same field. */
#include "src/lumina.h"
#include "src/line_jbar.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern int nlte_precompute_within_sl_frac_checked(
    NLTEConfig *, AtomicData *, PlasmaState *, int);

static void require(int condition, const char *message) {
    if (!condition) { fprintf(stderr,"[CMF-OWNER][ERROR] %s\n",message); exit(2); }
}
static void read_exact(FILE *f, void *p, size_t size, size_t n) {
    require(fread(p,size,n,f)==n,"truncated input");
}
static void *allocate(size_t n, size_t size) {
    void *p=calloc(n,size); require(p!=NULL,"allocation failed"); return p;
}
static size_t interval(const double *nu, size_t n, double x) {
    size_t lo=0,hi=n-1;
    while(hi-lo>1){size_t m=lo+(hi-lo)/2;if(nu[m]>x)hi=m;else lo=m;}
    return lo;
}
static double interpolate(double a,double b,double ya,double yb,double x) {
    double t=(x-a)/(b-a);return (1.0-t)*ya+t*yb;
}
static double mean_linear(const double *nu,const double *J,size_t n,
                          double lo,double hi) {
    if(lo<nu[0]||hi>nu[n-1]||!(hi>lo))return NAN;
    size_t i=interval(nu,n,lo);double area=0.0,x=lo;
    while(x<hi){
        double end=nu[i+1]<hi?nu[i+1]:hi;
        double a=interpolate(nu[i],nu[i+1],J[i],J[i+1],x);
        double b=interpolate(nu[i],nu[i+1],J[i],J[i+1],end);
        area+=0.5*(a+b)*(end-x);x=end;i++;
    }
    return area/(hi-lo);
}
static double profile_integral(const double *nu,const double *J,size_t n,
                               double line_nu) {
    double d=line_nu*LINE_JBAR_VDOPPLER_CMS/C_SPEED_OF_LIGHT;
    double lo=line_nu-4*d,hi=line_nu+4*d;
    if(lo<nu[0]||hi>nu[n-1])return NAN;
    size_t i=interval(nu,n,lo);double sum=0.0,x=lo;
    while(x<hi){
        double end=nu[i+1]<hi?nu[i+1]:hi;
        double a=interpolate(nu[i],nu[i+1],J[i],J[i+1],x);
        double b=interpolate(nu[i],nu[i+1],J[i],J[i+1],end);
        sum+=line_jbar_segment_phi_integral(line_nu,x,end,a,b,end-x);
        x=end;i++;
    }
    return sum;
}
static int profile_selftest(void) {
    double nu[]={0.99e15,0.99999e15,1.00007e15,1.01e15};
    double constant[]={3,3,3,3},linear[4];
    for(int i=0;i<4;i++)linear[i]=2.0+(nu[i]/1e15-1.0);
    double a=profile_integral(nu,constant,4,1e15);
    double b=profile_integral(nu,linear,4,1e15);
    require(fabs(a/3-1)<1e-10&&fabs(b/2-1)<1e-10,
            "registered constant/linear profile integral");
    require(isnan(profile_integral(nu,constant,4,2e15)),"uncovered profile accepted");
    require(fabs(mean_linear(nu,linear,4,0.999e15,1.001e15)-2)<1e-13,
            "continuum linear integral");
    puts("CMF_OWNER_PROFILE_SELFTEST PASS");return 0;
}
static int ion_index(const AtomicData *a,int Z,int charge) {
    for(int i=0;i<a->n_ion_pops;i++)
        if(a->ion_pop_Z[i]==Z&&a->ion_pop_stage[i]==charge)return i;
    return -1;
}
static int deck_shell_count(const char *deck) {
    char path[4096],header[16384];
    snprintf(path,sizeof(path),"%s/abundances.csv",deck);
    FILE *f=fopen(path,"r");require(f!=NULL,"abundance header missing");
    require(fgets(header,sizeof(header),f)!=NULL&&strchr(header,'\n')!=NULL,
            "abundance header is missing or oversized");fclose(f);
    int count=0;for(char *p=header;*p;p++)if(*p==',')count++;
    require(count>0&&count<=1000,"invalid deck shell count");return count;
}
static double actual_density(const NLTEConfig *n,const AtomicData *a,int Z,int charge) {
    if(charge==5){int ip=ion_index(a,Z,charge);require(ip>=0,"missing terminal");
        return a->ion_number_density[ip];}
    double value=0;
    for(int k=0;k<n->n_nlte_levels_total;k++){
        int gl=n->nlte_to_global_level[k];
        if(a->level_Z[gl]==Z&&a->level_ion[gl]==charge)
            value+=n->nlte_level_populations[k];
    }
    return value;
}

int main(int argc,char **argv) {
    if(argc==2&&!strcmp(argv[1],"--selftest-profile"))return profile_selftest();
    require(argc>=4&&argc<=6,
            "usage: bench_cmfgen_owner DECK INPUT.bin OUTPUT.csv [Z:depth,... [NEW_DUMP_DIR]]");
    const uint32_t endian=1;require(*(const unsigned char*)&endian==1,"little endian required");
    FILE *in=fopen(argv[2],"rb");require(in!=NULL,"cannot open input");
    char magic[8];uint32_t dims[4];double epoch;
    read_exact(in,magic,1,8);read_exact(in,dims,4,4);read_exact(in,&epoch,8,1);
    require(!memcmp(magic,"CMFOWN01",8)&&dims[0]>0&&dims[0]<=1000&&
            dims[1]>=2&&dims[1]<=2000000&&dims[2]>0&&dims[2]<=10000&&
            dims[3]==LUMINA_RADFIELD_N_BINS&&epoch>0&&isfinite(epoch),
            "invalid input header");
    size_t nd=dims[0],nn=dims[1],nc=dims[2],nb=dims[3];
    unsigned char *selected=allocate(2*nd,1);
    size_t requested=0,executed=0;
    uint64_t field_generation=0;
    if(argc>=5){
        const char *p=argv[4];
        require(*p!=0,"empty selected-cell list");
        while(*p){
            char *end;long Z=strtol(p,&end,10);
            require(end!=p&&*end==':'&&(Z==14||Z==16),"invalid selected Z");
            p=end+1;long depth=strtol(p,&end,10);
            require(end!=p&&depth>=1&&(size_t)depth<=nd,"invalid selected depth");
            size_t key=2*((size_t)depth-1)+(Z==16);
            require(!selected[key],"duplicate selected cell");selected[key]=1;requested++;
            require(!*end||*end==',',"invalid selected-cell separator");
            if(!*end)break;p=end+1;require(*p!=0,"trailing selected-cell comma");
        }
    }else{memset(selected,1,2*nd);requested=2*nd;}
    double *nu=allocate(nn,8),*pop=allocate(nc,8),*J=allocate(nn,8);
    int32_t *col=allocate(nc*3,4),*valid=allocate(nb,4),*col_nlte=allocate(nc,4);
    double *jcan=allocate(nb,8),*edges=allocate(nb+1,8);
    read_exact(in,nu,8,nn);read_exact(in,col,4,nc*3);
    for(size_t i=0;i<nn;i++)require(isfinite(nu[i])&&nu[i]>0&&
                                  (!i||nu[i]>nu[i-1]),"invalid native frequency");
    for(size_t i=0;i<=nb;i++)edges[i]=radiation_field_canonical_frequency_edge(i);

    /* Only explicit supported switches; no inherited experimental guards. */
    setenv("LUMINA_NLTE_ELEMENT_WIDE","1",1);
    setenv("LUMINA_NLTE_ELEMENT_WIDE_Z","14,16",1);
    setenv("LUMINA_NLTE_ELEMENT_WIDE_SHELL","-1",1);
    setenv("LUMINA_NLTE_ELEMENT_WIDE_COMMIT","1",1);
    setenv("LUMINA_NLTE_ELEMENT_WIDE_DUMP",argc==6?"1":"0",1);
    if(argc==6)setenv("LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR",argv[5],1);
    setenv("LUMINA_NLTE_BF_COLLISION","1",1);
    setenv("LUMINA_ARTIS_PARITY","1",1);
    setenv("LUMINA_OMEGA_CMFGEN","1",1);
    AtomicData atom;memset(&atom,0,sizeof(atom));
    require(load_atomic_data(&atom,argv[1],deck_shell_count(argv[1]))==0,"atomic load failed");
    /* Load/validate the original deck unchanged, then replace only material
     * work slabs by one-cell storage.  Every cell below gets its own true
     * CMFGEN elemental totals; no deck abundance is used as a physics input. */
    free(atom.abundances);free(atom.ion_number_density);free(atom.partition_functions);
    atom.abundances=allocate(atom.n_elements,sizeof(double));
    atom.ion_number_density=allocate(atom.n_ion_pops,sizeof(double));
    atom.partition_functions=allocate(atom.n_ion_pops,sizeof(double));
    char sigma_path[4096];snprintf(sigma_path,sizeof(sigma_path),"%s/cmfgen_sigma_bf.bin",argv[1]);
    require(load_cmfgen_sigma_bf(&atom,sigma_path)==0,"sealed BF sigma required");
    snprintf(sigma_path,sizeof(sigma_path),"%s/cmfgen_bf_channels.bin",argv[1]);
    require(load_cmfgen_bf_channels(&atom,sigma_path)==0&&atom.cmfgen_loaded&&
            atom.cmfgen_bf_channels_loaded&&atom.cmfgen_bf_channel_version>=4,
            "sealed grouped-SL BF tables required");
    require(load_ion_coldata_manifest(&atom,argv[1])>=0,"sealed collision import required");
    omega_cmfgen_arm(&atom);
    PlasmaState plasma;memset(&plasma,0,sizeof(plasma));plasma.n_shells=1;
    double Te=0,ne=0,rho=0;
    plasma.T_e=&Te;plasma.n_electron=&ne;plasma.rho=&rho;
    OpacityState opac;memset(&opac,0,sizeof(opac));
    opac.n_lines=atom.n_lines;opac.n_shells=1;opac.line_list_nu=atom.line_nu;
    opac.tau_sobolev=allocate(atom.n_lines,sizeof(double));
    opac.jbar_line=allocate(atom.n_lines,sizeof(double));
    opac.jbar_count=allocate(atom.n_lines,sizeof(int));
    NLTEConfig nlte;memset(&nlte,0,sizeof(nlte));
    require(nlte_init(&nlte,&atom,&opac,1)==0,"production NLTE init failed");
    /* No packet estimator exists for a deterministic CMFGEN input.  Keeping
     * a zero-valued allocated C2 buffer would select the wrong rate source. */
    free(nlte.bf_rate_estimator);nlte.bf_rate_estimator=NULL;
    /* Only Si/S are consumed; the production projection itself is unchanged. */
    int *map=allocate(atom.n_lines,sizeof(int));uint8_t *mask=allocate(atom.n_lines,1);
    for(int l=0;l<atom.n_lines;l++)map[l]=
        (atom.line_atomic_number[l]==14||atom.line_atomic_number[l]==16)?nlte.nlte_line_map[l]:-1;
    require(line_jbar_bb_domain_mask_build(mask,atom.n_lines,atom.line_nu,map,NULL,NULL)==0,
            "BB domain build failed");
    LineJbarQSet qset;memset(&qset,0,sizeof(qset));
    require(line_jbar_qset_build(&qset,atom.n_lines,atom.line_nu,map,mask)==0,"Q_g build failed");
    require(line_jbar_qset_profile_support_covered(&qset,nu[0],nu[nn-1],NULL)==0,
            "native field does not cover registered profile");
    nlte.line_qset=&qset;
    require(radiation_field_owner_init(&nlte.radiation_field,1)==0,"radiation owner init");
    double *jbar=allocate(qset.n_q,8);uint64_t *ids=allocate(qset.n_q,8);
    int32_t *line_valid=allocate(qset.n_q,4);
    for(size_t i=0;i<qset.n_q;i++)ids[i]=(uint64_t)qset.line_id[i];
    GammaDeposition gamma;gamma_deposition_init(&gamma,1);
    double no_nonthermal=0;
    require(gamma_deposition_publish(&gamma,GAMMA_PROVENANCE_EXTERNAL_FILE,epoch,
                                     NULL,NULL,NULL,&no_nonthermal)==0,"zero NT publication failed");
    for(size_t i=0;i<nc;i++){
        int Z=col[3*i],c=col[3*i+1],sl=col[3*i+2];col_nlte[i]=-1;
        require((Z==14||Z==16)&&c>=1&&c<=5,"invalid population key");
        if(c==5)continue;
        for(int k=0;k<nlte.n_nlte_ions;k++)if(nlte.nlte_Z[k]==Z&&nlte.nlte_ion[k]==c){
            require(sl>=0&&sl<nlte.nlte_ion_super_offset[k+1]-nlte.nlte_ion_super_offset[k],
                    "invalid active SL");col_nlte[i]=nlte.nlte_ion_super_offset[k]+sl;
        }
        require(col_nlte[i]>=0,"production target missing");
    }
    FILE *out=fopen(argv[3],"wx");require(out!=NULL,"output exists or cannot be opened");
    fprintf(out,"depth_1based,Z,stage,Te_K,ne_cm3,n_element,pre_fraction,candidate_fraction,committed_fraction,completed,solve_attempted,candidate_valid,verdict_pass,commit_performed,topology_pass,numerical_pass,boundary_pass,scaled_se_residual,conservation_residual,kappa_2,dimension,raw_rank,graph_components,legacy_topology_pass,legacy_numerical_pass,certificate_attempted,certificate_status,certificate_pass,certificate_probability_error_display,certificate_relative_error_display,certificate_stage_lower_display,certificate_stage_upper_display,adoption_path\n");
    for(size_t s=0;s<nd;s++){
        double p[5];read_exact(in,p,8,5);rho=p[0];Te=p[1];ne=p[2];
        /* Identifies this externally supplied temperature; it is not a
         * radiative-equilibrium publication or a convergence claim. */
        plasma.T_e_generation=s+1;
        require(rho>0&&Te>0&&ne>0&&p[3]>=0&&p[4]>p[3],"invalid plasma profile");
        read_exact(in,pop,8,nc);read_exact(in,jcan,8,nb);read_exact(in,valid,4,nb);
        read_exact(in,J,8,nn);
        for(size_t i=0;i<nn;i++)require(J[i]>=0&&isfinite(J[i]),"invalid J_nu");
        if(!selected[2*s]&&!selected[2*s+1])continue;
        field_generation++;
        memset(atom.ion_number_density,0,atom.n_ion_pops*sizeof(double));
        memset(nlte.nlte_level_populations,0,nlte.n_nlte_levels_total*sizeof(double));
        double totals[2]={0,0},before[2][5]={{0}};
        for(size_t i=0;i<nc;i++){
            require(pop[i]>=0&&isfinite(pop[i]),"invalid population");
            int e=col[3*i]==14?0:1,c=col[3*i+1],ip=ion_index(&atom,col[3*i],c);
            require(ip>=0,"ion population slot absent");
            atom.ion_number_density[ip]+=pop[i];totals[e]+=pop[i];before[e][c-1]+=pop[i];
        }
        for(int e=0;e<atom.n_elements;e++){
            int Z=atom.element_Z[e];
            atom.abundances[e]=(Z==14||Z==16)?totals[Z==14?0:1]*atom.element_mass_amu[e]*1.660539066e-24/rho:0;
        }
        lumina_oracle_prepare_partitions(&atom,&plasma,1);
        require(nlte_precompute_within_sl_frac_checked(&nlte,&atom,&plasma,1)==0,
                "production thermal SL weights failed");
        for(int k=0;k<nlte.n_nlte_levels_total;k++){
            int gl=nlte.nlte_to_global_level[k];
            if(atom.level_Z[gl]!=14&&atom.level_Z[gl]!=16)continue;
            int found=0;
            for(size_t i=0;i<nc;i++)if(col_nlte[i]==nlte.fl_to_super[k]){
                nlte.nlte_level_populations[k]=pop[i]*nlte.within_sl_frac[k];found++;
            }
            require(found==1,"SL input coverage is not exactly once");
        }
        for(int e=0;e<2;e++)for(int c=1;c<=5;c++)
            require(fabs(actual_density(&nlte,&atom,e?16:14,c)-before[e][c-1])<=
                    1e-12*totals[e],"initial SL/ion total was changed");
        for(int b=0;b<nlte.n_freq_bins;b++){
            double lo=nlte.nu_min*exp(b*nlte.d_log_nu),hi=lo*exp(nlte.d_log_nu);
            nlte.J_nu[b]=mean_linear(nu,J,nn,lo,hi);
        }
        for(size_t q=0;q<qset.n_q;q++){
            jbar[q]=profile_integral(nu,J,nn,qset.line_nu[q]);
            require(jbar[q]>=0&&isfinite(jbar[q]),"invalid registered Jbar integral");
            line_valid[q]=jbar[q]>0?LINE_JBAR_VALID:LINE_JBAR_EXACT_ZERO;
        }
        RadiationFieldCommitRequest request;memset(&request,0,sizeof(request));
        request.provenance_kind=RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY;
        request.producer="CMFGEN_FROZEN_OWNER_PIECEWISE_LINEAR";
        request.generation=field_generation;request.epoch=epoch;request.n_shells=1;
        request.v_inner=&p[3];request.v_outer=&p[4];request.source_n_bins=nb;
        request.source_frequency_bin_edges=edges;request.source_J_nu=jcan;
        request.source_validity=(const RadiationFieldValidityState*)valid;
        request.statistic_kind=RADIATION_FIELD_DETERMINISTIC;
        request.line_n=qset.n_q;request.line_id=ids;
        request.line_q_set_hash=qset.q_set_hash;request.line_set_kind=LINE_JBAR_SET_RATE_GRAPH;
        request.line_profile_id=qset.profile_id;request.line_profile_hash=qset.profile_hash;
        request.line_provenance_kind=RADIATION_FIELD_PROVENANCE_CMFGEN_LINE_PROFILE_INTEGRAL;
        request.line_producer=LUMINA_LINE_JBAR_DETERMINISTIC_PRODUCER;
        request.line_jbar=jbar;request.line_validity=line_valid;
        require(radiation_field_begin_mc(&nlte.radiation_field,&p[3],&p[4],1,epoch,field_generation)==0,
                "radiation generation begin failed");
        require(radiation_field_commit(&nlte.radiation_field,&request)==0,"field commit failed");
        nlte.radfield_view_status=radiation_field_read_view(&nlte.radiation_field,
            epoch,1,field_generation,&nlte.radfield_view);
        nlte.line_view_status=radiation_field_line_jbar_rate_view(&nlte.radiation_field,
            epoch,1,field_generation,qset.q_set_hash,qset.profile_id,qset.profile_hash,&nlte.line_view);
        require(nlte.radfield_view_status==RADIATION_FIELD_VIEW_OK&&
                nlte.line_view_status==LINE_JBAR_VIEW_OK,"published view acquisition failed");
        nlte.current_iter=(int)s+1;
        for(int e=0;e<2;e++){
            if(!selected[2*s+e])continue;
            NLTEEWResponse response;int Z=e?16:14;
            require(nlte_element_wide_run_response(&nlte,&atom,&plasma,&opac,Z,0,(int)s,
                        epoch,&gamma,1,&response)==0,"production owner operational failure");
            executed++;
            for(int c=1;c<=5;c++){
                fprintf(out,"%zu,%d,%d,%.17g,%.17g,%.17g,%.17g,",s+1,Z,c+1,Te,ne,
                        totals[e],before[e][c-1]/totals[e]);
                if(response.candidate_available)fprintf(out,"%.17g",response.candidate_density[c-1]/totals[e]);
                fprintf(out,",%.17g,%d,%d,%d,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%d,%d,%d",
                    actual_density(&nlte,&atom,Z,c)/totals[e],response.completed,
                    response.solve_attempted,response.candidate_valid,response.verdict_pass,
                    response.commit_performed,response.topology_pass,response.numerical_pass,
                    response.boundary_pass,response.scaled_se_residual,
                    response.conservation_residual,response.kappa_2,
                    response.dimension,response.raw_rank,response.graph_components);
                fprintf(out,",%d,%d,%d,%d,%d,%.21Le,%.21Le,%.21Le,%.21Le,CERTIFIED_NUMERICAL\n",
                    response.legacy_topology_pass,response.legacy_numerical_pass,
                    response.certificate_attempted,response.certificate_status,response.certificate_pass,
                    response.certificate_probability_error,response.certificate_relative_error,
                    response.certificate_stage_lower[c-1],response.certificate_stage_upper[c-1]);
            }
        }
        require(fflush(out)==0&&!ferror(out),"response output write failed");
    }
    require(fgetc(in)==EOF&&!ferror(in),"unexpected trailing input");fclose(in);
    require(fclose(out)==0,"response close failed");
    gamma_deposition_free(&gamma);radiation_field_owner_free(&nlte.radiation_field);
    nlte_free(&nlte);line_jbar_qset_free(&qset);free_atomic_data(&atom);
    free(opac.tau_sobolev);free(opac.jbar_line);free(opac.jbar_count);
    free(nu);free(pop);free(J);free(col);free(valid);free(col_nlte);
    free(jcan);free(edges);free(map);free(mask);free(jbar);free(ids);free(line_valid);
    free(selected);require(executed==requested,"selected solve coverage incomplete");
    if(argc>=5)printf("CMF_OWNER_SELECTED_RESPONSE_COMPLETE input_depths=%zu solves=%zu GPUh=0 final_goal=NOT_JUDGED\n",nd,executed);
    else printf("CMF_OWNER_RESPONSE_COMPLETE depths=%zu GPUh=0 final_goal=NOT_JUDGED\n",nd);
    return 0;
}
