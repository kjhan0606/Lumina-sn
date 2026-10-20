/* 실제 원자모형 등록에서 상위 이온 누락과 terminal 혼입을 확인한다. */
#include "lumina.h"

static int check(int yes, const char *message) {
    if (!yes) fprintf(stderr, "[MODEL-PROJECTION][FAIL] %s\n", message);
    return yes ? 0 : 1;
}

static int import_native_nodes(const char *path, AtomicData *atom,
                               NLTEConfig *nlte, int ns, const char *export_path,
                               const double *occupation) {
    FILE *fp=fopen(path,"rb");if (!fp) return 1;
    char magic[8];uint32_t nd,nc,nl;double epoch;
    if (fread(magic,1,8,fp)!=8 || (memcmp(magic,"LUMNOD01",8) && memcmp(magic,"LUMCEL01",8)) ||
        fread(&nd,4,1,fp)!=1 || fread(&nc,4,1,fp)!=1 || fread(&nl,4,1,fp)!=1 ||
        fread(&epoch,8,1,fp)!=1 || nd!=(uint32_t)ns || nl!=(uint32_t)atom->n_levels ||
        nc==0 || nc>10000 || epoch<=0) return 1;
    int32_t *cols=malloc((size_t)nc*3*sizeof(*cols));
    int stride=memcmp(magic,"LUMCEL01",8)==0?6:4;
    int t_at=stride==6?2:1,ne_at=t_at+1,rho_at=t_at+2;
    double *physical=malloc((size_t)ns*(size_t)stride*sizeof(double));
    double *values=malloc((size_t)ns*nc*sizeof(double));
    double *te=malloc((size_t)ns*sizeof(double)),*ne=malloc((size_t)ns*sizeof(double));
    double *rho=malloc((size_t)ns*sizeof(double));
    if (!cols || !physical || !values || !te || !ne || !rho ||
        fread(cols,4,(size_t)nc*3,fp)!=(size_t)nc*3 ||
        fread(physical,8,(size_t)ns*(size_t)stride,fp)!=(size_t)ns*(size_t)stride ||
        fread(values,8,(size_t)ns*nc,fp)!=(size_t)ns*nc || fgetc(fp)!=EOF) return 1;
    fclose(fp);
    for (int s=0;s<ns;s++) { te[s]=physical[stride*s+t_at];ne[s]=physical[stride*s+ne_at];rho[s]=physical[stride*s+rho_at]; }
    PlasmaState p={0};p.n_shells=ns;p.T_e=te;p.n_electron=ne;p.rho=rho;
    if (lumina_publish_seed_te(&p,"native material restart direct test")) return 1;
    int32_t saved_z=cols[0];cols[0]=999;
    if (lumina_import_superlevel_material_with_occupation(nlte,atom,&p,(int)nc,cols,values,occupation)!=POP_ATOMIC_MISSING ||
        atom->population_committed_generation || nlte->population_committed_generation) return 1;
    cols[0]=saved_z;
    double *fixed=atom->element_density.number_density;double saved_n=fixed[0];
    fixed[0]*=1.01;
    if (lumina_import_superlevel_material_with_occupation(nlte,atom,&p,(int)nc,cols,values,occupation)!=POP_SOLVE_FAILED ||
        atom->population_committed_generation || nlte->population_committed_generation) return 1;
    fixed[0]=saved_n;
    double saved_ne=ne[0];ne[0]*=1.001;
    if (lumina_import_superlevel_material_with_occupation(nlte,atom,&p,(int)nc,cols,values,occupation)!=POP_NE_NOT_CONVERGED ||
        atom->population_committed_generation || nlte->population_committed_generation) return 1;
    ne[0]=saved_ne;
    PopulationStatus status=lumina_import_superlevel_material_with_occupation(nlte,atom,&p,(int)nc,cols,values,occupation);
    if (check(status==POP_OK,population_status_name(status))) return 1;
    if (check(nlte->full_model_population_generation==1,"full material ownership")) return 1;
    double max_total=0,max_charge=0,max_full_level_total=0;
    for (int i=0;i<nlte->n_nlte_ions;i++) {
        int ip=-1;
        for (int j=0;j<atom->n_ion_pops;j++)
            if (atom->ion_pop_Z[j]==nlte->nlte_Z[i] &&
                atom->ion_pop_stage[j]==nlte->nlte_ion[i]) ip=j;
        if (ip<0) return 1;
        for (int s=0;s<ns;s++) {
            long double full=0;
            for (int g=nlte->nlte_ion_level_offset[i];g<nlte->nlte_ion_level_offset[i+1];g++)
                full+=nlte->nlte_level_populations[(size_t)g*ns+s];
            double reference=atom->ion_number_density[(size_t)ip*ns+s];
            double rel=reference>0 ? (double)fabsl(full/reference-1) : (double)fabsl(full);
            if (rel>max_full_level_total) max_full_level_total=rel;
        }
    }
    if (check(max_full_level_total<=1e-12,"full-level reconstruction changed ion totals")) return 1;
    for (int s=0;s<ns;s++) {
        double charge=0;
        for (int e=0;e<atom->n_elements;e++) {
            double sum=0;
            for (int ip=atom->elem_ion_offset[e];ip<atom->elem_ion_offset[e+1];ip++) {
                double n=atom->ion_number_density[(size_t)ip*ns+s];
                sum+=n;charge+=n*atom->ion_pop_stage[ip];
            }
            double expected=fixed[(size_t)e*ns+s];
            double error=fabs(sum/expected-1);if (error>max_total) max_total=error;
        }
        double error=fabs(charge/ne[s]-1);if (error>max_charge) max_charge=error;
        if (te[s]!=physical[stride*s+t_at] || ne[s]!=physical[stride*s+ne_at] || rho[s]!=physical[stride*s+rho_at]) return 1;
    }
    printf("[MATERIAL-IMPORT][ACTUAL] input=%.8s samples=%d SL_and_terminal=%u total_error=%.17g charge_error=%.17g Te_ne_rho=UNCHANGED negative_controls=3 RE_solved=0\n",
           magic,ns,nc,max_total,max_charge);
    printf("[MATERIAL-IMPORT][FULL-LEVEL] ion_total_relative_error=%.17g\n",max_full_level_total);
    int checked=0;
    for (int e=0;e<atom->n_elements;e++) {
        int chosen=-1;
        for (int l=0;l<atom->n_lines;l++)
            if (atom->line_atomic_number[l]==atom->element_Z[e]) { chosen=l;break; }
        if (chosen<0) return 1;
        int ip=-1;
        for (int i=0;i<atom->n_ion_pops;i++)
            if (atom->ion_pop_Z[i]==atom->line_atomic_number[chosen] &&
                atom->ion_pop_stage[i]==atom->line_ion_number[chosen]) ip=i;
        if (ip<0) return 1;
        int gl=atom->level_offset[ip]+atom->line_level_upper[chosen];
        int ng=nlte->global_to_nlte_level[gl];
        double upper=NAN;
        EmissivityStatus st=lumina_line_upper_population_for_tau(
            atom,&p,nlte,chosen,0,(size_t)ns,&upper);
        if (check((st==EMISS_OK || st==EMISS_EXACT_ZERO) && ng>=0 &&
                  upper==nlte->nlte_level_populations[(size_t)ng*ns],
                  "all-element line consumer did not use imported population")) return 1;
        nlte->full_model_population_generation=0;
        st=lumina_line_upper_population_for_tau(atom,&p,nlte,chosen,0,(size_t)ns,&upper);
        nlte->full_model_population_generation=1;
        if (check(st==EMISS_STALE_POP,"unowned full model fell back to LTE")) return 1;
        checked++;
    }
    printf("[MATERIAL-IMPORT][LINE-OWNER] elements=%d imported_NLTE=EXACT unowned_generation=REJECTED\n",checked);
    if (export_path) {
        FILE *out=fopen(export_path,"wbx");if (!out) return 1;
        uint32_t shape[2]={(uint32_t)atom->n_levels,(uint32_t)ns};
        int failed=fwrite("LMATFL01",1,8,out)!=8 || fwrite(shape,4,2,out)!=2;
        for (int gl=0;gl<atom->n_levels && !failed;gl++) {
            int at=nlte->global_to_nlte_level[gl];
            if (at<0 || fwrite(nlte->nlte_level_populations+(size_t)at*ns,
                    sizeof(double),(size_t)ns,out)!=(size_t)ns) failed=1;
        }
        if (fclose(out)!=0 || failed) return 1;
        printf("[MATERIAL-IMPORT][FULL-EXPORT] original_FL_order=%d samples=%d\n",atom->n_levels,ns);
    }
    free(cols);free(physical);free(values);free(te);free(ne);free(rho);
    return 0;
}

int main(int argc, char **argv) {
    if (argc>=2 && argc<=6) {
        AtomicData actual = {0}; OpacityState op = {0}; NLTEConfig model = {0};
        int ns=argc>=3 ? atoi(argv[2]):1;
        if (ns<=0 || ns>1000) return 2;
        /* 이 원자 view의 조성은 명시적으로 한 점만 저장한 조건부 입력이다.
         * 물리 쉘 격자 템플릿으로 쓰지 않고 원자 등록만 검사한다. */
        if (check(load_atomic_data(&actual,argv[1],ns)==0,"actual atomic load")) return 1;
        op.n_lines=actual.n_lines;
        if (check(nlte_init_model_atom(&model,&actual,&op,ns)==0,
                  "actual model projection")) return 1;
        if (check(model.n_nlte_levels_total==actual.n_levels,
                  "actual full-level coverage")) return 1;
        for (int l=0;l<actual.n_levels;l++)
            if (check(model.global_to_nlte_level[l]>=0,"actual full-level missing")) return 1;
        for (int l=0;l<actual.n_lines;l++)
            if (check(model.nlte_line_map[l]>=0,"actual line missing")) return 1;
        printf("[MODEL-PROJECTION][ACTUAL] bound_ions=%d full_levels=%d superlevels=%d lines=%d\n",
               model.n_nlte_ions,model.n_nlte_levels_total,model.n_super_total,actual.n_lines);
        double *occupation=NULL;
        if (argc==6) {
            FILE *input=fopen(argv[5],"rb");char magic[8];uint32_t shape[2];
            if (!input || fread(magic,1,8,input)!=8 || memcmp(magic,"LMATOCC1",8) ||
                fread(shape,4,2,input)!=2 || shape[0]!=(uint32_t)actual.n_levels || shape[1]!=(uint32_t)ns) return 1;
            size_t count=(size_t)shape[0]*shape[1];occupation=malloc(count*sizeof(double));
            if (!occupation || fread(occupation,sizeof(double),count,input)!=count || fgetc(input)!=EOF) return 1;
            fclose(input);
        }
        if (argc>=4 && import_native_nodes(argv[3],&actual,&model,ns,argc>=5?argv[4]:NULL,occupation)) return 1;
        free(occupation);
        if (actual.element_density.source==POP_ELEMENT_TOTAL_EXPLICIT_NUMBER) {
            PlasmaState p={0};p.n_shells=ns;
            double *rho=calloc((size_t)ns,sizeof(*rho));p.rho=rho;
            if (!rho) return 1;
            for (int e=0;e<actual.n_elements;e++)
                for (int s=0;s<ns;s++) {
                    size_t at=(size_t)e*ns+s;
                    double expected=actual.element_density.number_density[at];
                    actual.element_mass_amu[e]=NAN;
                    actual.abundances[at]=NAN;rho[s]=NAN;
                    if (check(lumina_element_number_density(&actual,&p,e,s,ns)==expected,
                              "actual inventory changed by mass/X/rho")) return 1;
                }
            free(rho);
            printf("[MODEL-PROJECTION][ACTUAL-NUMBER] elements=%d points=%d; poisoned mass/X/rho ignored\n",
                   actual.n_elements,ns);
        }
        nlte_free(&model); free_atomic_data(&actual);
        return 0;
    }
    enum { NI = 12, NL = 8, NS = 2, NLINE = 4 };
    int ion_z[NI] = {28,28,27,27,26,26,20,20,16,16,14,14};
    int stages[NI] = {5,6,5,6,5,6,4,5,4,5,4,5};
    int offsets[NI+1] = {0,1,1,2,2,3,3,4,4,5,5,8,8};
    int level_z[NL] = {28,27,26,20,16,14,14,14};
    int level_stage[NL] = {5,5,5,4,4,4,4,4};
    int level_num[NL] = {0,0,0,0,0,0,1,2};
    int super[NL] = {0,0,0,0,0,0,0,1};
    int metastable[NL] = {0};
    double energy[NL] = {0,0,0,0,0,0,1,2};
    int line_z[NLINE] = {14,20,26,28};
    int line_stage[NLINE] = {4,4,5,5};
    int line_upper[NLINE] = {1,0,0,0};
    AtomicData atom = {0}; OpacityState opacity = {0}; NLTEConfig nlte = {0};
    atom.n_ion_pops=NI; atom.n_levels=NL; atom.n_lines=NLINE;
    atom.ion_pop_Z=ion_z; atom.ion_pop_stage=stages; atom.level_offset=offsets;
    atom.level_Z=level_z; atom.level_ion=level_stage; atom.level_num=level_num;
    atom.level_super=super; atom.level_metastable=metastable;
    atom.level_energy_eV=energy;
    atom.line_atomic_number=line_z; atom.line_ion_number=line_stage;
    atom.line_level_upper=line_upper; opacity.n_lines=NLINE;
    if (check(nlte_init_model_atom(&nlte,&atom,&opacity,NS)==0,"initialization")) return 1;
    if (check(nlte.n_nlte_ions==6,"every bound ion, no terminal") ||
        check(nlte.n_nlte_levels_total==NL,"all full levels") ||
        check(nlte.n_super_total==7,"all superlevels") ||
        check(nlte.super_mode,"superlevel reconstruction active")) return 1;
    const int expected_z[6]={14,16,20,26,27,28};
    for (int i=0;i<6;i++)
        if (check(nlte.nlte_Z[i]==expected_z[i],"contiguous sorted projection")) return 1;
    for (int l=0;l<NL;l++)
        if (check(nlte.global_to_nlte_level[l]>=0,"unmapped full level")) return 1;
    for (int l=0;l<NLINE;l++)
        if (check(nlte.nlte_line_map[l]>=0,"unmapped high-ion line")) return 1;
    nlte_free(&nlte);
    offsets[NI]=NL-1;
    if (check(nlte_init_model_atom(&nlte,&atom,&opacity,NS)!=0,
              "inconsistent level ownership accepted")) return 1;
    offsets[NI]=NL; level_stage[0]=4;
    if (check(nlte_init_model_atom(&nlte,&atom,&opacity,NS)!=0,
              "wrong ion identity accepted")) return 1;
    level_stage[0]=5; super[0]=-1;
    if (check(nlte_init_model_atom(&nlte,&atom,&opacity,NS)!=0,
              "negative superlevel accepted")) return 1;
    puts("[MODEL-PROJECTION][PASS] full-level=8 bound-ion=6 terminal=6 excluded; identity failures rejected");
    return 0;
}
