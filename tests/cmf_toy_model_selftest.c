#include "cmf_toy_model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"TOY_MODEL_FAIL line=%d %s\n",__LINE__,#x);rc=1;goto done;}}while(0)
int main(int argc,char **argv){
    if(argc!=4)return 2;
    const double masses[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel m={0};int ns=atoi(argv[3]),rc=0;
    CHECK(!cmf_toy_model_prepare(&m,argv[1],argv[2],ns,masses));
    const AtomicData *a=&m.atom;const NLTEConfig *n=&m.nlte;const PlasmaState *p=&m.plasma;
    CHECK(m.prepared&&m.ejecta.n_source_shells==807&&m.geometry.time_explosion==172800);
    CHECK(m.geometry.r_inner[0]==0&&m.geometry.v_outer[ns-1]==40350e5);
    CHECK(!memcmp(p->T_e,m.ejecta.initial_temperature,(size_t)ns*sizeof(double)));
    CHECK(n->n_nlte_ions==a->n_ion_pops&&n->n_nlte_levels_total==a->n_levels&&n->super_mode);
    CHECK(!p->physical_history&&p->initial_steady_nlte==1&&n->radfield_view_status==RADIATION_FIELD_VIEW_DISABLED);
    CHECK(n->line_qset==&m.qset&&n->line_eset==&m.eset);
    double charge_error=0,ion_error=0,level_error=0,max_terminal=0,total_mass=0;
    for(int s=0;s<ns;s++){
        total_mass+=m.ejecta.mass_g[s];double charge=0;
        for(int e=0;e<a->n_elements;e++){
            double nuclei=p->rho[s]*a->abundances[e*ns+s]/(a->element_mass_amu[e]*AMU),sum=0;
            for(int ip=a->elem_ion_offset[e];ip<a->elem_ion_offset[e+1];ip++){
                double ni=a->ion_number_density[ip*ns+s];sum+=ni;charge+=ni*a->ion_pop_stage[ip];
                CHECK(ni>=0&&isfinite(ni));double full=0;
                int lo=a->level_offset[ip],hi=a->level_offset[ip+1];
                for(int gl=lo;gl<hi;gl++){
                    int il=n->global_to_nlte_level[gl];CHECK(il>=0);
                    double value=n->nlte_level_populations[(size_t)il*ns+s];
                    CHECK(value>=0&&isfinite(value));full+=value;
                }
                if(hi>lo){
                    if(ni>0)level_error=fmax(level_error,fabs(full/ni-1));else CHECK(full==0);
                }else if(nuclei>0)max_terminal=fmax(max_terminal,ni/nuclei);
            }
            if(nuclei>0)ion_error=fmax(ion_error,fabs(sum/nuclei-1));else CHECK(sum==0);
        }
        CHECK(charge>0);charge_error=fmax(charge_error,fabs(p->n_electron[s]/charge-1));
    }
    CHECK(total_mass>1.98e33&&total_mass<2e33);
    fprintf(stderr,"TOY_MODEL_CONSERVATION shells=%d charge=%.17g nuclei=%.17g levels=%.17g\n",
            ns,charge_error,ion_error,level_error);
    CHECK(charge_error<1e-8&&ion_error<1e-10&&level_error<1e-10);
    CHECK(!cmf_toy_model_write_seed(&m,"toy_initial_material.csv"));
    CHECK(cmf_toy_model_write_seed(&m,"toy_initial_material.csv")!=0);
    printf("TOY_MODEL_PREPARED shells=%d levels=%d super=%d ions=%d lines=%d coll_ions=%d mass_g=%.17g T_first=%.17g T_last=%.17g ne_first=%.17g ne_last=%.17g charge_error=%.17g ion_error=%.17g level_error=%.17g max_terminal_fraction=%.17g initial_NLTE_converged=0 spectrum_compared=0\n",
        ns,a->n_levels,n->n_super_total,a->n_ion_pops,a->n_lines,a->ncol_ions,total_mass,
        p->T_e[0],p->T_e[ns-1],p->n_electron[0],p->n_electron[ns-1],charge_error,ion_error,level_error,max_terminal);
done:
    cmf_toy_model_free(&m);return rc;
}
