#include "cmf_toy_model.h"
#include "nlte_line_index.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"TOY_MODEL_FAIL line=%d %s\n",__LINE__,#x);rc=1;goto done;}}while(0)
static int check_line_selection(const NLTEConfig *n,int count,int lo,int hi){
    NLTEPairLineIterator it=nlte_pair_line_iterator(n,lo,hi,count);
    for(int line=0;line<count;line++)if(n->nlte_line_map[line]>=lo&&n->nlte_line_map[line]<=hi)
        if(nlte_pair_line_next(&it)!=line)return -1;
    return nlte_pair_line_next(&it)==-1?0:-1;
}
int main(int argc,char **argv){
    if(argc!=4&&argc!=5&&argc!=6)return 2;
    const double masses[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel m={0};int ns=atoi(argv[3]),rc=0;
    {
        int map[9]={1,-1,0,2,1,0,3,1,2},offset[6]={0,2,5,7,8,8};
        int lines[8]={2,5,0,4,7,3,8,6};
        NLTEConfig sample={0};sample.n_nlte_ions=5;sample.nlte_line_map=map;
        sample.nlte_ion_line_offsets=offset;sample.nlte_ion_lines=lines;sample.nlte_line_index_count=9;
        for(int lo=0;lo<5;lo++)for(int hi=lo;hi<5;hi++)
            CHECK(!check_line_selection(&sample,9,lo,hi));
        /* 빈 terminal, 교차 순서, 미매핑 선, 넓은 범위 및 오래된 index fallback. */
        sample.nlte_line_index_count=10;
        CHECK(!check_line_selection(&sample,9,0,1));
        sample.nlte_line_index_count=9;sample.nlte_ion_lines=NULL;
        CHECK(!check_line_selection(&sample,9,0,1));
    }
    if(argc==5){
        char *end;double ratio=strtod(argv[4],&end);
        CHECK(*argv[4]&&!*end&&isfinite(ratio)&&ratio>1);
        CHECK(!cmf_toy_model_prepare_refined(&m,argv[1],argv[2],ns,masses,ratio));
        printf("TOY_GRID_REFINED base_shells=%d effective_shells=%d T_ratio=%.17g reference_input=0\n",
               ns,m.ejecta.n_shells,ratio);
        ns=m.ejecta.n_shells;
    }else CHECK(!cmf_toy_model_prepare_runtime(&m,argv[1],argv[2],ns,masses));
    ns=m.geometry.n_shells;
    const AtomicData *a=&m.atom;const NLTEConfig *n=&m.nlte;const PlasmaState *p=&m.plasma;
    CHECK(m.prepared&&m.ejecta.n_source_shells==807&&m.geometry.time_explosion==172800);
    CHECK(m.geometry.r_inner[0]==0&&m.geometry.v_outer[ns-1]==40350e5);
    CHECK(!memcmp(p->T_e,m.ejecta.initial_temperature,(size_t)ns*sizeof(double)));
    CHECK(n->n_nlte_ions==a->n_ion_pops&&n->n_nlte_levels_total==a->n_levels&&n->super_mode);
    CHECK(n->exclude_empirical_heavy_ce==1);
    CHECK(n->channel_owned_recombination==1);
    CHECK(!p->physical_history&&p->initial_steady_nlte==1&&n->radfield_view_status==RADIATION_FIELD_VIEW_DISABLED);
    CHECK(n->line_qset==&m.qset&&n->line_eset==&m.eset);
    CHECK(n->nlte_line_global&&n->nlte_line_global_count==a->n_lines);
    CHECK(n->nlte_ion_line_offsets&&n->nlte_ion_lines&&n->nlte_line_index_count==a->n_lines);
    for(int lo=0;lo<n->n_nlte_ions;lo++){
        CHECK(!check_line_selection(n,a->n_lines,lo,lo));
        if(lo+1<n->n_nlte_ions)CHECK(!check_line_selection(n,a->n_lines,lo,lo+1));
    }
    printf("TOY_ION_LINE_INDEX PASS ions=%d lines=%d exact_selection_and_order=1\n",n->n_nlte_ions,a->n_lines);
    clock_t lookup_start=clock();size_t matched=0;
    size_t collision_tiers[4]={0};int collision_probe=0;
    for(int line=0;line<a->n_lines;line++){
        int ip=-1,lo=-1,up=-1;
        for(int i=0;i<a->n_ion_pops;i++)if(a->ion_pop_Z[i]==a->line_atomic_number[line]&&
            a->ion_pop_stage[i]==a->line_ion_number[line]){ip=i;break;}
        if(ip>=0)for(int l=a->level_offset[ip];l<a->level_offset[ip+1];l++){
            if(a->level_num[l]==a->line_level_lower[line])lo=l;
            if(a->level_num[l]==a->line_level_upper[line])up=l;
            if(lo>=0&&up>=0)break;
        }
        CHECK(n->nlte_line_global[2*(size_t)line]==lo&&n->nlte_line_global[2*(size_t)line+1]==up);
        if(lo>=0&&up>=0)matched++;
        if(omega_cmfgen_enabled()){
            int tier=0;double ups=omega_cmfgen_line(a,line,5000,&tier);
            CHECK(tier>=1&&tier<=3&&isfinite(ups)&&ups>0);collision_tiers[tier]++;
            if(a->line_atomic_number[line]==14&&a->line_ion_number[line]==2&&
               a->line_level_lower[line]==19&&a->line_level_upper[line]==26){
                CHECK(lo>=0&&up>=0&&tier==2);
                double dE=fabs(a->level_energy_eV[up]-a->level_energy_eV[lo])*EV_TO_ERG;
                /* 봉인 toy 원자의 이 전이는5000K에서 다른-n gbar분기0.2다.
                 * SAME_N은별도미구현사항이며이번초기화회귀시험과혼합하지않는다. */
                double expected=47.972*0.2*a->line_f_lu[line]*a->level_g[lo]/(dE/(H_PLANCK*1e15));
                CHECK(expected>0.1&&fabs(ups/expected-1)<1e-12);collision_probe++;
            }
        }
    }
    if(omega_cmfgen_enabled()){
        CHECK(collision_tiers[1]>0&&collision_tiers[2]>0&&collision_tiers[3]>0&&collision_probe==1);
        printf("TOY_COLLISION_DISPATCH PASS tabulated=%zu strong=%zu weak=%zu probe_SiIII_19_26=1\n",
               collision_tiers[1],collision_tiers[2],collision_tiers[3]);
    }
    double old_lookup_seconds=(double)(clock()-lookup_start)/CLOCKS_PER_SEC;
    volatile size_t checksum=0;lookup_start=clock();
    for(int repeat=0;repeat<100;repeat++)for(int line=0;line<a->n_lines;line++)
        checksum+=(size_t)(n->nlte_line_global[2*(size_t)line]+n->nlte_line_global[2*(size_t)line+1]+2);
    printf("TOY_LINE_GLOBAL_CACHE PASS lines=%d matched=%zu old_search_one_pass_cpu_s=%.9g cached_100_passes_cpu_s=%.9g checksum=%zu\n",
        a->n_lines,matched,old_lookup_seconds,(double)(clock()-lookup_start)/CLOCKS_PER_SEC,(size_t)checksum);
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
    if(argc==6){
        int iteration=-1;char status[128];
        CHECK(!cmf_toy_initial_iterate_read(&m,argv[4],argv[5],&iteration));
        CHECK(iteration>=0&&!p->physical_history&&p->initial_steady_nlte);
        CHECK(n->radfield_view_status==RADIATION_FIELD_VIEW_DISABLED);
        CHECK(!n->radiation_field.field.generation.computed_generation);
        snprintf(status,sizeof status,"UNCONVERGED_INITIAL_ITERATE; completed_iterations=%d; physical_history=0",iteration);
        CHECK(!cmf_toy_model_write_material(&m,"iterate_material.csv",status));
        printf("TOY_ITERATE_INSPECTED completed_iterations=%d initial_NLTE_converged=0 radiation_recomputed=0 source_immutable=1\n",iteration);
    }
done:
    cmf_toy_model_free(&m);return rc;
}
