#include "cmf_toy_model.h"
#include "cmf_nonthermal_binding.h"
#include <assert.h>
#include <stdio.h>
#include <time.h>
/* 실제 2일 Saha predictor/원자 자료의 여기 연결·경계 민감도 시험.
 * 전리 경로를 포함하되 실제 침적·공동 NLTE 해는 사용하지 않으므로 SN 결과가 아니다. */
int main(int argc,char **argv){
    if(argc!=5)return 2;
    const double masses[6]={55.942132,55.939839,55.934937,40.078,32.065,28.0855};
    CMFToyModel m={0};int ns=atoi(argv[3]);
    assert(!cmf_toy_model_prepare(&m,argv[1],argv[2],ns,masses));
    AtomicData *a=&m.atom;NLTEConfig n=m.nlte;n.nlte_level_populations=calloc(a->n_levels,sizeof(double));
    assert(n.nlte_level_populations);PlasmaState p={0};double electron;p.n_shells=1;p.n_electron=&electron;
    double deposition=1e-10;GammaDeposition g={0};g.n_shells=1;g.generation=1;
    g.provenance=GAMMA_PROVENANCE_NONLOCAL_MC;g.epoch=m.geometry.time_explosion;g.heating_rate=&deposition;
    CMFNonthermalAtomicData raw={0};assert(!cmf_nonthermal_atomic_read(&raw,"input/arnaud_rothenflug_23jun15"));
    double baselines[4][3]={{0}};int points[]={128,256,320,256,256,256},limits[]={20,20,20,40,20,20};
    for(int c=0;c<6;c++){
        int ne=points[c],extra=c==2?64:0;double *E=malloc((size_t)ne*sizeof(double));assert(E);
        for(int i=0;i<extra;i++)E[i]=exp(log(.1)+(double)i/extra*log(10));
        for(int i=extra;i<ne;i++)E[i]=exp((double)(i-extra)/(ne-extra-1)*log(2000));
        CMFNonthermalExcitations x={0};clock_t started=clock();
        assert(!cmf_nonthermal_excitation_build(a,E,ne,limits[c],&x));
        assert(x.view.n_channels>1000&&x.n_bethe>0&&x.n_omega>0&&x.n_subthermal>0);
        printf("NT_ACTUAL_EXCITATIONS case=%d lower_limit=%d energy_points=%d Emin=%.17g channels=%d Bethe=%d Omega=%d below_boundary=%d minimum_dE=%.17g sigma_bytes=%zu\n",
               c,limits[c],ne,E[0],x.view.n_channels,x.n_bethe,x.n_omega,x.n_subthermal,
               x.minimum_threshold_eV,(size_t)ne*x.view.n_channels*sizeof(double));fflush(stdout);
        CMFNonthermalResolvedModel bound={0};const CMFNonthermalBoundModel *view=&x.view;
        if(c>=4){
            assert(!cmf_nonthermal_binding_build_policy(a,&raw,&x,argv[4],c==4?CMF_NT_SIGMA_SAME_EXCESS:CMF_NT_SIGMA_SOURCE_KINETIC,&bound));view=&bound.view;
            printf("NT_RESOLVED_MODEL policy=%d ion_channels=%d resolved_routes=%d unresolved_routes=%d missing_ions=%d max_threshold_mismatch_eV=%.17g complete_atomic_model=0\n",
                   bound.cross_section_policy,bound.n_ion_channels,bound.n_resolved_routes,bound.n_unresolved_routes,bound.n_missing_ions,bound.max_threshold_shift_eV);
            /* 봉인한 두 실제 원자 덱의 연결표. 상단 이온을 바꾼 덱에 이전
             * full-level ID/경로 수를 재사용하지 않는다. 다른 덱은 별도 조사한다. */
            if(a->n_levels==19627&&a->n_ion_pops==27)
                assert(bound.n_resolved_routes==41&&bound.n_unresolved_routes==10&&bound.n_missing_ions==2);
            else{
                assert(a->n_levels==24542&&a->n_ion_pops==33);
                assert(bound.n_resolved_routes==51&&bound.n_unresolved_routes==13&&bound.n_missing_ions==3);
            }
        }
        int shells[]={0,ns/3,2*ns/3,ns-1};
        for(int j=0;j<4;j++){
            int s=shells[j];electron=m.plasma.n_electron[s];
            for(int k=0;k<a->n_levels;k++)n.nlte_level_populations[k]=m.nlte.nlte_level_populations[(size_t)k*ns+s];
            CMFNonthermalRates r={0};assert(!cmf_nonthermal_bound_evaluate(view,a,&n,&p,&g,&r));
            double fraction=0,Si=0,S=0,ion=0,Si2=0,Si3=0,S2=0,S3=0;
            for(int k=0;k<r.n_transitions;k++){
                int lo=r.transitions[k].lower_global,up=r.transitions[k].upper_global;
                double rate=r.rate_per_particle[k],pop=n.nlte_level_populations[n.global_to_nlte_level[lo]];
                assert(rate>=0&&isfinite(rate));
                (void)up;double delta;int electrons;
                assert(!cmf_nonthermal_transition_energy(a,&r.transitions[k],&delta,&electrons));
                double power=rate*pop*delta*1.602176634e-12;
                fraction+=power/deposition;
                if(a->level_Z[lo]==14)Si+=power/deposition;
                if(a->level_Z[lo]==16)S+=power/deposition;
                if(electrons){
                    ion+=power/deposition;double event=rate*pop*1.602176634e-12/deposition;
                    if(a->level_Z[lo]==14&&a->level_ion[lo]==1)Si2+=event;
                    if(a->level_Z[lo]==14&&a->level_ion[lo]==2)Si3+=event;
                    if(a->level_Z[lo]==16&&a->level_ion[lo]==1)S2+=event;
                    if(a->level_Z[lo]==16&&a->level_ion[lo]==2)S3+=event;
                }
            }
            assert(fraction>=0&&fraction<=1+1e-10);
            if(c==1){baselines[j][0]=fraction;baselines[j][1]=Si;baselines[j][2]=S;}
            printf("NT_ACTUAL_SAMPLE case=%d shell=%d T=%.17g ne=%.17g internal_fraction=%.17g Si_fraction=%.17g S_fraction=%.17g difference_from_excitation256=%.17g Si_difference=%.17g S_difference=%.17g initial_NLTE_converged=0 physical_gamma=0\n",
                   c,s,m.plasma.T_e[s],electron,fraction,Si,S,c>1?fraction-baselines[j][0]:0,
                   c>1?Si-baselines[j][1]:0,c>1?S-baselines[j][2]:0);fflush(stdout);
            if(c>=4)printf("NT_RESOLVED_SPLIT case=%d shell=%d heat=%.17g ion=%.17g excitation=%.17g SiII_events_per_eV=%.17g SiIII_events_per_eV=%.17g SII_events_per_eV=%.17g SIII_events_per_eV=%.17g\n",
                           c,s,1-fraction,ion,fraction-ion,Si2,Si3,S2,S3);
            cmf_nonthermal_rates_free(&r);
        }
        printf("NT_ACTUAL_EXCITATION_CPU case=%d seconds=%.6g\n",c,(double)(clock()-started)/CLOCKS_PER_SEC);
        cmf_nonthermal_binding_free(&bound);cmf_nonthermal_excitation_free(&x);free(E);
    }
    cmf_nonthermal_atomic_free(&raw);free(n.nlte_level_populations);cmf_toy_model_free(&m);
    puts("CMF_NONTHERMAL_ACTUAL_SELFTEST_OK resolved_subset=1 complete_atomic_model=0 spectrum_compared=0");return 0;
}
