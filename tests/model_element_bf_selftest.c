/* 실제 전모형 조립 API의 terminal/율 보존 시험. 작은 합성 generator다. */
#include "nlte_model_se.h"
#include "nlte_model_bb.h"
#include <assert.h>

int main(void)
{
    AtomicData a={0};NLTEConfig n={0};PlasmaState p={0};
    int z[4]={16,16,16,16},q[4]={2,2,3,3},map[4]={0,1,2,3};
    int sl[4]={0,0,1,1},anchor[2]={0,2};
    int iz[3]={16,16,16},iq[3]={2,3,4},off[4]={0,2,4,4},ez[1]={16};
    double pop[4]={1.,2.,3.,4.},ions[3]={3.,7.,1.};
    double fraction[4]={.4,.6,.2,.8},te[1]={18000.},ne[1]={1.e10},total=11.;
    int csr[5]={0,1,1,2,2},target[2]={2,-1},terminal[2]={-1,2},grid[2]={1,1};
    double prob[2]={.25,.5},edge[2]={3.e15,3.e15},sigma[2]={7.e-18,7.e-18},saha[2]={.03,.02};
    BfPointRates rates[2]={{8.,2.,1.,-1.e-12,2.e-13,3.e-13},
                           {12.,3.,2.,1.e-12,3.e-13,4.e-13}};
    BfPointRateView view={.n_levels=4,.n_routes=2,.n_shells=1,
        .radiation_generation=5,.material_generation=9,.offset=csr,.target=target,
        .terminal_ion=terminal,.probability=prob,.threshold_Hz=edge,.sigma_edge=sigma,
        .phot_grid=grid,.saha_lower_per_upper=saha,.temperature=te,.electron_density=ne,.rates=rates};
    a.n_levels=4;a.n_ion_pops=3;a.level_Z=z;a.level_ion=q;a.level_offset=off;
    a.ion_pop_Z=iz;a.ion_pop_stage=iq;a.ion_number_density=ions;a.n_elements=1;a.element_Z=ez;
    a.element_density=(PopulationElementDensityView){POP_ELEMENT_TOTAL_EXPLICIT_NUMBER,1,1,&total};
    a.population_committed_generation=9;
    n.model_atom_projection=1;n.n_nlte_levels_total=4;n.n_super_total=2;
    n.population_committed_generation=9;n.nlte_to_global_level=map;n.global_to_nlte_level=map;
    n.fl_to_super=sl;n.super_anchor_global=anchor;n.within_sl_frac=fraction;
    n.nlte_level_populations=pop;n.point_bf_rates=&view;
    n.radfield_view_status=RADIATION_FIELD_VIEW_OK;n.radfield_view.generation=5;
    p.n_shells=1;p.T_e=te;p.n_electron=ne;
    NLTEModelElement element={0};
    assert(!nlte_model_element_init(&element,&n,&a,&p,16,0));
    assert(element.n_variables==3 && element.n_bound==2 && element.n_terminals==1);
    assert(element.initial_population[0]==3. && element.initial_population[1]==7. &&
           element.initial_population[2]==1. && element.total_nuclei==11.);
    assert(!setenv("LUMINA_NLTE_BF_COLLISION","0",1));
    NLTEModelBFBlock block={0};
    assert(!nlte_model_element_bf_assemble(&block,&element,&n,&a,&p));
    assert(block.routes==2 && block.terminal_routes==1 && block.n_variables==3);
    assert(fabs(block.radiative[1]-.8)<1.e-14);
    assert(fabs(block.radiative[3]-.75)<1.e-14);
    assert(fabs(block.radiative[5]-1.2)<1.e-14);
    assert(fabs(block.radiative[7]-2.5)<1.e-14);
    assert(fabs(block.radiative_heat[0]-(-.25e-12*.4))<1.e-27);
    assert(fabs(block.radiative_heat[1]-(-.25*5.e-13+.5e-12*.2))<1.e-27);
    assert(fabs(block.radiative_heat[2]-(-.5*7.e-13))<1.e-27);
    for(int j=0;j<3;j++){
        double sum=0.;for(int i=0;i<3;i++)sum+=block.radiative[j*3+i];
        assert(fabs(sum)<1.e-14);
    }
    double solution[3];PopulationGeneratorSolveDiagnostic diag;
    assert(population_generator_stationary_gth(block.radiative,3,11.,solution,&diag)==POP_OK);
    assert(diag.generator_recognized && diag.exact_generator_componentwise_residual<1.e-12);
    assert(fabs(solution[1]/solution[0]-.8/.75)<1.e-12);
    assert(fabs(solution[2]/solution[1]-1.2/2.5)<1.e-12);
    nlte_model_bf_block_free(&block);
    int sl_bb[4]={0,1,2,2},anchor_bb[3]={0,1,2};
    double bb_frac[4]={1.,1.,.2,.8};
    NLTEConfig bb_n=n;bb_n.fl_to_super=sl_bb;bb_n.super_anchor_global=anchor_bb;
    bb_n.n_super_total=3;bb_n.within_sl_frac=bb_frac;
    NLTEModelElement bb_element={0};
    assert(!nlte_model_element_init(&bb_element,&bb_n,&a,&p,16,0));
    int32_t bb_lo[2]={2,0},bb_up[2]={3,1};
    double bb_nu[2]={1.e15,2.e15},bb_abs[2]={5.,7.},bb_stim[2]={1.,2.},bb_spont[2]={2.,3.};
    uint8_t bb_status[2]={1,1};
    NLTEModelBBView bb_view={.n_levels=4,.n_lines=2,.n_shells=1,.native_shell=0,
        .radiation_generation=5,.material_generation=9,.lower=bb_lo,.upper=bb_up,
        .frequency_Hz=bb_nu,.absorption=bb_abs,.stimulated=bb_stim,
        .spontaneous=bb_spont,.status=bb_status};
    NLTEModelBBBlock bb={0};
    assert(!nlte_model_element_bb_assemble(&bb,&bb_element,&bb_n,&a,&p,&bb_view));
    assert(bb.lines==2 && bb.intra_super_lines==1 && bb.n_variables==4);
    assert(fabs(bb.radiative[1]-7.)<1.e-14 && fabs(bb.radiative[4]-5.)<1.e-14);
    assert(fabs(bb.radiative[0]+7.)<1.e-14 && fabs(bb.radiative[5]+5.)<1.e-14);
    assert(fabs(bb.radiative_heat[0]-6.62607015e-12*14.)<1.e-25);
    assert(fabs(bb.radiative_heat[1]+6.62607015e-12*10.)<1.e-25);
    assert(fabs(bb.radiative_heat[2]+6.62607015e-12*1.4)<1.e-25);
    nlte_model_bb_block_free(&bb);
    bb_status[1]=3;
    assert(nlte_model_element_bb_assemble(&bb,&bb_element,&bb_n,&a,&p,&bb_view)!=0 &&
           !bb.n_variables && !bb.radiative);
    bb_status[1]=2;
    assert(nlte_model_element_bb_assemble(&bb,&bb_element,&bb_n,&a,&p,&bb_view)!=0);
    bb_abs[1]=0.;bb_stim[1]=0.;
    assert(!nlte_model_element_bb_assemble(&bb,&bb_element,&bb_n,&a,&p,&bb_view));
    assert(bb.radiative[1]==0. && bb.radiative[4]==3.);
    nlte_model_bb_block_free(&bb);
    bb_abs[1]=7.;bb_stim[1]=2.;bb_status[1]=1;
    bb_view.radiation_generation=4;
    assert(nlte_model_element_bb_assemble(&bb,&bb_element,&bb_n,&a,&p,&bb_view)!=0);
    bb_view.radiation_generation=5;bb_view.native_shell=1;
    assert(nlte_model_element_bb_assemble(&bb,&bb_element,&bb_n,&a,&p,&bb_view)!=0);
    bb_view.native_shell=0;
    nlte_model_element_free(&bb_element);
    assert(!setenv("LUMINA_NLTE_BF_COLLISION","1",1));
    assert(!nlte_model_element_bf_assemble(&block,&element,&n,&a,&p));
    /* bound upper가 있으면 grid1 CI 제외; terminal이면 grid1도 포함. */
    assert(block.collisional[1]==0. && block.collisional[5]>0. && block.collisional[7]>0.);
    assert(fabs(block.collisional[7]/block.collisional[5]-saha[1]/fraction[2])<1.e-12);
    nlte_model_bf_block_free(&block);
    BfPointTransition tr={0},before=tr;terminal[1]=1;
    assert(nlte_point_bf_transition(&n,&a,&p,2,1,0,&tr)!=0 && !memcmp(&before,&tr,sizeof(tr)));
    assert(nlte_model_element_bf_assemble(&block,&element,&n,&a,&p)!=0 && !block.n_variables && !block.radiative);
    terminal[1]=2;
    /* catalog의 무관한 수는 scalar inverse를 임의로 재정규화하지 않는다. */
    int top_ip=2;double top_e=0.,top_g=80.;
    a.topion_n=1;a.topion_ion_index=&top_ip;a.topion_E_cm=&top_e;a.topion_g=&top_g;
    assert(!nlte_point_bf_transition(&n,&a,&p,2,1,0,&tr));
    assert(tr.target_level==-1 && tr.target_ion==2 && tr.radiative.spontaneous==1.5);
    assert(!memcmp(pop,(double[]){1.,2.,3.,4.},sizeof(pop)) && ions[2]==1.);
    nlte_model_element_free(&element);nlte_model_element_free(&element);
    puts("MODEL_ELEMENT_BF_SELFTEST PASS terminal_scalar=OWNED BF_CI_C3_heat=PASS GTH=EXISTING_SYNTHETIC_ONLY public_mutations=0 full_SE_RE_solved=0");
    return 0;
}
