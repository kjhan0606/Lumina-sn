/* 점값 BF를 실제 pair 행렬 조립기로 보내는 소형 시험. 물질 게시 없음. */
#include "lumina.h"
#include <assert.h>

int main(void)
{
    AtomicData a={0}; NLTEConfig n={0}; PlasmaState p={0}; OpacityState o={0};
    int z[4]={16,16,16,16}, ion[4]={2,2,3,3}, g[4]={1,3,2,8};
    int num[4]={0,1,0,1},super[4]={0,0,0,0},map[4]={0,1,2,3};
    int fl_sl[4]={0,0,1,1},anchor[2]={0,2};
    double energy[4]={0.,1.,0.,1.},fraction[4]={.4,.6,.2,.8},pop[4]={1.,2.,3.,4.};
    int iz[2]={16,16},ii[2]={2,3},off[3]={0,2,4};
    int elem_z[1]={16},elem_off[2]={0,2};
    double ie[2]={48.,70.},density[2]={3.,7.},partition[2]={4.,10.};
    double mass[1]={32.},abundance[1]={1.},rho[1]={32.*AMU*10.};
    double te[1]={18000.},ne[1]={1.e10};
    int roff[5]={0,1,1,1,1},target[1]={2},grid[1]={1};
    double prob[1]={.25},edge[1]={1.e18},sigma[1]={0.},saha[1]={.03};
    BfPointRates rates={8.,2.,1.,1.e-12,2.e-13,3.e-13};
    BfPointRateView view={.n_levels=4,.n_routes=1,.n_shells=1,
        .radiation_generation=5,.material_generation=9,.offset=roff,.target=target,
        .probability=prob,.threshold_Hz=edge,.sigma_edge=sigma,.phot_grid=grid,
        .saha_lower_per_upper=saha,.temperature=te,.electron_density=ne,.rates=&rates};
    a.n_levels=4; a.level_Z=z; a.level_ion=ion; a.level_g=g; a.level_num=num;
    a.level_super=super; a.level_energy_eV=energy;
    a.n_ionization=2; a.ioniz_Z=iz; a.ioniz_ion=ii; a.ioniz_energy_eV=ie;
    a.n_ion_pops=2; a.ion_pop_Z=iz; a.ion_pop_stage=ii; a.level_offset=off;
    a.ion_number_density=density; a.partition_functions=partition;
    a.n_elements=1; a.element_Z=elem_z; a.element_mass_amu=mass;
    a.abundances=abundance; a.elem_ion_offset=elem_off;
    a.population_committed_generation=9;
    n.n_nlte_ions=2; n.nlte_Z[0]=n.nlte_Z[1]=16;
    n.nlte_ion[0]=2; n.nlte_ion[1]=3;
    n.nlte_ion_level_offset[1]=2; n.nlte_ion_level_offset[2]=4;
    n.nlte_ion_super_offset[1]=1; n.nlte_ion_super_offset[2]=2;
    n.n_nlte_levels_total=4; n.n_super_total=2;
    n.nlte_to_global_level=map; n.global_to_nlte_level=map;
    n.fl_to_super=fl_sl; n.super_anchor_global=anchor;
    n.within_sl_frac=fraction; n.nlte_level_populations=pop;
    n.point_bf_rates=&view; n.radfield_view.generation=5;
    n.radfield_view_status=RADIATION_FIELD_VIEW_OK;
    n.nu_min=1.e15; n.nu_max=2.e15; n.n_freq_bins=1; n.d_log_nu=log(2.);
    p.n_shells=1; p.T_e=te; p.n_electron=ne; p.rho=rho;
    GammaDeposition gamma={0}; double heating[1]={0.};
    gamma_deposition_init(&gamma,1);
    assert(!gamma_deposition_publish(&gamma,GAMMA_PROVENANCE_EXTERNAL_FILE,
                                     1.e6,NULL,NULL,NULL,heating));
    double matrix[4]={0},rhs[2]={0},baseline[4]={0},baseline_rhs[2]={0};
    BfPointRates saved_rates=rates;
    memset(&rates,0,sizeof(rates));
    assert(!nlte_assemble_rate_matrix(&n,&a,&p,&o,0,1,0,1.e6,
                                     baseline,baseline_rhs,2,&gamma,NULL,-1));
    rates=saved_rates;
    int rc=nlte_assemble_rate_matrix(&n,&a,&p,&o,0,1,0,1.e6,
                                     matrix,rhs,2,&gamma,NULL,-1);
    assert(!rc);
    /* 마지막 행은 보존식. 첫 행은 순수 물리율로 남는다.
     * lower FL fraction만 적용; upper-SL fraction/Saha 중복 금지.
     * 문턱이 기존 bin 범위 밖이어도 점값 율은 소비한다. */
    assert(fabs(matrix[0]-baseline[0]+.25*8.*.4)<1.e-14);
    assert(fabs(matrix[2]-baseline[2]-.25*(2.+1.))<1.e-14);
    assert(matrix[1]==1. && matrix[3]==1.);
    assert(!memcmp(pop,(double[]){1.,2.,3.,4.},sizeof(pop)));
    /* 동일 PHOT grid 선택·점유확률 포함 Saha 비로 CI/C3를 한 번만 추가. */
    assert(!setenv("LUMINA_NLTE_BF_COLLISION","1",1));
    grid[0]=0; edge[0]=3.e15; sigma[0]=7.e-18;
    double u=H_PLANCK*edge[0]/(K_BOLTZMANN*te[0]);
    double ci=ne[0]*1.55e13/sqrt(te[0])*.3*sigma[0]*exp(-u)/u;
    memset(matrix,0,sizeof(matrix)); memset(rhs,0,sizeof(rhs));
    assert(!nlte_assemble_rate_matrix(&n,&a,&p,&o,0,1,0,1.e6,
                                     matrix,rhs,2,&gamma,NULL,-1));
    assert(fabs(matrix[0]-baseline[0]+prob[0]*(rates.photo+ci)*fraction[0])<1.e-14);
    assert(fabs(matrix[2]-baseline[2]-prob[0]*(rates.spontaneous+rates.stimulated+ci*saha[0]))<1.e-14);
    grid[0]=1; /* 다음 이온이 존재하므로 이 PHOT grid의 CI는 제외된다. */
    memset(matrix,0,sizeof(matrix)); memset(rhs,0,sizeof(rhs));
    assert(!nlte_assemble_rate_matrix(&n,&a,&p,&o,0,1,0,1.e6,
                                     matrix,rhs,2,&gamma,NULL,-1));
    assert(fabs(matrix[0]-baseline[0]+prob[0]*rates.photo*fraction[0])<1.e-14);
    assert(fabs(matrix[2]-baseline[2]-prob[0]*(rates.spontaneous+rates.stimulated))<1.e-14);
    n.radfield_view.generation=6;
    memset(matrix,0,sizeof(matrix)); memset(rhs,0,sizeof(rhs));
    assert(nlte_assemble_rate_matrix(&n,&a,&p,&o,0,1,0,1.e6,
                                     matrix,rhs,2,&gamma,NULL,-1)!=0);
    n.radfield_view.generation=5; target[0]=-1;
    memset(matrix,0,sizeof(matrix)); memset(rhs,0,sizeof(rhs));
    assert(nlte_assemble_rate_matrix(&n,&a,&p,&o,0,1,0,1.e6,
                                     matrix,rhs,2,&gamma,NULL,-1)!=0);
    gamma_deposition_free(&gamma);
    puts("BF_POINT_MATRIX_SELFTEST PASS (production pair; no publication)");
    return 0;
}
