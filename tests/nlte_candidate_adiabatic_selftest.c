#include "nlte_population_candidate.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int fail(const char *message)
{
    fprintf(stderr, "[NLTE-CANDIDATE-ADIABATIC][FAIL] %s\n", message);
    return 1;
}

static int time_ledger_check(const NLTEPopulationCandidate *candidate)
{
    const size_t ns=2;double edges[2]={1e14,2e14},zero[2]={0,0};
    A208Validity chi_status[8];EmissivityStatus eta_status[6];
    for(int i=0;i<8;i++)chi_status[i]=A208_EXACT_ZERO;
    for(int i=0;i<6;i++)eta_status[i]=EMISS_EXACT_ZERO;
    CpuOpacityPublication op={0};CpuEmissivityPublication em={0};
    op.n_shells=ns;op.n_bins=1;op.generation_committed=9;op.population_generation=7;
    op.te_generation=6;op.radiation_generation=8;op.line_jbar_generation=8;op.tau_generation=9;
    op.frequency_edges=edges;op.chi_bf=zero;op.chi_ff=zero;op.chi_validity=chi_status;
    em.n_shells=ns;em.n_bins=1;em.committed_emissivity_generation=9;em.opacity_generation=9;
    em.population_generation=7;em.te_generation=6;em.radfield_generation=8;em.line_view_generation=8;
    em.nu_edge=edges;em.eta_bf=zero;em.eta_ff=zero;em.component_status=eta_status;
    A210LineNetShell lines[2]={{0},{0}};
    for(int s=0;s<2;s++)lines[s].status=LINE_NET_EXACT_ZERO;
    A210LineNetPublication line={0};line.n_shells=ns;line.shell=lines;
    line.population_generation=7;line.te_generation=6;line.tau_generation=9;
    line.opacity_generation=9;line.radiation_generation=8;
    A210TrialLedgerInput input={.opacity=&op,.emissivity=&em,.j_nu=zero,
        .temperature_K=candidate->trial_te,.electron_density_cm3=candidate->electron_density,
        .time_work=candidate->time_work,.line_net=&line,.n_shells=ns};
    A210TermLedger out[2],saved[2];
    if(a210_trial_ledger_build(&input,out)!=RADEQ_OK)return fail("time ledger build");
    for(int s=0;s<2;s++)
        if(out[s].adiabatic_model!=A210_MATERIAL_TIME_HISTORY ||
           out[s].residual!=-candidate->time_work[s].signed_total ||
           out[s].material_thermal_time!=candidate->time_work[s].thermal_time ||
           out[s].material_atomic_time!=candidate->time_work[s].atomic_time ||
           out[s].adiabatic_temperature_gradient!=0 ||
           out[s].adiabatic_internal_energy_gradient!=0)
            return fail("actual candidate work did not reach residual");
    memcpy(saved,out,sizeof(out));CmfgenAdiabaticCell steady[2]={{0},{0}};input.adiabatic=steady;
    if(a210_trial_ledger_build(&input,out)!=RADEQ_TERM_SCHEMA || memcmp(saved,out,sizeof(out)))
        return fail("double-counting steady and time work accepted");
    input.adiabatic=NULL;input.time_work=NULL;
    if(a210_trial_ledger_build(&input,out)!=RADEQ_TERM_SCHEMA || memcmp(saved,out,sizeof(out)))
        return fail("absent material work accepted");
    printf("[MATERIAL-HISTORY] candidate_to_residual=PASS residual0=%.17g residual1=%.17g\n",
           saved[0].residual,saved[1].residual);
    return 0;
}

typedef struct {
    CmfgenMaterialEnergyState old,now;
    double deposition[2];
} TimeRootFixture;

static RadeqStatus time_root_residual(const double *T,size_t n,A210TermLedger *out,void *opaque)
{
    TimeRootFixture *fixture=opaque;CmfgenMaterialTimeWork work[2];
    if(n!=2)return RADEQ_TERM_SCHEMA;
    CmfgenMaterialEnergyState current=fixture->now;current.temperature_K=T;
    if(cmfgen_material_time_work(&current,&fixture->old,work)!=CMFGEN_ADIABATIC_OK)
        return RADEQ_NONFINITE;
    for(size_t s=0;s<n;++s){
        A210TermLedger *l=&out[s];memset(l,0,sizeof(*l));l->equation_kind=A210_RE_INTEGRAL;
        for(int i=0;i<A210_NHEAT;++i)l->heating_status[i]=A210_EXACT_ZERO;
        for(int i=0;i<A210_NCOOL;++i)l->cooling_status[i]=A210_EXACT_ZERO;
        l->m_line=1;l->radiative_line_included=1;
        l->heating[A210_GAMMA]=fixture->deposition[s];l->heating_status[A210_GAMMA]=A210_INCLUDED;
        RadeqStatus status=a210_apply_material_time_work(l,&work[s]);
        if(status!=RADEQ_OK)return status;
        status=a210_line_owner_finalize(l);if(status!=RADEQ_OK)return status;
    }
    return RADEQ_OK;
}

static int time_root_check(void)
{
    double rho0[2]={1e-13,2e-13},n0[2]={1e9,2e9},e0[2]={1e9,2e9},T0[2]={1e4,1.2e4},u[2]={1e-11,1e-11};
    double rho[2],n[2],ne[2];const double t0=1e5,t=1.1e5,dt=t-t0,q=pow(t0/t,3);
    for(int s=0;s<2;s++){rho[s]=q*rho0[s];n[s]=q*n0[s];ne[s]=q*e0[s];}
    TimeRootFixture fixture={{2,t0,rho0,T0,n0,e0,u},{2,t,rho,T0,n,ne,u},{1e-6,3e-6}};
    double lower[2]={3500,3500},upper[2]={140000,140000},T[2]={0,0},outne[2]={0,0};
    ElectronTemperaturePublication publication={0};char hash[65];memset(hash,'a',64);hash[64]=0;
    int rc=a210_solve_vector_candidate(lower,upper,ne,2,1,1,hash,time_root_residual,
        &fixture,&publication,T,outne);
    if(rc)return fail("existing temperature root rejected time work");
    for(int s=0;s<2;s++){
        double C=1.5*(n[s]+ne[s])*K_BOLTZMANN;
        double expected=(fixture.deposition[s]+C*T0[s]/dt)/(C/dt*(1+2*log(t/fixture.old.epoch_s)));
        if(fabs(T[s]/expected-1)>1e-8 || publication.ledger[s].adiabatic_model!=A210_MATERIAL_TIME_HISTORY ||
           publication.committed_te_generation!=0)return fail("time root disagrees with analytic gas energy");
        printf("[MATERIAL-HISTORY] root_shell=%d T=%.17g analytic_T=%.17g e_balance=%.17g\n",
               s,T[s],expected,publication.ledger[s].e_balance);
    }
    a210_publication_free(&publication);return 0;
}

int main(void)
{
    enum { NS = 2, NI = 2, NL = 2 };
    AtomicData atom;
    NLTEConfig nlte;
    PlasmaState plasma;
    memset(&atom, 0, sizeof(atom));
    memset(&nlte, 0, sizeof(nlte));
    memset(&plasma, 0, sizeof(plasma));

    int ion_Z[NI] = {1, 1};
    int ion_stage[NI] = {0, 1};
    int level_offset[NI + 1] = {0, 2, 2};
    int level_Z[NL] = {1, 1};
    int level_ion[NL] = {0, 0};
    int level_g[NL] = {2, 4};
    double level_energy_eV[NL] = {5.0, 7.0};
    int topion_index[1] = {1};
    double topion_energy_cm[1] = {100.0};
    double topion_g[1] = {1.0};
    int ioniz_Z[1] = {1};
    int ioniz_stage[1] = {0};
    double ioniz_energy[1] = {13.6};
    double te[NS] = {10000.0, 12000.0};
    double ne[NS] = {1.0, 2.0};
    double ion_population[NI * NS] = {3.0, 2.0, 1.0, 2.0};
    double partition[NI * NS] = {0};
    double nlte_population[NL * NS] = {2.0, 1.0, 1.0, 1.0};
    double within[NL * NS] = {1.0, 1.0, 1.0, 1.0};
    int global_to_nlte[NL] = {0, 1};

    atom.n_ion_pops = NI;
    atom.n_levels = NL;
    atom.ion_pop_Z = ion_Z;
    atom.ion_pop_stage = ion_stage;
    atom.level_offset = level_offset;
    atom.level_Z = level_Z;
    atom.level_ion = level_ion;
    atom.level_g = level_g;
    atom.level_energy_eV = level_energy_eV;
    atom.topion_n = 1;
    atom.topion_ion_index = topion_index;
    atom.topion_E_cm = topion_energy_cm;
    atom.topion_g = topion_g;
    atom.n_ionization = 1;
    atom.ioniz_Z = ioniz_Z;
    atom.ioniz_ion = ioniz_stage;
    atom.ioniz_energy_eV = ioniz_energy;
    atom.ion_number_density = ion_population;
    atom.partition_functions = partition;
    atom.population_committed_generation = 5;

    nlte.n_nlte_levels_total = NL;
    nlte.global_to_nlte_level = global_to_nlte;
    nlte.nlte_level_populations = nlte_population;
    nlte.within_sl_frac = within;
    nlte.population_committed_generation = 5;
    plasma.n_shells = NS;
    plasma.T_e = te;
    plasma.n_electron = ne;
    plasma.T_e_generation = 4;

    PopulationAtomicView view = {
        NI, NL, level_offset, level_energy_eV, level_g, NULL,
        level_Z, level_ion, 1, topion_index, topion_energy_cm, topion_g
    };
    if (population_partition_build(
            &view, te, NS, 5, 4, partition, &atom.partition_stamp) != POP_OK)
        return fail("partition fixture build");

    nlte.channel_owned_recombination=1;
    nlte.exclude_empirical_heavy_ce=1;
    AtomicData atom_before = atom;
    NLTEConfig nlte_before = nlte;
    PlasmaState plasma_before = plasma;
    double ion_before[NI * NS], level_before[NL * NS], ne_before[NS];
    memcpy(ion_before, ion_population, sizeof(ion_before));
    memcpy(level_before, nlte_population, sizeof(level_before));
    memcpy(ne_before, ne, sizeof(ne_before));

    NLTEPopulationCandidate candidate;
    if (nlte_population_candidate_begin(
            &candidate, &nlte, &atom, &plasma, te, NS, 4, 5) !=
        NLTE_CANDIDATE_OK)
        return fail("candidate begin");
    if(!candidate.nlte.channel_owned_recombination||!candidate.nlte.exclude_empirical_heavy_ce)
        return fail("candidate lost atomic process ownership");
    double radius[NS] = {1.0e14, 2.0e14};
    double velocity[NS] = {1.0e8, 2.0e8};
    if (nlte_population_candidate_prepare_adiabatic(
            &candidate, radius, velocity, 1.0e6) != NLTE_CANDIDATE_OK ||
        !candidate.adiabatic_active)
        return fail("combined internal-energy/adiabatic producer");

    const double ev_to_erg = 1.602176634e-12;
    double expected_u0 = 15.6 * ev_to_erg / 4.0;
    double expected_u1 = 29.2 * ev_to_erg / 4.0;
    if (candidate.n_atom[0] != 4.0 || candidate.n_atom[1] != 4.0 ||
        fabs(candidate.internal_energy_atom[0] - expected_u0) >
            1.0e-14 * expected_u0 ||
        fabs(candidate.internal_energy_atom[1] - expected_u1) >
            1.0e-14 * expected_u1 ||
        !isfinite(candidate.adiabatic[0].signed_total) ||
        !isfinite(candidate.adiabatic[1].signed_total) ||
        candidate.adiabatic[0].signed_total !=
            candidate.adiabatic[0].cooling -
            candidate.adiabatic[0].heating)
        return fail("combined known answer/sign split");

    if (memcmp(&atom, &atom_before, sizeof(atom)) != 0 ||
        memcmp(&nlte, &nlte_before, sizeof(nlte)) != 0 ||
        memcmp(&plasma, &plasma_before, sizeof(plasma)) != 0 ||
        memcmp(ion_population, ion_before, sizeof(ion_before)) != 0 ||
        memcmp(nlte_population, level_before, sizeof(level_before)) != 0 ||
        memcmp(ne, ne_before, sizeof(ne_before)) != 0)
        return fail("combined candidate mutated public state");
    nlte_population_candidate_free(&candidate);

    if (nlte_population_candidate_begin(
            &candidate, &nlte, &atom, &plasma, te, NS, 4, 5) !=
            NLTE_CANDIDATE_OK)
        return fail("negative candidate begin");
    double bad_velocity[NS] = {1.0e8, 2.1e8};
    if (nlte_population_candidate_prepare_adiabatic(
            &candidate, radius, bad_velocity, 1.0e6) !=
            NLTE_CANDIDATE_ADIABATIC_FAILED || candidate.active ||
        candidate.n_atom || candidate.internal_energy_atom ||
        candidate.adiabatic)
        return fail("nonhomology did not fail atomically");
    nlte_population_candidate_free(&candidate);

    /* Capture physical epoch, then change the public material for the next
     * epoch. Neither it nor repeated candidates may alter the saved past. */
    double rho[NS]={4*AMU,4*AMU};plasma.rho=rho;
    nlte.nlte_to_global_level=global_to_nlte;
    NLTEPhysicalHistory history={0};
    if(nlte_physical_history_capture(&history,&nlte,&atom,&plasma,9e5)!=NLTE_CANDIDATE_OK)
        return fail("physical history capture");
    double saved_history[(5+NI+NL)*NS];
    memcpy(saved_history,history.storage,sizeof(saved_history));
    const double q=pow(.9,3);
    for(int s=0;s<NS;++s){rho[s]*=q;ne[s]*=q;te[s]*=1.1;}
    for(int i=0;i<NI*NS;++i)ion_population[i]*=q;
    for(int i=0;i<NL*NS;++i)nlte_population[i]*=q;
    atom.population_committed_generation=nlte.population_committed_generation=7;
    plasma.T_e_generation=6;plasma.physical_history=&history;
    if(population_partition_build(&view,te,NS,7,6,partition,&atom.partition_stamp)!=POP_OK)
        return fail("new epoch partition");
    for(int repeat=0;repeat<2;++repeat){
        if(nlte_population_candidate_begin(&candidate,&nlte,&atom,&plasma,te,NS,6,7)!=NLTE_CANDIDATE_OK ||
           nlte_population_candidate_prepare_adiabatic(&candidate,radius,velocity,1e6)!=NLTE_CANDIDATE_OK ||
           !candidate.time_work || candidate.adiabatic || !candidate.adiabatic_active)
            return fail("candidate did not select physical history");
        for(int s=0;s<NS;++s){
            double expected=1.5*(candidate.n_atom[s]+candidate.electron_density[s])*K_BOLTZMANN*
                (te[s]-history.energy.temperature_K[s])/1e5;
            if(fabs(candidate.time_work[s].thermal_time-expected)>1e-12*fabs(expected) ||
               fabs(candidate.time_work[s].atomic_time)>1e-12*fabs(expected))
                return fail("time work used live state instead of frozen past");
        }
        if(time_ledger_check(&candidate))return 1;
        nlte_population_candidate_free(&candidate);
    }
    if(memcmp(saved_history,history.storage,sizeof(saved_history)) ||
       history.level_density==nlte.nlte_level_populations || history.ion_density==atom.ion_number_density)
        return fail("history mutated or aliased");
    rho[1]*=1.1;
    if(nlte_population_candidate_begin(&candidate,&nlte,&atom,&plasma,te,NS,6,7)!=NLTE_CANDIDATE_OK ||
       nlte_population_candidate_prepare_adiabatic(&candidate,radius,velocity,1e6)!=NLTE_CANDIDATE_ADIABATIC_FAILED ||
       candidate.time_work || candidate.adiabatic || candidate.active)
        return fail("inconsistent physical history did not fail atomically");
    nlte_population_candidate_free(&candidate);rho[1]/=1.1;
    te[0]=NAN;
    if(nlte_physical_history_capture(&history,&nlte,&atom,&plasma,1e6)==NLTE_CANDIDATE_OK ||
       memcmp(saved_history,history.storage,sizeof(saved_history)))
        return fail("failed new capture destroyed past");
    plasma.physical_history=NULL;nlte_physical_history_free(&history);

    if(time_root_check())return 1;
    printf("[NLTE-CANDIDATE-ADIABATIC][SELFTEST] status=PASS "
           "internal_energy=NEUTRAL_GROUND vector=CMFGEN_COMPLETE "
           "sign=SIGNED nonhomology=FAIL_CLOSED public_mutations=0\n");
    return 0;
}
