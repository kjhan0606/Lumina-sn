#include "lumina.h"
#include "population_contract.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int nlte_ew_test_solve_chain4(const double ion_rate[3],
                              const double recombination_rate[3],
                              double total, double population[4],
                              double *closure_residual, int bb_owner_count[4]);
int nlte_ew_test_terminal_scalar_rates(int Z, double terminal_g,
                                       double rates_out[4]);
int nlte_ew_test_grouped_superlevel_rates(
    int version, double upper_delta_eV, double rates_out[4]);
int nlte_ew_test_commit_fallback_blocks(
    int Z, int selected, int commit_enabled, int owner_status);
double bf_rate_sigma_log_weight(double sigma_bin_average, double nu_lo,
                                double nu_hi, double nu_thresh);
double nlte_bf_cmfgen_collisional_ionization_rate(
    double T_e, double n_e, double threshold_eV,
    double sigma_edge_cm2, int ion_number);
int nlte_ew_test_bf_collisional_policy(const char *override,
                                       int parity_enabled);
int nlte_ew_test_bf_ci_route_included(int phot_grid,
                                      int upper_ion_present);
int nlte_bf_collisional_enabled(void);

static double rel_error(double actual, double expected) {
    return fabs(actual - expected) / fmax(fabs(expected), 1.0e-300);
}

static int terminal_scalar_rate_test(int Z, double target_g) {
    const double nu_min = 1.0e15, dln = log(2.0), nu_threshold = 2.5e15;
    const double sigma[3] = {2.0e-18, 4.0e-18, 1.0e-18};
    const double T_e = 15000.0, n_e = 1.0e10;
    double rates[4] = {0.0, 0.0, 0.0, 0.0};
    if (nlte_ew_test_terminal_scalar_rates(Z, target_g, rates) != 0)
        return -1;

    double gamma = 0.0, milne = 0.0;
    int active_bins = 0;
    for (int b = 0; b < 3; ++b) {
        double nu_lo = nu_min * exp(b * dln);
        double nu_hi = nu_min * exp((b + 1) * dln);
        double weight = bf_rate_sigma_log_weight(
            sigma[b], nu_lo, nu_hi, nu_threshold);
        if (!(weight > 0.0)) continue;
        active_bins++;
        double nu_c = sqrt(nu_lo * nu_hi);
        double nu_milne = (nu_c < nu_threshold && nu_hi > nu_threshold)
            ? 0.5 * (nu_threshold + nu_hi) : nu_c;
        double x = H_PLANCK * nu_milne / (K_BOLTZMANN * T_e);
        double spont = 2.0 * H_PLANCK * nu_milne * nu_milne * nu_milne /
                       (C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT);
        double B = spont / expm1(x);
        double pref = 4.0 * M_PI_VAL / H_PLANCK * weight;
        gamma += pref * B;
        milne += pref * (spont + B) * exp(-x);
    }
    double chi_eV = H_PLANCK * nu_threshold / EV_TO_ERG;
    double kT = K_BOLTZMANN * T_e;
    double log_nstar = log(n_e) + 1.5 * log(H_PLANCK * H_PLANCK /
        (2.0 * M_PI_VAL * M_ELECTRON * kT)) +
        log(2.0 / (2.0 * target_g)) + chi_eV * EV_TO_ERG / kT;
    double recombination = exp(log_nstar) * milne;
    double coll_ion = nlte_bf_cmfgen_collisional_ionization_rate(
        T_e, n_e, chi_eV, 7.0e-18, 4);
    double coll_recombination = exp(log_nstar) * coll_ion;
    if (active_bins != 2 ||
        bf_rate_sigma_log_weight(sigma[0], nu_min, nu_min * exp(dln),
                                 nu_threshold) != 0.0 ||
        !(gamma > 0.0) || !(recombination > 0.0) ||
        !(coll_ion > 0.0) || !(coll_recombination > 0.0) ||
        rel_error(rates[0], gamma) > 2.0e-12 ||
        rel_error(rates[1], recombination) > 2.0e-12 ||
        rel_error(rates[2], coll_ion) > 2.0e-12 ||
        rel_error(rates[3], coll_recombination) > 2.0e-12) {
        fprintf(stderr,
                "terminal scalar rate mismatch Z=%d g=%.1f got=(%.17g,%.17g) "
                "expected=(%.17g,%.17g) active_bins=%d\n",
                Z, target_g, rates[0], rates[1], gamma, recombination,
                active_bins);
        return -1;
    }
    return 0;
}

static int cmfgen_collisional_ionization_rate_test(void) {
    const double T_e = 18000.0, n_e = 2.5e10;
    const double threshold_eV = 25.3, sigma_cm2 = 3.2e-18;
    const double lower_g = 4.0;
    double edge_15 = threshold_eV * EV_TO_ERG / H_PLANCK / 1.0e15;
    double T_4 = T_e / 1.0e4;
    double u = H_PLANCK * edge_15 * 1.0e15 /
               (K_BOLTZMANN * T_e);
    for (int ion_number = 0; ion_number <= 4; ++ion_number) {
        double zion = (double)ion_number + 1.0;
        double phot_const = 0.1 * fmin(zion, 3.0) * 3.23 / 8.63e-8;
        double phot_cross_scaled = sigma_cm2 * 1.0e10;
        double omega_diag = phot_const * phot_cross_scaled * lower_g *
                            T_4 / edge_15;
        double cmfgen_rate = 8.63e-8 * n_e * omega_diag / lower_g /
                             sqrt(T_4) * exp(-u);
        double lumina_rate = nlte_bf_cmfgen_collisional_ionization_rate(
            T_e, n_e, threshold_eV, sigma_cm2, ion_number);
        if (!(cmfgen_rate > 0.0) || !(lumina_rate > 0.0) ||
            rel_error(lumina_rate, cmfgen_rate) > 1.0e-3) {
            fprintf(stderr,
                    "CMFGEN collisional ionization mismatch ion=%d "
                    "Lumina=%.17g CMFGEN=%.17g rel=%.3e\n",
                    ion_number, lumina_rate, cmfgen_rate,
                    rel_error(lumina_rate, cmfgen_rate));
            return -1;
        }
    }
    if (nlte_ew_test_bf_collisional_policy("1", 0) != 1 ||
        nlte_ew_test_bf_collisional_policy("0", 1) != 0 ||
        nlte_ew_test_bf_collisional_policy(NULL, 1) != 1 ||
        nlte_ew_test_bf_collisional_policy(NULL, 0) != 0)
        return -1;

    /* CMFGEN PHOT_ID=1 selects only the first active upper-ion route; when
     * the upper ion is absent it folds every route into its ground reservoir. */
    {
        const int grids[] = {0, 1, 1};
        const double probability[] = {1.0, 0.2, 0.8};
        double active_sum = 0.0, terminal_sum = 0.0;
        for (int r = 0; r < 3; ++r) {
            if (nlte_ew_test_bf_ci_route_included(grids[r], 1))
                active_sum += probability[r];
            if (nlte_ew_test_bf_ci_route_included(grids[r], 0))
                terminal_sum += probability[r];
        }
        if (nlte_ew_test_bf_ci_route_included(-1, 0) ||
            fabs(active_sum - 1.0) > 1e-14 ||
            fabs(terminal_sum - 2.0) > 1e-14)
            return -1;
    }
    return 0;
}

static int four_stage_charge_closure_test(void) {
    const double amu = 1.660539066e-24;
    int ion_Z[] = {14,14,14,14,14,14};
    int ion_stage[] = {0,1,2,3,4,5};
    int element_Z[] = {14};
    int elem_offset[] = {0,6};
    double element_mass[] = {28.0};
    double abundance[] = {1.0};
    double rho[] = {28.0 * amu * 100.0};
    int nlte_Z[] = {14,14,14,14};
    int nlte_stage[] = {1,2,3,4};
    int level_offset[] = {0,1,2,3,4};
    double levels[] = {10.0,20.0,20.0,30.0};
    double ions[] = {0.0,1.0,2.0,3.0,4.0,20.0};
    double ne[] = {200.0};
    double residual = -1.0;
    int worst_shell = -1;

    if (population_nlte_stage_charge_close(
            6,ion_Z,ion_stage,1,element_Z,elem_offset,element_mass,
            abundance,rho,amu,4,nlte_Z,nlte_stage,level_offset,4,1,
            levels,ions,ne,1.0,&residual,&worst_shell) != POP_OK)
        return -1;
    if (ions[0] != 0.0 || ions[1] != 10.0 || ions[2] != 20.0 ||
        ions[3] != 20.0 || ions[4] != 30.0 || ions[5] != 20.0 ||
        ne[0] != 330.0 || residual != 0.0 || worst_shell != -1)
        return -1;
    return 0;
}

static int commit_fallback_policy_test(void) {
    if (!nlte_ew_test_commit_fallback_blocks(14, 1, 1, 0) ||
        !nlte_ew_test_commit_fallback_blocks(16, 1, 1, 0) ||
        nlte_ew_test_commit_fallback_blocks(14, 1, 1, 1) ||
        nlte_ew_test_commit_fallback_blocks(14, 1, 0, 0) ||
        nlte_ew_test_commit_fallback_blocks(14, 0, 1, 0) ||
        nlte_ew_test_commit_fallback_blocks(26, 1, 1, 0))
        return -1;
    return 0;
}

/* Exercise the public production owner with a compact, fully connected atom.
 * This intentionally goes through ew_run_impl -> nlte_assemble_rate_matrix,
 * including both pair BF channels and the PHOT-declared terminal scalar. */
static int production_owner_transaction_test(int Z, double element_mass_amu,
                                              double terminal_g,
                                              int corrupt_route,
                                              int channel_version) {
    enum { NLEV = 8, NIONPOP = 6, NLINES = 4, NFREQ = 5, NSHELL = 1 };
    const double amu = 1.660539066e-24;
    const double epoch = 1.0;
    const double n_element = 1.0e10;
    const double nu_min = 1.0e13, dlognu = 1.5;
    int ion_pop_Z[NIONPOP], ion_pop_stage[NIONPOP];
    int level_offset[NIONPOP + 1] = {0,0,2,4,6,8,8};
    double ion_density[NIONPOP];
    int level_Z[NLEV], level_ion[NLEV], level_num[NLEV], level_g[NLEV];
    int level_super[NLEV], level_metastable[NLEV] = {0}, has_sigma[NLEV];
    double level_energy[NLEV];
    int ioniz_Z[4], ioniz_stage[4];
    double ioniz_energy[4] = {8.0,6.0,4.0,3.0};
    int element_Z[1] = {0}, elem_ion_offset[2] = {0,NIONPOP};
    double element_mass[1], abundance[NSHELL] = {1.0};
    int line_Z[NLINES], line_stage[NLINES], line_lower[NLINES], line_upper[NLINES];
    double line_f[NLINES], line_lambda[NLINES], line_A[NLINES];
    double line_Blu[NLINES], line_Bul[NLINES], line_nu[NLINES];
    int nlte_to_global[NLEV], global_to_nlte[NLEV], nlte_line_map[NLINES];
    int fl_to_super[NLEV], super_anchor[NLEV] = {0,1,2,4,5,6,7,-1};
    static const int super_offsets[5] = {0,2,3,5,7};
    static const int full_to_super[NLEV] = {0,1,2,2,3,4,5,6};
    double within_sl[NLEV], populations[NLEV], j_nu[NFREQ] = {0};
    double cmfgen_sigma[NLEV * NFREQ];
    int channel_offset[NLEV + 1], channel_target[NLEV], channel_grid[NLEV];
    double channel_probability[NLEV], channel_target_energy[NLEV];
    double channel_target_g[NLEV], channel_sigma_edge[NLEV];
    double channel_sigma[NLEV * NFREQ];
    int ma_rr_offset[NLEV + 1], ma_rr_target[NLEV - 2];
    double ma_rr_probability[NLEV - 2];
    double T_e[NSHELL] = {15000.0}, n_e[NSHELL] = {1.0e8};
    double rho[NSHELL] = {0};
    const double initial_populations[NLEV] = {
        0.5e9,0.5e9,0.5e9,0.5e9,0.5e9,0.5e9,0.5e9,0.5e9
    };
    const double initial_ion_density[NIONPOP] = {
        0.0,2.0e9,2.0e9,2.0e9,2.0e9,2.0e9
    };
    NLTEConfig nlte;
    AtomicData atom;
    PlasmaState plasma;
    OpacityState opacity;
    GammaDeposition gamma;
    double external_heating[NSHELL] = {0.0};
    int verdict_pass = -1;
    int rc;

    memset(&nlte, 0, sizeof(nlte));
    memset(&atom, 0, sizeof(atom));
    memset(&plasma, 0, sizeof(plasma));
    memset(&opacity, 0, sizeof(opacity));
    memset(&gamma, 0, sizeof(gamma));
    memcpy(populations, initial_populations, sizeof(populations));
    memcpy(ion_density, initial_ion_density, sizeof(ion_density));

    for (int ip = 0; ip < NIONPOP; ++ip) {
        ion_pop_Z[ip] = Z;
        ion_pop_stage[ip] = ip;
    }
    for (int g = 0; g < NLEV; ++g) {
        int stage = g / 2 + 1;
        int within = g % 2;
        level_Z[g] = Z;
        level_ion[g] = stage;
        level_num[g] = within;
        level_g[g] = within ? 4 : 2;
        level_super[g] = stage == 2 ? 0 : within;
        level_energy[g] = within ? 1.0 : 0.0;
        has_sigma[g] = 1;
        nlte_to_global[g] = g;
        global_to_nlte[g] = g;
        fl_to_super[g] = full_to_super[g];
        within_sl[g] = stage == 2 ? (within ? 0.75 : 0.25) : 1.0;
        channel_offset[g] = g;
        channel_probability[g] = 1.0;
        channel_grid[g] = 0;
        channel_target_energy[g] = 0.0;
        channel_sigma_edge[g] = 1.0e-18;
        for (int b = 0; b < NFREQ; ++b) {
            cmfgen_sigma[g * NFREQ + b] = 1.0e-18;
            channel_sigma[g * NFREQ + b] = 1.0e-18;
        }
        if (stage < 4) {
            channel_target[g] = stage * 2;
            /* Stage-I routes land in one stage-II superlevel with g=2+4. */
            channel_target_g[g] = stage == 1 ? 6.0 : 2.0;
        } else {
            channel_target[g] = -1;
            channel_target_g[g] = terminal_g;
        }
    }
    channel_offset[NLEV] = NLEV;
    for (int ip = 0; ip < 4; ++ip) {
        ioniz_Z[ip] = Z;
        ioniz_stage[ip] = ip + 1;
    }
    for (int line = 0; line < NLINES; ++line) {
        int stage = line + 1;
        double nu = EV_TO_ERG / H_PLANCK;
        double lambda_A = C_SPEED_OF_LIGHT / nu * 1.0e8;
        double f_lu = 1.0e-8;
        double A_ul = 2.02613e18 * f_lu / (4.0 * lambda_A * lambda_A);
        line_Z[line] = Z;
        line_stage[line] = stage;
        line_lower[line] = 0;
        line_upper[line] = 1;
        line_f[line] = f_lu;
        line_lambda[line] = C_SPEED_OF_LIGHT / nu;
        line_A[line] = A_ul;
        line_Bul[line] = A_ul * C_SPEED_OF_LIGHT * C_SPEED_OF_LIGHT /
                         (2.0 * H_PLANCK * nu * nu * nu);
        line_Blu[line] = 2.0 * line_Bul[line];
        line_nu[line] = nu;
        nlte_line_map[line] = line;
    }
    for (int g = 0; g <= NLEV; ++g) {
        ma_rr_offset[g] = g < NLEV - 2 ? g : NLEV - 2;
    }
    for (int g = 0; g < NLEV - 2; ++g) {
        int stage = g / 2 + 1;
        ma_rr_target[g] = stage * 2;
        ma_rr_probability[g] = 1.0;
    }

    element_Z[0] = Z;
    element_mass[0] = element_mass_amu;
    rho[0] = element_mass_amu * amu * n_element;
    atom.n_levels = NLEV;
    atom.level_Z = level_Z;
    atom.level_ion = level_ion;
    atom.level_num = level_num;
    atom.level_g = level_g;
    atom.level_super = level_super;
    atom.level_metastable = level_metastable;
    atom.level_energy_eV = level_energy;
    atom.n_ionization = 4;
    atom.ioniz_Z = ioniz_Z;
    atom.ioniz_ion = ioniz_stage;
    atom.ioniz_energy_eV = ioniz_energy;
    atom.n_ion_pops = NIONPOP;
    atom.ion_pop_Z = ion_pop_Z;
    atom.ion_pop_stage = ion_pop_stage;
    atom.level_offset = level_offset;
    atom.ion_number_density = ion_density;
    atom.n_elements = 1;
    atom.element_Z = element_Z;
    atom.element_mass_amu = element_mass;
    atom.abundances = abundance;
    atom.elem_ion_offset = elem_ion_offset;
    atom.cmfgen_loaded = 1;
    atom.cmfgen_n_freq_bins = NFREQ;
    atom.cmfgen_nu_min = nu_min;
    atom.cmfgen_nu_max = nu_min * exp(NFREQ * dlognu);
    atom.cmfgen_has_sigma = has_sigma;
    atom.cmfgen_sigma_bf = cmfgen_sigma;
    atom.cmfgen_bf_channels_loaded = 1;
    atom.cmfgen_bf_channel_version = channel_version;
    if(channel_version>=5)
        for(int c=0;c<4;c++)atom.cmfgen_bf_recombination_policy[Z==14?0:1][c]=1;
    atom.cmfgen_bf_channels_n_routes = NLEV;
    atom.cmfgen_bf_channel_offset = channel_offset;
    atom.cmfgen_bf_channel_target = channel_target;
    atom.cmfgen_bf_channel_probability = channel_probability;
    atom.cmfgen_bf_channel_phot_grid = channel_grid;
    atom.cmfgen_bf_channel_target_energy_eV = channel_target_energy;
    atom.cmfgen_bf_channel_target_g = channel_target_g;
    atom.cmfgen_bf_channel_sigma_edge = channel_sigma_edge;
    atom.cmfgen_bf_channel_sigma = channel_sigma;
    atom.ma_rr_loaded = 1;
    atom.ma_rr_n_routes = NLEV - 2;
    atom.ma_rr_target_offset = ma_rr_offset;
    atom.ma_rr_targets = ma_rr_target;
    atom.ma_rr_probability = ma_rr_probability;
    /* v4 consumes its own PHOT target CSR. An absent, unused macro-atom map
     * must not reject an otherwise complete owner; corrupt CSR still fails. */
    if (channel_version >= 4) {
        atom.ma_rr_loaded = 0;
        atom.ma_rr_target_offset = NULL;
        atom.ma_rr_targets = NULL;
        atom.ma_rr_probability = NULL;
    }
    atom.n_lines = NLINES;
    atom.line_atomic_number = line_Z;
    atom.line_ion_number = line_stage;
    atom.line_level_lower = line_lower;
    atom.line_level_upper = line_upper;
    atom.line_f_lu = line_f;
    atom.line_wavelength_cm = line_lambda;
    atom.line_A_ul = line_A;
    atom.line_B_lu = line_Blu;
    atom.line_B_ul = line_Bul;
    atom.line_nu = line_nu;

    nlte.enabled = 1;
    nlte.n_freq_bins = NFREQ;
    nlte.nu_min = nu_min;
    nlte.d_log_nu = dlognu;
    nlte.nu_max = atom.cmfgen_nu_max;
    nlte.n_nlte_ions = 4;
    nlte.n_nlte_levels_total = NLEV;
    nlte.n_super_total = 7;
    nlte.nlte_to_global_level = nlte_to_global;
    nlte.global_to_nlte_level = global_to_nlte;
    nlte.nlte_line_map = nlte_line_map;
    nlte.nlte_level_populations = populations;
    nlte.J_nu = j_nu;
    nlte.fl_to_super = fl_to_super;
    nlte.super_anchor_global = super_anchor;
    nlte.within_sl_frac = within_sl;
    nlte.current_iter = 1;
    for (int stage = 0; stage < 4; ++stage) {
        nlte.nlte_Z[stage] = Z;
        nlte.nlte_ion[stage] = stage + 1;
        nlte.nlte_ion_level_offset[stage] = 2 * stage;
        nlte.nlte_ion_super_offset[stage] = super_offsets[stage];
    }
    nlte.nlte_ion_level_offset[4] = NLEV;
    nlte.nlte_ion_super_offset[4] = super_offsets[4];

    plasma.n_shells = NSHELL;
    plasma.rho = rho;
    plasma.n_electron = n_e;
    plasma.T_e = T_e;
    opacity.n_lines = NLINES;
    opacity.n_shells = NSHELL;

    gamma_deposition_init(&gamma, NSHELL);
    if (gamma_deposition_publish(&gamma, GAMMA_PROVENANCE_EXTERNAL_FILE,
                                 epoch, NULL, NULL, NULL,
                                 external_heating) != 0) {
        gamma_deposition_free(&gamma);
        return -1;
    }
    if (corrupt_route) channel_target[0] = -2;
    double saved_populations[NLEV];
    double saved_terminal = ion_density[5];
    memcpy(saved_populations, populations, sizeof(saved_populations));
    NLTEEWResponse response;
    rc = nlte_element_wide_run_response(&nlte, &atom, &plasma, &opacity,
                                        Z, 0, 0, epoch, &gamma, 1, &response);
    verdict_pass = response.verdict_pass;
    int failed = rc != 0;
    if (!corrupt_route) {
        double candidate_total = 0.0;
        for (int k = 0; k < 5; k++) candidate_total += response.candidate_density[k];
        if (!response.completed || !response.candidate_available ||
            !response.candidate_valid || !response.commit_performed ||
            response.scaled_se_residual > 1e-10 ||
            response.conservation_residual > 1e-12 ||
            fabs(candidate_total / n_element - 1.0) > 1e-12)
            failed = 1;
        double total = ion_density[5];
        for (int g = 0; g < NLEV; ++g) total += populations[g];
        if (!verdict_pass ||
            memcmp(saved_populations, populations, sizeof(saved_populations)) == 0 ||
            fabs(total - n_element) > 1.0e-10 * n_element ||
            ion_density[5] == saved_terminal) {
            fprintf(stderr,
                    "production EW commit failed Z=%d rc=%d pass=%d "
                    "total=%.17g terminal_before=%.17g terminal_after=%.17g\n",
                    Z, rc, verdict_pass, total, saved_terminal, ion_density[5]);
            failed = 1;
        }
    } else if (verdict_pass || response.commit_performed ||
               memcmp(saved_populations, populations, sizeof(saved_populations)) != 0 ||
               ion_density[5] != saved_terminal) {
        fprintf(stderr,
                "production EW failed-route transaction was not rolled back "
                "Z=%d rc=%d pass=%d terminal_before=%.17g terminal_after=%.17g\n",
                Z, rc, verdict_pass, saved_terminal, ion_density[5]);
        failed = 1;
    }
    gamma_deposition_free(&gamma);
    return failed ? -1 : 0;
}

int main(void) {
    extern int nlte_external_dr_fit_enabled(const AtomicData *,int,int);
    {
        AtomicData policy_atom={0};
        if(nlte_external_dr_fit_enabled(&policy_atom,14,1)!=1)return 1;
        policy_atom.cmfgen_bf_channel_version=5;
        policy_atom.cmfgen_bf_channels_loaded=1;
        if(nlte_external_dr_fit_enabled(&policy_atom,14,1)!=-1 ||
           nlte_external_dr_fit_enabled(&policy_atom,26,1)!=1)return 1;
        policy_atom.cmfgen_bf_recombination_policy[0][0]=1;
        if(nlte_external_dr_fit_enabled(&policy_atom,14,1)!=0 ||
           nlte_external_dr_fit_enabled(&policy_atom,16,1)!=-1 ||
           nlte_external_dr_fit_enabled(&policy_atom,14,2)!=-1)return 1;
    }
    _Static_assert(NLTE_EW_IONS == 36, "EW active-ion layout size changed");
    _Static_assert(NLTE_EW_IONS <= NLTE_MAX_IONS, "EW layout exceeds ion capacity");
    static const int want[][2] = {
        {0,1},{1,2},{2,3},{4,5},{6,7},{7,8},
        {9,10},{10,11},{11,12},{13,14},{15,16},{17,18},
        {19,20},{21,22},{23,24},{25,26},{27,28},{29,30},
        {31,32},{33,34},{34,35}
    };
    int pairs[NLTE_PAIR_COUNT][2] = {{0}};
    const char *names[NLTE_PAIR_COUNT] = {0};

    setenv("LUMINA_NLTE_ELEMENT_WIDE", "1", 1);
    setenv("LUMINA_NLTE_ELEMENT_WIDE_Z", "14,16", 1);
    setenv("LUMINA_NLTE_ELEMENT_WIDE_SHELL", "-1", 1);
    setenv("LUMINA_NLTE_ELEMENT_WIDE_COMMIT", "1", 1);
    setenv("LUMINA_NLTE_BF_JEQB", "1", 1);
    setenv("LUMINA_NLTE_BF_COLLISION", "1", 1);
    setenv("LUMINA_NLTE_JEQB", "1", 1);
    setenv("LUMINA_ARTIS_PARITY", "0", 1);
    unsetenv("LUMINA_NLTE_STAGE4");
    {
        const char *override = getenv("LUMINA_NLTE_BF_COLLISION");
        int expected = override ? (atoi(override) != 0) : 0;
        if (nlte_bf_collisional_enabled() != expected) {
            fprintf(stderr, "BF collisional override resolution failed\n");
            return 1;
        }
    }

    if (four_stage_charge_closure_test() != 0) {
        fprintf(stderr, "four-stage Si charge/abundance closure failed\n");
        return 1;
    }
    if (cmfgen_collisional_ionization_rate_test() != 0) {
        fprintf(stderr, "CMFGEN collisional ionization parity failed\n");
        return 1;
    }
    if (commit_fallback_policy_test() != 0) {
        fprintf(stderr, "Si/S owner commit fallback policy failed\n");
        return 1;
    }
    if (terminal_scalar_rate_test(14, 6.0) != 0 ||
        terminal_scalar_rate_test(16, 2.0) != 0) {
        fprintf(stderr, "Si/S terminal scalar BF detailed-balance test failed\n");
        return 1;
    }
    {
        double v3[4], v4[4], degenerate[4];
        double partition = 2.0 + 4.0 * exp(-EV_TO_ERG /
                                              (K_BOLTZMANN * 15000.0));
        if (nlte_ew_test_grouped_superlevel_rates(3, 1.0, v3) != 0 ||
            nlte_ew_test_grouped_superlevel_rates(4, 1.0, v4) != 0 ||
            nlte_ew_test_grouped_superlevel_rates(4, 0.0, degenerate) != 0 ||
            rel_error(v4[0], v3[0]) > 1.0e-12 ||
            rel_error(v4[2], v3[2]) > 1.0e-12 ||
            rel_error(v4[1], v3[1]) > 1.0e-12 ||
            rel_error(v4[3], v3[3]) > 1.0e-12 ||
            rel_error(v4[0], degenerate[0]) > 1.0e-12 ||
            rel_error(v4[2], degenerate[2]) > 1.0e-12 ||
            rel_error(v4[1] / degenerate[1], 6.0 / partition) > 1.0e-12 ||
            rel_error(v4[3] / degenerate[3], 6.0 / partition) > 1.0e-12) {
            fprintf(stderr,
                    "CMFGEN BF target-superlevel rate semantics failed\n");
            return 1;
        }
    }

    for (int pass = 0; pass < 2; ++pass) {
        int corrupt_route = pass != 0;
        if (production_owner_transaction_test(14, 28.085, 6.0,
                                               corrupt_route, 4) != 0 ||
            production_owner_transaction_test(16, 32.06, 2.0,
                                               corrupt_route, 4) != 0 ||
            production_owner_transaction_test(14, 28.085, 6.0,
                                               corrupt_route, 5) != 0 ||
            production_owner_transaction_test(16, 32.06, 2.0,
                                               corrupt_route, 5) != 0) {
            fprintf(stderr,
                    "Si/S production owner transaction %s test failed\n",
                    corrupt_route ? "rollback" : "commit");
            return 1;
        }
    }


    {
        const double ion_rate[] = {0.4,0.8,0.25};
        const double recombination_rate[] = {0.2,0.5,0.5};
        double population[4], closure = -1.0;
        int bb_owner_count[4];
        if (nlte_ew_test_solve_chain4(ion_rate,recombination_rate,100.0,
                                      population,&closure,bb_owner_count) != 0) {
            fprintf(stderr, "synthetic II-V owner solve failed\n");
            return 1;
        }
        double sum = 0.0;
        for (int q = 0; q < 4; q++) {
            sum += population[q];
            if (!(population[q] > 0.0) || bb_owner_count[q] != 1) {
                fprintf(stderr, "stage %d positivity/BB ownership failed\n",q+2);
                return 1;
            }
        }
        if (fabs(sum-100.0) > 1.0e-10 || closure > 1.0e-14 ||
            fabs(population[1]/population[0]-2.0) > 1.0e-10 ||
            fabs(population[2]/population[1]-1.6) > 1.0e-10 ||
            fabs(population[3]/population[2]-0.5) > 1.0e-10) {
            fprintf(stderr, "synthetic II-V closure/adjacent-rate balance failed\n");
            return 1;
        }
        double perturbed_rate[] = {0.4,4.0,0.25};
        double perturbed[4];
        if (nlte_ew_test_solve_chain4(perturbed_rate,recombination_rate,100.0,
                                      perturbed,&closure,bb_owner_count) != 0 ||
            fabs(perturbed[3]-population[3]) < 1.0) {
            fprintf(stderr, "synthetic IV-V rate perturbation has no response\n");
            return 1;
        }
    }

    if (!nlte_element_wide_layout_enabled() ||
        !nlte_element_wide_matches(14, 0) ||
        !nlte_element_wide_matches(14, 47) ||
        !nlte_element_wide_matches(16, 20) ||
        nlte_element_wide_matches(26, 8)) {
        fprintf(stderr, "all-shell Si/S ownership selection failed\n");
        return 1;
    }

    int n = nlte_get_pairs(pairs, names);
    if (n != (int)(sizeof(want) / sizeof(want[0])) || n != NLTE_EW_PAIRS) {
        fprintf(stderr, "EW pair count mismatch: got=%d expected=%zu macro=%d\n",
                n, sizeof(want) / sizeof(want[0]), NLTE_EW_PAIRS);
        return 1;
    }
    for (int p = 0; p < n; p++) {
        if (pairs[p][0] != want[p][0] || pairs[p][1] != want[p][1] ||
            pairs[p][1] != pairs[p][0] + 1 || !names[p]) {
            fprintf(stderr,
                    "EW pair %d mismatch: got=(%d,%d) expected=(%d,%d)\n",
                    p, pairs[p][0], pairs[p][1], want[p][0], want[p][1]);
            return 1;
        }
    }
    printf("EW owner PASS: v4 grouped-SL BF rates, Si/S terminal BF threshold+Milne, "
           "4-stage rate response, "
           "charge closure, BB single-owner; "
           "36 slots, %d adjacent pairs, all-shell Si/S\n", n);
    return 0;
}
