#define _POSIX_C_SOURCE 200809L
#include "physics_comparison.h"
#include "atomic_internal_energy.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static int comparison_lane_ok(const char *lane)
{
    if (!lane || !*lane) return 0;
    for (const unsigned char *p = (const unsigned char *)lane; *p; ++p)
        if (!isalnum(*p) && *p != '_' && *p != '-') return 0;
    return 1;
}

static int comparison_hex64(const char value[65])
{
    if (!value || value[64] != '\0') return 0;
    for (size_t i = 0; i < 64; ++i)
        if (!isxdigit((unsigned char)value[i])) return 0;
    return 1;
}

static int comparison_close(double a, double b)
{
    double scale = fmax(fmax(fabs(a), fabs(b)), 1.0);
    return fabs(a - b) <= 2.0e-12 * scale;
}

static int comparison_rebin_j(const RadiationFieldView *radiation,
                              const double *target_edges,
                              size_t target_bins,
                              size_t n_shells,
                              double *output)
{
    if (!radiation || !target_edges || !output ||
        radiation->n_shells != n_shells || radiation->n_bins == 0 ||
        !radiation->frequency_bin_edges || !radiation->J_nu ||
        !radiation->validity)
        return -1;
    for (size_t s = 0; s < n_shells; ++s) {
        size_t q = 0;
        for (size_t b = 0; b < target_bins; ++b) {
            double lo = target_edges[b], hi = target_edges[b+1];
            double integral = 0.0, covered = 0.0;
            while (q < radiation->n_bins &&
                   radiation->frequency_bin_edges[q+1] <= lo)
                ++q;
            size_t k = q;
            while (k < radiation->n_bins &&
                   radiation->frequency_bin_edges[k] < hi) {
                double left = fmax(lo, radiation->frequency_bin_edges[k]);
                double right = fmin(hi, radiation->frequency_bin_edges[k+1]);
                if (right > left) {
                    size_t cell = s*radiation->n_bins+k;
                    RadiationFieldValidityState validity =
                        radiation->validity[cell];
                    if (validity != RADIATION_FIELD_VALID &&
                        validity != RADIATION_FIELD_EXACT_ZERO)
                        return -1;
                    if (!isfinite(radiation->J_nu[cell]) ||
                        radiation->J_nu[cell] < 0.0)
                        return -1;
                    integral += radiation->J_nu[cell] * (right-left);
                    covered += right-left;
                }
                ++k;
            }
            if (!(hi > lo) ||
                fabs(covered-(hi-lo)) > 1.0e-10*(hi-lo))
                return -1;
            output[s*target_bins+b] = integral/(hi-lo);
        }
    }
    return 0;
}

static PhysicsComparisonStatus comparison_validate(
        const PhysicsComparisonSnapshotInput *in, double *j_rebinned)
{
    if (!in || !comparison_lane_ok(in->lane) || in->iteration < 0 ||
        !isfinite(in->epoch_s) || in->epoch_s <= 0.0 || in->n_shells < 2 ||
        !in->r_inner_cm || !in->r_outer_cm || !in->v_inner_cm_s ||
        !in->v_outer_cm_s || !in->temperature_K ||
        !in->electron_density_cm3 || !in->atom_density_cm3 ||
        !in->internal_energy_atom_erg || !in->radiation || !in->opacity ||
        !in->emissivity || !in->temperature_publication || !j_rebinned)
    {
        fprintf(stderr,
                "[PHYSICS_COMPARISON][BLOCKED] "
                "reason=COMPARISON_INPUT_INVALID site=99 "
                "lane=%s iteration=%d input_present=%d lane_valid=%d "
                "epoch_s=%.17g n_shells=%zu r_inner_cm=%p r_outer_cm=%p "
                "v_inner_cm_s=%p v_outer_cm_s=%p temperature_K=%p "
                "electron_density_cm3=%p atom_density_cm3=%p "
                "internal_energy_atom_erg=%p radiation=%p opacity=%p "
                "emissivity=%p temperature_publication=%p j_rebinned=%p "
                "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0\n",
                in && in->lane ? in->lane : "(null)",
                in ? in->iteration : -1,
                in != NULL,
                in ? comparison_lane_ok(in->lane) : 0,
                in ? in->epoch_s : NAN,
                in ? in->n_shells : 0,
                in ? (void *)in->r_inner_cm : NULL,
                in ? (void *)in->r_outer_cm : NULL,
                in ? (void *)in->v_inner_cm_s : NULL,
                in ? (void *)in->v_outer_cm_s : NULL,
                in ? (void *)in->temperature_K : NULL,
                in ? (void *)in->electron_density_cm3 : NULL,
                in ? (void *)in->atom_density_cm3 : NULL,
                in ? (void *)in->internal_energy_atom_erg : NULL,
                in ? (void *)in->radiation : NULL,
                in ? (void *)in->opacity : NULL,
                in ? (void *)in->emissivity : NULL,
                in ? (void *)in->temperature_publication : NULL,
                (void *)j_rebinned);
        return PHYSICS_COMPARISON_INVALID_ARGUMENT;
    }

    const CpuOpacityPublication *op = in->opacity;
    const CpuEmissivityPublication *em = in->emissivity;
    const ElectronTemperaturePublication *te = in->temperature_publication;
    size_t ns = in->n_shells, nb = op->n_bins;
    if (!nb || op->n_shells != ns || em->n_shells != ns || em->n_bins != nb ||
        te->n_shells != ns || !op->frequency_edges || !em->nu_edge ||
        !op->chi_es || !op->chi_bb || !op->chi_bf || !op->chi_ff ||
        !op->chi_total || !op->chi_validity || !em->eta_bb || !em->eta_bf ||
        !em->eta_ff || !em->eta_true_total || !em->cell_status ||
        !em->component_status || !te->ledger || !te->shell_status ||
        !te->residual_status)
    {
        fprintf(stderr,
                "[PHYSICS_COMPARISON][BLOCKED] "
                "reason=COMPARISON_PUBLICATION_LAYOUT_INVALID site=112 "
                "lane=%s iteration=%d n_shells=%zu n_bins=%zu "
                "op_n_shells=%zu em_n_shells=%zu em_n_bins=%zu te_n_shells=%zu "
                "frequency_edges=%p nu_edge=%p chi_es=%p chi_bb=%p "
                "chi_bf=%p chi_ff=%p chi_total=%p chi_validity=%p "
                "eta_bb=%p eta_bf=%p eta_ff=%p eta_true_total=%p "
                "cell_status=%p component_status=%p ledger=%p shell_status=%p "
                "residual_status=%p "
                "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0\n",
                in->lane ? in->lane : "(null)", in->iteration, ns, nb,
                op->n_shells, em->n_shells, em->n_bins, te->n_shells,
                (void *)op->frequency_edges, (void *)em->nu_edge,
                (void *)op->chi_es, (void *)op->chi_bb,
                (void *)op->chi_bf, (void *)op->chi_ff,
                (void *)op->chi_total, (void *)op->chi_validity,
                (void *)em->eta_bb, (void *)em->eta_bf,
                (void *)em->eta_ff, (void *)em->eta_true_total,
                (void *)em->cell_status, (void *)em->component_status,
                (void *)te->ledger, (void *)te->shell_status,
                (void *)te->residual_status);
        return PHYSICS_COMPARISON_INVALID_ARGUMENT;
    }

    uint64_t tg = te->committed_te_generation;
    uint64_t pg = te->population_generation;
    uint64_t rg = te->radfield_generation;
    uint64_t og = te->opacity_generation;
    uint64_t eg = te->emissivity_generation;
    if (!tg || !pg || !rg || !og || !eg ||
        te->required_te_generation != tg || op->te_generation != tg ||
        em->te_generation != tg || op->population_generation != pg ||
        em->population_generation != pg || op->radiation_generation != rg ||
        em->radfield_generation != rg || in->radiation->generation != rg ||
        op->generation_committed != og || op->generation_required != og ||
        em->opacity_generation != og ||
        em->committed_emissivity_generation != eg ||
        em->required_emissivity_generation != eg || eg != og)
        return PHYSICS_COMPARISON_STALE_GENERATION;
    if (!comparison_hex64(te->atomic_model_sha256) ||
        !comparison_hex64(te->geometry_sha256) ||
        !comparison_hex64(te->te_manifest_sha256) ||
        !comparison_hex64(em->grid_manifest_sha256))
    {
        fprintf(stderr,
                "[PHYSICS_COMPARISON][BLOCKED] "
                "reason=COMPARISON_HASH_INVALID site=133 "
                "lane=%s iteration=%d atomic_model_sha256_valid=%d "
                "geometry_sha256_valid=%d te_manifest_sha256_valid=%d "
                "grid_manifest_sha256_valid=%d atomic_model_sha256=%.64s "
                "geometry_sha256=%.64s te_manifest_sha256=%.64s "
                "grid_manifest_sha256=%.64s "
                "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0\n",
                in->lane ? in->lane : "(null)", in->iteration,
                comparison_hex64(te->atomic_model_sha256),
                comparison_hex64(te->geometry_sha256),
                comparison_hex64(te->te_manifest_sha256),
                comparison_hex64(em->grid_manifest_sha256),
                te->atomic_model_sha256, te->geometry_sha256,
                te->te_manifest_sha256, em->grid_manifest_sha256);
        return PHYSICS_COMPARISON_INVALID_ARGUMENT;
    }

    double *velocity_edges = malloc((ns+1)*sizeof(*velocity_edges));
    if (!velocity_edges) return PHYSICS_COMPARISON_IO_ERROR;
    velocity_edges[0] = in->v_inner_cm_s[0];
    for (size_t s = 0; s < ns; ++s) {
        if (!isfinite(in->r_inner_cm[s]) || !isfinite(in->r_outer_cm[s]) ||
            !isfinite(in->v_inner_cm_s[s]) ||
            !isfinite(in->v_outer_cm_s[s]) ||
            !(in->r_outer_cm[s] > in->r_inner_cm[s]) ||
            !(in->v_outer_cm_s[s] > in->v_inner_cm_s[s]) ||
            !comparison_close(in->r_inner_cm[s],
                              in->v_inner_cm_s[s]*in->epoch_s) ||
            !comparison_close(in->r_outer_cm[s],
                              in->v_outer_cm_s[s]*in->epoch_s) ||
            (s && !comparison_close(in->v_inner_cm_s[s],
                                    in->v_outer_cm_s[s-1]))) {
            free(velocity_edges);
            return PHYSICS_COMPARISON_INVALID_GRID;
        }
        velocity_edges[s+1] = in->v_outer_cm_s[s];
    }
    char geometry_sha256[65];
    RadeqStatus geometry_status = a210_geometry_sha256(
        velocity_edges, ns+1, geometry_sha256);
    free(velocity_edges);
    if (geometry_status != RADEQ_OK ||
        strcmp(geometry_sha256, te->geometry_sha256) != 0)
        return PHYSICS_COMPARISON_STALE_GENERATION;

    for (size_t b = 0; b <= nb; ++b) {
        if (!isfinite(op->frequency_edges[b]) ||
            !isfinite(em->nu_edge[b]) ||
            op->frequency_edges[b] != em->nu_edge[b] ||
            (b && !(op->frequency_edges[b] > op->frequency_edges[b-1])))
            return PHYSICS_COMPARISON_INVALID_GRID;
    }
    if (comparison_rebin_j(in->radiation, op->frequency_edges, nb, ns,
                           j_rebinned) != 0)
        return PHYSICS_COMPARISON_INVALID_GRID;

    for (size_t s = 0; s < ns; ++s) {
        const A210TermLedger *ledger = &te->ledger[s];
        if (!isfinite(in->temperature_K[s]) || in->temperature_K[s] <= 0.0 ||
            !isfinite(in->electron_density_cm3[s]) ||
            in->electron_density_cm3[s] < 0.0 ||
            !isfinite(in->atom_density_cm3[s]) ||
            in->atom_density_cm3[s] <= 0.0 ||
            !isfinite(in->internal_energy_atom_erg[s]) ||
            in->internal_energy_atom_erg[s] < 0.0 ||
            (te->shell_status[s] != RADEQ_OK &&
             te->shell_status[s] != RADEQ_EXACT_ZERO_BALANCE) ||
            te->residual_status[s] != RADEQ_OK ||
            ledger->adiabatic_model != A210_ADIABATIC_CMFGEN_COMPLETE ||
            !isfinite(ledger->adiabatic_temperature_gradient) ||
            !isfinite(ledger->adiabatic_velocity_divergence) ||
            !isfinite(ledger->adiabatic_electron_fraction_gradient) ||
            !isfinite(ledger->adiabatic_internal_energy_gradient) ||
            !isfinite(ledger->adiabatic_signed_total) ||
            !isfinite(ledger->sum_heating) ||
            !isfinite(ledger->sum_cooling) ||
            !isfinite(ledger->residual) ||
            !comparison_close(ledger->adiabatic_signed_total,
                ledger->cooling[A210_ADIABATIC]-
                ledger->heating[A210_ADIABATIC_H]) ||
            !comparison_close(ledger->residual,
                               ledger->sum_heating-ledger->sum_cooling))
            return PHYSICS_COMPARISON_INVALID_VALUE;
    }
    size_t cells = ns*nb;
    for (size_t i = 0; i < cells; ++i) {
        double chi_sum = op->chi_es[i]+op->chi_bb[i]+op->chi_bf[i]+op->chi_ff[i];
        double eta_sum = em->eta_bb[i]+em->eta_bf[i]+em->eta_ff[i];
        if (!isfinite(j_rebinned[i]) ||
            !isfinite(op->chi_es[i]) || !isfinite(op->chi_bb[i]) ||
            !isfinite(op->chi_bf[i]) || !isfinite(op->chi_ff[i]) ||
            !isfinite(op->chi_total[i]) || !isfinite(em->eta_bb[i]) ||
            !isfinite(em->eta_bf[i]) || !isfinite(em->eta_ff[i]) ||
            !isfinite(em->eta_true_total[i]) || em->eta_bb[i] < 0.0 ||
            em->eta_bf[i] < 0.0 || em->eta_ff[i] < 0.0 ||
            em->eta_true_total[i] < 0.0 ||
            !comparison_close(op->chi_total[i],chi_sum) ||
            !comparison_close(em->eta_true_total[i],eta_sum) ||
            (em->cell_status[i] != EMISS_OK &&
             em->cell_status[i] != EMISS_EXACT_ZERO))
            return PHYSICS_COMPARISON_INVALID_VALUE;
        for (size_t component = 0; component < 4; ++component) {
            A208Validity validity = op->chi_validity[component*cells+i];
            if (validity != A208_VALID && validity != A208_EXACT_ZERO)
                return PHYSICS_COMPARISON_INVALID_VALUE;
        }
        for (size_t component = 0; component < 3; ++component) {
            EmissivityStatus component_status =
                em->component_status[component*cells+i];
            if (component_status != EMISS_OK &&
                component_status != EMISS_EXACT_ZERO)
                return PHYSICS_COMPARISON_INVALID_VALUE;
        }
    }
    return PHYSICS_COMPARISON_OK;
}

static int comparison_path(char out[PATH_MAX], const char *dir,
                           const char *lane, int iteration,
                           const char *suffix, const char *temporary)
{
    int n = snprintf(out, PATH_MAX, "%s/physics_%s_iter%04d.%s%s",
                     dir, lane, iteration, suffix,
                     temporary ? temporary : "");
    return n > 0 && n < PATH_MAX ? 0 : -1;
}

static int comparison_finish(FILE *stream)
{
    return !stream || fflush(stream) != 0 || fsync(fileno(stream)) != 0 ||
           fclose(stream) != 0 ? -1 : 0;
}

PhysicsComparisonStatus physics_comparison_snapshot_write(
        const char *directory, const PhysicsComparisonSnapshotInput *in)
{
    if (!directory || !*directory || !in)
    {
        fprintf(stderr,
                "[PHYSICS_COMPARISON][BLOCKED] "
                "reason=SNAPSHOT_INPUT_MISSING site=255 "
                "directory_present=%d input_present=%d lane=%s iteration=%d "
                "n_bins=%zu n_shells=%zu "
                "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0\n",
                directory && *directory, in != NULL,
                in && in->lane ? in->lane : "(null)",
                in ? in->iteration : -1,
                in && in->opacity ? in->opacity->n_bins : 0,
                in ? in->n_shells : 0);
        return PHYSICS_COMPARISON_INVALID_ARGUMENT;
    }
    size_t nb = in->opacity ? in->opacity->n_bins : 0;
    if (!nb || !in->n_shells || in->n_shells > SIZE_MAX/nb)
    {
        fprintf(stderr,
                "[PHYSICS_COMPARISON][BLOCKED] "
                "reason=SNAPSHOT_BIN_OR_SHELL_INVALID site=258 "
                "lane=%s iteration=%d n_bins=%zu n_shells=%zu "
                "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0\n",
                in->lane ? in->lane : "(null)", in->iteration, nb,
                in->n_shells);
        return PHYSICS_COMPARISON_INVALID_ARGUMENT;
    }
    double *j_rebinned = malloc(in->n_shells*nb*sizeof(*j_rebinned));
    if (!j_rebinned) return PHYSICS_COMPARISON_IO_ERROR;
    PhysicsComparisonStatus status = comparison_validate(in, j_rebinned);
    if (status != PHYSICS_COMPARISON_OK) {
        free(j_rebinned);
        return status;
    }
    if (mkdir(directory, 0775) != 0 && errno != EEXIST) {
        free(j_rebinned);
        return PHYSICS_COMPARISON_IO_ERROR;
    }
    struct stat directory_stat;
    if (stat(directory, &directory_stat) != 0 ||
        !S_ISDIR(directory_stat.st_mode)) {
        free(j_rebinned);
        return PHYSICS_COMPARISON_IO_ERROR;
    }

    char tag[64];
    snprintf(tag, sizeof(tag), ".tmp.%ld", (long)getpid());
    char shell_path[PATH_MAX], spectral_path[PATH_MAX], manifest_path[PATH_MAX];
    char shell_tmp[PATH_MAX], spectral_tmp[PATH_MAX], manifest_tmp[PATH_MAX];
    if (comparison_path(shell_path,directory,in->lane,in->iteration,"shell.csv",NULL) ||
        comparison_path(spectral_path,directory,in->lane,in->iteration,"spectral.csv",NULL) ||
        comparison_path(manifest_path,directory,in->lane,in->iteration,"manifest.json",NULL) ||
        comparison_path(shell_tmp,directory,in->lane,in->iteration,"shell.csv",tag) ||
        comparison_path(spectral_tmp,directory,in->lane,in->iteration,"spectral.csv",tag) ||
        comparison_path(manifest_tmp,directory,in->lane,in->iteration,"manifest.json",tag)) {
        free(j_rebinned);
        return PHYSICS_COMPARISON_IO_ERROR;
    }

    FILE *shell = fopen(shell_tmp, "w");
    FILE *spectral = fopen(spectral_tmp, "w");
    if (!shell || !spectral) {
        if (shell) fclose(shell);
        if (spectral) fclose(spectral);
        remove(shell_tmp); remove(spectral_tmp); free(j_rebinned);
        return PHYSICS_COMPARISON_IO_ERROR;
    }
    fprintf(shell,
        "shell_id,r_inner_cm,r_outer_cm,v_inner_cm_s,v_outer_cm_s,T_e_K,n_e_cm3,"
        "n_atom_cm3,u_atom_erg,q_ad_temperature_gradient,q_ad_velocity_divergence,"
        "q_ad_electron_fraction_gradient,q_ad_internal_energy_gradient,"
        "q_ad_signed_total,q_ad_heating,q_ad_cooling,photo_heat,line_abs_heat,"
        "ff_abs_heat,compton_heat,gamma_heat,nonthermal_heat,recomb_cool,"
        "line_emit_cool,coll_line_cool,ff_emit_cool,compton_cool,sum_heating,"
        "sum_cooling,residual\n");
    for (size_t s = 0; s < in->n_shells; ++s) {
        const A210TermLedger *l = &in->temperature_publication->ledger[s];
        fprintf(shell,
            "%zu,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g\n",
            s,in->r_inner_cm[s],in->r_outer_cm[s],in->v_inner_cm_s[s],
            in->v_outer_cm_s[s],in->temperature_K[s],
            in->electron_density_cm3[s],in->atom_density_cm3[s],
            in->internal_energy_atom_erg[s],l->adiabatic_temperature_gradient,
            l->adiabatic_velocity_divergence,
            l->adiabatic_electron_fraction_gradient,
            l->adiabatic_internal_energy_gradient,l->adiabatic_signed_total,
            l->heating[A210_ADIABATIC_H],l->cooling[A210_ADIABATIC],
            l->heating[A210_PHOTO],l->heating[A210_LINE_ABS],
            l->heating[A210_FF_ABS],l->heating[A210_COMPTON_H],
            l->heating[A210_GAMMA],l->heating[A210_NONTHERMAL],
            l->cooling[A210_RECOMB],l->cooling[A210_LINE_EMIT],
            l->cooling[A210_COLL_LINE],l->cooling[A210_FF_EMIT],
            l->cooling[A210_COMPTON_C],l->sum_heating,l->sum_cooling,l->residual);
    }
    fprintf(spectral,
        "shell_id,bin_id,nu_lo_Hz,nu_hi_Hz,J_nu,chi_es_cm1,chi_bb_cm1,"
        "chi_bf_cm1,chi_ff_cm1,chi_total_cm1,eta_bb,eta_bf,eta_ff,eta_true_total\n");
    for (size_t s = 0; s < in->n_shells; ++s)
        for (size_t b = 0; b < nb; ++b) {
            size_t i = s*nb+b;
            fprintf(spectral,
                "%zu,%zu,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                "%.17g,%.17g,%.17g,%.17g,%.17g\n",
                s,b,in->opacity->frequency_edges[b],
                in->opacity->frequency_edges[b+1],j_rebinned[i],
                in->opacity->chi_es[i],in->opacity->chi_bb[i],
                in->opacity->chi_bf[i],in->opacity->chi_ff[i],
                in->opacity->chi_total[i],in->emissivity->eta_bb[i],
                in->emissivity->eta_bf[i],in->emissivity->eta_ff[i],
                in->emissivity->eta_true_total[i]);
        }
    if (comparison_finish(shell) != 0 || comparison_finish(spectral) != 0) {
        remove(shell_tmp); remove(spectral_tmp); free(j_rebinned);
        return PHYSICS_COMPARISON_IO_ERROR;
    }
    free(j_rebinned);

    const ElectronTemperaturePublication *te = in->temperature_publication;
    const char *te_reason =
        a210_temperature_publication_validate(te, in->n_shells);

    if (te_reason) {
        remove(shell_tmp);
        remove(spectral_tmp);
        remove(manifest_tmp);
        fprintf(stderr,
                "[PHYSICS_COMPARISON][BLOCKED] reason=%s\n",
                te_reason);
        return PHYSICS_COMPARISON_IO_ERROR;
    }

    FILE *manifest = fopen(manifest_tmp, "w");
    if (!manifest) {
        remove(shell_tmp); remove(spectral_tmp);
        return PHYSICS_COMPARISON_IO_ERROR;
    }
    fprintf(manifest,
        "{\n"
        "  \"schema\": \"LUMINA_PHYSICS_COMPARISON_V1\",\n"
        "  \"transaction_status\": \"COMMITTED\",\n"
        "  \"code\": \"LUMINA\",\n"
        "  \"lane\": \"%s\",\n"
        "  \"te_lane\": \"%s\",\n",
        in->lane, a210_te_lane_name(te->te_lane));

    if (a210_te_manifest_has_fixed_fields(te)) {
        fprintf(manifest,
        "  \"te_profile_sha256\": \"%s\",\n"
        "  \"pinned_shells\": %zu,\n"
        "  \"re_root_required\": %s,\n",
        te->te_profile_sha256,
        te->pinned_shells,
        te->re_root_required ? "true" : "false");
    }

    fprintf(manifest,
        "  \"iteration\": %d,\n"
        "  \"epoch_s\": %.17g,\n"
        "  \"n_shells\": %zu,\n"
        "  \"n_bins\": %zu,\n"
        "  \"frame\": \"SHELL_COMOVING\",\n"
        "  \"frequency_coordinate\": \"HZ\",\n"
        "  \"opacity_units\": \"CM^-1\",\n"
        "  \"emissivity_units\": \"ERG_S^-1_CM^-3_HZ^-1_SR^-1\",\n"
        "  \"volume_rate_units\": \"ERG_S^-1_CM^-3\",\n"
        "  \"eta_is_per_sr\": true,\n"
        "  \"radiative_integral_factor\": %.17g,\n"
        "  \"adiabatic_positive_is_cooling\": true,\n"
        "  \"shell_weight\": \"SPHERICAL_VOLUME\",\n"
        "  \"frequency_regrid\": \"INTEGRAL_PRESERVING_PIECEWISE_CONSTANT\",\n"
        "  \"atomic_model_sha256\": \"%s\",\n"
        "  \"geometry_sha256\": \"%s\",\n"
        "  \"te_manifest_sha256\": \"%s\",\n"
        "  \"grid_manifest_sha256\": \"%s\",\n"
        "  \"radiation_generation\": %llu,\n"
        "  \"population_generation\": %llu,\n"
        "  \"te_generation\": %llu,\n"
        "  \"opacity_generation\": %llu,\n"
        "  \"emissivity_generation\": %llu,\n"
        "  \"shell_file\": \"physics_%s_iter%04d.shell.csv\",\n"
        "  \"spectral_file\": \"physics_%s_iter%04d.spectral.csv\"\n"
        "}\n",
        in->iteration,in->epoch_s,in->n_shells,nb,
        4.0*M_PI_VAL,te->atomic_model_sha256,te->geometry_sha256,
        te->te_manifest_sha256,in->emissivity->grid_manifest_sha256,
        (unsigned long long)te->radfield_generation,
        (unsigned long long)te->population_generation,
        (unsigned long long)te->committed_te_generation,
        (unsigned long long)te->opacity_generation,
        (unsigned long long)te->emissivity_generation,
        in->lane,in->iteration,in->lane,in->iteration);
    if (comparison_finish(manifest) != 0) {
        remove(shell_tmp); remove(spectral_tmp); remove(manifest_tmp);
        return PHYSICS_COMPARISON_IO_ERROR;
    }
    if (rename(shell_tmp,shell_path) != 0 ||
        rename(spectral_tmp,spectral_path) != 0 ||
        rename(manifest_tmp,manifest_path) != 0) {
        remove(shell_tmp); remove(spectral_tmp); remove(manifest_tmp);
        return PHYSICS_COMPARISON_IO_ERROR;
    }
    return PHYSICS_COMPARISON_OK;
}

PhysicsComparisonStatus physics_comparison_dump_if_requested(
        const char *lane, int iteration, const Geometry *geometry,
        const AtomicData *atom, const PlasmaState *plasma,
        const OpacityState *opacity, const NLTEConfig *nlte)
{
    const char *directory = getenv("LUMINA_PHYSICS_COMPARISON_DIR");
    if (!directory || !*directory) return PHYSICS_COMPARISON_NOT_REQUESTED;
    if (!geometry) {
        fprintf(stderr,
                "[PHYSICS_COMPARISON][BLOCKED] reason=DUMP_GEOMETRY_MISSING "
                "site=448 lane=%s iteration=%d geometry=0 atom=%d plasma=%d "
                "opacity=%d nlte=%d geometry_n_shells=%d plasma_n_shells=%d "
                "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0\n",
                lane ? lane : "(null)", iteration, atom != NULL, plasma != NULL,
                opacity != NULL, nlte != NULL, -1,
                plasma ? plasma->n_shells : -1);
        return PHYSICS_COMPARISON_INVALID_ARGUMENT;
    }
    if (!atom) {
        fprintf(stderr,
                "[PHYSICS_COMPARISON][BLOCKED] reason=DUMP_ATOM_MISSING "
                "site=448 lane=%s iteration=%d geometry=%d atom=0 plasma=%d "
                "opacity=%d nlte=%d geometry_n_shells=%d plasma_n_shells=%d "
                "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0\n",
                lane ? lane : "(null)", iteration, geometry != NULL, plasma != NULL,
                opacity != NULL, nlte != NULL, geometry->n_shells,
                plasma ? plasma->n_shells : -1);
        return PHYSICS_COMPARISON_INVALID_ARGUMENT;
    }
    if (!plasma) {
        fprintf(stderr,
                "[PHYSICS_COMPARISON][BLOCKED] reason=DUMP_PLASMA_MISSING "
                "site=448 lane=%s iteration=%d geometry=%d atom=%d plasma=0 "
                "opacity=%d nlte=%d geometry_n_shells=%d plasma_n_shells=%d "
                "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0\n",
                lane ? lane : "(null)", iteration, geometry != NULL, atom != NULL,
                opacity != NULL, nlte != NULL, geometry->n_shells, -1);
        return PHYSICS_COMPARISON_INVALID_ARGUMENT;
    }
    if (!opacity) {
        fprintf(stderr,
                "[PHYSICS_COMPARISON][BLOCKED] reason=DUMP_OPACITY_MISSING "
                "site=448 lane=%s iteration=%d geometry=%d atom=%d plasma=%d "
                "opacity=0 nlte=%d geometry_n_shells=%d plasma_n_shells=%d "
                "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0\n",
                lane ? lane : "(null)", iteration, geometry != NULL, atom != NULL,
                plasma != NULL, nlte != NULL, geometry->n_shells,
                plasma->n_shells);
        return PHYSICS_COMPARISON_INVALID_ARGUMENT;
    }
    if (!nlte) {
        fprintf(stderr,
                "[PHYSICS_COMPARISON][BLOCKED] reason=DUMP_NLTE_MISSING "
                "site=448 lane=%s iteration=%d geometry=%d atom=%d plasma=%d "
                "opacity=%d nlte=0 geometry_n_shells=%d plasma_n_shells=%d "
                "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0\n",
                lane ? lane : "(null)", iteration, geometry != NULL, atom != NULL,
                plasma != NULL, opacity != NULL, geometry->n_shells,
                plasma->n_shells);
        return PHYSICS_COMPARISON_INVALID_ARGUMENT;
    }
    if (geometry->n_shells < 2) {
        fprintf(stderr,
                "[PHYSICS_COMPARISON][BLOCKED] reason=DUMP_SHELL_COUNT_TOO_SMALL "
                "site=448 lane=%s iteration=%d geometry=%d atom=%d plasma=%d "
                "opacity=%d nlte=%d geometry_n_shells=%d plasma_n_shells=%d "
                "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0\n",
                lane ? lane : "(null)", iteration, geometry != NULL, atom != NULL,
                plasma != NULL, opacity != NULL, nlte != NULL,
                geometry->n_shells, plasma->n_shells);
        return PHYSICS_COMPARISON_INVALID_ARGUMENT;
    }
    if (plasma->n_shells != geometry->n_shells) {
        fprintf(stderr,
                "[PHYSICS_COMPARISON][BLOCKED] reason=DUMP_SHELL_COUNT_MISMATCH "
                "site=448 lane=%s iteration=%d geometry=%d atom=%d plasma=%d "
                "opacity=%d nlte=%d geometry_n_shells=%d plasma_n_shells=%d "
                "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0\n",
                lane ? lane : "(null)", iteration, geometry != NULL, atom != NULL,
                plasma != NULL, opacity != NULL, nlte != NULL,
                geometry->n_shells, plasma->n_shells);
        return PHYSICS_COMPARISON_INVALID_ARGUMENT;
    }
    size_t ns = (size_t)geometry->n_shells;
    AtomicInternalEnergyCell *energy = calloc(ns,sizeof(*energy));
    double *n_atom = malloc(ns*sizeof(*n_atom));
    double *u_atom = malloc(ns*sizeof(*u_atom));
    if (!energy || !n_atom || !u_atom) {
        free(energy); free(n_atom); free(u_atom);
        return PHYSICS_COMPARISON_IO_ERROR;
    }
    AtomicInternalEnergyStatus energy_status = atomic_internal_energy_build(
        atom,nlte,plasma,ns,atom->population_committed_generation,
        plasma->T_e_generation,energy);
    if (energy_status != ATOMIC_INTERNAL_ENERGY_OK) {
        free(energy); free(n_atom); free(u_atom);
        return PHYSICS_COMPARISON_INVALID_VALUE;
    }
    for (size_t s = 0; s < ns; ++s) {
        n_atom[s] = energy[s].n_atom_cm3;
        u_atom[s] = energy[s].internal_energy_atom_erg;
    }
    PhysicsComparisonSnapshotInput input = {
        lane,iteration,geometry->time_explosion,ns,
        geometry->r_inner,geometry->r_outer,geometry->v_inner,geometry->v_outer,
        plasma->T_e,plasma->n_electron,n_atom,u_atom,&nlte->radfield_view,
        &opacity->cpu_opacity,&opacity->cpu_emissivity,
        &plasma->te_publication
    };
    PhysicsComparisonStatus status =
        physics_comparison_snapshot_write(directory,&input);
    free(energy); free(n_atom); free(u_atom);
    if (status == PHYSICS_COMPARISON_OK)
        fprintf(stderr,
                "[PHYSICS-COMPARISON] lane=%s iter=%d status=COMMITTED dir=%s\n",
                lane,iteration,directory);
    return status;
}

const char *physics_comparison_status_name(PhysicsComparisonStatus status)
{
    switch (status) {
    case PHYSICS_COMPARISON_OK: return "PHYSICS_COMPARISON_OK";
    case PHYSICS_COMPARISON_NOT_REQUESTED:
        return "PHYSICS_COMPARISON_NOT_REQUESTED";
    case PHYSICS_COMPARISON_INVALID_ARGUMENT:
        return "PHYSICS_COMPARISON_INVALID_ARGUMENT";
    case PHYSICS_COMPARISON_STALE_GENERATION:
        return "PHYSICS_COMPARISON_STALE_GENERATION";
    case PHYSICS_COMPARISON_INVALID_GRID:
        return "PHYSICS_COMPARISON_INVALID_GRID";
    case PHYSICS_COMPARISON_INVALID_VALUE:
        return "PHYSICS_COMPARISON_INVALID_VALUE";
    case PHYSICS_COMPARISON_IO_ERROR:
        return "PHYSICS_COMPARISON_IO_ERROR";
    default: return "PHYSICS_COMPARISON_UNKNOWN";
    }
}
