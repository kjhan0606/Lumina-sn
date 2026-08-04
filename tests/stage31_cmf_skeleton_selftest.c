#include "lumina_cmf_field.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int expect_status(int got, int expected, const char *label)
{
    if (got != expected) {
        fprintf(stderr, "%s: got %s expected %s\n", label,
                lumina_cmf_status_string(got), lumina_cmf_status_string(expected));
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    const double edge[] = {0.0, 0.5, 1.0};
    const double nu[] = {4.0, 2.0, 1.0};
    const double chi[] = {1.0,1.0,1.0,1.0,1.0,1.0};
    const double eta[] = {1.0,1.0,1.0,1.0,1.0,1.0};
    const double zeros[] = {0.0,0.0,0.0,0.0,0.0,0.0};
    const double boundary[] = {1.0,1.0,1.0};
    LCMFInput input;
    LCMFOptions options;
    LCMFError error;
    LCMFRayCache cache;
    double intensity = 0.0, exact;
    int failures = 0;
    memset(&input,0,sizeof(input)); memset(&options,0,sizeof(options));
    input.nr=2; input.nnu=3; input.r_edge=edge; input.nu=nu;
    input.chi_total=chi; input.eta_fixed=eta; input.chi_coherent=zeros;
    input.inner_bc=LCMF_BC_DIFFUSION; input.scatter_mode=LCMF_SCAT_NONE;
    input.B_inner=boundary; input.dB_dtau_inner=zeros; input.t_exp_s=1.0;
    options.n_mu=8; options.max_source_iter=2; options.source_rtol=1e-11;
    failures += expect_status(lumina_cmf_validate_input(&input,&options,&error),LCMF_OK,"valid");
    {
        double bad_nu[] = {4.0,2.0,2.1}; input.nu=bad_nu;
        failures += expect_status(lumina_cmf_validate_input(&input,&options,&error),LCMF_EGRID,"frequency order");
        input.nu=nu;
    }
    {
        double bad_chi[] = {1.0,1.0,-1.0,1.0,1.0,1.0}; input.chi_total=bad_chi;
        failures += expect_status(lumina_cmf_validate_input(&input,&options,&error),LCMF_ENEGATIVE,"negative chi");
        input.chi_total=chi;
    }
    {
        const double steep_positive_eta[] = {1.0,1.0,1.0,0.1,0.1,0.1};
        LCMFResult positive_result;
        memset(&positive_result, 0, sizeof(positive_result));
        input.eta_fixed = steep_positive_eta;
        failures += expect_status(lumina_cmf_field_solve(&input,&options,&positive_result),
                                  LCMF_OK,"positive geometric outer extrapolation");
        lumina_cmf_result_free(&positive_result);
        input.eta_fixed = eta;
    }
    {
        const double zero_outer_stencil_eta[] = {1.0,1.0,1.0,0.0,0.0,0.0};
        LCMFResult failed_result;
        memset(&failed_result, 0, sizeof(failed_result));
        input.eta_fixed = zero_outer_stencil_eta;
        failures += expect_status(lumina_cmf_field_solve(&input,&options,&failed_result),
                                  LCMF_ENEGATIVE,"zero outer log stencil");
        if (failed_result.error.radial_index != input.nr ||
            failed_result.error.frequency_index != 0u ||
            failed_result.error.value != 0.0 ||
            failed_result.bdf_eta_negative_count != 0u ||
            failed_result.solution_negative_excess_count != 0u ||
            failed_result.sign_uncertain_count != 0u ||
            strstr(failed_result.error.message,"positive stencil values at outer face") == NULL) {
            fprintf(stderr,"zero outer log-stencil failure was not recorded exactly\n");
            ++failures;
        }
        lumina_cmf_result_free(&failed_result);
        input.eta_fixed = eta;
    }
    {
        const double edge3[] = {0.0,0.25,0.5,1.0};
        const double chi3[] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
        const double zero3[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
        const double zero_inner_stencil_eta[] = {
            0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0
        };
        LCMFInput inner_input = input;
        LCMFResult failed_result;
        memset(&failed_result, 0, sizeof(failed_result));
        inner_input.nr = 3u;
        inner_input.r_edge = edge3;
        inner_input.chi_total = chi3;
        inner_input.eta_fixed = zero_inner_stencil_eta;
        inner_input.chi_coherent = zero3;
        failures += expect_status(lumina_cmf_field_solve(&inner_input,&options,&failed_result),
                                  LCMF_ENEGATIVE,"zero inner log stencil");
        if (failed_result.error.radial_index != 0u ||
            failed_result.error.frequency_index != 0u ||
            failed_result.error.value != 0.0 ||
            strstr(failed_result.error.message,"positive stencil values at inner face") == NULL) {
            fprintf(stderr,"zero inner log-stencil failure was not recorded exactly: "
                    "radial=%zu frequency=%zu value=%.17g message=%s\n",
                    failed_result.error.radial_index,
                    failed_result.error.frequency_index,
                    failed_result.error.value, failed_result.error.message);
            ++failures;
        }
        lumina_cmf_result_free(&failed_result);
    }
    input.scatter_mode=LCMF_SCAT_REDISTRIBUTION;
    failures += expect_status(lumina_cmf_validate_input(&input,&options,&error),LCMF_EUNSUPPORTED,"redistribution");
    input.scatter_mode=LCMF_SCAT_NONE;
    {
        char binary_path[96], manifest_path[96];
        FILE *stream;
        LCMFFrozenField frozen;
        (void)snprintf(binary_path,sizeof(binary_path),"/tmp/stage31_bad_%ld.bin",(long)getpid());
        (void)snprintf(manifest_path,sizeof(manifest_path),"/tmp/stage31_bad_%ld.manifest",(long)getpid());
        stream=fopen(binary_path,"wb");
        if (stream==NULL) return 2;
        (void)fwrite("BADMAGIC",1,8,stream);
        fclose(stream);
        stream=fopen(manifest_path,"wb");
        if (stream==NULL) return 2;
        (void)fprintf(stream,"sha256=0000000000000000000000000000000000000000000000000000000000000000\n");
        fclose(stream);
        failures += expect_status(lumina_cmf_frozen_load(binary_path,manifest_path,&frozen,&error),LCMF_ESCHEMA,"malformed schema");
        (void)remove(binary_path); (void)remove(manifest_path);
    }
    failures += expect_status(lumina_cmf_ray_cache_build(edge,2,1,8,&cache,&error),LCMF_OK,"ray cache");
    if (cache.n_mu==8) {
        double weight_sum=0.0;
        for (size_t i=0;i<cache.n_mu;++i) {
            weight_sum += cache.weight[i];
            if (!(cache.mu[i]>0.0 && cache.mu[i]<1.0 && cache.p[i]>=0.0)) ++failures;
        }
        if (fabs(weight_sum-1.0)>1e-14) ++failures;
    }
    lumina_cmf_ray_cache_free(&cache);
    failures += expect_status(lumina_cmf_sc_linear(0.3,1.2,1.2,0.7,&intensity),LCMF_OK,"SC constant source");
    exact=0.3*exp(-0.7)+1.2*(1.0-exp(-0.7));
    if (fabs(intensity-exact)>4.0e-15) { fprintf(stderr,"SC error %.17g\n",intensity-exact); ++failures; }
    {
        const LCMFInterval incoming = {0.3, 0.3, 0.3};
        const LCMFInterval source_up = {-0.1, -0.1, -0.1};
        const LCMFInterval source_down = {-0.2, -0.2, -0.2};
        LCMFInterval signed_result;
        double one_minus_e = -expm1(-0.7);
        double ratio = one_minus_e / 0.7;
        double psi_down = 1.0 - ratio;
        double psi_up = one_minus_e - psi_down;
        exact = 0.3 * exp(-0.7) - 0.1 * psi_up - 0.2 * psi_down;
        failures += expect_status(lumina_cmf_sc_linear_signed(&incoming,&source_up,
                                  &source_down,0.7,&signed_result),LCMF_OK,"signed SC analytic");
        if (fabs(signed_result.value-exact)>4.0e-15 ||
            signed_result.lower > exact || signed_result.upper < exact) {
            fprintf(stderr,"signed SC enclosure failed\n"); ++failures;
        }
    }
    {
        const LCMFInterval zero = {0.0, 0.0, 0.0};
        const LCMFInterval negative = {-1.0, -1.0, -1.0};
        const LCMFInterval uncertain = {0.0, -1.0e-15, 1.0e-15};
        LCMFInterval result_negative, result_uncertain;
        failures += expect_status(lumina_cmf_sc_linear_signed(&zero,&negative,&negative,
                                  0.7,&result_negative),LCMF_OK,"signed SC negative interval");
        failures += expect_status(lumina_cmf_sc_linear_signed(&uncertain,&zero,&zero,
                                  0.7,&result_uncertain),LCMF_OK,"signed SC straddling interval");
        if (!(result_negative.upper < 0.0) ||
            !(result_uncertain.lower < 0.0 && result_uncertain.upper > 0.0)) {
            fprintf(stderr,"signed SC guard branches failed\n"); ++failures;
        }
    }
    {
        const LCMFInterval start = {0.25, 0.25, 0.25};
        const LCMFInterval zero = {0.0, 0.0, 0.0};
        const LCMFInterval constant = {2.0, 2.0, 2.0};
        const LCMFInterval linear = {3.0, 3.0, 3.0};
        const LCMFInterval quadratic = {3.0, 3.0, 3.0};
        const LCMFInterval negative = {-2.0, -2.0, -2.0};
        const LCMFInterval uncertain = {0.0, -1.0e-15, 1.0e-15};
        LCMFInterval result;
        failures += expect_status(lumina_cmf_sc_quadratic_signed(
            &start, &constant, &zero, &zero, 0.4, 0.0, &result),
            LCMF_OK, "quadratic SC constant vacuum");
        if (fabs(result.value - 1.05) > 8.0e-15) ++failures;
        failures += expect_status(lumina_cmf_sc_quadratic_signed(
            &start, &constant, &linear, &zero, 0.4, 0.0, &result),
            LCMF_OK, "quadratic SC linear vacuum");
        if (fabs(result.value - 1.65) > 8.0e-15) ++failures;
        failures += expect_status(lumina_cmf_sc_quadratic_signed(
            &start, &constant, &linear, &quadratic, 0.4, 0.0, &result),
            LCMF_OK, "quadratic SC quadratic vacuum");
        if (fabs(result.value - 2.05) > 8.0e-15) ++failures;
        failures += expect_status(lumina_cmf_sc_quadratic_signed(
            &start, &constant, &linear, &quadratic, 0.4, 1.0e-12, &result),
            LCMF_OK, "quadratic SC tau to zero");
        if (fabs(result.value - 2.05) > 3.0e-12) ++failures;
        failures += expect_status(lumina_cmf_sc_quadratic_signed(
            &zero, &negative, &zero, &zero, 0.4, 0.7, &result),
            LCMF_OK, "quadratic SC negative interval");
        if (!(result.upper < 0.0)) ++failures;
        failures += expect_status(lumina_cmf_sc_quadratic_signed(
            &uncertain, &zero, &zero, &zero, 0.4, 0.7, &result),
            LCMF_OK, "quadratic SC straddling interval");
        if (!(result.lower < 0.0 && result.upper > 0.0)) ++failures;
    }
    {
        const double exact_positive_previous = 2.2972677906250876e-6;
        const double exact_positive_previous2 = 4.7547305910730000e-6;
        const double observed_previous = 1.1251344515546073e-7;
        const double observed_previous2 = 5.0547558929601476e-7;
        const double exact_tail_previous = 1.0;
        const double exact_tail_previous2 = 4.0117088489;
        const double exact_eta_history = 2.0 * exact_positive_previous
                                       - 0.5 * exact_positive_previous2;
        const double exact_tail_eta_history = 2.0 * exact_tail_previous
                                            - 0.5 * exact_tail_previous2;
        const double observed_eta_history = 2.0 * observed_previous
                                          - 0.5 * observed_previous2;
        if (!(exact_eta_history > 0.0) || !(exact_tail_eta_history < 0.0) ||
            !(observed_eta_history < 0.0) ||
            !(observed_previous2 / observed_previous > 4.0)) {
            fprintf(stderr,"BDF exact/observed history regression failed\n"); ++failures;
        }
    }
    {
        const double source[] = {0.4,1.1,0.7,1.8};
        const double dtau[] = {0.03,0.6,2.0,0.2};
        double recursive=0.25, closed=0.25*exp(-(dtau[0]+dtau[1]+dtau[2]+dtau[3]));
        for (size_t j=0;j<4u;++j) {
            failures += expect_status(lumina_cmf_sc_linear(recursive,source[j],source[j],dtau[j],&recursive),LCMF_OK,"piecewise SC");
            double downstream_tau=0.0;
            for (size_t q=j+1u;q<4u;++q) downstream_tau += dtau[q];
            closed += source[j]*(1.0-exp(-dtau[j]))*exp(-downstream_tau);
        }
        if (fabs(recursive-closed)>8.0e-15) { fprintf(stderr,"piecewise SC error %.17g\n",recursive-closed); ++failures; }
    }
    {
        const double coherent_edge[] = {0.0,0.25,0.5,0.75,1.0};
        const double coherent_nu[] = {1.0};
        const double coherent_chi[] = {1.0,1.0,1.0,1.0};
        const double coherent_eta[] = {0.2,0.2,0.2,0.2};
        const double coherent_opacity[] = {0.8,0.8,0.8,0.8};
        LCMFInput coherent_input;
        LCMFOptions coherent_options;
        LCMFResult coherent_result;
        int coherent_status;
        memset(&coherent_input, 0, sizeof(coherent_input));
        memset(&coherent_options, 0, sizeof(coherent_options));
        memset(&coherent_result, 0, sizeof(coherent_result));
        coherent_input.nr=4u; coherent_input.nnu=1u;
        coherent_input.r_edge=coherent_edge; coherent_input.nu=coherent_nu;
        coherent_input.chi_total=coherent_chi; coherent_input.eta_fixed=coherent_eta;
        coherent_input.chi_coherent=coherent_opacity; coherent_input.t_exp_s=1.0;
        coherent_input.inner_bc=LCMF_BC_IRRADIATION;
        coherent_input.scatter_mode=LCMF_SCAT_COHERENT;
        coherent_options.n_mu=4u; coherent_options.max_source_iter=200u;
        coherent_options.source_rtol=1.0e-11;
        coherent_status=lumina_cmf_field_solve(&coherent_input,&coherent_options,
                                               &coherent_result);
        failures += expect_status(coherent_status,LCMF_OK,"plain coherent source iteration");
        if (coherent_status==LCMF_OK &&
            (coherent_result.source_iterations<=1u ||
             coherent_result.source_resid_linf>1.0e-11 || coherent_result.J[0]<=0.0 ||
             coherent_result.clamp_count!=0u || coherent_result.nonfinite_count!=0u)) {
            fprintf(stderr,"plain coherent iteration contract failed\n"); ++failures;
        }
        lumina_cmf_result_free(&coherent_result);
    }
    {
        const double core_edge[] = {1.0e10,1.25e10,1.5e10,1.75e10,2.0e10};
        const double core_nu[] = {4.0,2.0,1.0};
        const double core_zero[12] = {0.0};
        const double core_boundary[] = {1.0,1.0,1.0};
        const double target[] = {2.0e10};
        const double shift = 0.1;
        const double dx = log(2.0);
        const double beta_length = shift * (3.0 + 2.0 / dx);
        const double source_ratio = (2.0 / dx - 3.0) / (3.0 + 2.0 / dx);
        const double trapezoidal_exact = exp(-beta_length) +
                                         source_ratio * (-expm1(-beta_length));
        LCMFInput core_input;
        LCMFOptions core_options;
        LCMFResult core_result;
        int core_status;
        memset(&core_input, 0, sizeof(core_input));
        memset(&core_options, 0, sizeof(core_options));
        memset(&core_result, 0, sizeof(core_result));
        core_input.nr = 4u;
        core_input.nnu = 3u;
        core_input.r_edge = core_edge;
        core_input.nu = core_nu;
        core_input.chi_total = core_zero;
        core_input.eta_fixed = core_zero;
        core_input.t_exp_s = 1.0e10 / (2.99792458e10 * shift);
        core_input.inner_bc = LCMF_BC_DIFFUSION;
        core_input.scatter_mode = LCMF_SCAT_NONE;
        core_input.B_inner = core_boundary;
        core_input.dB_dtau_inner = core_zero;
        core_options.n_mu = 1u;
        core_options.n_r_eval = 1u;
        core_options.r_eval = target;
        core_options.max_source_iter = 1u;
        core_options.source_rtol = 1.0e-12;
        core_options.store_intensity = 1;
        core_options.frequency_advection = 1;
        core_options.radial_characteristic = 1;
        core_status = lumina_cmf_field_solve(&core_input, &core_options, &core_result);
        failures += expect_status(core_status, LCMF_OK,
                                  "trapezoidal start and core-local stencil");
        if (core_status == LCMF_OK &&
            (fabs(core_result.I_plus[1] - trapezoidal_exact) > 2.0e-13 ||
             !isfinite(core_result.I_plus[2]) || core_result.I_plus[2] <= 0.0)) {
            fprintf(stderr, "trapezoidal/core-stencil regression failed\n");
            ++failures;
        }
        lumina_cmf_result_free(&core_result);
    }
    if (failures!=0) { fprintf(stderr,"stage31 skeleton failures=%d\n",failures); return 1; }
    if (argc == 3) {
        LCMFFrozenField frozen;
        failures += expect_status(lumina_cmf_frozen_load(argv[1],argv[2],&frozen,&error),LCMF_OK,"valid frozen field");
        if (failures==0 && (frozen.nr!=1u || frozen.nnu!=2u || frozen.iteration!=10u ||
                            frozen.field_generation!=7u)) ++failures;
        lumina_cmf_frozen_free(&frozen);
    }
    if (failures!=0) { fprintf(stderr,"stage31 loader failures=%d\n",failures); return 1; }
    puts("stage31 skeleton PASS");
    return 0;
}
