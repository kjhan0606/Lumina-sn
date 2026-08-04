#include "lumina_cmf_field.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KA3_C_LIGHT 2.99792458e10
#define KA3_SIGMA_X 0.04

typedef struct {
    double x0;
    double dx;
} GaussianBoundary;

static double stable_erf_difference(double high, double low)
{
    if (low >= 0.0) return erfc(low) - erfc(high);
    if (high <= 0.0) return erfc(-high) - erfc(-low);
    return erf(high) - erf(low);
}

static double gaussian_cell_average(void *opaque, double p, double mu, double nu)
{
    const GaussianBoundary *context = (const GaussianBoundary *)opaque;
    const double x = log(nu);
    const double root_two_sigma = sqrt(2.0) * KA3_SIGMA_X;
    const double high = (x + 0.5 * context->dx - context->x0) / root_two_sigma;
    const double low = (x - 0.5 * context->dx - context->x0) / root_two_sigma;
    (void)p; (void)mu;
    return KA3_SIGMA_X * sqrt(0.5 * 3.141592653589793238462643383279502884)
           * stable_erf_difference(high, low) / context->dx;
}

static int parse_size(const char *text, size_t *value)
{
    char *end=NULL; unsigned long long parsed;
    errno=0; parsed=strtoull(text,&end,10);
    if (errno!=0 || end==text || *end!='\0' || parsed==0 || parsed>SIZE_MAX) return 0;
    *value=(size_t)parsed; return 1;
}

static int parse_double(const char *text, double *value)
{
    char *end=NULL; errno=0; *value=strtod(text,&end);
    return errno==0 && end!=text && *end=='\0' && isfinite(*value);
}

static int run_ka1(size_t nr, size_t nmu, double tau_radius, const char *path)
{
    double *edge=NULL,*chi=NULL,*eta=NULL;
    const double nu[1]={1.0};
    LCMFInput input; LCMFOptions options; LCMFResult result; LCMFRayCache cache;
    FILE *stream=NULL; size_t i,m; int status=LCMF_OK;
    memset(&result,0,sizeof(result));
    edge=(double*)calloc(nr+1u,sizeof(double)); chi=(double*)calloc(nr,sizeof(double));
    eta=(double*)calloc(nr,sizeof(double));
    if (edge==NULL || chi==NULL || eta==NULL) { status=LCMF_ENOMEM; goto done; }
    for (i=0;i<=nr;++i) edge[i]=(double)i/(double)nr;
    for (i=0;i<nr;++i) {
        const double r=0.5*(edge[i]+edge[i+1u]);
        chi[i]=tau_radius; eta[i]=tau_radius*(1.0+0.5*r*r);
    }
    memset(&input,0,sizeof(input)); memset(&options,0,sizeof(options));
    input.nr=nr; input.nnu=1; input.r_edge=edge; input.nu=nu; input.chi_total=chi;
    input.eta_fixed=eta; input.t_exp_s=1.0; input.inner_bc=LCMF_BC_IRRADIATION;
    input.scatter_mode=LCMF_SCAT_NONE;
    options.n_mu=nmu; options.max_source_iter=1; options.source_rtol=1e-12;
    options.compute_hk=1; options.store_intensity=1; options.frequency_advection=0;
    options.n_r_eval=nr+1u; options.r_eval=edge;
    status=lumina_cmf_field_solve(&input,&options,&result);
    if (status!=LCMF_OK) {
        fprintf(stderr,"KA1 solve: %s: %s\n",lumina_cmf_status_string(status),result.error.message);
        goto done;
    }
    stream=fopen(path,"wb"); if (stream==NULL) { status=LCMF_EIO; goto done; }
    (void)fprintf(stream,"# nr=%zu nmu=%zu tau=%.17g residual=%.17g clamp=%llu bdf_eta_negative=%llu solution_negative_excess=%llu solution_subtruncation=%llu solution_sign_indeterminate_subtruncation=%llu solution_roundoff_enclosure_restart=%llu sign_uncertain=%llu nonfinite=%llu\n",
                  nr,nmu,tau_radius,result.transport_resid_linf,
                  (unsigned long long)result.clamp_count,
                  (unsigned long long)result.bdf_eta_negative_count,
                  (unsigned long long)result.solution_negative_excess_count,
                  (unsigned long long)result.solution_subtruncation_count,
                  (unsigned long long)result.solution_sign_indeterminate_subtruncation_count,
                  (unsigned long long)result.solution_roundoff_enclosure_restart_count,
                  (unsigned long long)result.sign_uncertain_count,
                  (unsigned long long)result.nonfinite_count);
    (void)fprintf(stream,"i\tm\tr\tmu\tIminus\tIplus\tJ\tH\tK\n");
    for (i=0;i<=nr;++i) {
        status=lumina_cmf_ray_cache_build_at_radius(edge[i],nmu,&cache,&result.error);
        if (status!=LCMF_OK) goto done;
        for (m=0;m<nmu;++m) {
            const size_t iq=(i*nmu+m);
            (void)fprintf(stream,"%zu\t%zu\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\n",
                          i,m,edge[i],cache.mu[m],result.I_minus[iq],
                          result.I_plus[iq],result.J[i],result.H[i],result.K[i]);
        }
        lumina_cmf_ray_cache_free(&cache);
    }
done:
    if (stream!=NULL && fclose(stream)!=0 && status==LCMF_OK) status=LCMF_EIO;
    lumina_cmf_result_free(&result); free(edge); free(chi); free(eta); return status;
}

static int run_ka2(size_t nr, size_t nmu, const char *path)
{
    const double epsilon = 0.2;
    const double chi0 = 1.0;
    const double b0 = 1.0;
    const double nu[1] = {1.0};
    double *edge = NULL, *chi = NULL, *eta_fixed = NULL, *chi_coherent = NULL;
    double *eta_converged = NULL;
    LCMFInput input, escape_input;
    LCMFOptions options, escape_options;
    LCMFResult result, escape_result;
    FILE *stream = NULL;
    double l_thermal = 0.0, l_absorbed = 0.0, l_escape, energy_closure;
    size_t i;
    int status = LCMF_OK;
    memset(&result, 0, sizeof(result));
    memset(&escape_result, 0, sizeof(escape_result));
    edge = (double *)calloc(nr + 1u, sizeof(double));
    chi = (double *)calloc(nr, sizeof(double));
    eta_fixed = (double *)calloc(nr, sizeof(double));
    chi_coherent = (double *)calloc(nr, sizeof(double));
    eta_converged = (double *)calloc(nr, sizeof(double));
    if (edge == NULL || chi == NULL || eta_fixed == NULL || chi_coherent == NULL ||
        eta_converged == NULL) { status = LCMF_ENOMEM; goto done; }
    for (i = 0u; i <= nr; ++i) edge[i] = (double)i / (double)nr;
    for (i = 0u; i < nr; ++i) {
        chi[i] = chi0;
        eta_fixed[i] = epsilon * chi0 * b0;
        chi_coherent[i] = (1.0 - epsilon) * chi0;
    }
    memset(&input, 0, sizeof(input)); memset(&options, 0, sizeof(options));
    input.nr = nr; input.nnu = 1u; input.r_edge = edge; input.nu = nu;
    input.chi_total = chi; input.eta_fixed = eta_fixed;
    input.chi_coherent = chi_coherent; input.t_exp_s = 1.0;
    input.inner_bc = LCMF_BC_IRRADIATION; input.scatter_mode = LCMF_SCAT_COHERENT;
    options.n_mu = nmu; options.max_source_iter = 500u; options.source_rtol = 1.0e-12;
    options.compute_hk = 1; options.store_intensity = 0; options.frequency_advection = 0;
    status = lumina_cmf_field_solve(&input, &options, &result);
    if (status != LCMF_OK) {
        fprintf(stderr, "KA2 coherent solve: %s: %s\n",
                lumina_cmf_status_string(status), result.error.message);
        goto done;
    }
    for (i = 0u; i < nr; ++i) eta_converged[i] = eta_fixed[i] + chi_coherent[i] * result.J[i];
    escape_input = input;
    escape_input.eta_fixed = eta_converged;
    escape_input.chi_coherent = NULL;
    escape_input.scatter_mode = LCMF_SCAT_NONE;
    memset(&escape_options, 0, sizeof(escape_options));
    escape_options.n_mu = nmu; escape_options.n_r_eval = nr + 1u;
    escape_options.r_eval = edge; escape_options.max_source_iter = 1u;
    escape_options.source_rtol = 1.0e-12; escape_options.compute_hk = 1;
    status = lumina_cmf_field_solve(&escape_input, &escape_options, &escape_result);
    if (status != LCMF_OK) {
        fprintf(stderr, "KA2 escape solve: %s: %s\n",
                lumina_cmf_status_string(status), escape_result.error.message);
        goto done;
    }
    for (i = 0u; i < nr; ++i) {
        const double volume = (4.0 * 3.141592653589793238462643383279502884 / 3.0) *
                              (edge[i + 1u] * edge[i + 1u] * edge[i + 1u] -
                               edge[i] * edge[i] * edge[i]);
        l_thermal += 4.0 * 3.141592653589793238462643383279502884 * eta_fixed[i] * volume;
        l_absorbed += 4.0 * 3.141592653589793238462643383279502884 *
                      epsilon * chi0 * result.J[i] * volume;
    }
    l_escape = 16.0 * 3.141592653589793238462643383279502884 *
               3.141592653589793238462643383279502884 * escape_result.H[nr];
    energy_closure = fabs(l_thermal - l_escape - l_absorbed) / l_thermal;
    stream = fopen(path, "wb"); if (stream == NULL) { status = LCMF_EIO; goto done; }
    (void)fprintf(stream, "# nr=%zu nmu=%zu source_iterations=%zu source_residual=%.17g transport_residual=%.17g energy_closure=%.17g Lthermal=%.17g Lescape=%.17g Labsorbed=%.17g clamp=%llu solution_negative_excess=%llu solution_subtruncation=%llu solution_sign_indeterminate_subtruncation=%llu solution_roundoff_enclosure_restart=%llu sign_uncertain=%llu nonfinite=%llu\n",
                  nr, nmu, result.source_iterations, result.source_resid_linf,
                  fmax(result.transport_resid_linf, escape_result.transport_resid_linf),
                  energy_closure, l_thermal, l_escape, l_absorbed,
                  (unsigned long long)result.clamp_count,
                  (unsigned long long)result.solution_negative_excess_count,
                  (unsigned long long)result.solution_subtruncation_count,
                  (unsigned long long)result.solution_sign_indeterminate_subtruncation_count,
                  (unsigned long long)result.solution_roundoff_enclosure_restart_count,
                  (unsigned long long)result.sign_uncertain_count,
                  (unsigned long long)result.nonfinite_count);
    (void)fprintf(stream, "i\tr\tJ\n");
    for (i = 0u; i <= nr; ++i) {
        (void)fprintf(stream, "%zu\t%.17g\t%.17g\n", i, edge[i], escape_result.J[i]);
    }
done:
    if (stream != NULL && fclose(stream) != 0 && status == LCMF_OK) status = LCMF_EIO;
    lumina_cmf_result_free(&escape_result); lumina_cmf_result_free(&result);
    free(eta_converged); free(chi_coherent); free(eta_fixed); free(chi); free(edge);
    return status;
}

static int run_ka3(size_t ns, size_t nnu, const char *path)
{
    const double inner = 1.0e10;
    const double outer = 2.0e10;
    const double shift = 0.1;
    const double x0 = 0.0;
    const double x_high = x0 + 8.0 * KA3_SIGMA_X;
    const double x_low = x0 - shift - 8.0 * KA3_SIGMA_X;
    const double dx = (x_high - x_low) / (double)(nnu - 1u);
    const double frequency_step = exp(dx);
    const double target_r[1] = {outer};
    double *edge = NULL, *nu = NULL, *chi = NULL, *eta = NULL;
    LCMFInput input; LCMFOptions options; LCMFResult result;
    GaussianBoundary boundary = {x0, dx};
    FILE *stream = NULL;
    size_t cells, i, k;
    int status = LCMF_OK;
    if (ns < 2u || nnu < 3u || ns > SIZE_MAX / nnu) return LCMF_EINVAL;
    cells = ns * nnu;
    memset(&result, 0, sizeof(result));
    edge = (double *)calloc(ns + 1u, sizeof(double));
    nu = (double *)calloc(nnu, sizeof(double));
    chi = (double *)calloc(cells, sizeof(double));
    eta = (double *)calloc(cells, sizeof(double));
    if (edge == NULL || nu == NULL || chi == NULL || eta == NULL) {
        status = LCMF_ENOMEM; goto done;
    }
    for (i = 0u; i <= ns; ++i) edge[i] = inner + (outer - inner) * (double)i / (double)ns;
    nu[0] = exp(x_high);
    for (k = 1u; k < nnu; ++k) nu[k] = nu[k - 1u] / frequency_step;
    memset(&input, 0, sizeof(input)); memset(&options, 0, sizeof(options));
    input.nr = ns; input.nnu = nnu; input.r_edge = edge; input.nu = nu;
    input.chi_total = chi; input.eta_fixed = eta;
    input.t_exp_s = (outer - inner) / (KA3_C_LIGHT * shift);
    input.inner_bc = LCMF_BC_IRRADIATION; input.scatter_mode = LCMF_SCAT_NONE;
    input.inner_irradiation = gaussian_cell_average; input.boundary_ctx = &boundary;
    options.n_mu = 1u; options.n_r_eval = 1u; options.r_eval = target_r;
    options.max_source_iter = 1u; options.source_rtol = 1.0e-12;
    options.store_intensity = 1; options.frequency_advection = 1;
    options.radial_characteristic = 1;
    status = lumina_cmf_field_solve(&input, &options, &result);
    if (status == LCMF_ENEGATIVE || status == LCMF_ESIGNUNCERTAIN ||
        status == LCMF_ENONFINITE) {
        fprintf(stderr, "KA3 measured guard: %s: %s ir=%zu inu=%zu ray=%zu segment=%zu substep=%zu value=%.17g interval=[%.17g,%.17g]\n",
                lumina_cmf_status_string(status), result.error.message,
                result.error.radial_index, result.error.frequency_index,
                result.error.ray_index, result.error.segment_index,
                result.error.substep_index, result.error.value,
                result.error.interval_lower, result.error.interval_upper);
    } else if (status != LCMF_OK) {
        fprintf(stderr, "KA3 solve: %s: %s\n",
                lumina_cmf_status_string(status), result.error.message);
        goto done;
    }
    stream = fopen(path, "wb"); if (stream == NULL) { status = LCMF_EIO; goto done; }
    (void)fprintf(stream, "# ns=%zu nnu=%zu A=%.17g dx=%.17g residual=%.17g clamp=%llu bdf_eta_negative=%llu bdf_eta_negative_planes=%llu solution_negative_excess=%llu solution_subtruncation=%llu solution_sign_indeterminate_subtruncation=%llu solution_roundoff_enclosure_restart=%llu sign_uncertain=%llu nonfinite=%llu bdf_eta_min=%.17g solution_min=%.17g solver_status=%d first_eval=%zu first_k=%zu first_ray=%zu first_segment=%zu first_substep=%zu first_endpoint=%zu first_eta=%.17g first_prev=%.17g first_prev2=%.17g first_decay_ratio=%.17g first_theoretical_limit=%.17g\n",
                  ns, nnu, shift, dx, result.transport_resid_linf,
                  (unsigned long long)result.clamp_count,
                  (unsigned long long)result.bdf_eta_negative_count,
                  (unsigned long long)result.bdf_eta_negative_plane_count,
                  (unsigned long long)result.solution_negative_excess_count,
                  (unsigned long long)result.solution_subtruncation_count,
                  (unsigned long long)result.solution_sign_indeterminate_subtruncation_count,
                  (unsigned long long)result.solution_roundoff_enclosure_restart_count,
                  (unsigned long long)result.sign_uncertain_count,
                  (unsigned long long)result.nonfinite_count,
                  result.bdf_eta_min, result.solution_min, status,
                  result.bdf_eta_first.radial_index,
                  result.bdf_eta_first.frequency_index,
                  result.bdf_eta_first.ray_index,
                  result.bdf_eta_first.segment_index,
                  result.bdf_eta_first.substep_index,
                  result.bdf_eta_first.endpoint_index,
                  result.bdf_eta_first.value,
                  result.bdf_eta_first.term_previous,
                  result.bdf_eta_first.term_previous2,
                  result.bdf_eta_first.decay_ratio,
                  result.bdf_eta_first.theoretical_limit);
    (void)fprintf(stream, "k\tx\tnu\tIin\tIout\n");
    for (k = 0u; k < nnu; ++k) {
        (void)fprintf(stream, "%zu\t%.17g\t%.17g\t%.17g\t%.17g\n",
                      k, log(nu[k]), nu[k], gaussian_cell_average(&boundary, 0.0, 1.0, nu[k]),
                      result.I_plus[k]);
    }
    status = LCMF_OK;
done:
    if (stream != NULL && fclose(stream) != 0 && status == LCMF_OK) status = LCMF_EIO;
    lumina_cmf_result_free(&result);
    free(edge); free(nu); free(chi); free(eta);
    return status;
}

int main(int argc, char **argv)
{
    size_t nr,nmu; double parameter; int status;
    if (argc == 5 && strcmp(argv[1], "ka3") == 0 && parse_size(argv[2], &nr) &&
        parse_size(argv[3], &nmu)) {
        status = run_ka3(nr, nmu, argv[4]);
        if (status != LCMF_OK) { fprintf(stderr,"driver failed: %s\n",lumina_cmf_status_string(status)); return 1; }
        return 0;
    }
    if (argc == 5 && strcmp(argv[1], "ka2") == 0 && parse_size(argv[2], &nr) &&
        parse_size(argv[3], &nmu)) {
        status = run_ka2(nr, nmu, argv[4]);
        if (status != LCMF_OK) { fprintf(stderr, "driver failed: %s\n", lumina_cmf_status_string(status)); return 1; }
        return 0;
    }
    if (argc!=6 || strcmp(argv[1],"ka1")!=0 || !parse_size(argv[2],&nr) ||
        !parse_size(argv[3],&nmu) || !parse_double(argv[4],&parameter)) {
        fprintf(stderr,"usage: %s ka1 NR NMU TAU_RADIUS OUTPUT.tsv | ka2 NR NMU OUTPUT.tsv | ka3 NS NNU OUTPUT.tsv\n",argv[0]); return 2;
    }
    status=run_ka1(nr,nmu,parameter,argv[5]);
    if (status!=LCMF_OK) { fprintf(stderr,"driver failed: %s\n",lumina_cmf_status_string(status)); return 1; }
    return 0;
}
