#include "a210_coupled_newton.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double cn_monotonic_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static double cn_realtime_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

void a210_coupled_newton_default_config(A210CoupledNewtonConfig *config,
                                        size_t n)
{
    if (!config)
        return;
    memset(config, 0, sizeof(*config));
    config->n = n;
    config->te_min = 0.0;
    config->te_max = 0.0;
    config->tol_te = 1e-12;
    config->tol_ne = 1e-8;
    config->fd_step = 1e-6;
    config->max_newton = 40;
    config->max_evals = 200;
    config->tag = "A2-10";
}

const char *a210_coupled_newton_status_name(A210CoupledNewtonStatus status)
{
    switch (status) {
    case A210_CN_CONVERGED:            return "CONVERGED";
    case A210_CN_INVALID_ARGUMENT:     return "INVALID_ARGUMENT";
    case A210_CN_ALLOCATION_FAILED:    return "ALLOCATION_FAILED";
    case A210_CN_INITIAL_EVAL_FAILED:  return "INITIAL_EVAL_FAILED";
    case A210_CN_JACOBIAN_EVAL_FAILED: return "JACOBIAN_EVAL_FAILED";
    case A210_CN_SINGULAR_JACOBIAN:    return "SINGULAR_JACOBIAN";
    case A210_CN_LINESEARCH_FAILED:    return "LINESEARCH_FAILED";
    case A210_CN_NOT_CONVERGED:        return "NOT_CONVERGED";
    }
    return "UNKNOWN";
}

typedef struct {
    const A210CoupledNewtonConfig *cfg;
    A210CoupledResidualFn fn;
    void *ctx;
    size_t n, m;
    double *te, *ne, *f_te, *f_ne;
    int evaluations, rejected;
    int iteration;
} CnState;

/* Evaluate F at x (m = 2n, interleaved ln T, ln n_e).  Returns 0 on a
 * complete, finite, in-domain evaluation and fills F, *phi. */
static int cn_eval(CnState *st, const double *x, double *F, double *phi,
                   void *aux, const char *kind, double lambda)
{
    const A210CoupledNewtonConfig *cfg = st->cfg;
    size_t n = st->n;
    int rc = 0;
    const char *reason = "OK";
    double t0 = cn_realtime_s(), m0 = cn_monotonic_s();

    for (size_t s = 0; s < n; s++) {
        double te = exp(x[2 * s]), ne = exp(x[2 * s + 1]);
        if (!isfinite(te) || !isfinite(ne) || ne <= 0.0) {
            rc = 1;
            reason = "NONFINITE_TRIAL";
            break;
        }
        if (te < cfg->te_min || te > cfg->te_max) {
            rc = 1;
            reason = "OUTSIDE_QUALIFIED_DOMAIN";
            break;
        }
        st->te[s] = te;
        st->ne[s] = ne;
    }
    if (rc == 0) {
        st->evaluations++;
        if (st->fn(st->te, st->ne, n, st->f_te, st->f_ne, aux, st->ctx) != 0) {
            rc = 1;
            reason = "CALLBACK_REJECTED";
        }
    }
    double sum = 0.0, max_te = 0.0, max_ne = 0.0;
    size_t worst = 0;
    if (rc == 0) {
        for (size_t s = 0; s < n; s++) {
            double a = st->f_te[s], b = st->f_ne[s];
            if (!isfinite(a) || !isfinite(b)) {
                rc = 1;
                reason = "NONFINITE_RESIDUAL";
                break;
            }
            F[2 * s] = a;
            F[2 * s + 1] = b;
            sum += a * a + b * b;
            if (fabs(a) > max_te) {
                max_te = fabs(a);
                worst = s;
            }
            if (fabs(b) > max_ne)
                max_ne = fabs(b);
        }
    }
    if (rc != 0)
        st->rejected++;
    else
        *phi = 0.5 * sum;
    if (cfg->log)
        fprintf(cfg->log,
                "[%s][COUPLED-NEWTON][EVAL] outer_iteration=%llu "
                "newton_iteration=%d evaluation=%d kind=%s lambda=%.17g "
                "t_start_unix=%.3f wall_s=%.3f rc=%d reason=%s "
                "max_abs_f_te=%.17g max_abs_f_ne=%.17g worst_shell_te=%zu "
                "phi=%.17g T0=%.17g ne0=%.17g\n",
                cfg->tag ? cfg->tag : "A2-10",
                (unsigned long long)cfg->outer_iteration, st->iteration,
                st->evaluations, kind, lambda, t0, cn_monotonic_s() - m0, rc,
                reason, max_te, max_ne, worst, rc == 0 ? *phi : NAN,
                st->te[0], st->ne[0]);
    return rc;
}

/* In-place LU with partial pivoting; solves A d = b (b overwritten by d). */
static int cn_lu_solve(double *A, double *b, size_t m)
{
    for (size_t k = 0; k < m; k++) {
        size_t p = k;
        double big = fabs(A[k * m + k]);
        for (size_t i = k + 1; i < m; i++)
            if (fabs(A[i * m + k]) > big) {
                big = fabs(A[i * m + k]);
                p = i;
            }
        if (!isfinite(big) || big <= DBL_MIN)
            return 1;
        if (p != k) {
            for (size_t j = 0; j < m; j++) {
                double t = A[k * m + j];
                A[k * m + j] = A[p * m + j];
                A[p * m + j] = t;
            }
            double t = b[k];
            b[k] = b[p];
            b[p] = t;
        }
        for (size_t i = k + 1; i < m; i++) {
            double l = A[i * m + k] / A[k * m + k];
            A[i * m + k] = l;
            for (size_t j = k + 1; j < m; j++)
                A[i * m + j] -= l * A[k * m + j];
            b[i] -= l * b[k];
        }
    }
    for (size_t k = m; k-- > 0;) {
        double s = b[k];
        for (size_t j = k + 1; j < m; j++)
            s -= A[k * m + j] * b[j];
        b[k] = s / A[k * m + k];
        if (!isfinite(b[k]))
            return 1;
    }
    return 0;
}

/* Two-colour forward-difference Jacobian (see header). */
static int cn_jacobian(CnState *st, const double *x, const double *F,
                       double *J, double *xp, double *Fp, void *aux_scratch)
{
    const A210CoupledNewtonConfig *cfg = st->cfg;
    size_t n = st->n, m = st->m;
    static const char *kinds[2][2] = {{"FD_EVEN_T", "FD_EVEN_NE"},
                                      {"FD_ODD_T", "FD_ODD_NE"}};
    double *hv = malloc(n * sizeof(*hv));
    if (!hv)
        return 1;
    memset(J, 0, m * m * sizeof(*J));
    for (int colour = 0; colour < 2; colour++) {
        for (int v = 0; v < 2; v++) {
            memcpy(xp, x, m * sizeof(*xp));
            for (size_t j = 0; j < n; j++) {
                hv[j] = 0.0;
                if ((int)(j % 2) != colour)
                    continue;
                double h = cfg->fd_step;
                if (v == 0 && exp(x[2 * j] + h) > cfg->te_max)
                    h = -h;
                hv[j] = h;
                xp[2 * j + v] += h;
            }
            double phi;
            if (st->evaluations >= cfg->max_evals ||
                cn_eval(st, xp, Fp, &phi, aux_scratch, kinds[colour][v],
                        0.0) != 0) {
                free(hv);
                return 1;
            }
            for (size_t s = 0; s < n; s++) {
                size_t j;
                if (s == 0)
                    j = colour == 0 ? 0 : 1;
                else
                    j = (int)(s % 2) == colour ? s : s - 1;
                for (int e = 0; e < 2; e++)
                    J[(2 * s + e) * m + (2 * j + v)] =
                        (Fp[2 * s + e] - F[2 * s + e]) / hv[j];
            }
        }
    }
    free(hv);
    return 0;
}

static int cn_converged(const A210CoupledNewtonConfig *cfg, const double *F,
                        size_t n)
{
    for (size_t s = 0; s < n; s++)
        if (!(fabs(F[2 * s]) <= cfg->tol_te) ||
            !(fabs(F[2 * s + 1]) <= cfg->tol_ne))
            return 0;
    return 1;
}

A210CoupledNewtonStatus a210_coupled_newton_solve(
    const A210CoupledNewtonConfig *cfg, A210CoupledResidualFn fn,
    void *ctx, double *te, double *ne, void *eval_out_at_root,
    A210CoupledNewtonReport *report)
{
    A210CoupledNewtonReport rep;
    memset(&rep, 0, sizeof(rep));
    double m0 = cn_monotonic_s();
    if (report)
        memset(report, 0, sizeof(*report));
    if (!cfg || !fn || !te || !ne || cfg->n < 2 ||
        cfg->n > ((size_t)1 << 20) || !(cfg->te_min > 0.0) ||
        !(cfg->te_max > cfg->te_min) || !(cfg->tol_te > 0.0) ||
        !(cfg->tol_ne > 0.0) || !(cfg->fd_step > 0.0) ||
        cfg->max_newton < 1 || cfg->max_evals < 1 ||
        (cfg->eval_out_bytes && !eval_out_at_root)) {
        rep.status = A210_CN_INVALID_ARGUMENT;
        if (report)
            *report = rep;
        return rep.status;
    }
    size_t n = cfg->n, m = 2 * n;
    CnState st;
    memset(&st, 0, sizeof(st));
    st.cfg = cfg;
    st.fn = fn;
    st.ctx = ctx;
    st.n = n;
    st.m = m;
    st.te = malloc(n * sizeof(double));
    st.ne = malloc(n * sizeof(double));
    st.f_te = malloc(n * sizeof(double));
    st.f_ne = malloc(n * sizeof(double));
    double *x = malloc(m * sizeof(double)), *xt = malloc(m * sizeof(double));
    double *xp = malloc(m * sizeof(double));
    double *F = malloc(m * sizeof(double)), *Ft = malloc(m * sizeof(double));
    double *Fp = malloc(m * sizeof(double)), *d = malloc(m * sizeof(double));
    double *J = malloc(m * m * sizeof(double));
    double *LU = malloc(m * m * sizeof(double));
    size_t ab = cfg->eval_out_bytes;
    unsigned char *aux_base = ab ? malloc(ab) : NULL;
    unsigned char *aux_trial = ab ? malloc(ab) : NULL;
    unsigned char *aux_scratch = ab ? malloc(ab) : NULL;
    A210CoupledNewtonStatus status = A210_CN_NOT_CONVERGED;
    if (!st.te || !st.ne || !st.f_te || !st.f_ne || !x || !xt || !xp || !F ||
        !Ft || !Fp || !d || !J || !LU ||
        (ab && (!aux_base || !aux_trial || !aux_scratch))) {
        status = A210_CN_ALLOCATION_FAILED;
        goto done;
    }
    for (size_t s = 0; s < n; s++) {
        if (!isfinite(te[s]) || !isfinite(ne[s]) || te[s] <= 0.0 ||
            ne[s] <= 0.0) {
            status = A210_CN_INVALID_ARGUMENT;
            goto done;
        }
        x[2 * s] = log(te[s]);
        x[2 * s + 1] = log(ne[s]);
    }

    double phi = 0.0;
    if (cn_eval(&st, x, F, &phi, aux_base, "BASE", 0.0) != 0) {
        status = A210_CN_INITIAL_EVAL_FAILED;
        goto done;
    }
    int need_jacobian = 1, jacobian_fresh = 0;
    for (;;) {
        if (cn_converged(cfg, F, n)) {
            status = A210_CN_CONVERGED;
            break;
        }
        if (st.iteration >= cfg->max_newton ||
            st.evaluations >= cfg->max_evals) {
            status = A210_CN_NOT_CONVERGED;
            break;
        }
        if (need_jacobian) {
            if (cn_jacobian(&st, x, F, J, xp, Fp, aux_scratch) != 0) {
                status = st.evaluations >= cfg->max_evals
                       ? A210_CN_NOT_CONVERGED
                       : A210_CN_JACOBIAN_EVAL_FAILED;
                break;
            }
            rep.jacobians++;
            need_jacobian = 0;
            jacobian_fresh = 1;
        }
        memcpy(LU, J, m * m * sizeof(*LU));
        for (size_t i = 0; i < m; i++)
            d[i] = -F[i];
        if (cn_lu_solve(LU, d, m) != 0) {
            if (!jacobian_fresh) {
                need_jacobian = 1;
                continue;
            }
            status = A210_CN_SINGULAR_JACOBIAN;
            break;
        }
        double max_dlnt = 0.0, max_dlnne = 0.0;
        for (size_t s = 0; s < n; s++) {
            max_dlnt = fmax(max_dlnt, fabs(d[2 * s]));
            max_dlnne = fmax(max_dlnne, fabs(d[2 * s + 1]));
        }
        /* Backtracking: a trial outside the qualified domain is rejected
         * before any evaluation (no cost); at most 12 evaluated trials. */
        double lambda = 1.0, phi_t = 0.0;
        int accepted = 0, evaluated = 0, domain_skips = 0;
        for (int k = 0; k < 60 && evaluated < 12; k++) {
            if (st.evaluations >= cfg->max_evals)
                break;
            int in_domain = 1;
            for (size_t i = 0; i < m; i++) {
                xt[i] = x[i] + lambda * d[i];
                if (i % 2 == 0) {
                    double t = exp(xt[i]);
                    if (!(t >= cfg->te_min && t <= cfg->te_max))
                        in_domain = 0;
                }
            }
            if (!in_domain) {
                domain_skips++;
                lambda *= 0.5;
                continue;
            }
            evaluated++;
            if (cn_eval(&st, xt, Ft, &phi_t, aux_trial, "LINESEARCH",
                        lambda) == 0 &&
                phi_t <= (1.0 - 1e-4 * lambda) * phi) {
                accepted = 1;
                break;
            }
            lambda *= 0.5;
        }
        if (cfg->log)
            fprintf(cfg->log,
                    "[%s][COUPLED-NEWTON][ITER] outer_iteration=%llu "
                    "newton_iteration=%d jacobian=%s accepted=%d "
                    "lambda=%.17g phi_old=%.17g phi_new=%.17g "
                    "max_abs_dlnT=%.17g max_abs_dlnne=%.17g "
                    "linesearch_evaluated=%d domain_skips=%d evaluations=%d\n",
                    cfg->tag ? cfg->tag : "A2-10",
                    (unsigned long long)cfg->outer_iteration, st.iteration,
                    jacobian_fresh ? "FRESH" : "REUSED", accepted, lambda,
                    phi, accepted ? phi_t : NAN, max_dlnt, max_dlnne,
                    evaluated, domain_skips, st.evaluations);
        st.iteration++;
        if (!accepted) {
            if (!jacobian_fresh && st.evaluations < cfg->max_evals) {
                need_jacobian = 1;
                continue;
            }
            status = st.evaluations >= cfg->max_evals
                   ? A210_CN_NOT_CONVERGED
                   : A210_CN_LINESEARCH_FAILED;
            break;
        }
        int strong = lambda == 1.0 && phi_t <= 0.25 * phi;
        memcpy(x, xt, m * sizeof(*x));
        memcpy(F, Ft, m * sizeof(*F));
        if (ab) {
            unsigned char *t = aux_base;
            aux_base = aux_trial;
            aux_trial = t;
        }
        phi = phi_t;
        need_jacobian = !strong;
        jacobian_fresh = 0;
    }

    rep.max_abs_f_te = 0.0;
    rep.max_abs_f_ne = 0.0;
    for (size_t s = 0; s < n; s++) {
        rep.max_abs_f_te = fmax(rep.max_abs_f_te, fabs(F[2 * s]));
        rep.max_abs_f_ne = fmax(rep.max_abs_f_ne, fabs(F[2 * s + 1]));
    }
    if (status == A210_CN_CONVERGED) {
        for (size_t s = 0; s < n; s++) {
            te[s] = exp(x[2 * s]);
            ne[s] = exp(x[2 * s + 1]);
        }
        if (ab)
            memcpy(eval_out_at_root, aux_base, ab);
    }

done:
    rep.status = status;
    rep.newton_iterations = st.iteration;
    rep.evaluations = st.evaluations;
    rep.rejected_trials = st.rejected;
    rep.wall_seconds = cn_monotonic_s() - m0;
    if (cfg && cfg->log)
        fprintf(cfg->log,
                "[%s][COUPLED-NEWTON][RESULT] outer_iteration=%llu status=%s "
                "newton_iterations=%d evaluations=%d jacobians=%d "
                "rejected_trials=%d max_abs_f_te=%.17g max_abs_f_ne=%.17g "
                "tol_te=%.17g tol_ne=%.17g wall_s=%.3f clamp=0 floor=0 "
                "fallback=0\n",
                cfg->tag ? cfg->tag : "A2-10",
                (unsigned long long)cfg->outer_iteration,
                a210_coupled_newton_status_name(status),
                rep.newton_iterations, rep.evaluations, rep.jacobians,
                rep.rejected_trials, rep.max_abs_f_te, rep.max_abs_f_ne,
                cfg->tol_te, cfg->tol_ne, rep.wall_seconds);
    free(st.te);
    free(st.ne);
    free(st.f_te);
    free(st.f_ne);
    free(x);
    free(xt);
    free(xp);
    free(F);
    free(Ft);
    free(Fp);
    free(d);
    free(J);
    free(LU);
    free(aux_base);
    free(aux_trial);
    free(aux_scratch);
    if (report)
        *report = rep;
    return status;
}
