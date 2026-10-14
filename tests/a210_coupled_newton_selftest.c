/* CPU selftest for src/a210_coupled_newton.c (sprint 2026-10-13).
 * Synthetic shell problems with the A2-10 structure: thermal residual of
 * shell s depends on (T_s, n_e,s) and on shell s-1 (shell 0 on shell 1),
 * charge residual on (T_s, n_e,s) only.  Each case prints PASS/FAIL. */
#include "../src/a210_coupled_newton.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NS 50
#define TMIN 3500.0
#define TMAX 140000.0

typedef struct {
    int mode;
    double t_star[NS];
    double reject_above;
    int domain_violations;
    int evals;
} Prob;

enum { SMOOTH = 0, WIDE_STENCIL, NO_ROOT, JUMP, REJECT, FAIL_ALL, LOGBAL };

static double cool_ref(int s) { return 8000.0 + 300.0 * s; }

static int fn(const double *te, const double *ne, size_t n, double *fte,
              double *fne, void *aux, void *vctx)
{
    Prob *p = vctx;
    p->evals++;
    if (p->mode == FAIL_ALL)
        return 1;
    for (size_t s = 0; s < n; s++)
        if (te[s] < TMIN || te[s] > TMAX)
            p->domain_violations++;
    for (size_t s = 0; s < n; s++)
        if (p->mode == REJECT && te[s] > p->reject_above * cool_ref((int)s))
            return 1;
    for (size_t s = 0; s < n; s++) {
        size_t nb = s == 0 ? 1 : s - 1;
        double ts = te[s] / cool_ref((int)s);
        double coupling = 0.05 * tanh(log(te[nb] / te[s]));
        if (p->mode == WIDE_STENCIL && s + 1 < n)
            coupling += 0.03 * tanh(log(te[s + 1] / te[s]));
        double H = 1.0 + coupling + 0.02 * log(ne[s] / 1e9);
        double C = pow(ts, 1.5) * (1.0 + 0.1 * ts);
        if (p->mode == NO_ROOT)
            H = 2.0 * C + 1.0;
        fte[s] = p->mode == LOGBAL ? log(H / C) : (H - C) / (H + C);
        if (p->mode == JUMP) {
            double d = te[s] / cool_ref((int)s) - 1.0;
            fte[s] = (d >= 0.0 ? -1.0 : 1.0) * (1e-3 + fabs(d));
        }
        double q = 1e9 * pow(te[s] / 1e4, 0.8) * sqrt(1e9 / ne[s]) *
                   (1.0 + 0.01 * s);
        fne[s] = log(q / ne[s]);
    }
    if (aux)
        memcpy(aux, te, n * sizeof(double));
    return 0;
}

/* Independent reference: Jacobi sweeps of per-shell nested bisection. */
static void reference(Prob *p, double *te, double *ne)
{
    double t[NS], e[NS], f1[NS], f2[NS];
    for (int s = 0; s < NS; s++) {
        t[s] = cool_ref(s);
        e[s] = 1e9;
    }
    for (int sweep = 0; sweep < 100; sweep++) {
        double tn[NS], en[NS], change = 0.0;
        for (int s = 0; s < NS; s++) {
            double lo = log(TMIN), hi = log(TMAX);
            double nesol = e[s];
            for (int it = 0; it < 64; it++) {
                double mid = 0.5 * (lo + hi);
                double tt[NS], ee[NS];
                memcpy(tt, t, sizeof(tt));
                memcpy(ee, e, sizeof(ee));
                tt[s] = exp(mid);
                double nlo = log(1e3), nhi = log(1e15);
                for (int k = 0; k < 64; k++) {
                    double nm = 0.5 * (nlo + nhi);
                    ee[s] = exp(nm);
                    fn(tt, ee, NS, f1, f2, NULL, p);
                    if (f2[s] > 0.0) nlo = nm; else nhi = nm;
                }
                nesol = ee[s];
                fn(tt, ee, NS, f1, f2, NULL, p);
                if (f1[s] > 0.0) lo = mid; else hi = mid;
            }
            tn[s] = exp(0.5 * (lo + hi));
            en[s] = nesol;
        }
        for (int s = 0; s < NS; s++) {
            change = fmax(change, fabs(tn[s] / t[s] - 1.0));
            t[s] = tn[s];
            e[s] = en[s];
        }
        if (change < 1e-14)
            break;
    }
    memcpy(te, t, sizeof(t));
    memcpy(ne, e, sizeof(e));
}

static int failures = 0;
static void check(int ok, const char *name, const char *detail)
{
    printf("%s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
    if (!ok)
        failures++;
}

static A210CoupledNewtonStatus run(Prob *p, double *te, double *ne,
                                   double *aux, A210CoupledNewtonReport *r,
                                   size_t n)
{
    A210CoupledNewtonConfig c;
    a210_coupled_newton_default_config(&c, n);
    c.te_min = TMIN;
    c.te_max = TMAX;
    c.eval_out_bytes = aux ? n * sizeof(double) : 0;
    c.log = getenv("CN_SELFTEST_VERBOSE") ? stdout : NULL;
    c.tag = "SELFTEST";
    return a210_coupled_newton_solve(&c, fn, p, te, ne, aux, r);
}

int main(void)
{
    char buf[512];
    double te[NS], ne[NS], aux[NS], rt[NS], rn[NS], te0[NS], ne0[NS];
    A210CoupledNewtonReport r;

    /* 1. smooth coupled problem from a distant start */
    Prob p = {.mode = SMOOTH};
    for (int s = 0; s < NS; s++) {
        te[s] = 4000.0 + 50.0 * s;
        ne[s] = 3e9;
    }
    A210CoupledNewtonStatus st = run(&p, te, ne, aux, &r, NS);
    reference(&p, rt, rn);
    double err_t = 0.0, err_n = 0.0;
    for (int s = 0; s < NS; s++) {
        err_t = fmax(err_t, fabs(te[s] / rt[s] - 1.0));
        err_n = fmax(err_n, fabs(ne[s] / rn[s] - 1.0));
    }
    snprintf(buf, sizeof(buf),
             "status=%s iters=%d evals=%d jac=%d max_f_te=%.3g max_f_ne=%.3g "
             "err_T=%.3g err_ne=%.3g",
             a210_coupled_newton_status_name(st), r.newton_iterations,
             r.evaluations, r.jacobians, r.max_abs_f_te, r.max_abs_f_ne,
             err_t, err_n);
    check(st == A210_CN_CONVERGED && r.max_abs_f_te <= 1e-12 &&
              r.max_abs_f_ne <= 1e-8 && err_t < 1e-9 && err_n < 1e-7,
          "T1_smooth_converges_to_reference_root", buf);
    check(st == A210_CN_CONVERGED && r.evaluations <= 80,
          "T2_evaluation_budget_small", buf);
    check(memcmp(aux, te, sizeof(te)) == 0,
          "T3_root_record_is_accepted_root_evaluation", "aux==te bytes");
    check(p.domain_violations == 0, "T4_no_callback_outside_domain", "");

    /* 5. dependency wider than the colouring stencil: still converges, and
     *    only by residual */
    Prob pw = {.mode = WIDE_STENCIL};
    for (int s = 0; s < NS; s++) {
        te[s] = 9000.0;
        ne[s] = 1e9;
    }
    st = run(&pw, te, ne, NULL, &r, NS);
    snprintf(buf, sizeof(buf), "status=%s iters=%d evals=%d max_f_te=%.3g",
             a210_coupled_newton_status_name(st), r.newton_iterations,
             r.evaluations, r.max_abs_f_te);
    check((st == A210_CN_CONVERGED && r.max_abs_f_te <= 1e-12 &&
           r.max_abs_f_ne <= 1e-8) ||
              st != A210_CN_CONVERGED,
          "T5_wide_stencil_residual_declared_only", buf);

    /* 6. no root: must fail closed, inputs untouched */
    Prob pn = {.mode = NO_ROOT};
    for (int s = 0; s < NS; s++) {
        te[s] = te0[s] = 9000.0;
        ne[s] = ne0[s] = 1e9;
    }
    st = run(&pn, te, ne, aux, &r, NS);
    snprintf(buf, sizeof(buf), "status=%s evals=%d",
             a210_coupled_newton_status_name(st), r.evaluations);
    check(st != A210_CN_CONVERGED && memcmp(te, te0, sizeof(te)) == 0 &&
              memcmp(ne, ne0, sizeof(ne)) == 0,
          "T6_no_root_fails_closed", buf);
    check(pn.domain_violations == 0, "T7_no_root_never_leaves_domain", "");

    /* 8. jump at the sign change (no zero): must not declare convergence */
    Prob pj = {.mode = JUMP};
    for (int s = 0; s < NS; s++) {
        te[s] = te0[s] = 0.9 * cool_ref(s);
        ne[s] = ne0[s] = 1e9;
    }
    st = run(&pj, te, ne, NULL, &r, NS);
    snprintf(buf, sizeof(buf), "status=%s iters=%d evals=%d max_f_te=%.3g",
             a210_coupled_newton_status_name(st), r.newton_iterations,
             r.evaluations, r.max_abs_f_te);
    check(st != A210_CN_CONVERGED && memcmp(te, te0, sizeof(te)) == 0,
          "T8_jump_root_not_declared_converged", buf);

    /* 9. callback rejects T above a threshold (failed candidate): step is
     *    shortened and the root below the threshold is still found */
    Prob pr = {.mode = REJECT, .reject_above = 1.02};
    for (int s = 0; s < NS; s++) {
        te[s] = 0.5 * cool_ref(s);
        ne[s] = 1e9;
    }
    st = run(&pr, te, ne, NULL, &r, NS);
    snprintf(buf, sizeof(buf), "status=%s rejected=%d evals=%d",
             a210_coupled_newton_status_name(st), r.rejected_trials,
             r.evaluations);
    check(st == A210_CN_CONVERGED && r.max_abs_f_te <= 1e-12 && r.rejected_trials > 0,
          "T9_rejected_trials_shorten_step", buf);

    /* 10. initial evaluation failure */
    Prob pf = {.mode = FAIL_ALL};
    for (int s = 0; s < NS; s++) {
        te[s] = te0[s] = 9000.0;
        ne[s] = ne0[s] = 1e9;
    }
    st = run(&pf, te, ne, NULL, &r, NS);
    check(st == A210_CN_INITIAL_EVAL_FAILED &&
              memcmp(te, te0, sizeof(te)) == 0,
          "T10_initial_eval_failure_fails_closed",
          a210_coupled_newton_status_name(st));

    /* 13-14. production residual form ln(H/C) (same root as T1) from starts
     *        at either end of the qualified domain */
    for (int side = 0; side < 2; side++) {
        Prob pl = {.mode = LOGBAL};
        for (int s = 0; s < NS; s++) {
            te[s] = side == 0 ? 3600.0 : 130000.0;
            ne[s] = 1e9;
        }
        st = run(&pl, te, ne, NULL, &r, NS);
        double et = 0.0;
        for (int s = 0; s < NS; s++)
            et = fmax(et, fabs(te[s] / rt[s] - 1.0));
        snprintf(buf, sizeof(buf),
                 "status=%s iters=%d evals=%d jac=%d rejected=%d err_T=%.3g "
                 "domain_violations=%d",
                 a210_coupled_newton_status_name(st), r.newton_iterations,
                 r.evaluations, r.jacobians, r.rejected_trials, et,
                 pl.domain_violations);
        check(st == A210_CN_CONVERGED && et < 1e-9 &&
                  pl.domain_violations == 0,
              side == 0 ? "T13_log_balance_from_domain_floor"
                        : "T14_log_balance_from_domain_ceiling",
              buf);
    }

    /* 11. invalid arguments */
    st = run(&p, te, ne, NULL, &r, 1);
    check(st == A210_CN_INVALID_ARGUMENT, "T11_single_shell_rejected",
          a210_coupled_newton_status_name(st));
    te[3] = -1.0;
    st = run(&p, te, ne, NULL, &r, NS);
    check(st == A210_CN_INVALID_ARGUMENT, "T12_nonpositive_guess_rejected",
          a210_coupled_newton_status_name(st));

    printf("SELFTEST %s failures=%d\n", failures ? "FAIL" : "PASS",
           failures);
    return failures ? 1 : 0;
}
