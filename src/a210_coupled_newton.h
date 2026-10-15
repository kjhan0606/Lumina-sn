#ifndef LUMINA_A210_COUPLED_NEWTON_H
#define LUMINA_A210_COUPLED_NEWTON_H

/* Coupled (T_e, n_e) Newton root for the A2-10 FREE_T material solve
 * (sprint 2026-10-13, astra design sprint_20261012/astra_freet_accel.md).
 *
 * Unknowns are x = (ln T_s, ln n_e,s) for every shell.  One callback
 * evaluation solves the populations at the prescribed (T, n_e) and returns
 *   f_te[s] = ln(H / C)                steady thermal balance
 *           = (H-C)/unsubtracted row norm for physical time evolution
 *   f_ne[s] = ln(q_closed / n_e)       charge closure
 * The Jacobian is a forward difference in x with two shell colours, built on
 * the assumption that row s depends on shells s and s-1 only (shell 0 on 0
 * and 1), the CMFGEN adiabatic stencil.  This is an approximation: the
 * population core's CE loop stops on an all-shell maximum, so a perturbation
 * that changes its iteration count moves every row.  An inexact Jacobian can
 * only slow or stop the iteration; it cannot declare convergence, which is
 * judged from evaluated residuals alone.
 *
 * Convergence is declared only from the residual at an evaluated point:
 * |f_te| <= tol_te and |f_ne| <= tol_ne in every shell.  A trial outside
 * [te_min, te_max] or a failed callback is rejected and the step halved; no
 * value is ever clipped to a bound, and there is no fallback solver.  Any
 * other outcome returns a nonzero status and leaves te/ne/eval_out untouched. */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef enum {
    A210_CN_CONVERGED = 0,
    A210_CN_INVALID_ARGUMENT,
    A210_CN_ALLOCATION_FAILED,
    A210_CN_INITIAL_EVAL_FAILED,
    A210_CN_JACOBIAN_EVAL_FAILED,
    A210_CN_SINGULAR_JACOBIAN,
    A210_CN_LINESEARCH_FAILED,
    A210_CN_NOT_CONVERGED
} A210CoupledNewtonStatus;

/* Return 0 on a complete evaluation; nonzero rejects the trial point.
 * eval_out receives the caller's per-evaluation record (eval_out_bytes). */
typedef int (*A210CoupledResidualFn)(const double *te, const double *ne,
                                     size_t n, double *f_te, double *f_ne,
                                     void *eval_out, void *ctx);

typedef struct {
    size_t n;
    double te_min, te_max;
    double tol_te, tol_ne;
    double fd_step;
    int max_newton;
    int max_evals;
    size_t eval_out_bytes;
    FILE *log;
    const char *tag;
    uint64_t outer_iteration;
} A210CoupledNewtonConfig;

typedef struct {
    A210CoupledNewtonStatus status;
    int newton_iterations, evaluations, jacobians, rejected_trials;
    double max_abs_f_te, max_abs_f_ne;
    double wall_seconds;
} A210CoupledNewtonReport;

void a210_coupled_newton_default_config(A210CoupledNewtonConfig *config,
                                        size_t n);

/* te, ne: initial guess in, root out (only on A210_CN_CONVERGED).
 * eval_out_at_root: the callback record of the accepted root evaluation. */
A210CoupledNewtonStatus a210_coupled_newton_solve(
    const A210CoupledNewtonConfig *config, A210CoupledResidualFn fn,
    void *ctx, double *te, double *ne, void *eval_out_at_root,
    A210CoupledNewtonReport *report);

const char *a210_coupled_newton_status_name(A210CoupledNewtonStatus status);

#endif
