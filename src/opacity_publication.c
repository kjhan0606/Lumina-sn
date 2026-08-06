#include "opacity_publication.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static A208Counters g_a208;

static A208Validity numeric_status(double x) {
    if (!isfinite(x)) return A208_NONFINITE;
    return x == 0.0 ? A208_EXACT_ZERO : A208_VALID;
}

A208ValueView a208_signed_sobolev(double coefficient, double f_lu,
                                  double lambda_cm, double time_explosion,
                                  double n_lower, double n_upper,
                                  double g_lower, double g_upper,
                                  uint64_t generation) {
    A208ValueView out = {NAN, A208_INVALID_POPULATION, generation};
    if (!isfinite(n_lower) || !isfinite(n_upper) || n_lower < 0.0 || n_upper < 0.0)
        return out;
    if (!(g_lower > 0.0) || !(g_upper > 0.0)) return out;
    double difference = n_lower - (g_lower / g_upper) * n_upper;
    out.value = coefficient * f_lu * lambda_cm * time_explosion * difference;
    out.validity = numeric_status(out.value);
    if (out.validity == A208_EXACT_ZERO) out.value = 0.0;
    if (out.validity == A208_VALID && out.value < 0.0)
        g_a208.negative_tau_line_shells++;
    if (out.validity == A208_NONFINITE) g_a208.nonfinite_failures++;
    return out;
}

A208ValueView a208_line_source(double prefactor, double n_lower,
                               double n_upper, double g_lower, double g_upper,
                               uint64_t generation) {
    A208ValueView out = {NAN, A208_INVALID_POPULATION, generation};
    if (!isfinite(n_lower) || !isfinite(n_upper) || n_lower < 0.0 || n_upper < 0.0 ||
        !(g_lower > 0.0) || !(g_upper > 0.0)) return out;
    double chi_factor = n_lower - (g_lower / g_upper) * n_upper;
    if (chi_factor == 0.0) {
        if (n_upper == 0.0) {
            out.value = 0.0; out.validity = A208_EXACT_ZERO;
            g_a208.source_exact_zero++;
        } else {
            out.validity = A208_SOURCE_CANCELLATION_SINGULAR;
            g_a208.source_cancellation_singular++;
        }
        return out;
    }
    out.value = prefactor * (g_lower / g_upper) * n_upper / chi_factor;
    out.validity = numeric_status(out.value);
    if (out.validity == A208_VALID) {
        g_a208.source_valid++;
        if (out.value < 0.0) g_a208.source_negative++;
    }
    return out;
}

int a208_bf_split(double gross, double stimulated_ratio, double exponent,
                  A208SignedBfNet *net,
                  A208NonnegativeEventMeasure *event_measure) {
    if (!net || !event_measure || !isfinite(gross) || gross < 0.0 ||
        !isfinite(stimulated_ratio) || stimulated_ratio < 0.0 ||
        !isfinite(exponent) || exponent < 0.0) return -1;
    event_measure->value = gross;
    event_measure->validity = numeric_status(gross);
    net->value = gross * (1.0 - stimulated_ratio * exponent);
    net->validity = numeric_status(net->value);
    if (net->validity == A208_NONFINITE) return -1;
    if (net->value < 0.0) g_a208.negative_bf_route_shell_bins++;
    return 0;
}

A208TauInteractionMeasure a208_tau_interaction_measure(A208ValueView tau) {
    A208TauInteractionMeasure out = {NAN, tau.validity};
    if (tau.validity != A208_VALID && tau.validity != A208_EXACT_ZERO) return out;
    /* Deliberate total-variation measure, not a transport coefficient. */
    out.value = fabs(tau.value);
    return out;
}

static void *zalloc(size_t n, size_t width) { return n ? calloc(n, width) : NULL; }

int a208_publication_init(CpuOpacityPublication *p, size_t ns, size_t nb,
                          size_t nl, size_t nr) {
    if (!p || !ns || !nb) return -1;
    memset(p, 0, sizeof(*p)); p->n_shells=ns; p->n_bins=nb; p->n_lines=nl; p->n_routes=nr;
    size_t cells=ns*nb, lines=ns*nl, routes=ns*nb*nr;
    p->frequency_edges=zalloc(nb+1,sizeof(double));
    p->chi_es=zalloc(cells,sizeof(double)); p->chi_bb=zalloc(cells,sizeof(double));
    p->chi_bf=zalloc(cells,sizeof(double)); p->chi_ff=zalloc(cells,sizeof(double));
    p->chi_total=zalloc(cells,sizeof(double));
    p->chi_validity=zalloc(4*cells,sizeof(A208Validity));
    p->tau_sobolev=zalloc(lines,sizeof(double)); p->line_source_S=zalloc(lines,sizeof(double));
    p->tau_validity=zalloc(lines,sizeof(A208Validity));
    p->line_source_validity=zalloc(lines,sizeof(A208Validity));
    p->bf_net_route=zalloc(routes,sizeof(double)); p->bf_event_measure=zalloc(routes,sizeof(double));
    p->bf_route_validity=zalloc(routes,sizeof(A208Validity));
    if (!p->frequency_edges || !p->chi_es || !p->chi_bb || !p->chi_bf ||
        !p->chi_ff || !p->chi_total || !p->chi_validity ||
        (lines && (!p->tau_sobolev || !p->line_source_S || !p->tau_validity || !p->line_source_validity)) ||
        (routes && (!p->bf_net_route || !p->bf_event_measure || !p->bf_route_validity))) {
        a208_publication_free(p); return -1;
    }
    return 0;
}

void a208_publication_free(CpuOpacityPublication *p) {
    if (!p) return;
    free(p->frequency_edges); free(p->chi_es); free(p->chi_bb); free(p->chi_bf);
    free(p->chi_ff); free(p->chi_total); free(p->chi_validity);
    free(p->tau_sobolev); free(p->line_source_S); free(p->tau_validity);
    free(p->line_source_validity); free(p->bf_net_route); free(p->bf_event_measure);
    free(p->bf_route_validity); memset(p,0,sizeof(*p));
}

int a208_publication_commit(CpuOpacityPublication *public_p,
                            CpuOpacityPublication *candidate) {
    if (!public_p || !candidate ||
        candidate->generation_required == 0 ||
        candidate->generation_committed != 0) return -1;
    size_t cells=candidate->n_shells*candidate->n_bins;
    for (size_t i=0;i<4*cells;i++)
        if (candidate->chi_validity[i] != A208_VALID &&
            candidate->chi_validity[i] != A208_EXACT_ZERO) return -1;
    size_t bad=0;
    if (a208_publication_max_closure(candidate,&bad) > 1e-10) return -1;
    candidate->generation_committed=candidate->generation_required;
    CpuOpacityPublication old=*public_p; *public_p=*candidate;
    memset(candidate,0,sizeof(*candidate)); a208_publication_free(&old);
    g_a208.generation_committed=public_p->generation_committed;
    return 0;
}

double a208_publication_max_closure(const CpuOpacityPublication *p,
                                    size_t *worst_cell) {
    if (!p) return INFINITY;
    size_t n=p->n_shells*p->n_bins, wi=0;
    double worst=0.0;
    for(size_t i=0;i<n;i++) {
        double sum=((p->chi_es[i]+p->chi_bb[i])+p->chi_bf[i])+p->chi_ff[i];
        double den=fmax(fabs(p->chi_es[i])+fabs(p->chi_bb[i])+fabs(p->chi_bf[i])+fabs(p->chi_ff[i]),DBL_MIN);
        double e=fabs(p->chi_total[i]-sum)/den;
        if(e>worst){worst=e;wi=i;}
    }
    if(worst_cell)*worst_cell=wi;
    return worst;
}

int a208_capability_check(A208ConsumerCapability capability,
                          const A208ValueView *values, size_t count,
                          const char *consumer, uint64_t *blocked_counter,
                          size_t *first_negative) {
    if (!values && count) return 5;
    for(size_t i=0;i<count;i++) {
        if(values[i].validity!=A208_VALID && values[i].validity!=A208_EXACT_ZERO) return 5;
        if(values[i].value<0.0 && capability==A208_BLOCK_UNSUPPORTED) {
            if(blocked_counter)(*blocked_counter)++;
            if(first_negative)*first_negative=i;
            fprintf(stderr,"[A2-08][BLOCKED] consumer=%s reason=BLOCKED_NEGATIVE_OPACITY_SEMANTICS identity=%zu rc=3\n",consumer?consumer:"unknown",i);
            return 3;
        }
    }
    return 0;
}

const char *a208_validity_name(A208Validity v) {
    static const char *n[]={"INVALID_ENUM","VALID","EXACT_ZERO","UNSAMPLED","OUT_OF_GRID","MISS","STALE_GENERATION","QHASH_MISMATCH","PROFILE_MISMATCH","INVALID_POPULATION","INVALID_PARTITION","INVALID_TE","INVALID_NE","NONFINITE","SOURCE_CANCELLATION_SINGULAR","EVENT_MEASURE_UNAVAILABLE","BLOCKED_NEGATIVE_OPACITY_SEMANTICS","FORBIDDEN_FALLBACK"};
    return v>=A208_VALID&&v<=A208_FORBIDDEN_FALLBACK?n[v]:n[0];
}
A208Counters *a208_counters(void){return &g_a208;}
void a208_counters_reset(void){memset(&g_a208,0,sizeof(g_a208));}
void a208_report_counters(void){
    printf("[A2-08][SIGNED-OPACITY] generation_required=%llu generation_committed=%llu shells_attempted=%llu shells_published=%llu cells_attempted=%llu cells_published=%llu es_terms=%llu bb_terms=%llu bf_terms=%llu ff_terms=%llu exact_zero_es=%llu exact_zero_bb=%llu exact_zero_bf=%llu exact_zero_ff=%llu negative_tau_line_shells=%llu negative_bb_line_shells=%llu negative_bf_route_shell_bins=%llu negative_bf_shell_bins=%llu negative_total_shell_bins=%llu blocked_negative_transport=%llu blocked_negative_formal=%llu blocked_negative_heating=%llu blocked_negative_transition=%llu blocked_stale=%llu blocked_unsampled=%llu blocked_oog=%llu blocked_miss=%llu blocked_profile=%llu blocked_qhash=%llu blocked_population=%llu blocked_te=%llu blocked_ne=%llu source_valid=%llu source_exact_zero=%llu source_negative=%llu source_cancellation_singular=%llu event_measure_unavailable=%llu closure_failures=%llu nonfinite_failures=%llu fallback_attempts=%llu abs_attempts=%llu zero_clamp_attempts=%llu floor_attempts=%llu raw_view_attempts=%llu partial_publish_attempts=%llu replay_line_blocks_attempted=%llu replay_line_blocks_committed=%llu\n",
    (unsigned long long)g_a208.generation_required,(unsigned long long)g_a208.generation_committed,(unsigned long long)g_a208.shells_attempted,(unsigned long long)g_a208.shells_published,(unsigned long long)g_a208.cells_attempted,(unsigned long long)g_a208.cells_published,(unsigned long long)g_a208.es_terms,(unsigned long long)g_a208.bb_terms,(unsigned long long)g_a208.bf_terms,(unsigned long long)g_a208.ff_terms,(unsigned long long)g_a208.exact_zero_es,(unsigned long long)g_a208.exact_zero_bb,(unsigned long long)g_a208.exact_zero_bf,(unsigned long long)g_a208.exact_zero_ff,(unsigned long long)g_a208.negative_tau_line_shells,(unsigned long long)g_a208.negative_bb_line_shells,(unsigned long long)g_a208.negative_bf_route_shell_bins,(unsigned long long)g_a208.negative_bf_shell_bins,(unsigned long long)g_a208.negative_total_shell_bins,(unsigned long long)g_a208.blocked_negative_transport,(unsigned long long)g_a208.blocked_negative_formal,(unsigned long long)g_a208.blocked_negative_heating,(unsigned long long)g_a208.blocked_negative_transition,(unsigned long long)g_a208.blocked_stale,(unsigned long long)g_a208.blocked_unsampled,(unsigned long long)g_a208.blocked_oog,(unsigned long long)g_a208.blocked_miss,(unsigned long long)g_a208.blocked_profile,(unsigned long long)g_a208.blocked_qhash,(unsigned long long)g_a208.blocked_population,(unsigned long long)g_a208.blocked_te,(unsigned long long)g_a208.blocked_ne,(unsigned long long)g_a208.source_valid,(unsigned long long)g_a208.source_exact_zero,(unsigned long long)g_a208.source_negative,(unsigned long long)g_a208.source_cancellation_singular,(unsigned long long)g_a208.event_measure_unavailable,(unsigned long long)g_a208.closure_failures,(unsigned long long)g_a208.nonfinite_failures,(unsigned long long)g_a208.fallback_attempts,(unsigned long long)g_a208.abs_attempts,(unsigned long long)g_a208.zero_clamp_attempts,(unsigned long long)g_a208.floor_attempts,(unsigned long long)g_a208.raw_view_attempts,(unsigned long long)g_a208.partial_publish_attempts,(unsigned long long)g_a208.replay_line_blocks_attempted,(unsigned long long)g_a208.replay_line_blocks_committed);
}
