#ifndef LUMINA_BF_EVENT_MEASURE_ACCESS_H
#define LUMINA_BF_EVENT_MEASURE_ACCESS_H

#include <math.h>

#ifdef __CUDACC__
#define LUMINA_BF_EVENT_HD static __host__ __device__ __forceinline__
#else
#define LUMINA_BF_EVENT_HD static inline
#endif

/* Raw lookup shared by the CPU and CUDA wrappers.  BfEventMeasureStatus and
 * BfEventMeasureProvenance are declared by lumina.h before this header is
 * included.  Consumer policy remains outside: this function only classifies
 * the value/domain and never substitutes another opacity quantity. */
LUMINA_BF_EVENT_HD BfEventMeasureStatus bf_event_measure_lookup_raw(
    const double *event_measure,
    int event_measure_provenance,
    int n_freq_bins,
    int n_shells,
    double nu_min,
    double nu_max,
    double d_log_nu,
    int shell,
    double nu,
    double *out)
{
    if (out) *out = 0.0;
    if (!out || !event_measure || n_freq_bins <= 0 || n_shells <= 0 ||
        !(nu_min > 0.0) || !(nu_max > nu_min) || !(d_log_nu > 0.0) ||
        (event_measure_provenance != EVENT_MEASURE_SPONTANEOUS &&
         event_measure_provenance != EVENT_MEASURE_LEGACY_ARGMAX))
        return BF_EVENT_MEASURE_UNAVAILABLE;
    if (shell < 0 || shell >= n_shells || !isfinite(nu) ||
        nu < nu_min || nu >= nu_max)
        return BF_EVENT_MEASURE_OUT_OF_GRID;
    double x = log(nu / nu_min) / d_log_nu;
    int bin = (int)x;
    if (bin < 0 || bin >= n_freq_bins)
        return BF_EVENT_MEASURE_OUT_OF_GRID;
    if (bin >= n_freq_bins - 1) {
        *out = event_measure[(size_t)shell * n_freq_bins + n_freq_bins - 1];
    } else {
        double fraction = x - (double)bin;
        double lower = event_measure[(size_t)shell * n_freq_bins + bin];
        double upper = event_measure[(size_t)shell * n_freq_bins + bin + 1];
        *out = lower + fraction * (upper - lower);
    }
    if (!isfinite(*out)) return BF_EVENT_MEASURE_UNAVAILABLE;
    if (*out < 0.0) return BF_EVENT_MEASURE_NEGATIVE;
    return BF_EVENT_MEASURE_OK;
}

#undef LUMINA_BF_EVENT_HD

#endif
