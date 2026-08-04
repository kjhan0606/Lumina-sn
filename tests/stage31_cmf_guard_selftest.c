#include "lumina_cmf_field.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static int same_bits(double left, double right)
{
    return memcmp(&left, &right, sizeof(left)) == 0;
}

int main(void)
{
    const uint64_t excess_seed = UINT64_C(0x7d01);
    const uint64_t subtruncation_seed = UINT64_C(0x7d02);
    const double h = 1.0 / 64.0;
    const double scale = 1.0;
    const double bound = LCMF_TRUNCATION_ERROR_COEFFICIENT * h * h * scale;
    const LCMFInterval excess = {-0.1, -0.100000000000001, -0.099999999999999};
    const LCMFInterval subtruncation = {-1.0e-4, -1.00000000001e-4,
                                       -9.9999999999e-5};
    const LCMFInterval nonfinite = {NAN, -1.0, 1.0};
    const double raw_subtruncation = subtruncation.value;
    LCMFResult excess_result, subtruncation_result;
    LCMFResult nonfinite_result;
    int excess_status, subtruncation_status, nonfinite_status;
    memset(&excess_result, 0, sizeof(excess_result));
    memset(&subtruncation_result, 0, sizeof(subtruncation_result));
    memset(&nonfinite_result, 0, sizeof(nonfinite_result));
    excess_status = lumina_cmf_solution_guard_probe(&excess, h, scale, scale,
                                                    &excess_result);
    subtruncation_status = lumina_cmf_solution_guard_probe(
        &subtruncation, h, scale, scale, &subtruncation_result);
    nonfinite_status = lumina_cmf_solution_guard_probe(
        &nonfinite, h, scale, scale, &nonfinite_result);

    (void)printf("seed=%" PRIu64 " class=excess status=%d count=%" PRIu64
                 " value=%.17g bound=%.17g\n",
                 excess_seed, excess_status,
                 excess_result.solution_negative_excess_count,
                 excess.value, bound);
    (void)printf("seed=%" PRIu64 " class=subtruncation status=%d count=%" PRIu64
                 " value=%.17g min=%.17g bound=%.17g coord=%zu/%zu/%zu/%zu/%zu raw_unchanged=%d\n",
                 subtruncation_seed, subtruncation_status,
                 subtruncation_result.solution_subtruncation_count,
                 subtruncation.value,
                 subtruncation_result.solution_subtruncation_min,
                 subtruncation_result.solution_subtruncation_first_bound,
                 subtruncation_result.solution_subtruncation_first.radial_index,
                 subtruncation_result.solution_subtruncation_first.frequency_index,
                 subtruncation_result.solution_subtruncation_first.ray_index,
                 subtruncation_result.solution_subtruncation_first.segment_index,
                 subtruncation_result.solution_subtruncation_first.substep_index,
                 same_bits(raw_subtruncation, subtruncation.value));
    (void)printf("class=nonfinite status=%d count=%" PRIu64 "\n",
                 nonfinite_status, nonfinite_result.nonfinite_count);

    if (excess_status != LCMF_ENEGATIVE ||
        excess_result.solution_negative_excess_count != 1u ||
        excess_result.solution_subtruncation_count != 0u ||
        subtruncation_status != LCMF_OK ||
        subtruncation_result.solution_negative_excess_count != 0u ||
        subtruncation_result.solution_subtruncation_count != 1u ||
        subtruncation_result.solution_subtruncation_min != subtruncation.value ||
        nonfinite_status != LCMF_ENONFINITE ||
        nonfinite_result.nonfinite_count != 1u ||
        !(fabs(subtruncation.value) <= bound) ||
        !same_bits(raw_subtruncation, subtruncation.value)) {
        return 1;
    }
    return 0;
}
