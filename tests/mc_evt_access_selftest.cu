#include "lumina.h"
#include "bf_event_measure_access.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__global__ static void classify_fixture(const double *grid, int *status,
                                        double *value)
{
    if (threadIdx.x || blockIdx.x) return;
    const int bins = 4, shells = 1;
    const double nu_min = 1.0, nu_max = 16.0;
    const double dlog = log(nu_max / nu_min) / bins;
    const double nu_ok = nu_min * exp(0.25 * dlog);
    const double nu_negative = nu_min * exp(2.25 * dlog);
    status[0] = (int)bf_event_measure_lookup_raw(
        grid, EVENT_MEASURE_SPONTANEOUS, bins, shells,
        nu_min, nu_max, dlog, 0, nu_ok, &value[0]);
    status[1] = (int)bf_event_measure_lookup_raw(
        grid, EVENT_MEASURE_SPONTANEOUS, bins, shells,
        nu_min, nu_max, dlog, 0, nu_negative, &value[1]);
    status[2] = (int)bf_event_measure_lookup_raw(
        grid, EVENT_MEASURE_SPONTANEOUS, bins, shells,
        nu_min, nu_max, dlog, 0, 0.5 * nu_min, &value[2]);
    status[3] = (int)bf_event_measure_lookup_raw(
        grid, EVENT_MEASURE_PROVENANCE_NONE, bins, shells,
        nu_min, nu_max, dlog, 0, nu_ok, &value[3]);
}

int main(void)
{
    const double host_grid[4] = {2.0, 2.0, -3.0, -3.0};
    double *device_grid = NULL, *device_value = NULL;
    int *device_status = NULL;
    int status[4] = {-1, -1, -1, -1};
    double value[4] = {0.0, 0.0, 0.0, 0.0};
    if (cudaMalloc(&device_grid, sizeof(host_grid)) != cudaSuccess ||
        cudaMalloc(&device_status, sizeof(status)) != cudaSuccess ||
        cudaMalloc(&device_value, sizeof(value)) != cudaSuccess ||
        cudaMemcpy(device_grid, host_grid, sizeof(host_grid),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return 70;
    classify_fixture<<<1, 1>>>(device_grid, device_status, device_value);
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(status, device_status, sizeof(status),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(value, device_value, sizeof(value),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return 71;
    cudaFree(device_grid);
    cudaFree(device_status);
    cudaFree(device_value);
    if (status[0] != BF_EVENT_MEASURE_OK || !(value[0] > 0.0) ||
        status[1] != BF_EVENT_MEASURE_NEGATIVE || !(value[1] < 0.0) ||
        status[2] != BF_EVENT_MEASURE_OUT_OF_GRID ||
        status[3] != BF_EVENT_MEASURE_UNAVAILABLE) {
        fprintf(stderr, "[E-NE2][FAIL] status=%d,%d,%d,%d "
                        "value=%.17g,%.17g,%.17g,%.17g\n",
                status[0], status[1], status[2], status[3],
                value[0], value[1], value[2], value[3]);
        return 72;
    }
    printf("[E-NE2][PASS] gpu_shared_classifier "
           "ok=%d negative=%d out_of_grid=%d unavailable=%d\n",
           status[0], status[1], status[2], status[3]);
    return 0;
}
