/**
 * LUMINA-SN GPU Transport Entry Point
 * gpu_transport.cu - CUDA kernel infrastructure for OpenMP-driven GPU acceleration
 *
 * Task Order #019: CUDA Infrastructure Setup
 *
 * Architecture:
 *   - Each OpenMP thread manages its own CUDA stream
 *   - Streams enable concurrent kernel execution from multiple CPU threads
 *   - Warmup kernels verify CUDA concurrency before production use
 *
 * Compilation:
 *   nvcc -arch=sm_70 -Xcompiler -fopenmp -c gpu_transport.cu -o gpu_transport.o
 *
 * Linkage:
 *   gcc ... gpu_transport.o ... -lcudart -L$(CUDA_HOME)/lib64
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "cuda_interface.h"
#include "cuda_shared.h"

/* ============================================================================
 * INTERNAL STATE (must be before any functions that use them)
 * ============================================================================ */

static int g_cuda_initialized = 0;
static int g_cuda_device_id = -1;
static cudaStream_t g_streams[CUDA_MAX_STREAMS];
static int g_stream_created[CUDA_MAX_STREAMS] = {0};

/* Thread safety for stream creation */
#include <pthread.h>
static pthread_mutex_t g_stream_mutex = PTHREAD_MUTEX_INITIALIZER;

/* ============================================================================
 * Task Order #020: TRACE_PACKET DEVICE FUNCTIONS
 * ============================================================================
 *
 * These device functions implement the Monte Carlo transport logic.
 * They mirror the CPU implementation in rpacket.c but are optimized for GPU.
 */

/**
 * trace_packet_device: Find next interaction point (device version)
 *
 * This is the HEART of Monte Carlo transport on GPU.
 * Each thread runs this for its assigned packet.
 *
 * @param pkt               Packet being traced (in registers)
 * @param r_inner           Inner radii array [n_shells]
 * @param r_outer           Outer radii array [n_shells]
 * @param line_list_nu      Sorted line frequencies [n_lines]
 * @param tau_sobolev       Optical depths [n_lines x n_shells] row-major
 * @param electron_density  Free electron density per shell [n_shells]
 * @param model             Model parameters (scalars)
 * @param plasma            Plasma parameters (scalars)
 * @param distance          [out] Distance to interaction
 * @param delta_shell       [out] Shell crossing direction
 * @return                  Interaction type
 */
__device__ GPUInteractionType trace_packet_device(
    RPacket_GPU *pkt,
    const double * __restrict__ r_inner,
    const double * __restrict__ r_outer,
    const double * __restrict__ line_list_nu,
    const double * __restrict__ tau_sobolev,
    const double * __restrict__ electron_density,
    const Model_GPU *model,
    const Plasma_GPU *plasma,
    double *distance,
    int *delta_shell)
{
    int64_t shell_id = pkt->current_shell_id;
    int full_rel = model->enable_full_relativity;

    /* Boundary distance */
    double d_boundary = calculate_distance_boundary_gpu(
        pkt->r, pkt->mu,
        r_inner[shell_id], r_outer[shell_id],
        delta_shell
    );

    /* Sample random optical depth for electron scattering */
    double tau_event = -log(gpu_rng_uniform(&pkt->rng_state) + 1e-30);

    /* Electron scattering distance */
    double d_electron = calculate_distance_electron_gpu(
        electron_density[shell_id], tau_event
    );

    /* Comoving frame frequency */
    double doppler = get_doppler_factor_gpu(
        pkt->r, pkt->mu, model->inv_time_explosion, full_rel
    );
    double comov_nu = pkt->nu * doppler;

    /* Find minimum distance among: boundary, electron, line */
    double d_min = d_boundary;
    GPUInteractionType itype = GPU_INTERACTION_BOUNDARY;

    if (d_electron < d_min) {
        d_min = d_electron;
        itype = GPU_INTERACTION_ESCATTERING;
    }

    /* Check lines (if not disabled) */
    if (!plasma->disable_line_scattering && plasma->n_lines > 0) {
        /* Find next line to check */
        int64_t line_id = pkt->next_line_id;
        double tau_trace = 0.0;

        /* Loop through lines in frequency order */
        while (line_id < plasma->n_lines && tau_trace < tau_event) {
            double nu_line = line_list_nu[line_id];

            /* Distance to this line resonance */
            int is_last = (line_id >= plasma->n_lines - 1);
            double d_line = calculate_distance_line_gpu(
                pkt->nu, comov_nu, is_last, nu_line,
                model->ct, pkt->r, pkt->mu, full_rel
            );

            /* If line is closer than current minimum */
            if (d_line < d_min) {
                /* Get Sobolev optical depth for this line in this shell */
                int64_t tau_idx = line_id * plasma->n_shells + shell_id;
                double tau_sob = tau_sobolev[tau_idx];

                /* Accumulate optical depth */
                tau_trace += tau_sob;

                if (tau_trace >= tau_event) {
                    /* Line interaction wins */
                    d_min = d_line;
                    itype = GPU_INTERACTION_LINE;
                    pkt->next_line_id = line_id;
                    break;
                }
            } else {
                /* Line is farther than current d_min, stop checking */
                break;
            }

            line_id++;
        }

        /* Update line pointer for next trace */
        if (itype != GPU_INTERACTION_LINE) {
            pkt->next_line_id = line_id;
        }
    }

    *distance = d_min;
    return itype;
}

/**
 * process_boundary_crossing_device: Handle shell boundary transition
 */
__device__ void process_boundary_crossing_device(
    RPacket_GPU *pkt,
    int delta_shell,
    int64_t n_shells)
{
    int64_t new_shell = pkt->current_shell_id + delta_shell;

    if (new_shell < 0) {
        /* Hit inner boundary -> reabsorbed */
        pkt->status = GPU_PACKET_REABSORBED;
    } else if (new_shell >= n_shells) {
        /* Escaped through outer boundary */
        pkt->status = GPU_PACKET_EMITTED;
    } else {
        /* Normal shell crossing */
        pkt->current_shell_id = new_shell;
    }
}

/**
 * thomson_scatter_device: Isotropic electron scattering
 */
__device__ void thomson_scatter_device(
    RPacket_GPU *pkt,
    const Model_GPU *model)
{
    /* Isotropic scattering in comoving frame */
    double mu_cmf = 2.0 * gpu_rng_uniform(&pkt->rng_state) - 1.0;

    /* Transform back to lab frame */
    pkt->mu = angle_aberration_CMF_to_LF_gpu(pkt->r, mu_cmf, model->ct);

    /* Update frequency (due to Doppler shift) */
    double inv_doppler = get_inverse_doppler_factor_gpu(
        pkt->r, pkt->mu, model->inv_time_explosion,
        model->enable_full_relativity
    );
    pkt->nu *= inv_doppler;

    pkt->last_interaction_type = 1;  /* Electron scatter */
}

/**
 * line_scatter_device: Resonant line scattering
 */
__device__ void line_scatter_device(
    RPacket_GPU *pkt,
    const double * __restrict__ line_list_nu,
    const Model_GPU *model,
    int line_interaction_type)
{
    /* Store incoming state */
    pkt->last_interaction_in_nu = pkt->nu;
    pkt->last_line_interaction_in_id = pkt->next_line_id;

    /* Isotropic re-emission in comoving frame */
    double mu_cmf = 2.0 * gpu_rng_uniform(&pkt->rng_state) - 1.0;

    /* Transform to lab frame */
    pkt->mu = angle_aberration_CMF_to_LF_gpu(pkt->r, mu_cmf, model->ct);

    /* For simple scatter mode: emit at line frequency */
    if (line_interaction_type == GPU_LINE_SCATTER) {
        double nu_line = line_list_nu[pkt->next_line_id];
        double inv_doppler = get_inverse_doppler_factor_gpu(
            pkt->r, pkt->mu, model->inv_time_explosion,
            model->enable_full_relativity
        );
        pkt->nu = nu_line * inv_doppler;
    }
    /* TODO: LINE_DOWNBRANCH and LINE_MACROATOM require additional data */

    pkt->last_interaction_type = 2;  /* Line scatter */
    pkt->last_line_interaction_out_id = pkt->next_line_id;

    /* Advance past this line */
    pkt->next_line_id++;
}

/* ============================================================================
 * MAIN TRANSPORT KERNEL
 * ============================================================================
 *
 * Task Order #020: trace_packet_kernel
 *
 * Each thread processes one packet through its complete lifecycle.
 * This is the "persistent thread" model - one thread per packet.
 */

__global__ void trace_packet_kernel(
    RPacket_GPU *packets,
    int n_packets,
    const double * __restrict__ r_inner,
    const double * __restrict__ r_outer,
    const double * __restrict__ line_list_nu,
    const double * __restrict__ tau_sobolev,
    const double * __restrict__ electron_density,
    Model_GPU model,
    Plasma_GPU plasma,
    GPUStats *stats,
    int max_iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_packets) return;

    /* Load packet from global memory to registers */
    RPacket_GPU pkt = packets[idx];

    /* Statistics counters (thread-local) */
    int64_t local_line_interactions = 0;
    int64_t local_electron_scatters = 0;
    int64_t local_boundary_crossings = 0;
    int iter = 0;

    /* Main transport loop */
    while (pkt.status == GPU_PACKET_IN_PROCESS && iter < max_iterations) {

        /* Find next interaction */
        double distance;
        int delta_shell;
        GPUInteractionType itype = trace_packet_device(
            &pkt, r_inner, r_outer, line_list_nu, tau_sobolev,
            electron_density, &model, &plasma, &distance, &delta_shell
        );

        /* Move packet to interaction point */
        move_packet_gpu(&pkt.r, &pkt.mu, distance);

        /* Process interaction */
        switch (itype) {
            case GPU_INTERACTION_BOUNDARY:
                process_boundary_crossing_device(&pkt, delta_shell, model.n_shells);
                local_boundary_crossings++;
                break;

            case GPU_INTERACTION_ESCATTERING:
                thomson_scatter_device(&pkt, &model);
                local_electron_scatters++;
                break;

            case GPU_INTERACTION_LINE:
                line_scatter_device(&pkt, line_list_nu, &model,
                                    plasma.line_interaction_type);
                local_line_interactions++;
                break;
        }

        iter++;
    }

    /* Write packet back to global memory */
    packets[idx] = pkt;

    /* Update global statistics atomically */
    if (stats != NULL) {
        atomicAdd((unsigned long long*)&stats->n_iterations_total, (unsigned long long)iter);
        atomicAdd((unsigned long long*)&stats->n_line_interactions, (unsigned long long)local_line_interactions);
        atomicAdd((unsigned long long*)&stats->n_electron_scatters, (unsigned long long)local_electron_scatters);
        atomicAdd((unsigned long long*)&stats->n_boundary_crossings, (unsigned long long)local_boundary_crossings);

        if (pkt.status == GPU_PACKET_EMITTED) {
            atomicAdd((unsigned long long*)&stats->n_emitted, 1ULL);
        } else if (pkt.status == GPU_PACKET_REABSORBED) {
            atomicAdd((unsigned long long*)&stats->n_reabsorbed, 1ULL);
        }
    }
}

/* ============================================================================
 * KERNEL LAUNCHER (C-callable)
 * ============================================================================ */

extern "C" {

/**
 * Launch trace_packet kernel
 *
 * @param d_packets         Device pointer to packet array
 * @param n_packets         Number of packets
 * @param d_r_inner         Device pointer to inner radii
 * @param d_r_outer         Device pointer to outer radii
 * @param d_line_list_nu    Device pointer to line frequencies
 * @param d_tau_sobolev     Device pointer to optical depths
 * @param d_electron_density Device pointer to electron densities
 * @param model             Model parameters (copied by value)
 * @param plasma            Plasma parameters (copied by value)
 * @param d_stats           Device pointer to statistics (can be NULL)
 * @param stream_id         CUDA stream to use
 * @param max_iterations    Safety limit on transport loop
 * @return 0 on success, -1 on failure
 */
int cuda_launch_trace_packet(
    void *d_packets,
    int n_packets,
    void *d_r_inner,
    void *d_r_outer,
    void *d_line_list_nu,
    void *d_tau_sobolev,
    void *d_electron_density,
    Model_GPU model,
    Plasma_GPU plasma,
    void *d_stats,
    int stream_id,
    int max_iterations)
{
    if (!g_cuda_initialized) {
        fprintf(stderr, "[CUDA] Error: CUDA not initialized\n");
        return -1;
    }

    cudaStream_t stream = (cudaStream_t)cuda_interface_get_stream(stream_id);

    int threads_per_block = 256;
    int num_blocks = (n_packets + threads_per_block - 1) / threads_per_block;

    trace_packet_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        (RPacket_GPU*)d_packets,
        n_packets,
        (const double*)d_r_inner,
        (const double*)d_r_outer,
        (const double*)d_line_list_nu,
        (const double*)d_tau_sobolev,
        (const double*)d_electron_density,
        model,
        plasma,
        (GPUStats*)d_stats,
        max_iterations
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: trace_packet_kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

/**
 * Allocate and initialize packet array on GPU
 */
int cuda_allocate_packets(void **d_packets, int n_packets)
{
    cudaError_t err = cudaMalloc(d_packets, n_packets * sizeof(RPacket_GPU));
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: Failed to allocate packets: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

/**
 * Copy packets from host to device
 */
int cuda_upload_packets(void *d_packets, const void *h_packets, int n_packets)
{
    cudaError_t err = cudaMemcpy(d_packets, h_packets,
                                  n_packets * sizeof(RPacket_GPU),
                                  cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * Copy packets from device to host
 */
int cuda_download_packets(void *h_packets, const void *d_packets, int n_packets)
{
    cudaError_t err = cudaMemcpy(h_packets, d_packets,
                                  n_packets * sizeof(RPacket_GPU),
                                  cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * Allocate statistics structure on GPU
 */
int cuda_allocate_stats(void **d_stats)
{
    cudaError_t err = cudaMalloc(d_stats, sizeof(GPUStats));
    if (err != cudaSuccess) return -1;

    cudaMemset(*d_stats, 0, sizeof(GPUStats));
    return 0;
}

/**
 * Download statistics from GPU
 */
int cuda_download_stats(void *h_stats, const void *d_stats)
{
    cudaError_t err = cudaMemcpy(h_stats, d_stats, sizeof(GPUStats),
                                  cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

} /* extern "C" - Task Order #020 additions */

/* ============================================================================
 * WARMUP KERNEL
 * ============================================================================
 *
 * Simple kernel that performs minimal computation to verify GPU access.
 * Each thread writes its ID to verify execution.
 */

__global__ void warmup_kernel(int *output, int n_elements, int stream_id)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        /* Simple computation: store thread index + stream ID */
        output[idx] = idx + stream_id * 1000000;
    }

    /* First thread in first block prints verification */
    if (idx == 0) {
        printf("[GPU] Warmup kernel executed: stream_id=%d, n_elements=%d, blockDim=%d\n",
               stream_id, n_elements, blockDim.x);
    }
}

/**
 * More sophisticated warmup that does actual work
 * Useful for profiling stream concurrency
 */
__global__ void warmup_compute_kernel(float *data, int n_elements, int iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        float val = (float)idx;

        /* Perform some computation to keep GPU busy */
        for (int i = 0; i < iterations; i++) {
            val = sinf(val) * cosf(val) + 0.1f;
        }

        data[idx] = val;
    }
}

/* ============================================================================
 * C-CALLABLE INTERFACE IMPLEMENTATION
 * ============================================================================ */

extern "C" {

int cuda_interface_init(int device_id)
{
    if (g_cuda_initialized) {
        return 0;  /* Already initialized */
    }

    /* Query device count */
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "[CUDA] Error: No CUDA devices found: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    /* Validate device ID */
    if (device_id < 0 || device_id >= device_count) {
        fprintf(stderr, "[CUDA] Error: Invalid device ID %d (have %d devices)\n",
                device_id, device_count);
        return -1;
    }

    /* Set device */
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: Failed to set device %d: %s\n",
                device_id, cudaGetErrorString(err));
        return -1;
    }

    /* Print device info */
    cuda_interface_print_device_info(device_id);

    /* Initialize stream tracking */
    memset(g_streams, 0, sizeof(g_streams));
    memset(g_stream_created, 0, sizeof(g_stream_created));

    g_cuda_device_id = device_id;
    g_cuda_initialized = 1;

    printf("[CUDA] Initialization complete (device %d)\n", device_id);
    return 0;
}

int cuda_interface_is_available(void)
{
    return g_cuda_initialized;
}

int cuda_interface_get_device_count(void)
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess) ? count : 0;
}

void cuda_interface_print_device_info(int device_id)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);

    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: Cannot get device properties: %s\n",
                cudaGetErrorString(err));
        return;
    }

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║           LUMINA-SN CUDA Device Information                   ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Device %d: %-49s ║\n", device_id, prop.name);
    printf("║  Compute Capability:  %d.%d                                     ║\n",
           prop.major, prop.minor);
    printf("║  Multiprocessors:     %-42d ║\n", prop.multiProcessorCount);
    printf("║  Global Memory:       %-39.1f GB ║\n",
           prop.totalGlobalMem / 1e9);
    printf("║  Shared Mem/Block:    %-39zu KB ║\n",
           prop.sharedMemPerBlock / 1024);
    printf("║  Max Threads/Block:   %-42d ║\n", prop.maxThreadsPerBlock);
    printf("║  Warp Size:           %-42d ║\n", prop.warpSize);
    printf("║  Concurrent Kernels:  %-42s ║\n",
           prop.concurrentKernels ? "Yes" : "No");
    printf("║  Async Engines:       %-42d ║\n", prop.asyncEngineCount);
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
}

void cuda_interface_shutdown(void)
{
    if (!g_cuda_initialized) return;

    /* Destroy all streams */
    cuda_interface_destroy_streams();

    /* Reset device */
    cudaDeviceReset();

    g_cuda_initialized = 0;
    g_cuda_device_id = -1;

    printf("[CUDA] Shutdown complete\n");
}

/* ============================================================================
 * STREAM MANAGEMENT
 * ============================================================================ */

void* cuda_interface_get_stream(int stream_id)
{
    if (!g_cuda_initialized) {
        fprintf(stderr, "[CUDA] Error: CUDA not initialized\n");
        return NULL;
    }

    if (stream_id < 0 || stream_id >= CUDA_MAX_STREAMS) {
        fprintf(stderr, "[CUDA] Error: Invalid stream ID %d (max %d)\n",
                stream_id, CUDA_MAX_STREAMS - 1);
        return NULL;
    }

    /* Thread-safe stream creation */
    pthread_mutex_lock(&g_stream_mutex);

    if (!g_stream_created[stream_id]) {
        cudaError_t err = cudaStreamCreate(&g_streams[stream_id]);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA] Error: Failed to create stream %d: %s\n",
                    stream_id, cudaGetErrorString(err));
            pthread_mutex_unlock(&g_stream_mutex);
            return NULL;
        }
        g_stream_created[stream_id] = 1;
        printf("[CUDA] Created stream %d\n", stream_id);
    }

    pthread_mutex_unlock(&g_stream_mutex);

    return (void*)g_streams[stream_id];
}

int cuda_interface_stream_sync(int stream_id)
{
    if (!g_cuda_initialized || stream_id < 0 || stream_id >= CUDA_MAX_STREAMS) {
        return -1;
    }

    if (!g_stream_created[stream_id]) {
        return 0;  /* Stream not created, nothing to sync */
    }

    cudaError_t err = cudaStreamSynchronize(g_streams[stream_id]);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: Stream %d sync failed: %s\n",
                stream_id, cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int cuda_interface_sync_all_streams(void)
{
    if (!g_cuda_initialized) return -1;

    int failures = 0;

    for (int i = 0; i < CUDA_MAX_STREAMS; i++) {
        if (g_stream_created[i]) {
            if (cuda_interface_stream_sync(i) < 0) {
                failures++;
            }
        }
    }

    return (failures == 0) ? 0 : -1;
}

void cuda_interface_destroy_streams(void)
{
    pthread_mutex_lock(&g_stream_mutex);

    for (int i = 0; i < CUDA_MAX_STREAMS; i++) {
        if (g_stream_created[i]) {
            cudaStreamDestroy(g_streams[i]);
            g_stream_created[i] = 0;
        }
    }

    pthread_mutex_unlock(&g_stream_mutex);
}

/* ============================================================================
 * WARMUP / DIAGNOSTIC KERNELS
 * ============================================================================ */

int cuda_interface_launch_warmup(int stream_id, int n_elements)
{
    if (!g_cuda_initialized) {
        fprintf(stderr, "[CUDA] Error: CUDA not initialized\n");
        return -1;
    }

    /* Get or create stream */
    cudaStream_t stream = (cudaStream_t)cuda_interface_get_stream(stream_id);
    if (stream == NULL && stream_id != 0) {
        /* Stream 0 is the default stream, NULL is valid */
        return -1;
    }

    /* Allocate device memory for output */
    int *d_output = NULL;
    cudaError_t err = cudaMalloc(&d_output, n_elements * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: Failed to allocate warmup buffer: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    /* Launch kernel */
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    warmup_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_output, n_elements, stream_id
    );

    /* Check for launch errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: Warmup kernel launch failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_output);
        return -1;
    }

    /* Free device memory (will complete after kernel finishes) */
    cudaFreeAsync(d_output, stream);

    return 0;
}

int cuda_interface_launch_warmup_sync(int stream_id, int n_elements)
{
    int result = cuda_interface_launch_warmup(stream_id, n_elements);
    if (result < 0) return result;

    return cuda_interface_stream_sync(stream_id);
}

int cuda_interface_test_concurrency(int n_streams)
{
    if (!g_cuda_initialized) {
        fprintf(stderr, "[CUDA] Error: CUDA not initialized\n");
        return -1;
    }

    if (n_streams <= 0 || n_streams > CUDA_MAX_STREAMS) {
        fprintf(stderr, "[CUDA] Error: Invalid stream count %d\n", n_streams);
        return -1;
    }

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║           CUDA Concurrency Test (%2d streams)                  ║\n",
           n_streams);
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    /* Allocate work buffers for each stream */
    const int n_elements = 1024 * 1024;  /* 1M elements per stream */
    const int iterations = 100;           /* Work per element */

    float **d_buffers = (float**)malloc(n_streams * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(n_streams * sizeof(cudaStream_t));

    for (int i = 0; i < n_streams; i++) {
        cudaMalloc(&d_buffers[i], n_elements * sizeof(float));
        streams[i] = (cudaStream_t)cuda_interface_get_stream(i);
    }

    /* Create timing events */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Record start time */
    cudaEventRecord(start);

    /* Launch kernels on all streams */
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;

    printf("[CUDA] Launching %d kernels concurrently...\n", n_streams);

    for (int i = 0; i < n_streams; i++) {
        warmup_compute_kernel<<<num_blocks, threads_per_block, 0, streams[i]>>>(
            d_buffers[i], n_elements, iterations
        );
    }

    /* Record stop time (after all kernels) */
    cudaEventRecord(stop);

    /* Wait for all streams to complete */
    cudaEventSynchronize(stop);

    /* Calculate elapsed time */
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    printf("[CUDA] All kernels completed in %.2f ms\n", elapsed_ms);
    printf("[CUDA] Throughput: %.2f M elements/stream/ms\n",
           (float)n_elements / 1e6 / elapsed_ms * n_streams);

    /* Cleanup */
    for (int i = 0; i < n_streams; i++) {
        cudaFree(d_buffers[i]);
    }
    free(d_buffers);
    free(streams);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("[CUDA] Concurrency test PASSED\n\n");
    return 0;
}

/* ============================================================================
 * MEMORY MANAGEMENT
 * ============================================================================ */

void* cuda_interface_malloc(size_t size_bytes)
{
    if (!g_cuda_initialized) return NULL;

    void *d_ptr = NULL;
    cudaError_t err = cudaMalloc(&d_ptr, size_bytes);

    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error: cudaMalloc(%zu) failed: %s\n",
                size_bytes, cudaGetErrorString(err));
        return NULL;
    }

    return d_ptr;
}

void cuda_interface_free(void *d_ptr)
{
    if (d_ptr) {
        cudaFree(d_ptr);
    }
}

int cuda_interface_memcpy_h2d(void *d_dst, const void *h_src, size_t size)
{
    cudaError_t err = cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_interface_memcpy_d2h(void *h_dst, const void *d_src, size_t size)
{
    cudaError_t err = cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_interface_memcpy_h2d_async(void *d_dst, const void *h_src,
                                     size_t size, int stream_id)
{
    if (!g_cuda_initialized || stream_id < 0 || stream_id >= CUDA_MAX_STREAMS) {
        return -1;
    }

    cudaStream_t stream = g_stream_created[stream_id] ?
                          g_streams[stream_id] : 0;

    cudaError_t err = cudaMemcpyAsync(d_dst, h_src, size,
                                       cudaMemcpyHostToDevice, stream);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_interface_memcpy_d2h_async(void *h_dst, const void *d_src,
                                     size_t size, int stream_id)
{
    if (!g_cuda_initialized || stream_id < 0 || stream_id >= CUDA_MAX_STREAMS) {
        return -1;
    }

    cudaStream_t stream = g_stream_created[stream_id] ?
                          g_streams[stream_id] : 0;

    cudaError_t err = cudaMemcpyAsync(h_dst, d_src, size,
                                       cudaMemcpyDeviceToHost, stream);
    return (err == cudaSuccess) ? 0 : -1;
}

void* cuda_interface_malloc_host(size_t size_bytes)
{
    void *h_ptr = NULL;
    cudaError_t err = cudaMallocHost(&h_ptr, size_bytes);
    return (err == cudaSuccess) ? h_ptr : NULL;
}

void cuda_interface_free_host(void *h_ptr)
{
    if (h_ptr) {
        cudaFreeHost(h_ptr);
    }
}

/* ============================================================================
 * ERROR HANDLING
 * ============================================================================ */

const char* cuda_interface_get_error(void)
{
    return cudaGetErrorString(cudaGetLastError());
}

int cuda_interface_check_error(void)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Error detected: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

} /* extern "C" */

/* ============================================================================
 * STANDALONE TEST DRIVER (when compiled directly)
 * ============================================================================ */

#ifdef GPU_TRANSPORT_STANDALONE

#include <omp.h>

int main(int argc, char *argv[])
{
    printf("LUMINA-SN GPU Transport Test\n");
    printf("============================\n\n");

    /* Initialize CUDA */
    if (cuda_interface_init(0) < 0) {
        fprintf(stderr, "Failed to initialize CUDA\n");
        return 1;
    }

    /* Test basic warmup */
    printf("Testing single warmup kernel...\n");
    if (cuda_interface_launch_warmup_sync(0, 1024) < 0) {
        fprintf(stderr, "Warmup failed\n");
        return 1;
    }
    printf("Single warmup: PASSED\n\n");

    /* Test concurrency */
    int n_threads = 4;
    #ifdef _OPENMP
    n_threads = omp_get_max_threads();
    if (n_threads > 8) n_threads = 8;  /* Limit for test */
    #endif

    printf("Testing OpenMP + CUDA concurrency (%d threads)...\n", n_threads);

    #pragma omp parallel num_threads(n_threads)
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif

        printf("[Thread %d] Launching warmup kernel on stream %d\n", tid, tid);
        cuda_interface_launch_warmup(tid, 1024);
    }

    /* Sync all streams */
    cuda_interface_sync_all_streams();
    printf("OpenMP concurrency test: PASSED\n\n");

    /* Test stream concurrency with profiling */
    cuda_interface_test_concurrency(n_threads);

    /* Cleanup */
    cuda_interface_shutdown();

    printf("All tests PASSED\n");
    return 0;
}

#endif /* GPU_TRANSPORT_STANDALONE */
