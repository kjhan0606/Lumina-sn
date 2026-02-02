/**
 * LUMINA-SN CUDA Interface for C Code
 * cuda_interface.h - Pure C header for CUDA interoperability
 *
 * This header provides C-callable wrappers for CUDA functions.
 * It does NOT include any CUDA headers, so it can be safely
 * included from pure C code compiled with gcc.
 *
 * Task Order #019: CUDA Infrastructure Setup
 *
 * Architecture:
 *   - OpenMP threads on CPU each manage their own CUDA stream
 *   - CUDA streams enable concurrent kernel execution
 *   - C code calls through this interface to launch GPU work
 */

#ifndef CUDA_INTERFACE_H
#define CUDA_INTERFACE_H

#include <stdint.h>
#include <stddef.h>

/* Include shared struct definitions (C-compatible) */
#include "cuda_shared.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CUDA DEVICE MANAGEMENT
 * ============================================================================ */

/**
 * Initialize CUDA and select device
 * @param device_id  GPU device index (0 for single-GPU systems)
 * @return 0 on success, -1 on failure
 */
int cuda_interface_init(int device_id);

/**
 * Check if CUDA is available and initialized
 * @return 1 if CUDA is ready, 0 otherwise
 */
int cuda_interface_is_available(void);

/**
 * Get number of available CUDA devices
 * @return number of devices, or 0 if CUDA not available
 */
int cuda_interface_get_device_count(void);

/**
 * Print CUDA device properties
 * @param device_id  GPU device index
 */
void cuda_interface_print_device_info(int device_id);

/**
 * Shutdown CUDA and release all resources
 */
void cuda_interface_shutdown(void);

/* ============================================================================
 * CUDA STREAM MANAGEMENT
 * ============================================================================
 *
 * Each OpenMP thread should use its own CUDA stream for concurrent execution.
 * Streams are created lazily and cached for reuse.
 */

/**
 * Maximum number of concurrent streams (one per OpenMP thread)
 */
#define CUDA_MAX_STREAMS 64

/**
 * Get or create a CUDA stream for the given ID
 * Streams are cached internally for reuse.
 *
 * @param stream_id  Stream identifier (typically omp_get_thread_num())
 * @return opaque stream handle, or NULL on failure
 */
void* cuda_interface_get_stream(int stream_id);

/**
 * Synchronize a specific stream (wait for all operations to complete)
 * @param stream_id  Stream identifier
 * @return 0 on success, -1 on failure
 */
int cuda_interface_stream_sync(int stream_id);

/**
 * Synchronize all active streams
 * @return 0 on success, -1 on failure
 */
int cuda_interface_sync_all_streams(void);

/**
 * Destroy all cached streams
 */
void cuda_interface_destroy_streams(void);

/* ============================================================================
 * WARMUP / DIAGNOSTIC KERNELS
 * ============================================================================
 *
 * These functions are used to verify CUDA concurrency from OpenMP threads.
 */

/**
 * Launch a warmup kernel on the specified stream
 * This kernel performs minimal work but verifies GPU access.
 *
 * @param stream_id   Stream identifier (typically omp_get_thread_num())
 * @param n_elements  Number of elements to process (for scaling test)
 * @return 0 on success, -1 on failure
 */
int cuda_interface_launch_warmup(int stream_id, int n_elements);

/**
 * Launch warmup kernel and wait for completion
 * Combines launch + sync for testing.
 *
 * @param stream_id   Stream identifier
 * @param n_elements  Number of elements to process
 * @return 0 on success, -1 on failure
 */
int cuda_interface_launch_warmup_sync(int stream_id, int n_elements);

/**
 * Run concurrency test from multiple threads
 * Launches kernels from n_streams threads and verifies parallel execution.
 *
 * @param n_streams  Number of concurrent streams to test
 * @return 0 on success, -1 on failure
 */
int cuda_interface_test_concurrency(int n_streams);

/* ============================================================================
 * DEVICE MEMORY MANAGEMENT
 * ============================================================================
 *
 * Generic memory allocation wrappers for C code.
 */

/**
 * Allocate device memory
 * @param size_bytes  Number of bytes to allocate
 * @return device pointer, or NULL on failure
 */
void* cuda_interface_malloc(size_t size_bytes);

/**
 * Free device memory
 * @param d_ptr  Device pointer to free
 */
void cuda_interface_free(void *d_ptr);

/**
 * Copy data from host to device
 * @param d_dst   Device destination pointer
 * @param h_src   Host source pointer
 * @param size    Number of bytes to copy
 * @return 0 on success, -1 on failure
 */
int cuda_interface_memcpy_h2d(void *d_dst, const void *h_src, size_t size);

/**
 * Copy data from device to host
 * @param h_dst   Host destination pointer
 * @param d_src   Device source pointer
 * @param size    Number of bytes to copy
 * @return 0 on success, -1 on failure
 */
int cuda_interface_memcpy_d2h(void *h_dst, const void *d_src, size_t size);

/**
 * Async copy from host to device on a stream
 * @param d_dst      Device destination pointer
 * @param h_src      Host source pointer (must be pinned memory)
 * @param size       Number of bytes to copy
 * @param stream_id  Stream to use for async transfer
 * @return 0 on success, -1 on failure
 */
int cuda_interface_memcpy_h2d_async(void *d_dst, const void *h_src,
                                     size_t size, int stream_id);

/**
 * Async copy from device to host on a stream
 * @param h_dst      Host destination pointer (must be pinned memory)
 * @param d_src      Device source pointer
 * @param size       Number of bytes to copy
 * @param stream_id  Stream to use for async transfer
 * @return 0 on success, -1 on failure
 */
int cuda_interface_memcpy_d2h_async(void *h_dst, const void *d_src,
                                     size_t size, int stream_id);

/**
 * Allocate pinned (page-locked) host memory for async transfers
 * @param size_bytes  Number of bytes to allocate
 * @return host pointer, or NULL on failure
 */
void* cuda_interface_malloc_host(size_t size_bytes);

/**
 * Free pinned host memory
 * @param h_ptr  Host pointer to free
 */
void cuda_interface_free_host(void *h_ptr);

/* ============================================================================
 * ERROR HANDLING
 * ============================================================================ */

/**
 * Get the last CUDA error message
 * @return error string (static buffer, do not free)
 */
const char* cuda_interface_get_error(void);

/**
 * Check and clear any pending CUDA errors
 * @return 0 if no error, -1 if error occurred
 */
int cuda_interface_check_error(void);

/* ============================================================================
 * TASK ORDER #020: TRACE PACKET KERNEL INTERFACE
 * ============================================================================ */

/**
 * Launch trace_packet kernel on GPU
 *
 * @param d_packets          Device pointer to RPacket_GPU array
 * @param n_packets          Number of packets to process
 * @param d_r_inner          Device pointer to inner radii [n_shells]
 * @param d_r_outer          Device pointer to outer radii [n_shells]
 * @param d_line_list_nu     Device pointer to line frequencies [n_lines]
 * @param d_tau_sobolev      Device pointer to optical depths [n_lines x n_shells]
 * @param d_electron_density Device pointer to electron densities [n_shells]
 * @param model              Model parameters (passed by value)
 * @param plasma             Plasma parameters (passed by value)
 * @param d_stats            Device pointer to GPUStats (can be NULL)
 * @param stream_id          CUDA stream to use
 * @param max_iterations     Safety limit on transport loop
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
    int max_iterations);

/**
 * Allocate packet array on GPU
 * @param d_packets  [out] Device pointer
 * @param n_packets  Number of packets
 * @return 0 on success, -1 on failure
 */
int cuda_allocate_packets(void **d_packets, int n_packets);

/**
 * Copy packets from host to device
 */
int cuda_upload_packets(void *d_packets, const void *h_packets, int n_packets);

/**
 * Copy packets from device to host
 */
int cuda_download_packets(void *h_packets, const void *d_packets, int n_packets);

/**
 * Allocate statistics structure on GPU
 */
int cuda_allocate_stats(void **d_stats);

/**
 * Download statistics from GPU
 */
int cuda_download_stats(void *h_stats, const void *d_stats);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_INTERFACE_H */
