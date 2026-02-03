/**
 * LUMINA-SN CUDA Atomic Data Loader
 * cuda_atomic.cu - Memory mapping from CPU to GPU structures
 *
 * Implements flattening of hierarchical atomic data into contiguous
 * GPU memory layouts optimized for coalesced access patterns.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cuda_atomic.h"
#include "atomic_data.h"
#include "plasma_physics.h"
#include "simulation_state.h"

/* ============================================================================
 * CONSTANT MEMORY SYMBOL DECLARATIONS
 * ============================================================================ */

extern __constant__ CudaSimConfig d_config;
extern __constant__ CudaPartitionCache d_partition;

/* ============================================================================
 * MEMORY MAPPING: CPU → GPU
 * ============================================================================ */

/**
 * Flatten and upload element data to GPU
 */
static int upload_element_data(CudaDeviceMemory *mem, const AtomicData *atomic)
{
    CudaElementData h_elements;
    memset(&h_elements, 0, sizeof(h_elements));

    h_elements.n_elements = atomic->n_elements;
    if (h_elements.n_elements > CUDA_MAX_ELEMENTS) {
        fprintf(stderr, "CUDA: Too many elements (%d > %d)\n",
                h_elements.n_elements, CUDA_MAX_ELEMENTS);
        return -1;
    }

    /* Flatten element data to SOA layout */
    for (int i = 0; i < atomic->n_elements; i++) {
        const Element *elem = &atomic->elements[i];
        int Z = elem->atomic_number;

        if (Z >= 1 && Z <= CUDA_MAX_ELEMENTS) {
            h_elements.atomic_number[Z - 1] = Z;
            h_elements.mass[Z - 1] = elem->mass;
            strncpy(h_elements.symbol[Z - 1], elem->symbol, 2);
        }
    }

    /* Build ion index ranges for each element */
    int ion_idx = 0;
    for (int Z = 1; Z <= CUDA_MAX_ELEMENTS; Z++) {
        h_elements.ion_start[Z] = ion_idx;

        /* Count ions for this element */
        for (int i = 0; i < atomic->n_ions; i++) {
            if (atomic->ions[i].atomic_number == Z) {
                ion_idx++;
            }
        }

        h_elements.ion_end[Z] = ion_idx;
    }

    /* Allocate and copy to device */
    cudaError_t err = cudaMalloc(&mem->d_elements, sizeof(CudaElementData));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: Failed to allocate element data: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(mem->d_elements, &h_elements, sizeof(CudaElementData),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    mem->total_memory += sizeof(CudaElementData);
    printf("[CUDA] Uploaded %d elements (%.2f KB)\n",
           h_elements.n_elements, sizeof(CudaElementData) / 1024.0);

    return 0;
}

/**
 * Flatten and upload ion data to GPU
 */
static int upload_ion_data(CudaDeviceMemory *mem, const AtomicData *atomic)
{
    CudaIonData h_ions;
    memset(&h_ions, 0, sizeof(h_ions));

    h_ions.n_ions = atomic->n_ions;
    if (h_ions.n_ions > CUDA_MAX_IONS) {
        fprintf(stderr, "CUDA: Too many ions (%d > %d)\n",
                h_ions.n_ions, CUDA_MAX_IONS);
        return -1;
    }

    /* Flatten ion data to SOA layout */
    int level_idx = 0;
    int line_idx = 0;

    for (int i = 0; i < atomic->n_ions; i++) {
        const Ion *ion = &atomic->ions[i];

        h_ions.atomic_number[i] = ion->atomic_number;
        h_ions.ion_number[i] = ion->ion_number;
        h_ions.ionization_energy[i] = ion->ionization_energy;
        h_ions.n_levels[i] = ion->n_levels;

        /* Level index range */
        h_ions.level_start[i] = level_idx;
        level_idx += ion->n_levels;
        h_ions.level_end[i] = level_idx;

        /* Line index range (count lines for this ion) */
        h_ions.line_start[i] = line_idx;
        for (int64_t j = 0; j < atomic->n_lines; j++) {
            if (atomic->lines[j].atomic_number == ion->atomic_number &&
                atomic->lines[j].ion_number == ion->ion_number) {
                line_idx++;
            }
        }
        h_ions.line_end[i] = line_idx;
    }

    /* Allocate and copy to device */
    cudaError_t err = cudaMalloc(&mem->d_ions, sizeof(CudaIonData));
    if (err != cudaSuccess) return -1;

    err = cudaMemcpy(mem->d_ions, &h_ions, sizeof(CudaIonData),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -1;

    mem->total_memory += sizeof(CudaIonData);
    printf("[CUDA] Uploaded %d ions (%.2f KB)\n",
           h_ions.n_ions, sizeof(CudaIonData) / 1024.0);

    return 0;
}

/**
 * Flatten and upload level data to GPU
 */
static int upload_level_data(CudaDeviceMemory *mem, const AtomicData *atomic)
{
    CudaLevelData *h_levels = (CudaLevelData *)calloc(1, sizeof(CudaLevelData));
    if (!h_levels) return -1;

    h_levels->n_levels = atomic->n_levels;
    if (h_levels->n_levels > CUDA_MAX_LEVELS) {
        fprintf(stderr, "CUDA: Too many levels (%d > %d)\n",
                h_levels->n_levels, CUDA_MAX_LEVELS);
        free(h_levels);
        return -1;
    }

    /* Flatten level data to SOA layout */
    for (int i = 0; i < atomic->n_levels; i++) {
        const Level *level = &atomic->levels[i];

        h_levels->atomic_number[i] = level->atomic_number;
        h_levels->ion_number[i] = level->ion_number;
        h_levels->level_number[i] = level->level_number;
        h_levels->energy[i] = level->energy;
        h_levels->g[i] = level->g;
        h_levels->metastable[i] = level->metastable;
    }

    /* Allocate and copy to device */
    cudaError_t err = cudaMalloc(&mem->d_levels, sizeof(CudaLevelData));
    if (err != cudaSuccess) {
        free(h_levels);
        return -1;
    }

    err = cudaMemcpy(mem->d_levels, h_levels, sizeof(CudaLevelData),
                     cudaMemcpyHostToDevice);
    free(h_levels);
    if (err != cudaSuccess) return -1;

    mem->total_memory += sizeof(CudaLevelData);
    printf("[CUDA] Uploaded %d levels (%.2f MB)\n",
           atomic->n_levels, sizeof(CudaLevelData) / 1e6);

    return 0;
}

/**
 * Flatten and upload line data to GPU (frequency-sorted)
 */
static int upload_line_data(CudaDeviceMemory *mem, const AtomicData *atomic)
{
    /* Allocate host buffer */
    CudaLineData *h_lines = (CudaLineData *)calloc(1, sizeof(CudaLineData));
    if (!h_lines) return -1;

    int64_t n_lines = atomic->n_lines;
    if (n_lines > CUDA_MAX_LINES) {
        fprintf(stderr, "CUDA: Too many lines (%ld > %d)\n",
                (long)n_lines, CUDA_MAX_LINES);
        free(h_lines);
        return -1;
    }

    h_lines->n_lines = n_lines;

    /* Create sorted index array */
    int64_t *sort_idx = (int64_t *)malloc(n_lines * sizeof(int64_t));
    double *sort_nu = (double *)malloc(n_lines * sizeof(double));

    for (int64_t i = 0; i < n_lines; i++) {
        sort_idx[i] = i;
        sort_nu[i] = atomic->lines[i].nu;
    }

    /* Sort by frequency (simple insertion sort for now) */
    for (int64_t i = 1; i < n_lines; i++) {
        int64_t key_idx = sort_idx[i];
        double key_nu = sort_nu[i];
        int64_t j = i - 1;

        while (j >= 0 && sort_nu[j] > key_nu) {
            sort_idx[j + 1] = sort_idx[j];
            sort_nu[j + 1] = sort_nu[j];
            j--;
        }
        sort_idx[j + 1] = key_idx;
        sort_nu[j + 1] = key_nu;
    }

    /* Flatten to SOA layout in sorted order */
    for (int64_t i = 0; i < n_lines; i++) {
        int64_t orig_idx = sort_idx[i];
        const Line *line = &atomic->lines[orig_idx];

        h_lines->nu[i] = line->nu;
        h_lines->wavelength[i] = line->wavelength;
        h_lines->f_lu[i] = line->f_lu;
        h_lines->A_ul[i] = line->A_ul;
        h_lines->atomic_number[i] = line->atomic_number;
        h_lines->ion_number[i] = line->ion_number;
        h_lines->level_lower[i] = line->level_number_lower;
        h_lines->level_upper[i] = line->level_number_upper;

        /* Precompute Sobolev factor: SOBOLEV_CONST * f_lu * λ */
        h_lines->sobolev_factor[i] = 2.6540281e-2 * line->f_lu * line->wavelength;
    }

    free(sort_idx);
    free(sort_nu);

    /* Allocate and copy to device */
    cudaError_t err = cudaMalloc(&mem->d_lines, sizeof(CudaLineData));
    if (err != cudaSuccess) {
        free(h_lines);
        return -1;
    }

    err = cudaMemcpy(mem->d_lines, h_lines, sizeof(CudaLineData),
                     cudaMemcpyHostToDevice);
    free(h_lines);
    if (err != cudaSuccess) return -1;

    mem->total_memory += sizeof(CudaLineData);
    printf("[CUDA] Uploaded %ld lines (%.2f MB, frequency-sorted)\n",
           (long)n_lines, sizeof(CudaLineData) / 1e6);

    return 0;
}

/**
 * Upload shell state and active lines to GPU
 */
static int upload_shell_data(CudaDeviceMemory *mem, const SimulationState *state)
{
    int n_shells = state->n_shells;

    /* Allocate host shell array */
    CudaShellState *h_shells = (CudaShellState *)calloc(n_shells, sizeof(CudaShellState));
    if (!h_shells) return -1;

    /* Count total active lines across all shells */
    int64_t total_active = 0;
    for (int i = 0; i < n_shells; i++) {
        total_active += state->shells[i].n_active_lines;
    }

    /* Allocate host active lines array */
    CudaActiveLine *h_active = (CudaActiveLine *)malloc(total_active * sizeof(CudaActiveLine));
    if (!h_active) {
        free(h_shells);
        return -1;
    }

    /* Flatten shell and active line data */
    int64_t active_offset = 0;
    for (int i = 0; i < n_shells; i++) {
        const ShellState *shell = &state->shells[i];

        h_shells[i].r_inner = shell->r_inner;
        h_shells[i].r_outer = shell->r_outer;
        h_shells[i].v_inner = shell->v_inner;
        h_shells[i].v_outer = shell->v_outer;
        h_shells[i].T = shell->plasma.T;
        h_shells[i].rho = shell->plasma.rho;
        h_shells[i].n_e = shell->plasma.n_e;
        h_shells[i].sigma_thomson_ne = shell->sigma_thomson_ne;
        h_shells[i].tau_electron = shell->tau_electron;

        /* Copy ion fractions for major species (Z=1..30, ions 0..5) */
        for (int Z = 1; Z <= CUDA_MAX_ELEMENTS; Z++) {
            for (int ion = 0; ion < 6 && ion <= Z; ion++) {
                h_shells[i].ion_fraction[Z][ion] = shell->plasma.ion_fraction[Z][ion];
                h_shells[i].n_ion[Z][ion] = shell->plasma.n_ion[Z][ion];
            }
        }

        /* Copy active lines for this shell */
        h_shells[i].n_active_lines = shell->n_active_lines;
        h_shells[i].active_line_start = active_offset;

        for (int64_t j = 0; j < shell->n_active_lines; j++) {
            h_active[active_offset + j].line_idx = shell->active_lines[j].line_idx;
            h_active[active_offset + j].nu = shell->active_lines[j].nu;
            h_active[active_offset + j].tau_sobolev = shell->active_lines[j].tau_sobolev;
        }

        active_offset += shell->n_active_lines;
    }

    /* Allocate device memory for shells */
    cudaError_t err = cudaMalloc(&mem->d_shells, n_shells * sizeof(CudaShellState));
    if (err != cudaSuccess) {
        free(h_shells);
        free(h_active);
        return -1;
    }

    err = cudaMemcpy(mem->d_shells, h_shells, n_shells * sizeof(CudaShellState),
                     cudaMemcpyHostToDevice);
    free(h_shells);
    if (err != cudaSuccess) {
        free(h_active);
        return -1;
    }

    /* Allocate device memory for active lines */
    err = cudaMalloc(&mem->d_active_lines, total_active * sizeof(CudaActiveLine));
    if (err != cudaSuccess) {
        free(h_active);
        return -1;
    }

    err = cudaMemcpy(mem->d_active_lines, h_active,
                     total_active * sizeof(CudaActiveLine),
                     cudaMemcpyHostToDevice);
    free(h_active);
    if (err != cudaSuccess) return -1;

    mem->total_active_lines = total_active;
    mem->total_memory += n_shells * sizeof(CudaShellState);
    mem->total_memory += total_active * sizeof(CudaActiveLine);

    printf("[CUDA] Uploaded %d shells with %ld active lines (%.2f MB)\n",
           n_shells, (long)total_active,
           (n_shells * sizeof(CudaShellState) + total_active * sizeof(CudaActiveLine)) / 1e6);

    return 0;
}

/* ============================================================================
 * PUBLIC API IMPLEMENTATION
 * ============================================================================ */

/**
 * Allocate and populate all device memory from CPU structures
 */
int cuda_allocate_atomic_data(CudaDeviceMemory *mem,
                               const void *cpu_atomic,
                               const void *cpu_state)
{
    const AtomicData *atomic = (const AtomicData *)cpu_atomic;
    const SimulationState *state = (const SimulationState *)cpu_state;

    memset(mem, 0, sizeof(CudaDeviceMemory));

    printf("\n[CUDA] Uploading atomic data to GPU...\n");

    /* Upload each data structure */
    if (upload_element_data(mem, atomic) != 0) return -1;
    if (upload_ion_data(mem, atomic) != 0) return -1;
    if (upload_level_data(mem, atomic) != 0) return -1;
    if (upload_line_data(mem, atomic) != 0) return -1;
    if (upload_shell_data(mem, state) != 0) return -1;

    printf("[CUDA] Total device memory: %.2f MB\n", mem->total_memory / 1e6);

    return 0;
}

/**
 * Task Order #038-Revised: Upload macro-atom downbranch table to GPU
 */
int cuda_upload_downbranch_data(CudaDeviceMemory *mem, const void *cpu_atomic)
{
    const AtomicData *atomic = (const AtomicData *)cpu_atomic;

    /* Check if downbranch data exists */
    if (atomic->downbranch.total_emission_entries == 0) {
        printf("[CUDA] WARNING: Downbranch table is empty - macro-atom will fallback to scatter\n");
        mem->d_downbranch = NULL;
        mem->total_emission_entries = 0;
        return 0;  /* Not an error, just no macro-atom data */
    }

    printf("[CUDA] Uploading downbranch table: %ld entries for %ld lines\n",
           (long)atomic->downbranch.total_emission_entries,
           (long)atomic->n_lines);

    /* Allocate device structure */
    CudaDownbranchData *h_downbranch = (CudaDownbranchData *)calloc(1, sizeof(CudaDownbranchData));
    if (!h_downbranch) return -1;

    h_downbranch->n_lines = atomic->n_lines;
    h_downbranch->total_emission_entries = atomic->downbranch.total_emission_entries;

    /* Allocate and copy emission_start array */
    size_t start_size = atomic->n_lines * sizeof(int64_t);
    cudaError_t err = cudaMalloc(&h_downbranch->emission_start, start_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: Failed to allocate emission_start\n");
        free(h_downbranch);
        return -1;
    }
    err = cudaMemcpy(h_downbranch->emission_start, atomic->downbranch.emission_line_start,
                     start_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(h_downbranch->emission_start);
        free(h_downbranch);
        return -1;
    }
    mem->total_memory += start_size;

    /* Allocate and copy emission_count array */
    size_t count_size = atomic->n_lines * sizeof(int32_t);
    err = cudaMalloc(&h_downbranch->emission_count, count_size);
    if (err != cudaSuccess) {
        cudaFree(h_downbranch->emission_start);
        free(h_downbranch);
        return -1;
    }
    err = cudaMemcpy(h_downbranch->emission_count, atomic->downbranch.emission_line_count,
                     count_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(h_downbranch->emission_start);
        cudaFree(h_downbranch->emission_count);
        free(h_downbranch);
        return -1;
    }
    mem->total_memory += count_size;

    /* Allocate and copy emission_line_id array */
    int64_t n_entries = atomic->downbranch.total_emission_entries;
    size_t line_id_size = n_entries * sizeof(int64_t);
    err = cudaMalloc(&h_downbranch->emission_line_id, line_id_size);
    if (err != cudaSuccess) {
        cudaFree(h_downbranch->emission_start);
        cudaFree(h_downbranch->emission_count);
        free(h_downbranch);
        return -1;
    }
    err = cudaMemcpy(h_downbranch->emission_line_id, atomic->downbranch.emission_line_ids,
                     line_id_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(h_downbranch->emission_start);
        cudaFree(h_downbranch->emission_count);
        cudaFree(h_downbranch->emission_line_id);
        free(h_downbranch);
        return -1;
    }
    mem->total_memory += line_id_size;

    /* Allocate and copy branching_prob array */
    size_t prob_size = n_entries * sizeof(double);
    err = cudaMalloc(&h_downbranch->branching_prob, prob_size);
    if (err != cudaSuccess) {
        cudaFree(h_downbranch->emission_start);
        cudaFree(h_downbranch->emission_count);
        cudaFree(h_downbranch->emission_line_id);
        free(h_downbranch);
        return -1;
    }
    err = cudaMemcpy(h_downbranch->branching_prob, atomic->downbranch.branching_probs,
                     prob_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(h_downbranch->emission_start);
        cudaFree(h_downbranch->emission_count);
        cudaFree(h_downbranch->emission_line_id);
        cudaFree(h_downbranch->branching_prob);
        free(h_downbranch);
        return -1;
    }
    mem->total_memory += prob_size;

    /* Allocate and copy the structure itself */
    err = cudaMalloc(&mem->d_downbranch, sizeof(CudaDownbranchData));
    if (err != cudaSuccess) {
        cudaFree(h_downbranch->emission_start);
        cudaFree(h_downbranch->emission_count);
        cudaFree(h_downbranch->emission_line_id);
        cudaFree(h_downbranch->branching_prob);
        free(h_downbranch);
        return -1;
    }
    err = cudaMemcpy(mem->d_downbranch, h_downbranch, sizeof(CudaDownbranchData),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(mem->d_downbranch);
        cudaFree(h_downbranch->emission_start);
        cudaFree(h_downbranch->emission_count);
        cudaFree(h_downbranch->emission_line_id);
        cudaFree(h_downbranch->branching_prob);
        free(h_downbranch);
        return -1;
    }
    mem->total_memory += sizeof(CudaDownbranchData);

    mem->total_emission_entries = n_entries;
    free(h_downbranch);

    printf("[CUDA] Uploaded downbranch data (%.2f MB)\n",
           (start_size + count_size + line_id_size + prob_size) / 1e6);

    return 0;
}

/**
 * Upload partition function cache to constant memory
 */
int cuda_upload_partition_cache(const void *cpu_cache)
{
    const PartitionFunctionCache *cache = (const PartitionFunctionCache *)cpu_cache;

    CudaPartitionCache h_cache;
    memset(&h_cache, 0, sizeof(h_cache));

    /* Copy temperature grid */
    for (int i = 0; i < CUDA_PARTITION_N_TEMPS; i++) {
        h_cache.T_grid[i] = cache->T_grid[i];
    }

    h_cache.log_T_min = log(CUDA_PARTITION_T_MIN);
    h_cache.log_T_max = log(CUDA_PARTITION_T_MAX);
    h_cache.d_log_T = (h_cache.log_T_max - h_cache.log_T_min) /
                       (CUDA_PARTITION_N_TEMPS - 1);

    /* Copy partition functions */
    for (int Z = 1; Z <= CUDA_MAX_ELEMENTS; Z++) {
        for (int ion = 0; ion <= Z; ion++) {
            for (int t = 0; t < CUDA_PARTITION_N_TEMPS; t++) {
                h_cache.U[Z][ion][t] = cache->U[Z][ion][t];
            }
        }
    }

    cudaError_t err = cudaMemcpyToSymbol(d_partition, &h_cache,
                                          sizeof(CudaPartitionCache));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: Failed to upload partition cache: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    printf("[CUDA] Uploaded partition function cache to constant memory\n");
    return 0;
}
