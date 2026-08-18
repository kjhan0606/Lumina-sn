#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

struct Transform {
    double transmission;
    double emission;
};

struct WindowNode {
    Transform value;
    Transform aggregate;
};

struct LogicalMap {
    int epoch;
    int offset;
    int output_bin;
    int boundary_bin;
    int q_fold_index;
    int front_fold_index;
    int back_fold_index;
};

enum TraceChain {
    TRACE_Q = 0,
    TRACE_F = 1,
    TRACE_P = 2,
    TRACE_G = 3
};

enum TracePrimitive {
    TRACE_MUL_T = 0,
    TRACE_MUL_E = 1,
    TRACE_ADD_E = 2
};

struct PrimitiveTrace {
    int mode_slot;
    int epoch;
    int chain;
    int node;
    int primitive;
    unsigned long long lhs_bits;
    unsigned long long rhs_bits;
    unsigned long long result_bits;
};

struct ProofCounts {
    long long cases;
    long long mode_cases;
    long long records;
    long long values;
};

enum G6FailureCode {
    G6_OK = 0,
    G6_INVALID_MODE = 1,
    G6_INVALID_INDEX = 2,
    G6_WORKSPACE_TOO_SMALL = 3,
    G6_ALLOCATION_FAILURE = 4,
    G6_CUDA_FAILURE = 5,
    G6_NONFINITE_TRANSFORM = 6,
    G6_COMPOSE_FAILURE = 7
};

struct G6FailureDetail {
    int code;
    int mode;
    int epoch;
    int chain;
    int node;
    int source_index;
};

enum Rounding {
    ROUND_LOWER = -1,
    ROUND_NEAREST = 0,
    ROUND_UPPER = 1
};

static const int kModes[3] = {ROUND_LOWER, ROUND_NEAREST, ROUND_UPPER};

static void fail(const char *message)
{
    std::fprintf(stderr, "CMF_EXACT_EPOCH_SCAN_SELFTEST FAIL %s\n", message);
    std::exit(1);
}

static void cuda_require(cudaError_t status, const char *where)
{
    if (status == cudaSuccess) return;
    std::fprintf(stderr,
                 "CMF_EXACT_EPOCH_SCAN_SELFTEST FAIL cuda where=%s error=%s\n",
                 where, cudaGetErrorString(status));
    std::exit(1);
}

static uint64_t bits_of(double value)
{
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static bool transform_bits_equal(Transform a, Transform b)
{
    return bits_of(a.transmission) == bits_of(b.transmission) &&
           bits_of(a.emission) == bits_of(b.emission);
}

static bool host_finite_nonnegative(double value)
{
    return std::isfinite(value) && value >= 0.0;
}

static bool host_add_bound(double a, double b, int rounding, double *result)
{
    if (!result || !host_finite_nonnegative(a) ||
        !host_finite_nonnegative(b)) return false;
    double sum = a + b;
    if (!host_finite_nonnegative(sum)) return false;
    if (rounding != ROUND_NEAREST && a != 0.0 && b != 0.0) {
        double b_virtual = sum - a;
        double error = (a - (sum - b_virtual)) + (b - b_virtual);
        if (rounding == ROUND_UPPER && error > 0.0)
            sum = std::nextafter(sum, INFINITY);
        else if (rounding == ROUND_LOWER && error < 0.0)
            sum = std::nextafter(sum, 0.0);
    }
    if (!host_finite_nonnegative(sum)) return false;
    *result = sum;
    return true;
}

static bool host_multiply_bound(double a, double b, int rounding,
                                double *result)
{
    if (!result || !host_finite_nonnegative(a) ||
        !host_finite_nonnegative(b)) return false;
    if (a == 0.0 || b == 0.0) {
        *result = 0.0;
        return true;
    }
    double product = a * b;
    if (!host_finite_nonnegative(product)) return false;
    if (rounding == ROUND_UPPER)
        product = std::nextafter(product, INFINITY);
    else if (rounding == ROUND_LOWER && product != 0.0)
        product = std::nextafter(product, 0.0);
    if (!host_finite_nonnegative(product)) return false;
    *result = product;
    return true;
}

static bool host_reverse_compose(Transform a, Transform b, int rounding,
                                 Transform *result)
{
    if (!result) return false;
    double attenuated;
    return host_multiply_bound(b.transmission, a.transmission, rounding,
                               &result->transmission) &&
           host_multiply_bound(b.transmission, a.emission, rounding,
                               &attenuated) &&
           host_add_bound(b.emission, attenuated, rounding,
                          &result->emission);
}

static void append_host_trace(std::vector<PrimitiveTrace> *trace,
                              int mode_slot, int epoch, int chain, int node,
                              int primitive, double lhs, double rhs,
                              double result)
{
    PrimitiveTrace entry;
    entry.mode_slot = mode_slot;
    entry.epoch = epoch;
    entry.chain = chain;
    entry.node = node;
    entry.primitive = primitive;
    entry.lhs_bits = static_cast<unsigned long long>(bits_of(lhs));
    entry.rhs_bits = static_cast<unsigned long long>(bits_of(rhs));
    entry.result_bits = static_cast<unsigned long long>(bits_of(result));
    trace->push_back(entry);
}

static bool host_reverse_compose_traced(
    Transform a, Transform b, int rounding, Transform *result,
    std::vector<PrimitiveTrace> *trace, int mode_slot, int epoch,
    int chain, int node)
{
    if (!result || !trace) return false;
    double attenuated;
    if (!host_multiply_bound(b.transmission, a.transmission, rounding,
                             &result->transmission)) return false;
    append_host_trace(trace, mode_slot, epoch, chain, node, TRACE_MUL_T,
                      b.transmission, a.transmission,
                      result->transmission);
    if (!host_multiply_bound(b.transmission, a.emission, rounding,
                             &attenuated)) return false;
    append_host_trace(trace, mode_slot, epoch, chain, node, TRACE_MUL_E,
                      b.transmission, a.emission, attenuated);
    if (!host_add_bound(b.emission, attenuated, rounding,
                        &result->emission)) return false;
    append_host_trace(trace, mode_slot, epoch, chain, node, TRACE_ADD_E,
                      b.emission, attenuated, result->emission);
    return true;
}

__device__ __forceinline__ bool device_finite_nonnegative(double value)
{
    return isfinite(value) && value >= 0.0;
}

__device__ __forceinline__ double device_next_up(double value)
{
    if (value == 0.0) return __longlong_as_double(1LL);
    unsigned long long bits =
        static_cast<unsigned long long>(__double_as_longlong(value));
    return __longlong_as_double(static_cast<long long>(bits + 1ULL));
}

__device__ __forceinline__ double device_next_down(double value)
{
    if (value == 0.0) return 0.0;
    unsigned long long bits =
        static_cast<unsigned long long>(__double_as_longlong(value));
    return __longlong_as_double(static_cast<long long>(bits - 1ULL));
}

__device__ __forceinline__ bool device_add_bound(
    double a, double b, int rounding, double *result)
{
    if (!result || !device_finite_nonnegative(a) ||
        !device_finite_nonnegative(b)) return false;
    double sum = a + b;
    if (!device_finite_nonnegative(sum)) return false;
    if (rounding != ROUND_NEAREST && a != 0.0 && b != 0.0) {
        double b_virtual = sum - a;
        double error = (a - (sum - b_virtual)) + (b - b_virtual);
        if (rounding == ROUND_UPPER && error > 0.0)
            sum = device_next_up(sum);
        else if (rounding == ROUND_LOWER && error < 0.0)
            sum = device_next_down(sum);
    }
    if (!device_finite_nonnegative(sum)) return false;
    *result = sum;
    return true;
}

__device__ __forceinline__ bool device_multiply_bound(
    double a, double b, int rounding, double *result)
{
    if (!result || !device_finite_nonnegative(a) ||
        !device_finite_nonnegative(b)) return false;
    if (a == 0.0 || b == 0.0) {
        *result = 0.0;
        return true;
    }
    double product = a * b;
    if (!device_finite_nonnegative(product)) return false;
    if (rounding == ROUND_UPPER)
        product = device_next_up(product);
    else if (rounding == ROUND_LOWER && product != 0.0)
        product = device_next_down(product);
    if (!device_finite_nonnegative(product)) return false;
    *result = product;
    return true;
}

__device__ __forceinline__ bool device_reverse_compose(
    Transform a, Transform b, int rounding, Transform *result)
{
    if (!result) return false;
    double attenuated;
    return device_multiply_bound(b.transmission, a.transmission, rounding,
                                 &result->transmission) &&
           device_multiply_bound(b.transmission, a.emission, rounding,
                                 &attenuated) &&
           device_add_bound(b.emission, attenuated, rounding,
                            &result->emission);
}

__device__ __forceinline__ unsigned long long device_bits(double value)
{
    return static_cast<unsigned long long>(__double_as_longlong(value));
}

__device__ __forceinline__ bool append_device_trace(
    PrimitiveTrace *trace, int *trace_count, int trace_capacity,
    int mode_slot, int epoch, int chain, int node, int primitive,
    double lhs, double rhs, double result)
{
    if (!trace || !trace_count || trace_capacity <= 0) return false;
    int index = atomicAdd(trace_count, 1);
    if (index < 0 || index >= trace_capacity) return false;
    PrimitiveTrace entry;
    entry.mode_slot = mode_slot;
    entry.epoch = epoch;
    entry.chain = chain;
    entry.node = node;
    entry.primitive = primitive;
    entry.lhs_bits = device_bits(lhs);
    entry.rhs_bits = device_bits(rhs);
    entry.result_bits = device_bits(result);
    trace[index] = entry;
    return true;
}

__device__ __forceinline__ bool device_reverse_compose_traced(
    Transform a, Transform b, int rounding, Transform *result,
    PrimitiveTrace *trace, int *trace_count, int trace_capacity,
    int mode_slot, int epoch, int chain, int node)
{
    if (!result) return false;
    double attenuated;
    if (!device_multiply_bound(b.transmission, a.transmission, rounding,
                               &result->transmission) ||
        !append_device_trace(trace, trace_count, trace_capacity,
                             mode_slot, epoch, chain, node, TRACE_MUL_T,
                             b.transmission, a.transmission,
                             result->transmission)) return false;
    if (!device_multiply_bound(b.transmission, a.emission, rounding,
                               &attenuated) ||
        !append_device_trace(trace, trace_count, trace_capacity,
                             mode_slot, epoch, chain, node, TRACE_MUL_E,
                             b.transmission, a.emission,
                             attenuated)) return false;
    if (!device_add_bound(b.emission, attenuated, rounding,
                          &result->emission) ||
        !append_device_trace(trace, trace_count, trace_capacity,
                             mode_slot, epoch, chain, node, TRACE_ADD_E,
                             b.emission, attenuated,
                             result->emission)) return false;
    return true;
}

__device__ __forceinline__ int bounded_bin_index(int index, int bins)
{
    if (index < 0) return 0;
    if (index >= bins) return bins - 1;
    return index;
}

__global__ void nonassociation_kernel(const Transform *witnesses,
                                      Transform *results, int *failure)
{
    int mode_slot = static_cast<int>(threadIdx.x);
    if (mode_slot >= 3) return;
    int rounding = mode_slot - 1;
    const Transform *triple = witnesses + 3 * mode_slot;
    Transform first;
    Transform second;
    Transform left;
    Transform right;
    if (!device_reverse_compose(triple[0], triple[1], rounding, &first) ||
        !device_reverse_compose(first, triple[2], rounding, &left) ||
        !device_reverse_compose(triple[1], triple[2], rounding, &second) ||
        !device_reverse_compose(triple[0], second, rounding, &right)) {
        atomicCAS(failure, 0, 1);
        return;
    }
    results[2 * mode_slot] = left;
    results[2 * mode_slot + 1] = right;
}

__global__ void identity_window_kernel(int bins, Transform *outputs,
                                       LogicalMap *mapping, int *failure)
{
    int output_bin = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int mode_slot = static_cast<int>(blockIdx.y);
    if (mode_slot >= 3 || output_bin >= bins) return;
    int index = mode_slot * bins + output_bin;
    outputs[index] = Transform{1.0, 0.0};
    mapping[index] = LogicalMap{-1, 0, output_bin, output_bin,
                                -1, -1, -1};
    if (output_bin < 0) atomicCAS(failure, 0, 2);
}

__global__ void epoch_aggregate_kernel(const Transform *values, int bins,
                                       int window, Transform *outputs,
                                       LogicalMap *mapping, int *failure)
{
    extern __shared__ double shared_doubles[];
    Transform *front = reinterpret_cast<Transform *>(shared_doubles);
    Transform *new_back = front + window;
    Transform *boundary_value = new_back + window;

    int epoch = static_cast<int>(blockIdx.x);
    int mode_slot = static_cast<int>(blockIdx.y);
    if (mode_slot >= 3 || window <= 0 || bins <= 0) {
        if (threadIdx.x == 0) atomicCAS(failure, 0, 3);
        return;
    }
    int rounding = mode_slot - 1;
    int boundary_bin = bins - 1 - epoch * window;
    if (boundary_bin < 0) {
        if (threadIdx.x == 0) atomicCAS(failure, 0, 4);
        return;
    }
    int epoch_outputs = window;
    if (boundary_bin + 1 < epoch_outputs) epoch_outputs = boundary_bin + 1;

    bool okay = true;
    if (threadIdx.x == 0) {
        Transform aggregate =
            values[bounded_bin_index(boundary_bin + window, bins)];
        for (int index = boundary_bin + window - 1;
             index > boundary_bin; --index) {
            Transform next;
            okay = okay && device_reverse_compose(
                aggregate, values[bounded_bin_index(index, bins)],
                rounding, &next);
            aggregate = next;
        }
        *boundary_value = aggregate;
    } else if (threadIdx.x == 1) {
        Transform aggregate =
            values[bounded_bin_index(boundary_bin + 1, bins)];
        front[0] = aggregate;
        for (int index = 1; index < window; ++index) {
            Transform next;
            okay = okay && device_reverse_compose(
                values[bounded_bin_index(boundary_bin + 1 + index, bins)],
                aggregate, rounding, &next);
            aggregate = next;
            front[index] = aggregate;
        }
    } else if (threadIdx.x == 2 && epoch_outputs > 1) {
        Transform aggregate = values[bounded_bin_index(boundary_bin, bins)];
        new_back[0] = aggregate;
        for (int index = 1; index < epoch_outputs - 1; ++index) {
            Transform next;
            okay = okay && device_reverse_compose(
                aggregate,
                values[bounded_bin_index(boundary_bin - index, bins)],
                rounding, &next);
            aggregate = next;
            new_back[index] = aggregate;
        }
    }
    if (!okay) atomicCAS(failure, 0, 5);
    __syncthreads();
    if (*failure != 0) return;

    for (int offset = static_cast<int>(threadIdx.x);
         offset < epoch_outputs; offset += static_cast<int>(blockDim.x)) {
        int output_bin = boundary_bin - offset;
        int output_index = mode_slot * bins + output_bin;
        Transform aggregate;
        if (offset == 0) {
            aggregate = *boundary_value;
        } else if (!device_reverse_compose(
                       front[window - offset - 1], new_back[offset - 1],
                       rounding, &aggregate)) {
            atomicCAS(failure, 0, 6);
            continue;
        }
        outputs[output_index] = aggregate;
        mapping[output_index] = LogicalMap{
            epoch, offset, output_bin, boundary_bin,
            offset == 0 ? window - 1 : -1,
            offset == 0 ? -1 : window - offset - 1,
            offset == 0 ? -1 : offset - 1};
    }
}

__global__ void epoch_trace_kernel(
    const Transform *values, int bins, int window,
    PrimitiveTrace *trace, int *trace_count, int trace_capacity,
    int *failure)
{
    extern __shared__ double shared_doubles[];
    Transform *front = reinterpret_cast<Transform *>(shared_doubles);
    Transform *new_back = front + window;
    Transform *boundary_value = new_back + window;

    int epoch = static_cast<int>(blockIdx.x);
    int mode_slot = static_cast<int>(blockIdx.y);
    int rounding = mode_slot - 1;
    int boundary_bin = bins - 1 - epoch * window;
    if (mode_slot >= 3 || window <= 0 || bins <= 0 || boundary_bin < 0) {
        if (threadIdx.x == 0) atomicCAS(failure, 0, 20);
        return;
    }
    int epoch_outputs = window;
    if (boundary_bin + 1 < epoch_outputs) epoch_outputs = boundary_bin + 1;
    bool has_next_epoch = boundary_bin - window >= 0;

    bool okay = true;
    if (threadIdx.x == 0) {
        Transform aggregate =
            values[bounded_bin_index(boundary_bin + window, bins)];
        for (int node = 1; node < window; ++node) {
            Transform next;
            okay = okay && device_reverse_compose_traced(
                aggregate,
                values[bounded_bin_index(boundary_bin + window - node,
                                         bins)],
                rounding, &next, trace, trace_count, trace_capacity,
                mode_slot, epoch, TRACE_Q, node);
            aggregate = next;
        }
        *boundary_value = aggregate;
    } else if (threadIdx.x == 1 && epoch_outputs > 1) {
        Transform aggregate =
            values[bounded_bin_index(boundary_bin + 1, bins)];
        front[0] = aggregate;
        for (int node = 1; node < window; ++node) {
            Transform next;
            okay = okay && device_reverse_compose_traced(
                values[bounded_bin_index(boundary_bin + 1 + node, bins)],
                aggregate, rounding, &next,
                trace, trace_count, trace_capacity,
                mode_slot, epoch, TRACE_F, node);
            aggregate = next;
            front[node] = aggregate;
        }
    } else if (threadIdx.x == 2) {
        int p_values = epoch_outputs > 1 ? epoch_outputs - 1 : 0;
        if (has_next_epoch) p_values = window;
        if (p_values > 0) {
            Transform aggregate =
                values[bounded_bin_index(boundary_bin, bins)];
            new_back[0] = aggregate;
            for (int node = 1; node < p_values; ++node) {
                Transform next;
                okay = okay && device_reverse_compose_traced(
                    aggregate,
                    values[bounded_bin_index(boundary_bin - node, bins)],
                    rounding, &next,
                    trace, trace_count, trace_capacity,
                    mode_slot, epoch, TRACE_P, node);
                aggregate = next;
                new_back[node] = aggregate;
            }
        }
    }
    if (!okay) atomicCAS(failure, 0, 21);
    __syncthreads();
    if (*failure != 0) return;

    for (int offset = static_cast<int>(threadIdx.x) + 1;
         offset < epoch_outputs; offset += static_cast<int>(blockDim.x)) {
        Transform aggregate;
        if (!device_reverse_compose_traced(
                front[window - offset - 1], new_back[offset - 1],
                rounding, &aggregate,
                trace, trace_count, trace_capacity,
                mode_slot, epoch, TRACE_G, offset))
            atomicCAS(failure, 0, 22);
    }
}

__device__ __forceinline__ bool device_two_product_sum(
    double a, double x, double b, double y, int rounding, double *result)
{
    if (!result) return false;
    if (rounding == ROUND_NEAREST) {
        double value = a * x + b * y;
        if (!device_finite_nonnegative(value)) return false;
        *result = value;
        return true;
    }
    double ax;
    double by;
    return device_multiply_bound(a, x, rounding, &ax) &&
           device_multiply_bound(b, y, rounding, &by) &&
           device_add_bound(ax, by, rounding, result);
}

__device__ __forceinline__ bool device_window_push(
    WindowNode *back, int capacity, int *back_size, Transform value,
    int rounding)
{
    if (!back || !back_size || *back_size < 0 || *back_size >= capacity ||
        !device_finite_nonnegative(value.transmission) ||
        !device_finite_nonnegative(value.emission)) return false;
    WindowNode &node = back[*back_size];
    node.value = value;
    if (*back_size == 0)
        node.aggregate = value;
    else if (!device_reverse_compose(back[*back_size - 1].aggregate,
                                     value, rounding, &node.aggregate))
        return false;
    ++*back_size;
    return true;
}

__device__ __forceinline__ bool device_window_transfer(
    WindowNode *front, WindowNode *back, int capacity,
    int *front_size, int *back_size, int rounding)
{
    if (!front || !back || !front_size || !back_size ||
        *front_size != 0) return false;
    while (*back_size != 0) {
        Transform value = back[--*back_size].value;
        if (*front_size >= capacity) return false;
        WindowNode &node = front[*front_size];
        node.value = value;
        if (*front_size == 0)
            node.aggregate = value;
        else if (!device_reverse_compose(
                     value, front[*front_size - 1].aggregate,
                     rounding, &node.aggregate))
            return false;
        ++*front_size;
    }
    return true;
}

__device__ __forceinline__ bool device_window_aggregate(
    const WindowNode *front, const WindowNode *back,
    int front_size, int back_size, int rounding, Transform *aggregate)
{
    if (!aggregate || front_size < 0 || back_size < 0) return false;
    if (front_size == 0 && back_size == 0) {
        *aggregate = Transform{1.0, 0.0};
        return true;
    }
    if (front_size == 0) {
        *aggregate = back[back_size - 1].aggregate;
        return true;
    }
    if (back_size == 0) {
        *aggregate = front[front_size - 1].aggregate;
        return true;
    }
    return device_reverse_compose(front[front_size - 1].aggregate,
                                  back[back_size - 1].aggregate,
                                  rounding, aggregate);
}

__device__ bool device_serial_segment_values(
    int bins, double beta, const double *dt1, const double *t1,
    const double *source, const double *source_cell,
    const double *upstream, bool upstream_zero, double *output,
    WindowNode *front, WindowNode *back, int workspace_capacity,
    int rounding)
{
    int q = static_cast<int>(floor(beta));
    double phi = beta - static_cast<double>(q);
    if (q < 0 || !(phi >= 0.0) || !(phi < 1.0) || !isfinite(phi))
        return false;
    if (beta <= 0.5) {
        for (int bin = 0; bin < bins; ++bin) {
            double intensity = 0.0;
            if (!upstream_zero) {
                int i0 = bounded_bin_index(bin + q, bins);
                int i1 = bounded_bin_index(bin + q + 1, bins);
                if (!device_two_product_sum(
                        1.0 - phi, upstream[i0], phi, upstream[i1],
                        rounding, &intensity)) return false;
            }
            double transmission = exp(-dt1[bin] * beta);
            if (!device_two_product_sum(
                    transmission, intensity, 1.0 - transmission,
                    source[bin], rounding, &output[bin])) return false;
        }
        return true;
    }

    int qtop;
    double psi;
    if (phi < 0.5) {
        qtop = q;
        psi = phi + 0.5;
    } else {
        qtop = q + 1;
        psi = phi - 0.5;
    }
    int window = qtop >= 2 ? qtop - 1 : 0;
    if (window > workspace_capacity ||
        (window != 0 && (!front || !back))) return false;
    int front_size = 0;
    int back_size = 0;
    int highest = bins - 1;
    for (int index = highest + window; index > highest; --index) {
        int source_index = bounded_bin_index(index, bins);
        if (!device_window_push(
                back, window, &back_size,
                Transform{t1[source_index], source_cell[source_index]},
                rounding)) return false;
    }
    for (int bin = highest; bin >= 0; --bin) {
        double intensity = 0.0;
        if (!upstream_zero) {
            int i0 = bounded_bin_index(bin + q, bins);
            int i1 = bounded_bin_index(bin + q + 1, bins);
            if (!device_two_product_sum(
                    1.0 - phi, upstream[i0], phi, upstream[i1],
                    rounding, &intensity)) return false;
        }
        int top = bounded_bin_index(bin + qtop, bins);
        double transmission = exp(-psi * dt1[top]);
        if (!device_two_product_sum(
                transmission, intensity, 1.0 - transmission,
                source[top], rounding, &intensity)) return false;
        Transform aggregate;
        if (!device_window_aggregate(front, back, front_size, back_size,
                                     rounding, &aggregate) ||
            !device_two_product_sum(
                aggregate.transmission, intensity, 1.0,
                aggregate.emission, rounding, &intensity)) return false;
        double half = sqrt(t1[bin]);
        if (!device_two_product_sum(
                half, intensity, 1.0 - half, source[bin],
                rounding, &output[bin])) return false;
        if (bin == 0 || window == 0) continue;
        if (front_size == 0 &&
            !device_window_transfer(front, back, window,
                                    &front_size, &back_size, rounding))
            return false;
        if (front_size == 0) return false;
        --front_size;
        if (!device_window_push(
                back, window, &back_size,
                Transform{t1[bin], source_cell[bin]}, rounding))
            return false;
    }
    return true;
}

__global__ void serial_segment_kernel(
    int bins, double beta, const double *dt1, const double *t1,
    const double *source, const double *source_cell_by_mode,
    const double *upstream, int upstream_mode_stride, int upstream_zero,
    double *output,
    WindowNode *workspace, int workspace_capacity, int *failure)
{
    int mode_slot = static_cast<int>(threadIdx.x);
    if (blockIdx.x != 0 || mode_slot >= 3) return;
    WindowNode *front = workspace +
        static_cast<size_t>(mode_slot) * 2U *
            static_cast<size_t>(workspace_capacity);
    WindowNode *back = front + workspace_capacity;
    const double *mode_upstream = upstream_mode_stride == 0
        ? upstream
        : upstream + static_cast<size_t>(mode_slot) *
                         static_cast<size_t>(upstream_mode_stride);
    if (!device_serial_segment_values(
            bins, beta, dt1, t1, source,
            source_cell_by_mode + static_cast<size_t>(mode_slot) * bins,
            mode_upstream, upstream_zero != 0,
            output + static_cast<size_t>(mode_slot) * bins,
            front, back, workspace_capacity, mode_slot - 1))
        atomicCAS(failure, 0, 30 + mode_slot);
}

__global__ void zero_window_segment_kernel(
    int bins, double beta, const double *dt1, const double *t1,
    const double *source, const double *upstream,
    int upstream_mode_stride, int upstream_zero, double *output,
    int *failure)
{
    int bin = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int mode_slot = static_cast<int>(blockIdx.y);
    if (bin >= bins || mode_slot >= 3) return;
    int rounding = mode_slot - 1;
    const double *mode_upstream = upstream_mode_stride == 0
        ? upstream
        : upstream + static_cast<size_t>(mode_slot) *
                         static_cast<size_t>(upstream_mode_stride);
    int q = static_cast<int>(floor(beta));
    double phi = beta - static_cast<double>(q);
    double intensity = 0.0;
    if (!upstream_zero) {
        int i0 = bounded_bin_index(bin + q, bins);
        int i1 = bounded_bin_index(bin + q + 1, bins);
        if (!device_two_product_sum(
                1.0 - phi, mode_upstream[i0], phi, mode_upstream[i1],
                rounding, &intensity)) {
            atomicCAS(failure, 0, 40);
            return;
        }
    }
    double value;
    if (beta <= 0.5) {
        double transmission = exp(-dt1[bin] * beta);
        if (!device_two_product_sum(
                transmission, intensity, 1.0 - transmission, source[bin],
                rounding, &value)) {
            atomicCAS(failure, 0, 41);
            return;
        }
    } else {
        int qtop;
        double psi;
        if (phi < 0.5) {
            qtop = q;
            psi = phi + 0.5;
        } else {
            qtop = q + 1;
            psi = phi - 0.5;
        }
        int top = bounded_bin_index(bin + qtop, bins);
        double transmission = exp(-psi * dt1[top]);
        if (!device_two_product_sum(
                transmission, intensity, 1.0 - transmission, source[top],
                rounding, &intensity)) {
            atomicCAS(failure, 0, 42);
            return;
        }
        if (!device_two_product_sum(
                1.0, intensity, 1.0, 0.0,
                rounding, &intensity)) {
            atomicCAS(failure, 0, 44);
            return;
        }
        double half = sqrt(t1[bin]);
        if (!device_two_product_sum(
                half, intensity, 1.0 - half, source[bin],
                rounding, &value)) {
            atomicCAS(failure, 0, 43);
            return;
        }
    }
    output[static_cast<size_t>(mode_slot) * bins + bin] = value;
}

__global__ void epoch_segment_kernel(
    int bins, int window, int epoch_begin, double beta,
    const double *dt1, const double *t1, const double *source,
    const double *source_cell_by_mode, const double *upstream,
    int upstream_mode_stride, int upstream_zero,
    double *output, int *failure)
{
    extern __shared__ double shared_doubles[];
    Transform *front = reinterpret_cast<Transform *>(shared_doubles);
    Transform *new_back = front + window;
    Transform *boundary_value = new_back + window;
    int epoch = epoch_begin + static_cast<int>(blockIdx.x);
    int mode_slot = static_cast<int>(blockIdx.y);
    int rounding = mode_slot - 1;
    int boundary_bin = bins - 1 - epoch * window;
    if (mode_slot >= 3 || window <= 0 || boundary_bin < 0) {
        if (threadIdx.x == 0) atomicCAS(failure, 0, 50);
        return;
    }
    int epoch_outputs = window;
    if (boundary_bin + 1 < epoch_outputs) epoch_outputs = boundary_bin + 1;
    const double *source_cell = source_cell_by_mode +
        static_cast<size_t>(mode_slot) * bins;
    const double *mode_upstream = upstream_mode_stride == 0
        ? upstream
        : upstream + static_cast<size_t>(mode_slot) *
                         static_cast<size_t>(upstream_mode_stride);
    bool okay = true;
    if (threadIdx.x == 0) {
        int source_index = bounded_bin_index(boundary_bin + window, bins);
        Transform aggregate{t1[source_index], source_cell[source_index]};
        for (int node = 1; node < window; ++node) {
            source_index =
                bounded_bin_index(boundary_bin + window - node, bins);
            Transform next;
            okay = okay && device_reverse_compose(
                aggregate,
                Transform{t1[source_index], source_cell[source_index]},
                rounding, &next);
            aggregate = next;
        }
        *boundary_value = aggregate;
    } else if (threadIdx.x == 1 && epoch_outputs > 1) {
        int source_index = bounded_bin_index(boundary_bin + 1, bins);
        Transform aggregate{t1[source_index], source_cell[source_index]};
        front[0] = aggregate;
        for (int node = 1; node < window; ++node) {
            source_index =
                bounded_bin_index(boundary_bin + 1 + node, bins);
            Transform next;
            okay = okay && device_reverse_compose(
                Transform{t1[source_index], source_cell[source_index]},
                aggregate, rounding, &next);
            aggregate = next;
            front[node] = aggregate;
        }
    } else if (threadIdx.x == 2 && epoch_outputs > 1) {
        int source_index = bounded_bin_index(boundary_bin, bins);
        Transform aggregate{t1[source_index], source_cell[source_index]};
        new_back[0] = aggregate;
        for (int node = 1; node < epoch_outputs - 1; ++node) {
            source_index = bounded_bin_index(boundary_bin - node, bins);
            Transform next;
            okay = okay && device_reverse_compose(
                aggregate,
                Transform{t1[source_index], source_cell[source_index]},
                rounding, &next);
            aggregate = next;
            new_back[node] = aggregate;
        }
    }
    if (!okay) atomicCAS(failure, 0, 51);
    __syncthreads();
    if (*failure != 0) return;

    int q = static_cast<int>(floor(beta));
    double phi = beta - static_cast<double>(q);
    int qtop;
    double psi;
    if (phi < 0.5) {
        qtop = q;
        psi = phi + 0.5;
    } else {
        qtop = q + 1;
        psi = phi - 0.5;
    }
    for (int offset = static_cast<int>(threadIdx.x);
         offset < epoch_outputs; offset += static_cast<int>(blockDim.x)) {
        int bin = boundary_bin - offset;
        Transform aggregate;
        if (offset == 0)
            aggregate = *boundary_value;
        else if (!device_reverse_compose(
                     front[window - offset - 1], new_back[offset - 1],
                     rounding, &aggregate)) {
            atomicCAS(failure, 0, 52);
            continue;
        }
        double intensity = 0.0;
        if (!upstream_zero) {
            int i0 = bounded_bin_index(bin + q, bins);
            int i1 = bounded_bin_index(bin + q + 1, bins);
            if (!device_two_product_sum(
                    1.0 - phi, mode_upstream[i0], phi, mode_upstream[i1],
                    rounding, &intensity)) {
                atomicCAS(failure, 0, 53);
                continue;
            }
        }
        int top = bounded_bin_index(bin + qtop, bins);
        double transmission = exp(-psi * dt1[top]);
        if (!device_two_product_sum(
                transmission, intensity, 1.0 - transmission, source[top],
                rounding, &intensity) ||
            !device_two_product_sum(
                aggregate.transmission, intensity, 1.0,
                aggregate.emission, rounding, &intensity)) {
            atomicCAS(failure, 0, 54);
            continue;
        }
        double half = sqrt(t1[bin]);
        double value;
        if (!device_two_product_sum(
                half, intensity, 1.0 - half, source[bin],
                rounding, &value)) {
            atomicCAS(failure, 0, 55);
            continue;
        }
        output[static_cast<size_t>(mode_slot) * bins + bin] = value;
    }
}

__device__ double device_direct_segment_value(
    int bin, int bins, double beta, const double *dt1,
    const double *source, const double *upstream, bool upstream_zero)
{
    int q = static_cast<int>(floor(beta));
    double phi = beta - static_cast<double>(q);
    double intensity = 0.0;
    if (!upstream_zero) {
        int i0 = bounded_bin_index(bin + q, bins);
        int i1 = bounded_bin_index(bin + q + 1, bins);
        intensity = (1.0 - phi) * upstream[i0] + phi * upstream[i1];
    }
    if (beta <= 0.5) {
        double transmission = exp(-dt1[bin] * beta);
        return intensity * transmission +
               (1.0 - transmission) * source[bin];
    }
    double x = static_cast<double>(bin) + beta;
    int index = static_cast<int>(floor(x + 0.5));
    double cursor = x;
    for (;;) {
        double cell_edge = static_cast<double>(index) - 0.5;
        double lower = cell_edge > static_cast<double>(bin)
                     ? cell_edge : static_cast<double>(bin);
        double length = cursor - lower;
        if (length > 0.0) {
            int source_index = bounded_bin_index(index, bins);
            double transmission = exp(-dt1[source_index] * length);
            intensity = intensity * transmission +
                        (1.0 - transmission) * source[source_index];
        }
        if (lower <= static_cast<double>(bin) + 1.0e-12) break;
        cursor = lower;
        --index;
    }
    return intensity;
}

__global__ void direct_segment_kernel(
    int bins, double beta, const double *dt1, const double *source,
    const double *upstream, int upstream_zero, double *output,
    int *failure)
{
    int bin = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (bin >= bins) return;
    double value = device_direct_segment_value(
        bin, bins, beta, dt1, source, upstream, upstream_zero != 0);
    if (!device_finite_nonnegative(value)) {
        atomicCAS(failure, 0, 60);
        return;
    }
    output[bin] = value;
}

__device__ __forceinline__ void record_g6_failure(
    G6FailureDetail *detail, int code, int mode, int epoch,
    int chain, int node, int source_index)
{
    if (!detail) return;
    if (atomicCAS(&detail->code, G6_OK, code) == G6_OK) {
        detail->mode = mode;
        detail->epoch = epoch;
        detail->chain = chain;
        detail->node = node;
        detail->source_index = source_index;
    }
}

__global__ void g6_transaction_probe_kernel(
    const Transform *values, int bins, int window, int rounding,
    Transform *staging, G6FailureDetail *detail)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (rounding < ROUND_LOWER || rounding > ROUND_UPPER) {
        record_g6_failure(detail, G6_INVALID_MODE, rounding,
                          0, TRACE_Q, 0, -1);
        return;
    }
    int boundary_bin = bins - 1;
    int source_index = bounded_bin_index(boundary_bin + window, bins);
    Transform aggregate = values[source_index];
    if (!device_finite_nonnegative(aggregate.transmission) ||
        !device_finite_nonnegative(aggregate.emission)) {
        record_g6_failure(detail, G6_NONFINITE_TRANSFORM, rounding,
                          0, TRACE_Q, 0, source_index);
        return;
    }
    for (int node = 1; node < window; ++node) {
        source_index =
            bounded_bin_index(boundary_bin + window - node, bins);
        Transform value = values[source_index];
        if (!device_finite_nonnegative(value.transmission) ||
            !device_finite_nonnegative(value.emission)) {
            record_g6_failure(detail, G6_NONFINITE_TRANSFORM, rounding,
                              0, TRACE_Q, node, source_index);
            return;
        }
        Transform next;
        if (!device_reverse_compose(aggregate, value, rounding, &next)) {
            record_g6_failure(detail, G6_COMPOSE_FAILURE, rounding,
                              0, TRACE_Q, node, source_index);
            return;
        }
        aggregate = next;
    }
    staging[0] = aggregate;
}

static int host_bounded_bin_index(int index, int bins)
{
    if (index < 0) return 0;
    if (index >= bins) return bins - 1;
    return index;
}

static bool push_back(std::vector<WindowNode> *back, Transform value,
                      int rounding)
{
    WindowNode node;
    node.value = value;
    if (back->empty())
        node.aggregate = value;
    else if (!host_reverse_compose(back->back().aggregate, value, rounding,
                                   &node.aggregate))
        return false;
    back->push_back(node);
    return true;
}

static bool transfer(std::vector<WindowNode> *front,
                     std::vector<WindowNode> *back, int rounding)
{
    if (!front->empty()) return false;
    while (!back->empty()) {
        WindowNode node;
        node.value = back->back().value;
        back->pop_back();
        if (front->empty())
            node.aggregate = node.value;
        else if (!host_reverse_compose(node.value, front->back().aggregate,
                                       rounding, &node.aggregate))
            return false;
        front->push_back(node);
    }
    return true;
}

static bool aggregate_window(const std::vector<WindowNode> &front,
                             const std::vector<WindowNode> &back,
                             int rounding, Transform *aggregate)
{
    if (!aggregate) return false;
    if (front.empty() && back.empty()) {
        *aggregate = Transform{1.0, 0.0};
        return true;
    }
    if (front.empty()) {
        *aggregate = back.back().aggregate;
        return true;
    }
    if (back.empty()) {
        *aggregate = front.back().aggregate;
        return true;
    }
    return host_reverse_compose(front.back().aggregate,
                                back.back().aggregate, rounding, aggregate);
}

static bool serial_aggregates(const std::vector<Transform> &values,
                              int window, int rounding,
                              std::vector<Transform> *outputs)
{
    int bins = static_cast<int>(values.size());
    std::vector<WindowNode> front;
    std::vector<WindowNode> back;
    front.reserve(static_cast<size_t>(window));
    back.reserve(static_cast<size_t>(window));
    outputs->assign(static_cast<size_t>(bins), Transform{NAN, NAN});
    for (int index = bins - 1 + window; index > bins - 1; --index) {
        if (!push_back(&back, values[host_bounded_bin_index(index, bins)],
                       rounding)) return false;
    }
    for (int output_bin = bins - 1; output_bin >= 0; --output_bin) {
        if (!aggregate_window(front, back, rounding,
                              &(*outputs)[output_bin])) return false;
        if (output_bin == 0 || window == 0) continue;
        if (front.empty() && !transfer(&front, &back, rounding)) return false;
        if (front.empty()) return false;
        front.pop_back();
        if (!push_back(&back, values[output_bin], rounding)) return false;
    }
    return true;
}

static bool serial_primitive_trace(const std::vector<Transform> &values,
                                   int window, int mode_slot,
                                   std::vector<PrimitiveTrace> *trace)
{
    if (!trace || values.empty() || window <= 0) return false;
    int bins = static_cast<int>(values.size());
    int rounding = kModes[mode_slot];
    std::vector<WindowNode> front;
    std::vector<WindowNode> back;
    front.reserve(static_cast<size_t>(window));
    back.reserve(static_cast<size_t>(window));

    for (int index = bins - 1 + window; index > bins - 1; --index) {
        WindowNode node;
        node.value = values[host_bounded_bin_index(index, bins)];
        if (back.empty())
            node.aggregate = node.value;
        else if (!host_reverse_compose_traced(
                     back.back().aggregate, node.value, rounding,
                     &node.aggregate, trace, mode_slot, 0, TRACE_Q,
                     static_cast<int>(back.size())))
            return false;
        back.push_back(node);
    }

    int highest = bins - 1;
    for (int output_bin = highest; output_bin >= 0; --output_bin) {
        int epoch = (highest - output_bin) / window;
        int offset = highest - epoch * window - output_bin;
        Transform aggregate;
        if (front.empty() && back.empty()) {
            return false;
        } else if (front.empty()) {
            aggregate = back.back().aggregate;
        } else if (back.empty()) {
            aggregate = front.back().aggregate;
        } else if (!host_reverse_compose_traced(
                       front.back().aggregate, back.back().aggregate,
                       rounding, &aggregate, trace, mode_slot, epoch,
                       TRACE_G, offset)) {
            return false;
        }
        if (!host_finite_nonnegative(aggregate.transmission) ||
            !host_finite_nonnegative(aggregate.emission)) return false;
        if (output_bin == 0) continue;

        if (front.empty()) {
            while (!back.empty()) {
                WindowNode node;
                node.value = back.back().value;
                back.pop_back();
                if (front.empty())
                    node.aggregate = node.value;
                else if (!host_reverse_compose_traced(
                             node.value, front.back().aggregate, rounding,
                             &node.aggregate, trace, mode_slot, epoch,
                             TRACE_F, static_cast<int>(front.size())))
                    return false;
                front.push_back(node);
            }
        }
        if (front.empty()) return false;
        front.pop_back();

        WindowNode incoming;
        incoming.value = values[output_bin];
        if (back.empty())
            incoming.aggregate = incoming.value;
        else if (!host_reverse_compose_traced(
                     back.back().aggregate, incoming.value, rounding,
                     &incoming.aggregate, trace, mode_slot, epoch,
                     TRACE_P, static_cast<int>(back.size())))
            return false;
        back.push_back(incoming);
    }

    std::vector<PrimitiveTrace> actual = *trace;
    int epochs = (bins + window - 1) / window;
    for (int epoch = 1; epoch < epochs; ++epoch) {
        int aliased = 0;
        for (const PrimitiveTrace &entry : actual) {
            if (entry.mode_slot == mode_slot &&
                entry.epoch == epoch - 1 &&
                entry.chain == TRACE_P &&
                entry.node >= 1 && entry.node < window) {
                PrimitiveTrace alias = entry;
                alias.epoch = epoch;
                alias.chain = TRACE_Q;
                trace->push_back(alias);
                ++aliased;
            }
        }
        if (aliased != 3 * (window - 1)) return false;
    }
    return true;
}

static bool trace_less(const PrimitiveTrace &a, const PrimitiveTrace &b)
{
    if (a.mode_slot != b.mode_slot) return a.mode_slot < b.mode_slot;
    if (a.epoch != b.epoch) return a.epoch < b.epoch;
    if (a.chain != b.chain) return a.chain < b.chain;
    if (a.node != b.node) return a.node < b.node;
    return a.primitive < b.primitive;
}

static bool trace_equal(const PrimitiveTrace &a, const PrimitiveTrace &b)
{
    return a.mode_slot == b.mode_slot && a.epoch == b.epoch &&
           a.chain == b.chain && a.node == b.node &&
           a.primitive == b.primitive &&
           a.lhs_bits == b.lhs_bits && a.rhs_bits == b.rhs_bits &&
           a.result_bits == b.result_bits;
}

static LogicalMap expected_mapping(int bins, int window, int output_bin)
{
    if (window == 0)
        return LogicalMap{-1, 0, output_bin, output_bin, -1, -1, -1};
    int epoch = (bins - 1 - output_bin) / window;
    int boundary_bin = bins - 1 - epoch * window;
    int offset = boundary_bin - output_bin;
    return LogicalMap{epoch, offset, output_bin, boundary_bin,
                      offset == 0 ? window - 1 : -1,
                      offset == 0 ? -1 : window - offset - 1,
                      offset == 0 ? -1 : offset - 1};
}

static bool mapping_equal(const LogicalMap &a, const LogicalMap &b)
{
    return a.epoch == b.epoch && a.offset == b.offset &&
           a.output_bin == b.output_bin &&
           a.boundary_bin == b.boundary_bin &&
           a.q_fold_index == b.q_fold_index &&
           a.front_fold_index == b.front_fold_index &&
           a.back_fold_index == b.back_fold_index;
}

static uint64_t xorshift64(uint64_t *state)
{
    uint64_t value = *state;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    return value;
}

static double random_unit(uint64_t *state)
{
    uint64_t payload = xorshift64(state) >> 11;
    return static_cast<double>(payload) * 0x1.0p-53;
}

static void make_values(int bins, int trial, uint64_t *state,
                        std::vector<Transform> *values)
{
    values->resize(static_cast<size_t>(bins));
    for (int index = 0; index < bins; ++index) {
        Transform value;
        if (trial == 0) {
            value = Transform{1.0, 0.0};
        } else if (trial == 1) {
            value = Transform{std::nextafter(1.0, 0.0),
                              std::nextafter(0.0, 1.0)};
        } else {
            int exponent = -1020 +
                static_cast<int>(xorshift64(state) % 1041ULL);
            value = Transform{random_unit(state),
                              std::ldexp(random_unit(state), exponent)};
        }
        if (!host_finite_nonnegative(value.transmission) ||
            !host_finite_nonnegative(value.emission))
            fail("generator-produced-invalid-transform");
        (*values)[index] = value;
    }
}

static ProofCounts verify_logical_traces(bool sanitizer_smoke)
{
    const int full_bins[] = {2, 3, 17, 31, 32, 33, 63, 64, 65, 96};
    const int smoke_bins[] = {2, 32, 33, 96};
    const int *bin_counts = sanitizer_smoke ? smoke_bins : full_bins;
    int bin_count_size = sanitizer_smoke
        ? static_cast<int>(sizeof(smoke_bins) / sizeof(smoke_bins[0]))
        : static_cast<int>(sizeof(full_bins) / sizeof(full_bins[0]));
    int trials = sanitizer_smoke ? 2 : 4;
    constexpr int trace_capacity = 16384;
    constexpr int max_bins = 96;

    Transform *device_values = nullptr;
    PrimitiveTrace *device_trace = nullptr;
    int *device_trace_count = nullptr;
    int *device_failure = nullptr;
    cuda_require(cudaMalloc(&device_values, max_bins * sizeof(Transform)),
                 "g2-values-malloc");
    cuda_require(cudaMalloc(&device_trace,
                            trace_capacity * sizeof(PrimitiveTrace)),
                 "g2-trace-malloc");
    cuda_require(cudaMalloc(&device_trace_count, sizeof(int)),
                 "g2-trace-count-malloc");
    cuda_require(cudaMalloc(&device_failure, sizeof(int)),
                 "g2-failure-malloc");

    uint64_t random_state = 0x67325f7472616365ULL;
    std::vector<Transform> values;
    std::vector<PrimitiveTrace> expected;
    std::vector<PrimitiveTrace> actual(
        static_cast<size_t>(trace_capacity));
    ProofCounts counts{0, 0, 0, 0};
    for (int bin_slot = 0; bin_slot < bin_count_size; ++bin_slot) {
        int bins = bin_counts[bin_slot];
        std::set<int> windows = {
            1, 2, 3, bins - 1, bins, bins + 1, 2 * bins + 3
        };
        windows.erase(0);
        for (int window : windows) {
            if (window <= 0) continue;
            for (int trial = 0; trial < trials; ++trial) {
                make_values(bins, trial, &random_state, &values);
                expected.clear();
                for (int mode_slot = 0; mode_slot < 3; ++mode_slot) {
                    if (!serial_primitive_trace(values, window, mode_slot,
                                                &expected))
                        fail("g2-cpu-serial-trace-construction");
                }
                std::sort(expected.begin(), expected.end(), trace_less);
                if (expected.size() >
                    static_cast<size_t>(trace_capacity))
                    fail("g2-host-trace-capacity-model");

                cuda_require(cudaMemcpy(
                                 device_values, values.data(),
                                 static_cast<size_t>(bins) * sizeof(Transform),
                                 cudaMemcpyHostToDevice),
                             "g2-values-copy");
                cuda_require(cudaMemset(device_trace_count, 0, sizeof(int)),
                             "g2-trace-count-clear");
                cuda_require(cudaMemset(device_failure, 0, sizeof(int)),
                             "g2-failure-clear");
                int epochs = (bins + window - 1) / window;
                dim3 grid(epochs, 3, 1);
                size_t shared_bytes =
                    static_cast<size_t>(2 * window + 1) *
                    sizeof(Transform);
                epoch_trace_kernel<<<grid, 128, shared_bytes>>>(
                    device_values, bins, window, device_trace,
                    device_trace_count, trace_capacity, device_failure);
                cuda_require(cudaGetLastError(), "g2-trace-launch");
                int actual_count = 0;
                int failure = 0;
                cuda_require(cudaMemcpy(
                                 &actual_count, device_trace_count,
                                 sizeof(int), cudaMemcpyDeviceToHost),
                             "g2-trace-count-copy");
                cuda_require(cudaMemcpy(
                                 &failure, device_failure,
                                 sizeof(int), cudaMemcpyDeviceToHost),
                             "g2-failure-copy");
                if (failure != 0 || actual_count < 0 ||
                    actual_count > trace_capacity)
                    fail("g2-device-trace-failure");
                actual.resize(static_cast<size_t>(actual_count));
                cuda_require(cudaMemcpy(
                                 actual.data(), device_trace,
                                 actual.size() * sizeof(PrimitiveTrace),
                                 cudaMemcpyDeviceToHost),
                             "g2-trace-copy");
                std::sort(actual.begin(), actual.end(), trace_less);
                if (actual.size() != expected.size())
                    fail("g2-trace-record-count-mismatch");
                for (size_t index = 0; index < expected.size(); ++index) {
                    if (!trace_equal(expected[index], actual[index]))
                        fail("g2-primitive-trace-bit-mismatch");
                }
                counts.records += actual_count;
                counts.mode_cases += 3;
                ++counts.cases;
                actual.resize(static_cast<size_t>(trace_capacity));
            }
        }
    }

    cuda_require(cudaFree(device_failure), "g2-failure-free");
    cuda_require(cudaFree(device_trace_count), "g2-trace-count-free");
    cuda_require(cudaFree(device_trace), "g2-trace-free");
    cuda_require(cudaFree(device_values), "g2-values-free");
    return counts;
}

static int segment_window(double beta)
{
    if (beta <= 0.5) return 0;
    int q = static_cast<int>(std::floor(beta));
    double phi = beta - static_cast<double>(q);
    int qtop = phi < 0.5 ? q : q + 1;
    return qtop >= 2 ? qtop - 1 : 0;
}

static void make_segment_inputs(
    int bins, int trial, std::vector<double> *dt1,
    std::vector<double> *t1, std::vector<double> *source,
    std::vector<double> *source_cell_by_mode,
    std::vector<double> *upstream)
{
    dt1->resize(static_cast<size_t>(bins));
    t1->resize(static_cast<size_t>(bins));
    source->resize(static_cast<size_t>(bins));
    upstream->resize(static_cast<size_t>(bins));
    source_cell_by_mode->resize(static_cast<size_t>(3 * bins));
    for (int bin = 0; bin < bins; ++bin) {
        double depth;
        double src;
        double up;
        if (trial == 0) {
            depth = 0.0;
            src = std::ldexp(1.0 + 0.03125 * (bin % 7), -20);
            up = std::ldexp(1.0 + 0.0625 * (bin % 5), -18);
        } else if (trial == 1) {
            depth = std::ldexp(1.0 + 0.0625 * (bin % 11), -20);
            src = std::ldexp(1.0 + 0.03125 * (bin % 13), -1000);
            up = std::ldexp(1.0 + 0.015625 * (bin % 17), -900);
        } else {
            depth = 0.00390625 * (1.0 + static_cast<double>(bin % 23));
            src = std::ldexp(1.0 + 0.03125 * (bin % 19), -24 + bin % 9);
            up = std::ldexp(1.0 + 0.015625 * (bin % 29), -22 + bin % 7);
        }
        double transmission = std::exp(-depth);
        if (!host_finite_nonnegative(depth) ||
            !host_finite_nonnegative(transmission) ||
            transmission > 1.0 || !host_finite_nonnegative(src) ||
            !host_finite_nonnegative(up))
            fail("g3-invalid-generated-input");
        (*dt1)[bin] = depth;
        (*t1)[bin] = transmission;
        (*source)[bin] = src;
        (*upstream)[bin] = up;
        for (int mode_slot = 0; mode_slot < 3; ++mode_slot) {
            double cell;
            if (kModes[mode_slot] == ROUND_NEAREST)
                cell = (1.0 - transmission) * src;
            else if (!host_multiply_bound(
                         1.0 - transmission, src, kModes[mode_slot],
                         &cell))
                fail("g3-source-cell-generation");
            (*source_cell_by_mode)[
                static_cast<size_t>(mode_slot) * bins + bin] = cell;
        }
    }
}

static ProofCounts verify_segment_identity(bool sanitizer_smoke)
{
    const int full_bins[] = {1, 2, 3, 17, 31, 32, 33, 63, 64, 65, 96};
    const int smoke_bins[] = {2, 32, 33, 96};
    const int *bin_counts = sanitizer_smoke ? smoke_bins : full_bins;
    int bin_count_size = sanitizer_smoke
        ? static_cast<int>(sizeof(smoke_bins) / sizeof(smoke_bins[0]))
        : static_cast<int>(sizeof(full_bins) / sizeof(full_bins[0]));
    int trials = sanitizer_smoke ? 2 : 3;
    constexpr int max_bins = 96;
    constexpr int max_window = 2 * max_bins + 3;

    double *device_dt1 = nullptr;
    double *device_t1 = nullptr;
    double *device_source = nullptr;
    double *device_source_cell = nullptr;
    double *device_upstream = nullptr;
    double *device_serial = nullptr;
    double *device_epoch = nullptr;
    WindowNode *device_workspace = nullptr;
    int *device_failure = nullptr;
    cuda_require(cudaMalloc(&device_dt1, max_bins * sizeof(double)),
                 "g3-dt1-malloc");
    cuda_require(cudaMalloc(&device_t1, max_bins * sizeof(double)),
                 "g3-t1-malloc");
    cuda_require(cudaMalloc(&device_source, max_bins * sizeof(double)),
                 "g3-source-malloc");
    cuda_require(cudaMalloc(&device_source_cell,
                            3 * max_bins * sizeof(double)),
                 "g3-source-cell-malloc");
    cuda_require(cudaMalloc(&device_upstream, max_bins * sizeof(double)),
                 "g3-upstream-malloc");
    cuda_require(cudaMalloc(&device_serial, 3 * max_bins * sizeof(double)),
                 "g3-serial-malloc");
    cuda_require(cudaMalloc(&device_epoch, 3 * max_bins * sizeof(double)),
                 "g3-epoch-malloc");
    cuda_require(cudaMalloc(
                     &device_workspace,
                     static_cast<size_t>(3 * 2 * max_window) *
                         sizeof(WindowNode)),
                 "g3-workspace-malloc");
    cuda_require(cudaMalloc(&device_failure, sizeof(int)),
                 "g3-failure-malloc");

    std::vector<double> dt1;
    std::vector<double> t1;
    std::vector<double> source;
    std::vector<double> source_cell_by_mode;
    std::vector<double> upstream;
    std::vector<double> serial(static_cast<size_t>(3 * max_bins));
    std::vector<double> epoch(static_cast<size_t>(3 * max_bins));
    ProofCounts counts{0, 0, 0, 0};
    for (int bin_slot = 0; bin_slot < bin_count_size; ++bin_slot) {
        int bins = bin_counts[bin_slot];
        std::set<double> betas;
        if (sanitizer_smoke) {
            betas = {
                0.5, std::nextafter(0.5, INFINITY),
                std::nextafter(1.5, 0.0), 1.5, 2.5, 32.5,
                static_cast<double>(bins) + 0.5,
                2.0 * static_cast<double>(bins) + 3.5
            };
        } else {
            betas = {
                0.0, 0.5, std::nextafter(0.5, INFINITY),
                std::nextafter(1.5, 0.0), 1.5,
                std::nextafter(1.5, INFINITY), 2.5, 3.5,
                31.5, 32.5, 33.5, 63.5, 64.5, 65.5,
                127.5, 128.5, 129.5,
                static_cast<double>(bins) - 0.5,
                static_cast<double>(bins) + 0.5,
                static_cast<double>(bins) + 1.5,
                2.0 * static_cast<double>(bins) + 3.5
            };
        }
        for (int trial = 0; trial < trials; ++trial) {
            make_segment_inputs(bins, trial, &dt1, &t1, &source,
                                &source_cell_by_mode, &upstream);
            cuda_require(cudaMemcpy(device_dt1, dt1.data(),
                                    static_cast<size_t>(bins) * sizeof(double),
                                    cudaMemcpyHostToDevice),
                         "g3-dt1-copy");
            cuda_require(cudaMemcpy(device_t1, t1.data(),
                                    static_cast<size_t>(bins) * sizeof(double),
                                    cudaMemcpyHostToDevice),
                         "g3-t1-copy");
            cuda_require(cudaMemcpy(device_source, source.data(),
                                    static_cast<size_t>(bins) * sizeof(double),
                                    cudaMemcpyHostToDevice),
                         "g3-source-copy");
            cuda_require(cudaMemcpy(
                             device_source_cell, source_cell_by_mode.data(),
                             static_cast<size_t>(3 * bins) * sizeof(double),
                             cudaMemcpyHostToDevice),
                         "g3-source-cell-copy");
            cuda_require(cudaMemcpy(
                             device_upstream, upstream.data(),
                             static_cast<size_t>(bins) * sizeof(double),
                             cudaMemcpyHostToDevice),
                         "g3-upstream-copy");

            for (double beta : betas) {
                if (!(beta >= 0.0) || !std::isfinite(beta))
                    fail("g3-invalid-beta-matrix");
                int window = segment_window(beta);
                if (window > max_window)
                    fail("g3-window-capacity-model");
                for (int upstream_zero = 0; upstream_zero <= 1;
                     ++upstream_zero) {
                    cuda_require(cudaMemset(device_failure, 0, sizeof(int)),
                                 "g3-failure-clear");
                    serial_segment_kernel<<<1, 3>>>(
                        bins, beta, device_dt1, device_t1, device_source,
                        device_source_cell, device_upstream, 0,
                        upstream_zero, device_serial, device_workspace, max_window,
                        device_failure);
                    cuda_require(cudaGetLastError(), "g3-serial-launch");
                    if (window == 0) {
                        dim3 grid((bins + 127) / 128, 3, 1);
                        zero_window_segment_kernel<<<grid, 128>>>(
                            bins, beta, device_dt1, device_t1, device_source,
                            device_upstream, 0, upstream_zero,
                            device_epoch, device_failure);
                    } else {
                        int epochs = (bins + window - 1) / window;
                        dim3 grid(epochs, 3, 1);
                        size_t shared_bytes =
                            static_cast<size_t>(2 * window + 1) *
                            sizeof(Transform);
                        epoch_segment_kernel<<<grid, 128, shared_bytes>>>(
                            bins, window, 0, beta, device_dt1, device_t1,
                            device_source, device_source_cell,
                            device_upstream, 0, upstream_zero,
                            device_epoch, device_failure);
                    }
                    cuda_require(cudaGetLastError(), "g3-epoch-launch");
                    int failure = 0;
                    cuda_require(cudaMemcpy(
                                     serial.data(), device_serial,
                                     static_cast<size_t>(3 * bins) *
                                         sizeof(double),
                                     cudaMemcpyDeviceToHost),
                                 "g3-serial-copy");
                    cuda_require(cudaMemcpy(
                                     epoch.data(), device_epoch,
                                     static_cast<size_t>(3 * bins) *
                                         sizeof(double),
                                     cudaMemcpyDeviceToHost),
                                 "g3-epoch-copy");
                    cuda_require(cudaMemcpy(
                                     &failure, device_failure, sizeof(int),
                                     cudaMemcpyDeviceToHost),
                                 "g3-failure-copy");
                    if (failure != 0)
                        fail("g3-segment-kernel-reported-failure");
                    for (int mode_slot = 0; mode_slot < 3; ++mode_slot) {
                        for (int bin = 0; bin < bins; ++bin) {
                            int index = mode_slot * bins + bin;
                            if (bits_of(serial[index]) != bits_of(epoch[index])) {
                                std::fprintf(
                                    stderr,
                                    "CMF_EXACT_EPOCH_SCAN_SELFTEST "
                                    "G3_MISMATCH bins=%d beta=%a trial=%d "
                                    "upstream_zero=%d window=%d mode=%d "
                                    "bin=%d serial=%a serial_bits=%016llx "
                                    "epoch=%a epoch_bits=%016llx\n",
                                    bins, beta, trial, upstream_zero, window,
                                    kModes[mode_slot], bin, serial[index],
                                    static_cast<unsigned long long>(
                                        bits_of(serial[index])), epoch[index],
                                    static_cast<unsigned long long>(
                                        bits_of(epoch[index])));
                                fail("g3-serial-epoch-segment-bit-mismatch");
                            }
                            if (!host_finite_nonnegative(epoch[index]))
                                fail("g3-nonfinite-or-negative-output");
                        }
                    }
                    for (int bin = 0; bin < bins; ++bin) {
                        if (!(epoch[bin] <= epoch[bins + bin] &&
                              epoch[bins + bin] <= epoch[2 * bins + bin]))
                            fail("g3-directed-ordering-violation");
                    }
                    counts.values += 3LL * bins;
                    counts.mode_cases += 3;
                    ++counts.cases;
                }
            }
        }
    }

    cuda_require(cudaFree(device_failure), "g3-failure-free");
    cuda_require(cudaFree(device_workspace), "g3-workspace-free");
    cuda_require(cudaFree(device_epoch), "g3-epoch-free");
    cuda_require(cudaFree(device_serial), "g3-serial-free");
    cuda_require(cudaFree(device_upstream), "g3-upstream-free");
    cuda_require(cudaFree(device_source_cell), "g3-source-cell-free");
    cuda_require(cudaFree(device_source), "g3-source-free");
    cuda_require(cudaFree(device_t1), "g3-t1-free");
    cuda_require(cudaFree(device_dt1), "g3-dt1-free");
    return counts;
}

static bool host_two_product_sum(double a, double x, double b, double y,
                                 int rounding, double *result)
{
    if (!result) return false;
    if (rounding == ROUND_NEAREST) {
        double value = a * x + b * y;
        if (!host_finite_nonnegative(value)) return false;
        *result = value;
        return true;
    }
    double ax;
    double by;
    return host_multiply_bound(a, x, rounding, &ax) &&
           host_multiply_bound(b, y, rounding, &by) &&
           host_add_bound(ax, by, rounding, result);
}

static bool reconstruct_positive_j(
    int rays, int segments, int bins, const double *mu_by_ray,
    const std::vector<double> &in, const std::vector<double> &out,
    std::vector<double> *j)
{
    j->assign(static_cast<size_t>(3 * segments * bins), NAN);
    for (int mode_slot = 0; mode_slot < 3; ++mode_slot) {
        int rounding = kModes[mode_slot];
        for (int segment = 0; segment < segments; ++segment) {
            for (int bin = 0; bin < bins; ++bin) {
                double sum = 0.0;
                double previous_mu = 0.0;
                double previous_j = 0.0;
                bool first = true;
                for (int ray = rays - 1; ray >= 0; --ray) {
                    size_t slot = static_cast<size_t>(ray * segments + segment);
                    size_t at = slot * static_cast<size_t>(3 * bins) +
                        static_cast<size_t>(mode_slot * bins + bin);
                    double current_j;
                    if (!host_two_product_sum(
                            0.5, out[at], 0.5, in[at], rounding,
                            &current_j)) return false;
                    if (first) {
                        previous_j = current_j;
                        first = false;
                    }
                    double average;
                    double weighted;
                    double next_sum;
                    if (!host_add_bound(previous_j, current_j, rounding,
                                        &average) ||
                        !host_multiply_bound(
                            0.5 * (mu_by_ray[ray] - previous_mu), average,
                            rounding, &weighted) ||
                        !host_add_bound(sum, weighted, rounding,
                                        &next_sum))
                        return false;
                    sum = next_sum;
                    previous_mu = mu_by_ray[ray];
                    previous_j = current_j;
                }
                (*j)[static_cast<size_t>(mode_slot * segments * bins +
                                          segment * bins + bin)] = sum;
            }
        }
    }
    return true;
}

static bool reconstruct_direct_j(
    int rays, int segments, int bins, const double *mu_by_ray,
    const std::vector<double> &in, const std::vector<double> &out,
    std::vector<double> *j)
{
    j->assign(static_cast<size_t>(segments * bins), NAN);
    for (int segment = 0; segment < segments; ++segment) {
        for (int bin = 0; bin < bins; ++bin) {
            double sum = 0.0;
            double previous_mu = 0.0;
            double previous_j = 0.0;
            bool first = true;
            for (int ray = rays - 1; ray >= 0; --ray) {
                size_t slot = static_cast<size_t>(ray * segments + segment);
                size_t at = slot * static_cast<size_t>(bins) + bin;
                double current_j = 0.5 * (out[at] + in[at]);
                if (first) {
                    previous_j = current_j;
                    first = false;
                }
                sum += 0.5 * (previous_j + current_j) *
                       (mu_by_ray[ray] - previous_mu);
                previous_mu = mu_by_ray[ray];
                previous_j = current_j;
            }
            if (!host_finite_nonnegative(sum)) return false;
            (*j)[static_cast<size_t>(segment * bins + bin)] = sum;
        }
    }
    return true;
}

static void launch_epoch_segment_path(
    int bins, double beta, const double *device_dt1,
    const double *device_t1, const double *device_source,
    const double *device_source_cell, const double *device_upstream,
    int upstream_mode_stride, int upstream_zero, double *device_output,
    int *device_failure, int block_size = 128, int epoch_batch = 0)
{
    int window = segment_window(beta);
    if (window == 0) {
        dim3 grid((bins + block_size - 1) / block_size, 3, 1);
        zero_window_segment_kernel<<<grid, block_size>>>(
            bins, beta, device_dt1, device_t1, device_source,
            device_upstream, upstream_mode_stride, upstream_zero,
            device_output, device_failure);
    } else {
        int epochs = (bins + window - 1) / window;
        size_t shared_bytes =
            static_cast<size_t>(2 * window + 1) * sizeof(Transform);
        int batch = epoch_batch > 0 ? epoch_batch : epochs;
        for (int epoch_begin = 0; epoch_begin < epochs;
             epoch_begin += batch) {
            int count = epochs - epoch_begin;
            if (count > batch) count = batch;
            dim3 grid(count, 3, 1);
            epoch_segment_kernel<<<grid, block_size, shared_bytes>>>(
                bins, window, epoch_begin, beta,
                device_dt1, device_t1, device_source,
                device_source_cell, device_upstream, upstream_mode_stride,
                upstream_zero, device_output, device_failure);
            cuda_require(cudaGetLastError(), "epoch-batch-launch");
        }
    }
    cuda_require(cudaGetLastError(), "g4-epoch-segment-launch");
}

static ProofCounts verify_full_small_sweep(bool sanitizer_smoke)
{
    constexpr int rays = 4;
    constexpr int segments = 4;
    constexpr int bins = 96;
    constexpr int slots = rays * segments;
    constexpr int max_window = 2 * bins + 3;
    const int trials = sanitizer_smoke ? 1 : 3;
    const double beta[rays][segments] = {
        {0.5, std::nextafter(0.5, INFINITY), 1.5, 2.5},
        {3.5, 31.5, 32.5, 33.5},
        {63.5, 64.5, 65.5, 96.5},
        {127.5, 128.5, 129.5, 195.5}
    };
    const double mu_by_ray[rays] = {0.95, 0.70, 0.45, 0.20};

    double *device_dt1 = nullptr;
    double *device_t1 = nullptr;
    double *device_source = nullptr;
    double *device_source_cell = nullptr;
    double *device_inner = nullptr;
    double *device_inner_by_mode = nullptr;
    double *device_direct_in = nullptr;
    double *device_direct_out = nullptr;
    double *device_serial_in = nullptr;
    double *device_serial_out = nullptr;
    double *device_epoch_in = nullptr;
    double *device_epoch_out = nullptr;
    WindowNode *device_workspace = nullptr;
    int *device_failure = nullptr;
    cuda_require(cudaMalloc(&device_dt1,
                            segments * bins * sizeof(double)),
                 "g4-dt1-malloc");
    cuda_require(cudaMalloc(&device_t1,
                            segments * bins * sizeof(double)),
                 "g4-t1-malloc");
    cuda_require(cudaMalloc(&device_source,
                            segments * bins * sizeof(double)),
                 "g4-source-malloc");
    cuda_require(cudaMalloc(&device_source_cell,
                            segments * 3 * bins * sizeof(double)),
                 "g4-source-cell-malloc");
    cuda_require(cudaMalloc(&device_inner, bins * sizeof(double)),
                 "g4-inner-malloc");
    cuda_require(cudaMalloc(&device_inner_by_mode, 3 * bins * sizeof(double)),
                 "g4-inner-mode-malloc");
    cuda_require(cudaMalloc(&device_direct_in,
                            slots * bins * sizeof(double)),
                 "g4-direct-in-malloc");
    cuda_require(cudaMalloc(&device_direct_out,
                            slots * bins * sizeof(double)),
                 "g4-direct-out-malloc");
    cuda_require(cudaMalloc(&device_serial_in,
                            slots * 3 * bins * sizeof(double)),
                 "g4-serial-in-malloc");
    cuda_require(cudaMalloc(&device_serial_out,
                            slots * 3 * bins * sizeof(double)),
                 "g4-serial-out-malloc");
    cuda_require(cudaMalloc(&device_epoch_in,
                            slots * 3 * bins * sizeof(double)),
                 "g4-epoch-in-malloc");
    cuda_require(cudaMalloc(&device_epoch_out,
                            slots * 3 * bins * sizeof(double)),
                 "g4-epoch-out-malloc");
    cuda_require(cudaMalloc(
                     &device_workspace,
                     static_cast<size_t>(3 * 2 * max_window) *
                         sizeof(WindowNode)),
                 "g4-workspace-malloc");
    cuda_require(cudaMalloc(&device_failure, sizeof(int)),
                 "g4-failure-malloc");

    std::vector<double> all_dt1(static_cast<size_t>(segments * bins));
    std::vector<double> all_t1(static_cast<size_t>(segments * bins));
    std::vector<double> all_source(static_cast<size_t>(segments * bins));
    std::vector<double> all_source_cell(
        static_cast<size_t>(segments * 3 * bins));
    std::vector<double> inner(static_cast<size_t>(bins));
    std::vector<double> inner_by_mode(static_cast<size_t>(3 * bins));
    std::vector<double> scratch_upstream;
    std::vector<double> segment_dt1;
    std::vector<double> segment_t1;
    std::vector<double> segment_source;
    std::vector<double> segment_source_cell;
    std::vector<double> direct_in(static_cast<size_t>(slots * bins));
    std::vector<double> direct_out(static_cast<size_t>(slots * bins));
    std::vector<double> serial_in(static_cast<size_t>(slots * 3 * bins));
    std::vector<double> serial_out(static_cast<size_t>(slots * 3 * bins));
    std::vector<double> epoch_in(static_cast<size_t>(slots * 3 * bins));
    std::vector<double> epoch_out(static_cast<size_t>(slots * 3 * bins));
    std::vector<double> direct_j;
    std::vector<double> serial_j;
    std::vector<double> epoch_j;
    ProofCounts counts{0, 0, 0, 0};

    for (int trial = 0; trial < trials; ++trial) {
        for (int segment = 0; segment < segments; ++segment) {
            make_segment_inputs(
                bins, (trial + segment) % 3,
                &segment_dt1, &segment_t1, &segment_source,
                &segment_source_cell, &scratch_upstream);
            std::copy(segment_dt1.begin(), segment_dt1.end(),
                      all_dt1.begin() + segment * bins);
            std::copy(segment_t1.begin(), segment_t1.end(),
                      all_t1.begin() + segment * bins);
            std::copy(segment_source.begin(), segment_source.end(),
                      all_source.begin() + segment * bins);
            std::copy(segment_source_cell.begin(), segment_source_cell.end(),
                      all_source_cell.begin() + segment * 3 * bins);
        }
        for (int bin = 0; bin < bins; ++bin) {
            inner[bin] = std::ldexp(
                1.0 + 0.015625 * static_cast<double>(bin % 17), -19);
            for (int mode_slot = 0; mode_slot < 3; ++mode_slot)
                inner_by_mode[mode_slot * bins + bin] = inner[bin];
        }
        cuda_require(cudaMemcpy(
                         device_dt1, all_dt1.data(),
                         all_dt1.size() * sizeof(double),
                         cudaMemcpyHostToDevice),
                     "g4-dt1-copy");
        cuda_require(cudaMemcpy(
                         device_t1, all_t1.data(),
                         all_t1.size() * sizeof(double),
                         cudaMemcpyHostToDevice),
                     "g4-t1-copy");
        cuda_require(cudaMemcpy(
                         device_source, all_source.data(),
                         all_source.size() * sizeof(double),
                         cudaMemcpyHostToDevice),
                     "g4-source-copy");
        cuda_require(cudaMemcpy(
                         device_source_cell, all_source_cell.data(),
                         all_source_cell.size() * sizeof(double),
                         cudaMemcpyHostToDevice),
                     "g4-source-cell-copy");
        cuda_require(cudaMemcpy(device_inner, inner.data(),
                                inner.size() * sizeof(double),
                                cudaMemcpyHostToDevice),
                     "g4-inner-copy");
        cuda_require(cudaMemcpy(device_inner_by_mode, inner_by_mode.data(),
                                inner_by_mode.size() * sizeof(double),
                                cudaMemcpyHostToDevice),
                     "g4-inner-mode-copy");
        cuda_require(cudaMemset(device_failure, 0, sizeof(int)),
                     "g4-failure-clear");

        for (int ray = 0; ray < rays; ++ray) {
            for (int segment = 0; segment < segments; ++segment) {
                int slot = ray * segments + segment;
                const double *dt = device_dt1 + segment * bins;
                const double *tr = device_t1 + segment * bins;
                const double *src = device_source + segment * bins;
                const double *src_cell =
                    device_source_cell + segment * 3 * bins;
                const double *direct_upstream = segment == 0
                    ? device_inner
                    : device_direct_in + (slot - 1) * bins;
                const double *serial_upstream = segment == 0
                    ? device_inner_by_mode
                    : device_serial_in + (slot - 1) * 3 * bins;
                const double *epoch_upstream = segment == 0
                    ? device_inner_by_mode
                    : device_epoch_in + (slot - 1) * 3 * bins;
                direct_segment_kernel<<<1, 128>>>(
                    bins, beta[ray][segment], dt, src, direct_upstream,
                    segment == 0 ? 1 : 0,
                    device_direct_in + slot * bins, device_failure);
                serial_segment_kernel<<<1, 3>>>(
                    bins, beta[ray][segment], dt, tr, src, src_cell,
                    serial_upstream, bins, segment == 0 ? 1 : 0,
                    device_serial_in + slot * 3 * bins,
                    device_workspace, max_window, device_failure);
                cuda_require(cudaGetLastError(), "g4-inward-launch");
                launch_epoch_segment_path(
                    bins, beta[ray][segment], dt, tr, src, src_cell,
                    epoch_upstream, bins, segment == 0 ? 1 : 0,
                    device_epoch_in + slot * 3 * bins, device_failure);
            }
            for (int segment = segments - 1; segment >= 0; --segment) {
                int slot = ray * segments + segment;
                const double *dt = device_dt1 + segment * bins;
                const double *tr = device_t1 + segment * bins;
                const double *src = device_source + segment * bins;
                const double *src_cell =
                    device_source_cell + segment * 3 * bins;
                bool core = ray < 2;
                const double *direct_upstream;
                const double *serial_upstream;
                const double *epoch_upstream;
                if (segment == segments - 1) {
                    direct_upstream = core ? device_inner
                        : device_direct_in + slot * bins;
                    serial_upstream = core ? device_inner_by_mode
                        : device_serial_in + slot * 3 * bins;
                    epoch_upstream = core ? device_inner_by_mode
                        : device_epoch_in + slot * 3 * bins;
                } else {
                    direct_upstream = device_direct_out + (slot + 1) * bins;
                    serial_upstream =
                        device_serial_out + (slot + 1) * 3 * bins;
                    epoch_upstream =
                        device_epoch_out + (slot + 1) * 3 * bins;
                }
                direct_segment_kernel<<<1, 128>>>(
                    bins, beta[ray][segment], dt, src, direct_upstream, 0,
                    device_direct_out + slot * bins, device_failure);
                serial_segment_kernel<<<1, 3>>>(
                    bins, beta[ray][segment], dt, tr, src, src_cell,
                    serial_upstream, bins, 0,
                    device_serial_out + slot * 3 * bins,
                    device_workspace, max_window, device_failure);
                cuda_require(cudaGetLastError(), "g4-outward-launch");
                launch_epoch_segment_path(
                    bins, beta[ray][segment], dt, tr, src, src_cell,
                    epoch_upstream, bins, 0,
                    device_epoch_out + slot * 3 * bins, device_failure);
            }
        }
        int failure = 0;
        cuda_require(cudaMemcpy(&failure, device_failure, sizeof(int),
                                cudaMemcpyDeviceToHost),
                     "g4-failure-copy");
        if (failure != 0) fail("g4-sweep-kernel-reported-failure");
        cuda_require(cudaMemcpy(
                         direct_in.data(), device_direct_in,
                         direct_in.size() * sizeof(double),
                         cudaMemcpyDeviceToHost),
                     "g4-direct-in-copy");
        cuda_require(cudaMemcpy(
                         direct_out.data(), device_direct_out,
                         direct_out.size() * sizeof(double),
                         cudaMemcpyDeviceToHost),
                     "g4-direct-out-copy");
        cuda_require(cudaMemcpy(
                         serial_in.data(), device_serial_in,
                         serial_in.size() * sizeof(double),
                         cudaMemcpyDeviceToHost),
                     "g4-serial-in-copy");
        cuda_require(cudaMemcpy(
                         serial_out.data(), device_serial_out,
                         serial_out.size() * sizeof(double),
                         cudaMemcpyDeviceToHost),
                     "g4-serial-out-copy");
        cuda_require(cudaMemcpy(
                         epoch_in.data(), device_epoch_in,
                         epoch_in.size() * sizeof(double),
                         cudaMemcpyDeviceToHost),
                     "g4-epoch-in-copy");
        cuda_require(cudaMemcpy(
                         epoch_out.data(), device_epoch_out,
                         epoch_out.size() * sizeof(double),
                         cudaMemcpyDeviceToHost),
                     "g4-epoch-out-copy");
        for (size_t index = 0; index < serial_in.size(); ++index) {
            if (bits_of(serial_in[index]) != bits_of(epoch_in[index]) ||
                bits_of(serial_out[index]) != bits_of(epoch_out[index]))
                fail("g4-full-sweep-segment-bit-mismatch");
        }
        if (!reconstruct_direct_j(rays, segments, bins, mu_by_ray,
                                  direct_in, direct_out, &direct_j) ||
            !reconstruct_positive_j(rays, segments, bins, mu_by_ray,
                                    serial_in, serial_out, &serial_j) ||
            !reconstruct_positive_j(rays, segments, bins, mu_by_ray,
                                    epoch_in, epoch_out, &epoch_j))
            fail("g4-angular-reconstruction");
        for (size_t index = 0; index < epoch_j.size(); ++index) {
            if (bits_of(serial_j[index]) != bits_of(epoch_j[index]))
                fail("g4-serial-epoch-j-bit-mismatch");
        }
        size_t cells = static_cast<size_t>(segments * bins);
        for (size_t index = 0; index < cells; ++index) {
            double lower = epoch_j[index];
            double nearest = epoch_j[cells + index];
            double upper = epoch_j[2 * cells + index];
            double direct = direct_j[index];
            if (!(lower <= nearest && nearest <= upper) ||
                !host_finite_nonnegative(direct) ||
                !(lower <= direct && direct <= upper))
                fail("g4-direct-not-covered-by-componentwise-sweep-bound");
            ++counts.records;
        }
        counts.values += static_cast<long long>(epoch_j.size());
        counts.mode_cases += 3;
        ++counts.cases;
    }

    cuda_require(cudaFree(device_failure), "g4-failure-free");
    cuda_require(cudaFree(device_workspace), "g4-workspace-free");
    cuda_require(cudaFree(device_epoch_out), "g4-epoch-out-free");
    cuda_require(cudaFree(device_epoch_in), "g4-epoch-in-free");
    cuda_require(cudaFree(device_serial_out), "g4-serial-out-free");
    cuda_require(cudaFree(device_serial_in), "g4-serial-in-free");
    cuda_require(cudaFree(device_direct_out), "g4-direct-out-free");
    cuda_require(cudaFree(device_direct_in), "g4-direct-in-free");
    cuda_require(cudaFree(device_inner_by_mode), "g4-inner-mode-free");
    cuda_require(cudaFree(device_inner), "g4-inner-free");
    cuda_require(cudaFree(device_source_cell), "g4-source-cell-free");
    cuda_require(cudaFree(device_source), "g4-source-free");
    cuda_require(cudaFree(device_t1), "g4-t1-free");
    cuda_require(cudaFree(device_dt1), "g4-dt1-free");
    return counts;
}

static ProofCounts verify_schedule_invariance(bool sanitizer_smoke)
{
    constexpr int bins = 96;
    const double full_betas[] = {
        0.5, std::nextafter(0.5, INFINITY), 1.5, 2.5,
        3.5, 31.5, 32.5, 33.5, 63.5, 64.5, 65.5,
        127.5, 128.5, 129.5, 195.5
    };
    const double smoke_betas[] = {0.5, 2.5, 32.5, 195.5};
    const int full_blocks[] = {32, 64, 128, 256};
    const int smoke_blocks[] = {32, 256};
    const int full_batches[] = {1, 2, 7, 0};
    const int smoke_batches[] = {1, 0};
    const double *betas = sanitizer_smoke ? smoke_betas : full_betas;
    int beta_count = sanitizer_smoke
        ? static_cast<int>(sizeof(smoke_betas) / sizeof(smoke_betas[0]))
        : static_cast<int>(sizeof(full_betas) / sizeof(full_betas[0]));
    const int *blocks = sanitizer_smoke ? smoke_blocks : full_blocks;
    int block_count = sanitizer_smoke
        ? static_cast<int>(sizeof(smoke_blocks) / sizeof(smoke_blocks[0]))
        : static_cast<int>(sizeof(full_blocks) / sizeof(full_blocks[0]));
    const int *batches = sanitizer_smoke ? smoke_batches : full_batches;
    int batch_count = sanitizer_smoke
        ? static_cast<int>(sizeof(smoke_batches) / sizeof(smoke_batches[0]))
        : static_cast<int>(sizeof(full_batches) / sizeof(full_batches[0]));
    int trials = sanitizer_smoke ? 1 : 3;

    double *device_dt1 = nullptr;
    double *device_t1 = nullptr;
    double *device_source = nullptr;
    double *device_source_cell = nullptr;
    double *device_upstream = nullptr;
    double *device_output = nullptr;
    int *device_failure = nullptr;
    cuda_require(cudaMalloc(&device_dt1, bins * sizeof(double)),
                 "g5-dt1-malloc");
    cuda_require(cudaMalloc(&device_t1, bins * sizeof(double)),
                 "g5-t1-malloc");
    cuda_require(cudaMalloc(&device_source, bins * sizeof(double)),
                 "g5-source-malloc");
    cuda_require(cudaMalloc(&device_source_cell, 3 * bins * sizeof(double)),
                 "g5-source-cell-malloc");
    cuda_require(cudaMalloc(&device_upstream, bins * sizeof(double)),
                 "g5-upstream-malloc");
    cuda_require(cudaMalloc(&device_output, 3 * bins * sizeof(double)),
                 "g5-output-malloc");
    cuda_require(cudaMalloc(&device_failure, sizeof(int)),
                 "g5-failure-malloc");

    std::vector<double> dt1;
    std::vector<double> t1;
    std::vector<double> source;
    std::vector<double> source_cell;
    std::vector<double> upstream;
    std::vector<double> baseline(static_cast<size_t>(3 * bins));
    std::vector<double> actual(static_cast<size_t>(3 * bins));
    ProofCounts counts{0, 0, 0, 0};
    for (int trial = 0; trial < trials; ++trial) {
        make_segment_inputs(bins, trial, &dt1, &t1, &source,
                            &source_cell, &upstream);
        cuda_require(cudaMemcpy(device_dt1, dt1.data(),
                                bins * sizeof(double), cudaMemcpyHostToDevice),
                     "g5-dt1-copy");
        cuda_require(cudaMemcpy(device_t1, t1.data(),
                                bins * sizeof(double), cudaMemcpyHostToDevice),
                     "g5-t1-copy");
        cuda_require(cudaMemcpy(device_source, source.data(),
                                bins * sizeof(double), cudaMemcpyHostToDevice),
                     "g5-source-copy");
        cuda_require(cudaMemcpy(device_source_cell, source_cell.data(),
                                3 * bins * sizeof(double),
                                cudaMemcpyHostToDevice),
                     "g5-source-cell-copy");
        cuda_require(cudaMemcpy(device_upstream, upstream.data(),
                                bins * sizeof(double),
                                cudaMemcpyHostToDevice),
                     "g5-upstream-copy");
        for (int beta_slot = 0; beta_slot < beta_count; ++beta_slot) {
            double beta = betas[beta_slot];
            int window = segment_window(beta);
            for (int upstream_zero = 0; upstream_zero <= 1;
                 ++upstream_zero) {
                cuda_require(cudaMemset(device_failure, 0, sizeof(int)),
                             "g5-baseline-failure-clear");
                launch_epoch_segment_path(
                    bins, beta, device_dt1, device_t1, device_source,
                    device_source_cell, device_upstream, 0, upstream_zero,
                    device_output, device_failure, 128, 0);
                int failure = 0;
                cuda_require(cudaMemcpy(
                                 baseline.data(), device_output,
                                 3 * bins * sizeof(double),
                                 cudaMemcpyDeviceToHost),
                             "g5-baseline-copy");
                cuda_require(cudaMemcpy(
                                 &failure, device_failure, sizeof(int),
                                 cudaMemcpyDeviceToHost),
                             "g5-baseline-failure-copy");
                if (failure != 0) fail("g5-baseline-kernel-failure");
                for (int block_slot = 0; block_slot < block_count;
                     ++block_slot) {
                    for (int batch_slot = 0; batch_slot < batch_count;
                         ++batch_slot) {
                        int block_size = blocks[block_slot];
                        int epoch_batch = window == 0
                            ? 0 : batches[batch_slot];
                        if (block_size == 128 && epoch_batch == 0) continue;
                        if (window == 0 && batch_slot != 0) continue;
                        cuda_require(cudaMemset(
                                         device_failure, 0, sizeof(int)),
                                     "g5-failure-clear");
                        launch_epoch_segment_path(
                            bins, beta, device_dt1, device_t1, device_source,
                            device_source_cell, device_upstream, 0,
                            upstream_zero, device_output, device_failure,
                            block_size, epoch_batch);
                        cuda_require(cudaMemcpy(
                                         actual.data(), device_output,
                                         3 * bins * sizeof(double),
                                         cudaMemcpyDeviceToHost),
                                     "g5-output-copy");
                        cuda_require(cudaMemcpy(
                                         &failure, device_failure, sizeof(int),
                                         cudaMemcpyDeviceToHost),
                                     "g5-failure-copy");
                        if (failure != 0)
                            fail("g5-scheduled-kernel-failure");
                        for (int index = 0; index < 3 * bins; ++index) {
                            if (bits_of(actual[index]) !=
                                bits_of(baseline[index]))
                                fail("g5-scheduling-bit-mismatch");
                        }
                        counts.values += 3LL * bins;
                        counts.mode_cases += 3;
                        ++counts.cases;
                    }
                }
                ++counts.records;
            }
        }
    }

    cuda_require(cudaFree(device_failure), "g5-failure-free");
    cuda_require(cudaFree(device_output), "g5-output-free");
    cuda_require(cudaFree(device_upstream), "g5-upstream-free");
    cuda_require(cudaFree(device_source_cell), "g5-source-cell-free");
    cuda_require(cudaFree(device_source), "g5-source-free");
    cuda_require(cudaFree(device_t1), "g5-t1-free");
    cuda_require(cudaFree(device_dt1), "g5-dt1-free");
    return counts;
}

static uint64_t digest_double_bits(const std::vector<double> &values)
{
    uint64_t digest = 1469598103934665603ULL;
    for (double value : values) {
        uint64_t bits = bits_of(value);
        for (int byte = 0; byte < 8; ++byte) {
            digest ^= (bits >> (8 * byte)) & 0xffU;
            digest *= 1099511628211ULL;
        }
    }
    return digest;
}

static ProofCounts verify_multidevice_invariance(uint64_t *result_digest)
{
    constexpr int rays = 4;
    constexpr int bins = 96;
    const double betas[rays] = {2.5, 32.5, 65.5, 195.5};
    int visible = 0;
    cuda_require(cudaGetDeviceCount(&visible), "g5m-device-count");
    if (visible < 4) fail("g5m-requires-four-visible-devices");

    std::vector<double> dt1;
    std::vector<double> t1;
    std::vector<double> source;
    std::vector<double> source_cell;
    std::vector<double> upstream;
    make_segment_inputs(bins, 2, &dt1, &t1, &source,
                        &source_cell, &upstream);
    std::vector<double> ray_values(
        static_cast<size_t>(rays * 3 * bins));
    std::vector<double> baseline;
    ProofCounts counts{0, 0, 0, 0};
    const int device_counts[] = {1, 2, 4};
    for (int device_count : device_counts) {
        std::fill(ray_values.begin(), ray_values.end(), NAN);
        for (int shard = 0; shard < device_count; ++shard) {
            int ray_begin = (rays * shard) / device_count;
            int ray_end = (rays * (shard + 1)) / device_count;
            cuda_require(cudaSetDevice(shard), "g5m-set-device");
            double *device_dt1 = nullptr;
            double *device_t1 = nullptr;
            double *device_source = nullptr;
            double *device_source_cell = nullptr;
            double *device_upstream = nullptr;
            double *device_output = nullptr;
            int *device_failure = nullptr;
            cuda_require(cudaMalloc(&device_dt1, bins * sizeof(double)),
                         "g5m-dt1-malloc");
            cuda_require(cudaMalloc(&device_t1, bins * sizeof(double)),
                         "g5m-t1-malloc");
            cuda_require(cudaMalloc(&device_source, bins * sizeof(double)),
                         "g5m-source-malloc");
            cuda_require(cudaMalloc(&device_source_cell,
                                    3 * bins * sizeof(double)),
                         "g5m-source-cell-malloc");
            cuda_require(cudaMalloc(&device_upstream, bins * sizeof(double)),
                         "g5m-upstream-malloc");
            cuda_require(cudaMalloc(&device_output,
                                    3 * bins * sizeof(double)),
                         "g5m-output-malloc");
            cuda_require(cudaMalloc(&device_failure, sizeof(int)),
                         "g5m-failure-malloc");
            cuda_require(cudaMemcpy(device_dt1, dt1.data(),
                                    bins * sizeof(double),
                                    cudaMemcpyHostToDevice),
                         "g5m-dt1-copy");
            cuda_require(cudaMemcpy(device_t1, t1.data(),
                                    bins * sizeof(double),
                                    cudaMemcpyHostToDevice),
                         "g5m-t1-copy");
            cuda_require(cudaMemcpy(device_source, source.data(),
                                    bins * sizeof(double),
                                    cudaMemcpyHostToDevice),
                         "g5m-source-copy");
            cuda_require(cudaMemcpy(device_source_cell, source_cell.data(),
                                    3 * bins * sizeof(double),
                                    cudaMemcpyHostToDevice),
                         "g5m-source-cell-copy");
            cuda_require(cudaMemcpy(device_upstream, upstream.data(),
                                    bins * sizeof(double),
                                    cudaMemcpyHostToDevice),
                         "g5m-upstream-copy");
            for (int ray = ray_begin; ray < ray_end; ++ray) {
                cuda_require(cudaMemset(device_failure, 0, sizeof(int)),
                             "g5m-failure-clear");
                launch_epoch_segment_path(
                    bins, betas[ray], device_dt1, device_t1, device_source,
                    device_source_cell, device_upstream, 0, 0,
                    device_output, device_failure, 128, 2);
                int failure = 0;
                cuda_require(cudaMemcpy(
                                 ray_values.data() +
                                     static_cast<size_t>(ray * 3 * bins),
                                 device_output, 3 * bins * sizeof(double),
                                 cudaMemcpyDeviceToHost),
                             "g5m-output-copy");
                cuda_require(cudaMemcpy(&failure, device_failure, sizeof(int),
                                        cudaMemcpyDeviceToHost),
                             "g5m-failure-copy");
                if (failure != 0) fail("g5m-kernel-failure");
            }
            cuda_require(cudaFree(device_failure), "g5m-failure-free");
            cuda_require(cudaFree(device_output), "g5m-output-free");
            cuda_require(cudaFree(device_upstream), "g5m-upstream-free");
            cuda_require(cudaFree(device_source_cell),
                         "g5m-source-cell-free");
            cuda_require(cudaFree(device_source), "g5m-source-free");
            cuda_require(cudaFree(device_t1), "g5m-t1-free");
            cuda_require(cudaFree(device_dt1), "g5m-dt1-free");
        }

        std::vector<double> reduced(static_cast<size_t>(3 * bins), 0.0);
        for (int mode_slot = 0; mode_slot < 3; ++mode_slot) {
            int rounding = kModes[mode_slot];
            for (int bin = 0; bin < bins; ++bin) {
                double sum = 0.0;
                for (int ray = rays - 1; ray >= 0; --ray) {
                    double value = ray_values[
                        static_cast<size_t>(ray * 3 * bins +
                                            mode_slot * bins + bin)];
                    double weighted;
                    double next;
                    if (!host_multiply_bound(0.25, value, rounding,
                                             &weighted) ||
                        !host_add_bound(sum, weighted, rounding, &next))
                        fail("g5m-canonical-reduction");
                    sum = next;
                }
                reduced[static_cast<size_t>(mode_slot * bins + bin)] = sum;
            }
        }
        for (int bin = 0; bin < bins; ++bin) {
            if (!(reduced[bin] <= reduced[bins + bin] &&
                  reduced[bins + bin] <= reduced[2 * bins + bin]))
                fail("g5m-directed-ordering");
        }
        if (baseline.empty())
            baseline = reduced;
        else {
            for (size_t index = 0; index < reduced.size(); ++index) {
                if (bits_of(reduced[index]) != bits_of(baseline[index]))
                    fail("g5m-device-count-result-bit-mismatch");
            }
        }
        counts.values += static_cast<long long>(reduced.size());
        counts.mode_cases += 3;
        ++counts.cases;
    }
    cuda_require(cudaSetDevice(0), "g5m-restore-device-zero");
    if (result_digest) *result_digest = digest_double_bits(baseline);
    counts.records = rays * 3LL * bins;
    return counts;
}

enum G6Injection {
    G6_INJECT_NONE = 0,
    G6_INJECT_INVALID_MODE,
    G6_INJECT_INVALID_INDEX,
    G6_INJECT_WORKSPACE,
    G6_INJECT_ALLOCATION,
    G6_INJECT_CUDA,
    G6_INJECT_NONFINITE
};

static bool run_g6_transaction(G6Injection injection,
                               Transform *public_output,
                               G6FailureDetail *detail)
{
    if (!public_output || !detail) return false;
    *detail = G6FailureDetail{G6_OK, 0, -1, -1, -1, -1};
    int bins = 4;
    int window = 2;
    int rounding = ROUND_NEAREST;
    int provided_workspace = 2;
    if (injection == G6_INJECT_INVALID_MODE) rounding = 2;
    if (injection == G6_INJECT_INVALID_INDEX) bins = 0;
    if (injection == G6_INJECT_WORKSPACE) provided_workspace = 1;
    if (rounding < ROUND_LOWER || rounding > ROUND_UPPER) {
        *detail = G6FailureDetail{
            G6_INVALID_MODE, rounding, 0, TRACE_Q, 0, -1};
        return false;
    }
    if (bins <= 0) {
        *detail = G6FailureDetail{
            G6_INVALID_INDEX, rounding, 0, TRACE_Q, 0, -1};
        return false;
    }
    if (provided_workspace < window) {
        *detail = G6FailureDetail{
            G6_WORKSPACE_TOO_SMALL, rounding, 0, TRACE_Q, 0, -1};
        return false;
    }
    if (injection == G6_INJECT_ALLOCATION) {
        *detail = G6FailureDetail{
            G6_ALLOCATION_FAILURE, rounding, 0, TRACE_Q, 0, -1};
        return false;
    }

    Transform values[4] = {
        {0x1.f0p-1, 0x1.0p-20},
        {0x1.e0p-1, 0x1.2p-20},
        {0x1.d0p-1, 0x1.4p-20},
        {0x1.c0p-1, 0x1.6p-20}
    };
    if (injection == G6_INJECT_NONFINITE)
        values[3].transmission = NAN;
    Transform *device_values = nullptr;
    Transform *device_staging = nullptr;
    G6FailureDetail *device_detail = nullptr;
    cuda_require(cudaMalloc(&device_values, sizeof(values)),
                 "g6-values-malloc");
    cuda_require(cudaMalloc(&device_staging, sizeof(Transform)),
                 "g6-staging-malloc");
    cuda_require(cudaMalloc(&device_detail, sizeof(G6FailureDetail)),
                 "g6-detail-malloc");
    if (injection == G6_INJECT_CUDA) {
        *detail = G6FailureDetail{
            G6_CUDA_FAILURE, rounding, 0, TRACE_Q, 0, -1};
        cuda_require(cudaFree(device_detail), "g6-detail-free-injected");
        cuda_require(cudaFree(device_staging), "g6-staging-free-injected");
        cuda_require(cudaFree(device_values), "g6-values-free-injected");
        return false;
    }
    G6FailureDetail clear{G6_OK, rounding, -1, -1, -1, -1};
    cuda_require(cudaMemcpy(device_values, values, sizeof(values),
                            cudaMemcpyHostToDevice),
                 "g6-values-copy");
    cuda_require(cudaMemcpy(device_detail, &clear, sizeof(clear),
                            cudaMemcpyHostToDevice),
                 "g6-detail-clear");
    g6_transaction_probe_kernel<<<1, 1>>>(
        device_values, bins, window, rounding, device_staging,
        device_detail);
    cuda_require(cudaGetLastError(), "g6-probe-launch");
    cuda_require(cudaMemcpy(detail, device_detail, sizeof(*detail),
                            cudaMemcpyDeviceToHost),
                 "g6-detail-copy");
    bool success = detail->code == G6_OK;
    if (success) {
        Transform staged;
        cuda_require(cudaMemcpy(&staged, device_staging, sizeof(staged),
                                cudaMemcpyDeviceToHost),
                     "g6-staging-copy");
        *public_output = staged;
    }
    cuda_require(cudaFree(device_detail), "g6-detail-free");
    cuda_require(cudaFree(device_staging), "g6-staging-free");
    cuda_require(cudaFree(device_values), "g6-values-free");
    return success;
}

static ProofCounts verify_fail_closed_contract()
{
    const G6Injection failures[] = {
        G6_INJECT_INVALID_MODE,
        G6_INJECT_INVALID_INDEX,
        G6_INJECT_WORKSPACE,
        G6_INJECT_ALLOCATION,
        G6_INJECT_CUDA,
        G6_INJECT_NONFINITE
    };
    const int expected_codes[] = {
        G6_INVALID_MODE,
        G6_INVALID_INDEX,
        G6_WORKSPACE_TOO_SMALL,
        G6_ALLOCATION_FAILURE,
        G6_CUDA_FAILURE,
        G6_NONFINITE_TRANSFORM
    };
    ProofCounts counts{0, 0, 0, 0};
    for (size_t index = 0;
         index < sizeof(failures) / sizeof(failures[0]); ++index) {
        Transform public_output{0x1.23456789abcdep+20,
                                0x1.abcdef0123456p-20};
        Transform before = public_output;
        G6FailureDetail detail;
        if (run_g6_transaction(failures[index], &public_output, &detail))
            fail("g6-negative-control-published");
        if (std::memcmp(&public_output, &before, sizeof(before)) != 0)
            fail("g6-failure-mutated-public-output");
        if (detail.code != expected_codes[index])
            fail("g6-failure-code-mismatch");
        if (failures[index] == G6_INJECT_NONFINITE &&
            !(detail.mode == ROUND_NEAREST && detail.epoch == 0 &&
              detail.chain == TRACE_Q && detail.node == 0 &&
              detail.source_index == 3))
            fail("g6-nonfinite-logical-provenance-mismatch");
        ++counts.records;
        ++counts.cases;
    }

    Transform public_output{0x1.23456789abcdep+20,
                            0x1.abcdef0123456p-20};
    Transform before = public_output;
    G6FailureDetail detail;
    if (!run_g6_transaction(G6_INJECT_NONE, &public_output, &detail) ||
        detail.code != G6_OK ||
        std::memcmp(&public_output, &before, sizeof(before)) == 0 ||
        !host_finite_nonnegative(public_output.transmission) ||
        !host_finite_nonnegative(public_output.emission))
        fail("g6-success-did-not-publish");
    ++counts.values;
    ++counts.cases;
    return counts;
}

static void verify_nonassociation()
{
    const Transform witnesses[9] = {
        {0x1.fda66fd3b058fp-4, 0x1.42f617f05525ap-3},
        {0x1.4f9fee1fe330ep-7, 0x1.fb0e76b3ec976p-14},
        {0x1.b9d3be38204abp-4, 0x1.7e82febc2e09cp-24},
        {0x1.7ef1e3b8709dcp-7, 0x1.14368367ce85ap-30},
        {0x1.f47d6ab739746p-3, 0x1.2916788bc7ab5p-12},
        {0x1.f858297efd535p-2, 0x1.94c196e83936ep+1},
        {0x1.c0a2e8eeb73d8p-4, 0x1.742d6735617fep+4},
        {0x1.05e72455d5868p-4, 0x1.f7875c0da4e2ap-1},
        {0x1.64b1e4f1221f8p-7, 0x1.05e51e0c95836p+7}
    };
    Transform expected[6];
    for (int mode_slot = 0; mode_slot < 3; ++mode_slot) {
        Transform first;
        Transform second;
        if (!host_reverse_compose(witnesses[3 * mode_slot],
                                  witnesses[3 * mode_slot + 1],
                                  kModes[mode_slot], &first) ||
            !host_reverse_compose(first, witnesses[3 * mode_slot + 2],
                                  kModes[mode_slot],
                                  &expected[2 * mode_slot]) ||
            !host_reverse_compose(witnesses[3 * mode_slot + 1],
                                  witnesses[3 * mode_slot + 2],
                                  kModes[mode_slot], &second) ||
            !host_reverse_compose(witnesses[3 * mode_slot], second,
                                  kModes[mode_slot],
                                  &expected[2 * mode_slot + 1]))
            fail("host-nonassociation-evaluation");
        if (transform_bits_equal(expected[2 * mode_slot],
                                 expected[2 * mode_slot + 1]))
            fail("host-nonassociation-witness-collapsed");
    }

    Transform *device_witnesses = nullptr;
    Transform *device_results = nullptr;
    int *device_failure = nullptr;
    Transform actual[6];
    cuda_require(cudaMalloc(&device_witnesses, sizeof(witnesses)),
                 "nonassoc-witness-malloc");
    cuda_require(cudaMalloc(&device_results, sizeof(actual)),
                 "nonassoc-result-malloc");
    cuda_require(cudaMalloc(&device_failure, sizeof(int)),
                 "nonassoc-failure-malloc");
    cuda_require(cudaMemcpy(device_witnesses, witnesses, sizeof(witnesses),
                            cudaMemcpyHostToDevice),
                 "nonassoc-witness-copy");
    cuda_require(cudaMemset(device_failure, 0, sizeof(int)),
                 "nonassoc-failure-clear");
    nonassociation_kernel<<<1, 3>>>(device_witnesses, device_results,
                                    device_failure);
    cuda_require(cudaGetLastError(), "nonassoc-launch");
    int failure = 0;
    cuda_require(cudaMemcpy(actual, device_results, sizeof(actual),
                            cudaMemcpyDeviceToHost),
                 "nonassoc-result-copy");
    cuda_require(cudaMemcpy(&failure, device_failure, sizeof(int),
                            cudaMemcpyDeviceToHost),
                 "nonassoc-failure-copy");
    if (failure != 0) fail("device-nonassociation-evaluation");
    for (int index = 0; index < 6; ++index) {
        if (!transform_bits_equal(expected[index], actual[index]))
            fail("host-device-nonassociation-bit-mismatch");
    }
    cuda_require(cudaFree(device_failure), "nonassoc-failure-free");
    cuda_require(cudaFree(device_results), "nonassoc-result-free");
    cuda_require(cudaFree(device_witnesses), "nonassoc-witness-free");
}

int main(int argc, char **argv)
{
    bool sanitizer_smoke = false;
    bool multi_device = false;
    if (argc == 2 && std::strcmp(argv[1], "--sanitizer-smoke") == 0)
        sanitizer_smoke = true;
    else if (argc == 2 && std::strcmp(argv[1], "--multi-device") == 0)
        multi_device = true;
    else if (argc != 1)
        fail("usage: selftest_cmf_exact_epoch_scan "
             "[--sanitizer-smoke|--multi-device]");

    int device_count = 0;
    cuda_require(cudaGetDeviceCount(&device_count), "device-count");
    if (device_count < 1) fail("no-cuda-device");
    cuda_require(cudaSetDevice(0), "set-device");
    if (multi_device) {
        uint64_t digest = 0;
        ProofCounts cross = verify_multidevice_invariance(&digest);
        std::printf(
            "CMF_EXACT_EPOCH_MULTIDEVICE PASS device_counts=1/2/4 "
            "reductions=%lld mode_reductions=%lld reduced_values=%lld "
            "ray_values=%lld result_digest=%016llx "
            "numerical_repairs=0 floor=0 cap=0 clamp=0 jitter=0\n",
            cross.cases, cross.mode_cases, cross.values, cross.records,
            static_cast<unsigned long long>(digest));
        return 0;
    }
    verify_nonassociation();

    const int full_bins[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        31, 32, 33, 63, 64, 65, 96
    };
    const int smoke_bins[] = {1, 2, 17, 31, 32, 33, 63, 64, 65, 96};
    const int *bin_counts = sanitizer_smoke ? smoke_bins : full_bins;
    int bin_count_size = sanitizer_smoke
        ? static_cast<int>(sizeof(smoke_bins) / sizeof(smoke_bins[0]))
        : static_cast<int>(sizeof(full_bins) / sizeof(full_bins[0]));
    int trials = sanitizer_smoke ? 3 : 12;
    const int max_bins = 96;
    const int mode_count = 3;

    Transform *device_values = nullptr;
    Transform *device_outputs = nullptr;
    LogicalMap *device_mapping = nullptr;
    int *device_failure = nullptr;
    cuda_require(cudaMalloc(&device_values,
                            max_bins * static_cast<int>(sizeof(Transform))),
                 "values-malloc");
    cuda_require(cudaMalloc(&device_outputs,
                            mode_count * max_bins *
                                static_cast<int>(sizeof(Transform))),
                 "outputs-malloc");
    cuda_require(cudaMalloc(&device_mapping,
                            mode_count * max_bins *
                                static_cast<int>(sizeof(LogicalMap))),
                 "mapping-malloc");
    cuda_require(cudaMalloc(&device_failure, sizeof(int)), "failure-malloc");

    std::vector<Transform> values;
    std::vector<Transform> serial[3];
    std::vector<Transform> actual(static_cast<size_t>(mode_count * max_bins));
    std::vector<LogicalMap> actual_mapping(
        static_cast<size_t>(mode_count * max_bins));
    uint64_t random_state = 0x20260810d1a6f00dULL;
    long long base_cases = 0;
    long long mode_cases = 0;
    long long aggregate_pairs = 0;
    long long mapping_entries = 0;
    size_t max_shared_bytes = 0;

    for (int bin_slot = 0; bin_slot < bin_count_size; ++bin_slot) {
        int bins = bin_counts[bin_slot];
        std::set<int> windows = {
            0, 1, 2, 3, bins - 1, bins, bins + 1, 2 * bins + 3
        };
        windows.erase(-1);
        for (int window : windows) {
            if (window < 0) continue;
            for (int trial = 0; trial < trials; ++trial) {
                make_values(bins, trial, &random_state, &values);
                for (int mode_slot = 0; mode_slot < mode_count; ++mode_slot) {
                    if (!serial_aggregates(values, window, kModes[mode_slot],
                                           &serial[mode_slot]))
                        fail("cpu-serial-reference-evaluation");
                }
                for (int output_bin = 0; output_bin < bins; ++output_bin) {
                    if (!(serial[0][output_bin].transmission <=
                              serial[1][output_bin].transmission &&
                          serial[1][output_bin].transmission <=
                              serial[2][output_bin].transmission &&
                          serial[0][output_bin].emission <=
                              serial[1][output_bin].emission &&
                          serial[1][output_bin].emission <=
                              serial[2][output_bin].emission))
                        fail("directed-ordering-violation");
                }

                cuda_require(cudaMemcpy(
                                 device_values, values.data(),
                                 static_cast<size_t>(bins) * sizeof(Transform),
                                 cudaMemcpyHostToDevice),
                             "values-copy");
                cuda_require(cudaMemset(device_failure, 0, sizeof(int)),
                             "failure-clear");
                if (window == 0) {
                    dim3 grid((bins + 127) / 128, mode_count, 1);
                    identity_window_kernel<<<grid, 128>>>(
                        bins, device_outputs, device_mapping, device_failure);
                } else {
                    int epochs = (bins + window - 1) / window;
                    dim3 grid(epochs, mode_count, 1);
                    size_t shared_bytes =
                        static_cast<size_t>(2 * window + 1) *
                        sizeof(Transform);
                    if (shared_bytes > max_shared_bytes)
                        max_shared_bytes = shared_bytes;
                    epoch_aggregate_kernel<<<grid, 128, shared_bytes>>>(
                        device_values, bins, window, device_outputs,
                        device_mapping, device_failure);
                }
                cuda_require(cudaGetLastError(), "epoch-launch");
                int failure = 0;
                cuda_require(cudaMemcpy(
                                 actual.data(), device_outputs,
                                 static_cast<size_t>(mode_count * bins) *
                                     sizeof(Transform),
                                 cudaMemcpyDeviceToHost),
                             "outputs-copy");
                cuda_require(cudaMemcpy(
                                 actual_mapping.data(), device_mapping,
                                 static_cast<size_t>(mode_count * bins) *
                                     sizeof(LogicalMap),
                                 cudaMemcpyDeviceToHost),
                             "mapping-copy");
                cuda_require(cudaMemcpy(&failure, device_failure, sizeof(int),
                                        cudaMemcpyDeviceToHost),
                             "failure-copy");
                if (failure != 0) fail("epoch-kernel-reported-failure");

                for (int mode_slot = 0; mode_slot < mode_count; ++mode_slot) {
                    for (int output_bin = 0; output_bin < bins; ++output_bin) {
                        int index = mode_slot * bins + output_bin;
                        if (!transform_bits_equal(actual[index],
                                                  serial[mode_slot][output_bin]))
                            fail("cpu-cuda-aggregate-bit-mismatch");
                        LogicalMap expected =
                            expected_mapping(bins, window, output_bin);
                        if (!mapping_equal(actual_mapping[index], expected))
                            fail("logical-output-mapping-mismatch");
                        ++aggregate_pairs;
                        ++mapping_entries;
                    }
                    ++mode_cases;
                }
                ++base_cases;
            }
        }
    }

    cuda_require(cudaFree(device_failure), "failure-free");
    cuda_require(cudaFree(device_mapping), "mapping-free");
    cuda_require(cudaFree(device_outputs), "outputs-free");
    cuda_require(cudaFree(device_values), "values-free");
    cuda_require(cudaDeviceSynchronize(), "final-synchronize");

    ProofCounts g2 = verify_logical_traces(sanitizer_smoke);
    ProofCounts g3 = verify_segment_identity(sanitizer_smoke);
    ProofCounts g4 = verify_full_small_sweep(sanitizer_smoke);
    ProofCounts g5 = verify_schedule_invariance(sanitizer_smoke);
    ProofCounts g6 = verify_fail_closed_contract();
    cuda_require(cudaDeviceSynchronize(), "g26-final-synchronize");

    std::printf(
        "CMF_EXACT_EPOCH_SCAN_SELFTEST PASS profile=%s "
        "base_cases=%lld mode_cases=%lld aggregate_pairs=%lld "
        "mapping_entries=%lld nonassociation_modes=3 block_size=128 "
        "max_shared_bytes=%zu g2_trace_cases=%lld "
        "g2_mode_cases=%lld g2_primitive_records=%lld "
        "g3_segment_cases=%lld g3_mode_cases=%lld "
        "g3_output_values=%lld g4_sweeps=%lld g4_mode_sweeps=%lld "
        "g4_j_values=%lld g4_direct_covered=%lld "
        "g5_schedule_runs=%lld g5_mode_runs=%lld "
        "g5_values=%lld g5_baselines=%lld "
        "g6_cases=%lld g6_preserved_outputs=%lld "
        "g6_success_publications=%lld "
        "numerical_repairs=0 floor=0 cap=0 clamp=0 "
        "jitter=0\n",
        sanitizer_smoke ? "sanitizer-smoke" : "full",
        base_cases, mode_cases, aggregate_pairs, mapping_entries,
        max_shared_bytes, g2.cases, g2.mode_cases, g2.records,
        g3.cases, g3.mode_cases, g3.values,
        g4.cases, g4.mode_cases, g4.values, g4.records,
        g5.cases, g5.mode_cases, g5.values, g5.records,
        g6.cases, g6.records, g6.values);
    return 0;
}
