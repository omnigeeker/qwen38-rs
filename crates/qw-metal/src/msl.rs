//! MSL kernel sources.  Compiled at runtime by `GpuDevice::pipeline`.
//!
//! Naming/ABI conventions used by every kernel in this file:
//!   * weights are MLX-style affine quantisation: `w = q * scale + bias`
//!     with `q` an unsigned integer in `0..2^bits`, packed little-endian into
//!     `uint32` words (8 values per word for 4-bit), `group_size = 64`,
//!     `scales`/`biases` one entry per group per row.
//!   * **`scales`/`biases` are BF16** — the converter quantises bf16 weights, so
//!     the side tensors keep bf16 (verified in the shard headers). They are read
//!     as `ushort` and widened with `as_type<float>(bits << 16)`.
//!   * activations are fp16, accumulators are fp32.

/// Shared prelude injected in front of every kernel body.
pub const COMMON: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

#define GROUP_SIZE 64
#define Q4_WORDS_PER_GROUP 8

// Contiguous 8-values-per-word affine 4-bit GEMV: one threadgroup per row,
// 32 lanes cooperating over the row's groups.
kernel void q4_gemv(
    device const uint*   w      [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const half*   x      [[buffer(3)]],
    device half*         y      [[buffer(4)]],
    constant int&        K      [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const int n_groups = K / GROUP_SIZE;
    device const uint*   wp = w      + (size_t)row * (size_t)(K / 8);
    device const ushort* sp = scales + (size_t)row * (size_t)n_groups;
    device const ushort* bp = biases + (size_t)row * (size_t)n_groups;

    float acc = 0.0f;
    for (int g = (int)lane; g < n_groups; g += 32) {
        const float s = as_type<float>((uint)sp[g] << 16);
        const float b = as_type<float>((uint)bp[g] << 16);
        device const uint* gw = wp + g * Q4_WORDS_PER_GROUP;
        device const half* gx = x + g * GROUP_SIZE;
        #pragma unroll
        for (int wi = 0; wi < Q4_WORDS_PER_GROUP; ++wi) {
            const uint word = gw[wi];
            const float4 x0 = float4(*(device const half4*)(gx + wi * 8 + 0));
            const float4 x1 = float4(*(device const half4*)(gx + wi * 8 + 4));
            float4 q0 = float4((float)( word        & 0xFu),
                               (float)((word >>  4) & 0xFu),
                               (float)((word >>  8) & 0xFu),
                               (float)((word >> 12) & 0xFu));
            float4 q1 = float4((float)((word >> 16) & 0xFu),
                               (float)((word >> 20) & 0xFu),
                               (float)((word >> 24) & 0xFu),
                               (float)((word >> 28) & 0xFu));
            acc += dot(q0 * s + b, x0) + dot(q1 * s + b, x1);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = (half)acc;
    }
}

// Same kernel as q4_gemv_h with both of its loads widened: the weights arrive as two
// 16-byte uint4 instead of eight 4-byte words, and the eight halves the row needs for a
// word come from one 16-byte uint4 instead of two 8-byte half4.  This is the path the
// three MTP draft steps and the prefill take.  Same bits, same order of operations.
kernel void q4_gemv_hx(
    device const uint*   w      [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const half*   x      [[buffer(3)]],
    device half*         y      [[buffer(4)]],
    constant int&        K      [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const int n_groups = K / GROUP_SIZE;
    device const uint*   wp = w      + (size_t)row * (size_t)(K / 8);
    device const ushort* sp = scales + (size_t)row * (size_t)n_groups;
    device const ushort* bp = biases + (size_t)row * (size_t)n_groups;

    float acc = 0.0f;
    for (int g = (int)lane; g < n_groups; g += 32) {
        const half s = (half)as_type<float>((uint)sp[g] << 16);
        const half b = (half)as_type<float>((uint)bp[g] << 16);
        device const uint4* gw4 = (device const uint4*)(wp + g * Q4_WORDS_PER_GROUP);
        device const half*  gx  = x + g * GROUP_SIZE;
        #pragma unroll
        for (int wi2 = 0; wi2 < Q4_WORDS_PER_GROUP / 4; ++wi2) {
            const uint4 w4 = gw4[wi2];
            const uint wds[4] = {w4.x, w4.y, w4.z, w4.w};
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint word = wds[c];
                const uint4 xv = *(device const uint4*)(gx + (wi2 * 4 + c) * 8);
                const half4 x0 = as_type<half4>(xv.xy);
                const half4 x1 = as_type<half4>(xv.zw);
                const half4 q0 = half4((half)( word        & 0xFu),
                                       (half)((word >>  4) & 0xFu),
                                       (half)((word >>  8) & 0xFu),
                                       (half)((word >> 12) & 0xFu));
                const half4 q1 = half4((half)((word >> 16) & 0xFu),
                                       (half)((word >> 20) & 0xFu),
                                       (half)((word >> 24) & 0xFu),
                                       (half)((word >> 28) & 0xFu));
                acc += (float)(dot(q0 * s + b, x0) + dot(q1 * s + b, x1));
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = (half)acc;
    }
}

// Batch serving by GRID mapping, not by per-thread accumulators.
//
// The body below is q4_gemv_hx unchanged: one accumulator, one 16-byte x load and
// two dots per weight chunk.  Round 055 measured what happens when instead each
// thread holds one accumulator per activation (q4_gemv_k16_u4hx): the x loads go
// from 1 to 16 per weight chunk, the kernel stops being bandwidth-bound (22.7 GB/s
// against 466), and the whole batch comes out 22% slower than serial.
//
// Here the parallelism is in the grid: `B` consecutive threadgroup ids cover the
// same output row for `B` different activations, so the weight row is fetched from
// DRAM once and served the other B-1 times out of L2, while every thread keeps the
// optimal one-accumulator shape.  If L2 does not hold the row the traffic is the
// same as B separate sweeps, so the downside is bounded.
kernel void q4_gemv_b16(
    device const uint*   w      [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const half*   x      [[buffer(3)]],
    device half*         y      [[buffer(4)]],
    constant int&        K      [[buffer(5)]],
    constant int&        B      [[buffer(6)]],
    constant int&        out_f  [[buffer(7)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const int row = (int)(tg / (uint)B);
    const int xr  = (int)(tg % (uint)B);
    const int n_groups = K / GROUP_SIZE;
    device const uint*   wp = w      + (size_t)row * (size_t)(K / 8);
    device const ushort* sp = scales + (size_t)row * (size_t)n_groups;
    device const ushort* bp = biases + (size_t)row * (size_t)n_groups;
    device const half*   xb = x      + (size_t)xr  * (size_t)K;

    float acc = 0.0f;
    for (int g = (int)lane; g < n_groups; g += 32) {
        const half s = (half)as_type<float>((uint)sp[g] << 16);
        const half b = (half)as_type<float>((uint)bp[g] << 16);
        device const uint4* gw4 = (device const uint4*)(wp + g * Q4_WORDS_PER_GROUP);
        device const half*  gx  = xb + g * GROUP_SIZE;
        #pragma unroll
        for (int wi2 = 0; wi2 < Q4_WORDS_PER_GROUP / 4; ++wi2) {
            const uint4 w4 = gw4[wi2];
            const uint wds[4] = {w4.x, w4.y, w4.z, w4.w};
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint word = wds[c];
                const uint4 xv = *(device const uint4*)(gx + (wi2 * 4 + c) * 8);
                const half4 x0 = as_type<half4>(xv.xy);
                const half4 x1 = as_type<half4>(xv.zw);
                const half4 q0 = half4((half)( word        & 0xFu),
                                       (half)((word >>  4) & 0xFu),
                                       (half)((word >>  8) & 0xFu),
                                       (half)((word >> 12) & 0xFu));
                const half4 q1 = half4((half)((word >> 16) & 0xFu),
                                       (half)((word >> 20) & 0xFu),
                                       (half)((word >> 24) & 0xFu),
                                       (half)((word >> 28) & 0xFu));
                acc += (float)(dot(q0 * s + b, x0) + dot(q1 * s + b, x1));
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[(size_t)xr * (size_t)out_f + row] = (half)acc;
    }
}

// Same kernel and the same accumulation order as q4_gemv, but with the inner
// product in half.  The plain (k=1) and spec (k=3) paths use different kernels for
// the same linear, so their arithmetic has to stay identical or the two paths part
// company on the 300-token gate; this is the k=1 half of that pair, and
// q4_gemv_k3_u4h is the k=3 half.  Both widen only the group partial sum.
kernel void q4_gemv_h(
    device const uint*   w      [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const half*   x      [[buffer(3)]],
    device half*         y      [[buffer(4)]],
    constant int&        K      [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const int n_groups = K / GROUP_SIZE;
    device const uint*   wp = w      + (size_t)row * (size_t)(K / 8);
    device const ushort* sp = scales + (size_t)row * (size_t)n_groups;
    device const ushort* bp = biases + (size_t)row * (size_t)n_groups;

    float acc = 0.0f;
    for (int g = (int)lane; g < n_groups; g += 32) {
        const half s = (half)as_type<float>((uint)sp[g] << 16);
        const half b = (half)as_type<float>((uint)bp[g] << 16);
        device const uint* gw = wp + g * Q4_WORDS_PER_GROUP;
        device const half* gx = x + g * GROUP_SIZE;
        #pragma unroll
        for (int wi = 0; wi < Q4_WORDS_PER_GROUP; ++wi) {
            const uint word = gw[wi];
            const half4 x0 = *(device const half4*)(gx + wi * 8 + 0);
            const half4 x1 = *(device const half4*)(gx + wi * 8 + 4);
            half4 q0 = half4((half)( word        & 0xFu),
                             (half)((word >>  4) & 0xFu),
                             (half)((word >>  8) & 0xFu),
                             (half)((word >> 12) & 0xFu));
            half4 q1 = half4((half)((word >> 16) & 0xFu),
                             (half)((word >> 20) & 0xFu),
                             (half)((word >> 24) & 0xFu),
                             (half)((word >> 28) & 0xFu));
            acc += (float)(dot(q0 * s + b, x0) + dot(q1 * s + b, x1));
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = (half)acc;
    }
}

// Weight-stationary multi-token GEMV: the weights of a row are read once and
// reused across `k` activation vectors, so the bandwidth-limited part of a
// speculative verification pass is amortised over k tokens.
//
// Measured on M5 Max (497 linears, 14.41 GB): k=1 27.7 ms (520 GB/s);
// k=2 46.4 ms -> 43.1 tok/s equivalent; k=3 60.3 ms -> 49.8; k=4 74.6 ms -> 53.6.
// The marginal cost of an extra token stays below the cost of a full pass, but
// it is far from free: re-reads of x from L2 grow with k (a 4-row register
// blocked variant was measured *slower*, 62.7 ms at k=2, because it breaks the
// coalescing of the weight loads).
kernel void q4_gemv_k(
    device const uint*   w      [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const half*   x      [[buffer(3)]],   // [k][K]
    device half*         y      [[buffer(4)]],   // [k][out_f]
    constant int&        K      [[buffer(5)]],
    constant int&        k      [[buffer(6)]],
    constant int&        out_f  [[buffer(7)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const int n_groups = K / GROUP_SIZE;
    device const uint*   wp = w      + (size_t)row * (size_t)(K / 8);
    device const ushort* sp = scales + (size_t)row * (size_t)n_groups;
    device const ushort* bp = biases + (size_t)row * (size_t)n_groups;

    float acc[4];
    for (int t = 0; t < k; ++t) acc[t] = 0.0f;

    for (int g = (int)lane; g < n_groups; g += 32) {
        const float s  = as_type<float>((uint)sp[g] << 16);
        const float bb = as_type<float>((uint)bp[g] << 16);
        device const uint* gw = wp + g * Q4_WORDS_PER_GROUP;
        #pragma unroll
        for (int wi = 0; wi < Q4_WORDS_PER_GROUP; ++wi) {
            const uint word = gw[wi];
            const float4 w0 = float4((float)( word        & 0xFu),
                                     (float)((word >>  4) & 0xFu),
                                     (float)((word >>  8) & 0xFu),
                                     (float)((word >> 12) & 0xFu)) * s + bb;
            const float4 w1 = float4((float)((word >> 16) & 0xFu),
                                     (float)((word >> 20) & 0xFu),
                                     (float)((word >> 24) & 0xFu),
                                     (float)((word >> 28) & 0xFu)) * s + bb;
            for (int t = 0; t < k; ++t) {
                device const half* gx = x + (size_t)t * K + g * GROUP_SIZE + wi * 8;
                const float4 x0 = float4(*(device const half4*)(gx));
                const float4 x1 = float4(*(device const half4*)(gx + 4));
                acc[t] += dot(w0, x0) + dot(w1, x1);
            }
        }
    }
    for (int t = 0; t < k; ++t) {
        const float a = simd_sum(acc[t]);
        if (lane == 0) y[(size_t)t * out_f + row] = (half)a;
    }
}

// Vector-load variant of q4_gemv_k: the 4-bit words of a group are fetched as
// two 16-byte `uint4` loads instead of eight 4-byte loads.  The round-3 kernel
// issues one load instruction per 4-byte weight word, so its instruction stream
// is dominated by tiny loads; wider loads are the cheapest way to cut it.
// (The `R` scalar from the row-blocking experiment is kept for ABI stability and
// ignored.)
kernel void q4_gemv_kr(
    device const uint*   w      [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const half*   x      [[buffer(3)]],
    device half*         y      [[buffer(4)]],
    constant int&        K      [[buffer(5)]],
    constant int&        k      [[buffer(6)]],
    constant int&        out_f  [[buffer(7)]],
    constant int&        R      [[buffer(8)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    (void)R;
    const int n_groups = K / GROUP_SIZE;
    device const ushort* sp = scales + (size_t)row * (size_t)n_groups;
    device const ushort* bp = biases + (size_t)row * (size_t)n_groups;
    device const uint4*  wp = (device const uint4*)(w + (size_t)row * (size_t)(K / 8));

    float acc[4];
    for (int t = 0; t < k; ++t) acc[t] = 0.0f;

    for (int g = (int)lane; g < n_groups; g += 32) {
        const float s  = as_type<float>((uint)sp[g] << 16);
        const float bb = as_type<float>((uint)bp[g] << 16);
        device const uint4* gw = wp + (size_t)g * (Q4_WORDS_PER_GROUP / 4);
        #pragma unroll
        for (int wi = 0; wi < Q4_WORDS_PER_GROUP / 4; ++wi) {
            const uint4 w4 = gw[wi];
            const uint wds[4] = {w4.x, w4.y, w4.z, w4.w};
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint word = wds[c];
                const float4 w0 = float4((float)( word        & 0xFu),
                                         (float)((word >>  4) & 0xFu),
                                         (float)((word >>  8) & 0xFu),
                                         (float)((word >> 12) & 0xFu)) * s + bb;
                const float4 w1 = float4((float)((word >> 16) & 0xFu),
                                         (float)((word >> 20) & 0xFu),
                                         (float)((word >> 24) & 0xFu),
                                         (float)((word >> 28) & 0xFu)) * s + bb;
                for (int t = 0; t < k; ++t) {
                    device const half* gx = x + (size_t)t * K + g * GROUP_SIZE
                                          + (wi * 4 + c) * 8;
                    const float4 x0 = float4(*(device const half4*)(gx));
                    const float4 x1 = float4(*(device const half4*)(gx + 4));
                    acc[t] += dot(w0, x0) + dot(w1, x1);
                }
            }
        }
    }
    for (int t = 0; t < k; ++t) {
        const float a = simd_sum(acc[t]);
        if (lane == 0) y[(size_t)t * out_f + row] = (half)a;
    }
}

// The kernel count of q4_gemv_k is a runtime value, which forces `acc[k]` to be
// dynamically indexed and spills the accumulators to thread-local memory.  These
// specialisations have the token count as a literal so the loops fully unroll and
// every accumulator stays in a register.  Same ABI as q4_gemv_kr (R unused).
#define Q4_GEMV_KS(NAME, NK)                                                              \
kernel void NAME(                                                                         \
    device const uint*   w      [[buffer(0)]],                                            \
    device const ushort* scales [[buffer(1)]],                                            \
    device const ushort* biases [[buffer(2)]],                                            \
    device const half*   x      [[buffer(3)]],                                            \
    device half*         y      [[buffer(4)]],                                            \
    constant int&        K      [[buffer(5)]],                                            \
    constant int&        k      [[buffer(6)]],                                            \
    constant int&        out_f  [[buffer(7)]],                                            \
    constant int&        R      [[buffer(8)]],                                            \
    uint row  [[threadgroup_position_in_grid]],                                           \
    uint lane [[thread_index_in_threadgroup]])                                            \
{                                                                                         \
    const int n_groups = K / GROUP_SIZE;                                                  \
    device const uint*   wp = w      + (size_t)row * (size_t)(K / 8);                     \
    device const ushort* sp = scales + (size_t)row * (size_t)n_groups;                     \
    device const ushort* bp = biases + (size_t)row * (size_t)n_groups;                     \
    float acc[NK];                                                                        \
    _Pragma("unroll") for (int t = 0; t < NK; ++t) acc[t] = 0.0f;                         \
    for (int g = (int)lane; g < n_groups; g += 32) {                                      \
        const float s  = as_type<float>((uint)sp[g] << 16);                                \
        const float bb = as_type<float>((uint)bp[g] << 16);                                \
        device const uint* gw = wp + g * Q4_WORDS_PER_GROUP;                               \
        _Pragma("unroll") for (int wi = 0; wi < Q4_WORDS_PER_GROUP; ++wi) {                \
            const uint word = gw[wi];                                                      \
            const float4 w0 = float4((float)( word        & 0xFu),                         \
                                     (float)((word >>  4) & 0xFu),                         \
                                     (float)((word >>  8) & 0xFu),                         \
                                     (float)((word >> 12) & 0xFu)) * s + bb;               \
            const float4 w1 = float4((float)((word >> 16) & 0xFu),                         \
                                     (float)((word >> 20) & 0xFu),                         \
                                     (float)((word >> 24) & 0xFu),                         \
                                     (float)((word >> 28) & 0xFu)) * s + bb;               \
            _Pragma("unroll") for (int t = 0; t < NK; ++t) {                               \
                device const half* gx = x + (size_t)t * K + g * GROUP_SIZE + wi * 8;       \
                const float4 x0 = float4(*(device const half4*)(gx));                      \
                const float4 x1 = float4(*(device const half4*)(gx + 4));                  \
                acc[t] += dot(w0, x0) + dot(w1, x1);                                       \
            }                                                                              \
        }                                                                                  \
    }                                                                                      \
    _Pragma("unroll") for (int t = 0; t < NK; ++t) {                                       \
        const float a = simd_sum(acc[t]);                                                  \
        if (lane == 0) y[(size_t)t * out_f + row] = (half)a;                               \
    }                                                                                      \
}
Q4_GEMV_KS(q4_gemv_k2, 2)
Q4_GEMV_KS(q4_gemv_k3, 3)
Q4_GEMV_KS(q4_gemv_k4, 4)

// Same specialisation, but the eight words of a group arrive as two 16-byte
// loads instead of eight 4-byte ones: with a 12:1 x-to-weight byte ratio in this
// sweep the load *instruction* count is a first-order cost, not just the bytes.
#define Q4_GEMV_KS_U4(NAME, NK)                                                          \
kernel void NAME(                                                                        \
    device const uint*   w      [[buffer(0)]],                                           \
    device const ushort* scales [[buffer(1)]],                                           \
    device const ushort* biases [[buffer(2)]],                                           \
    device const half*   x      [[buffer(3)]],                                           \
    device half*         y      [[buffer(4)]],                                           \
    constant int&        K      [[buffer(5)]],                                           \
    constant int&        k      [[buffer(6)]],                                           \
    constant int&        out_f  [[buffer(7)]],                                           \
    constant int&        R      [[buffer(8)]],                                           \
    uint row  [[threadgroup_position_in_grid]],                                          \
    uint lane [[thread_index_in_threadgroup]])                                           \
{                                                                                        \
    (void)R;                                                                             \
    const int n_groups = K / GROUP_SIZE;                                                 \
    device const uint4*  wp = (device const uint4*)(w + (size_t)row * (size_t)(K / 8));  \
    device const ushort* sp = scales + (size_t)row * (size_t)n_groups;                    \
    device const ushort* bp = biases + (size_t)row * (size_t)n_groups;                    \
    float acc[NK];                                                                       \
    _Pragma("unroll") for (int t = 0; t < NK; ++t) acc[t] = 0.0f;                        \
    for (int g = (int)lane; g < n_groups; g += 32) {                                     \
        const float s  = as_type<float>((uint)sp[g] << 16);                               \
        const float bb = as_type<float>((uint)bp[g] << 16);                               \
        device const uint4* gw = wp + (size_t)g * (Q4_WORDS_PER_GROUP / 4);               \
        _Pragma("unroll") for (int wi = 0; wi < Q4_WORDS_PER_GROUP / 4; ++wi) {           \
            const uint4 w4 = gw[wi];                                                      \
            const uint wds[4] = {w4.x, w4.y, w4.z, w4.w};                                 \
            _Pragma("unroll") for (int c = 0; c < 4; ++c) {                               \
                const uint word = wds[c];                                                 \
                const float4 w0 = float4((float)( word        & 0xFu),                    \
                                         (float)((word >>  4) & 0xFu),                    \
                                         (float)((word >>  8) & 0xFu),                    \
                                         (float)((word >> 12) & 0xFu)) * s + bb;          \
                const float4 w1 = float4((float)((word >> 16) & 0xFu),                    \
                                         (float)((word >> 20) & 0xFu),                    \
                                         (float)((word >> 24) & 0xFu),                    \
                                         (float)((word >> 28) & 0xFu)) * s + bb;          \
                _Pragma("unroll") for (int t = 0; t < NK; ++t) {                          \
                    device const half* gx = x + (size_t)t * K + g * GROUP_SIZE            \
                                          + (wi * 4 + c) * 8;                             \
                    const float4 x0 = float4(*(device const half4*)(gx));                 \
                    const float4 x1 = float4(*(device const half4*)(gx + 4));             \
                    acc[t] += dot(w0, x0) + dot(w1, x1);                                  \
                }                                                                         \
            }                                                                             \
        }                                                                                 \
    }                                                                                     \
    _Pragma("unroll") for (int t = 0; t < NK; ++t) {                                      \
        const float a = simd_sum(acc[t]);                                                 \
        if (lane == 0) y[(size_t)t * out_f + row] = (half)a;                              \
    }                                                                                     \
}
Q4_GEMV_KS_U4(q4_gemv_k3_u4, 3)

// Same as Q4_GEMV_KS_U4, but the inner product runs in half.  The fp32 form spends
// 24 of its instructions per (group, word, lane) just widening x (half4 -> float4)
// and 8 more widening the nibbles, against only 24 dots; keeping the loads, the
// dequantised weights and the dots all in half4 removes those conversions, at the
// cost of rounding the dequantised weight to half (the oracle was produced with
// mlx-lm, whose own 4-bit path dequantises in the compute dtype, so this is not
// obviously further from the reference).  Only the group partial sums are widened,
// and they accumulate in fp32.
#define Q4_GEMV_KS_U4H(NAME, NK)                                                          \
kernel void NAME(                                                                          \
    device const uint*   w      [[buffer(0)]],                                             \
    device const ushort* scales [[buffer(1)]],                                             \
    device const ushort* biases [[buffer(2)]],                                             \
    device const half*   x      [[buffer(3)]],                                             \
    device half*         y      [[buffer(4)]],                                             \
    constant int&        K      [[buffer(5)]],                                             \
    constant int&        k      [[buffer(6)]],                                             \
    constant int&        out_f  [[buffer(7)]],                                             \
    constant int&        R      [[buffer(8)]],                                             \
    uint row  [[threadgroup_position_in_grid]],                                             \
    uint lane [[thread_index_in_threadgroup]])                                             \
{                                                                                          \
    (void)k;                                                                               \
    (void)R;                                                                               \
    const int n_groups = K / GROUP_SIZE;                                                   \
    device const uint4*  wp = (device const uint4*)(w + (size_t)row * (size_t)(K / 8));     \
    device const ushort* sp = scales + (size_t)row * (size_t)n_groups;                      \
    device const ushort* bp = biases + (size_t)row * (size_t)n_groups;                      \
    float acc[NK];                                                                         \
    _Pragma("unroll") for (int t = 0; t < NK; ++t) acc[t] = 0.0f;                          \
    for (int g = (int)lane; g < n_groups; g += 32) {                                       \
        const half sh = (half)as_type<float>((uint)sp[g] << 16);                            \
        const half bh = (half)as_type<float>((uint)bp[g] << 16);                            \
        device const uint4* gw = wp + (size_t)g * (Q4_WORDS_PER_GROUP / 4);                 \
        _Pragma("unroll") for (int wi = 0; wi < Q4_WORDS_PER_GROUP / 4; ++wi) {             \
            const uint4 w4 = gw[wi];                                                        \
            const uint wds[4] = {w4.x, w4.y, w4.z, w4.w};                                   \
            _Pragma("unroll") for (int c = 0; c < 4; ++c) {                                 \
                const uint word = wds[c];                                                   \
                const half4 w0 = half4((half)( word        & 0xFu),                         \
                                       (half)((word >>  4) & 0xFu),                         \
                                       (half)((word >>  8) & 0xFu),                         \
                                       (half)((word >> 12) & 0xFu)) * sh + bh;              \
                const half4 w1 = half4((half)((word >> 16) & 0xFu),                         \
                                       (half)((word >> 20) & 0xFu),                         \
                                       (half)((word >> 24) & 0xFu),                         \
                                       (half)((word >> 28) & 0xFu)) * sh + bh;              \
                _Pragma("unroll") for (int t = 0; t < NK; ++t) {                            \
                    device const half* gx = x + (size_t)t * K + g * GROUP_SIZE              \
                                          + (wi * 4 + c) * 8;                               \
                    const half4 x0 = *(device const half4*)(gx);                            \
                    const half4 x1 = *(device const half4*)(gx + 4);                        \
                    acc[t] += (float)(dot(w0, x0) + dot(w1, x1));                           \
                }                                                                           \
            }                                                                               \
        }                                                                                   \
    }                                                                                       \
    _Pragma("unroll") for (int t = 0; t < NK; ++t) {                                        \
        const float a = simd_sum(acc[t]);                                                   \
        if (lane == 0) y[(size_t)t * out_f + row] = (half)a;                                \
    }                                                                                       \
}
Q4_GEMV_KS_U4H(q4_gemv_k3_u4h, 3)
Q4_GEMV_KS_U4H(q4_gemv_k4_u4h, 4)

// The row loop of the half form pays a convert plus an fp32 add for every
// (group, word, row): `acc[t] += (float)(dot(w0,x0) + dot(w1,x1))`.  A lane owns
// only n_groups/32 groups (two or three), so those partial sums are short and can
// stay in half until the single simd_sum at the end - which removes the convert and
// the fp32 add from the hot loop and leaves `hacc[t] += dot + dot`.
#define Q4_GEMV_KS_U4HH(NAME, NK)                                                         \
kernel void NAME(                                                                         \
    device const uint*   w      [[buffer(0)]],                                            \
    device const ushort* scales [[buffer(1)]],                                            \
    device const ushort* biases [[buffer(2)]],                                            \
    device const half*   x      [[buffer(3)]],                                            \
    device half*         y      [[buffer(4)]],                                            \
    constant int&        K      [[buffer(5)]],                                            \
    constant int&        k      [[buffer(6)]],                                            \
    constant int&        out_f  [[buffer(7)]],                                            \
    constant int&        R      [[buffer(8)]],                                            \
    uint row  [[threadgroup_position_in_grid]],                                            \
    uint lane [[thread_index_in_threadgroup]])                                            \
{                                                                                         \
    (void)k;                                                                              \
    (void)R;                                                                              \
    const int n_groups = K / GROUP_SIZE;                                                  \
    device const uint4*  wp = (device const uint4*)(w + (size_t)row * (size_t)(K / 8));    \
    device const ushort* sp = scales + (size_t)row * (size_t)n_groups;                     \
    device const ushort* bp = biases + (size_t)row * (size_t)n_groups;                     \
    half hacc[NK];                                                                        \
    _Pragma("unroll") for (int t = 0; t < NK; ++t) hacc[t] = 0.0h;                         \
    for (int g = (int)lane; g < n_groups; g += 32) {                                      \
        const half sh = (half)as_type<float>((uint)sp[g] << 16);                           \
        const half bh = (half)as_type<float>((uint)bp[g] << 16);                           \
        device const uint4* gw = wp + (size_t)g * (Q4_WORDS_PER_GROUP / 4);                \
        _Pragma("unroll") for (int wi = 0; wi < Q4_WORDS_PER_GROUP / 4; ++wi) {            \
            const uint4 w4 = gw[wi];                                                       \
            const uint wds[4] = {w4.x, w4.y, w4.z, w4.w};                                  \
            _Pragma("unroll") for (int c = 0; c < 4; ++c) {                                \
                const uint word = wds[c];                                                  \
                const half4 w0 = half4((half)( word        & 0xFu),                        \
                                       (half)((word >>  4) & 0xFu),                        \
                                       (half)((word >>  8) & 0xFu),                        \
                                       (half)((word >> 12) & 0xFu)) * sh + bh;             \
                const half4 w1 = half4((half)((word >> 16) & 0xFu),                        \
                                       (half)((word >> 20) & 0xFu),                        \
                                       (half)((word >> 24) & 0xFu),                        \
                                       (half)((word >> 28) & 0xFu)) * sh + bh;             \
                _Pragma("unroll") for (int t = 0; t < NK; ++t) {                           \
                    device const half* gx = x + (size_t)t * K + g * GROUP_SIZE             \
                                          + (wi * 4 + c) * 8;                              \
                    const half4 x0 = *(device const half4*)(gx);                           \
                    const half4 x1 = *(device const half4*)(gx + 4);                       \
                    hacc[t] += dot(w0, x0) + dot(w1, x1);                                  \
                }                                                                          \
            }                                                                              \
        }                                                                                  \
    }                                                                                      \
    _Pragma("unroll") for (int t = 0; t < NK; ++t) {                                       \
        const float a = simd_sum((float)hacc[t]);                                          \
        if (lane == 0) y[(size_t)t * out_f + row] = (half)a;                               \
    }                                                                                      \
}
Q4_GEMV_KS_U4HH(q4_gemv_k3_u4hh, 3)
Q4_GEMV_KS_U4HH(q4_gemv_k4_u4hh, 4)

// The x loads dominate the kernel's memory ops: per (group, word) the row loop issues
// two 8-byte half4 loads per row, while the weights need only one 16-byte load for the
// whole pass - 28.8 G loads against 3.6 G.  The eight halves a lane needs are sixteen
// contiguous bytes, so one uint4 load covers them and the two half4s are reinterprets.
// Same bits, same arithmetic, half the x loads.
#define Q4_GEMV_KS_U4HX(NAME, NK)                                                         \
kernel void NAME(                                                                         \
    device const uint*   w      [[buffer(0)]],                                            \
    device const ushort* scales [[buffer(1)]],                                            \
    device const ushort* biases [[buffer(2)]],                                            \
    device const half*   x      [[buffer(3)]],                                            \
    device half*         y      [[buffer(4)]],                                            \
    constant int&        K      [[buffer(5)]],                                            \
    constant int&        k      [[buffer(6)]],                                            \
    constant int&        out_f  [[buffer(7)]],                                            \
    constant int&        R      [[buffer(8)]],                                            \
    uint row  [[threadgroup_position_in_grid]],                                            \
    uint lane [[thread_index_in_threadgroup]])                                            \
{                                                                                         \
    (void)k;                                                                              \
    (void)R;                                                                              \
    const int n_groups = K / GROUP_SIZE;                                                  \
    device const uint4*  wp = (device const uint4*)(w + (size_t)row * (size_t)(K / 8));    \
    device const ushort* sp = scales + (size_t)row * (size_t)n_groups;                     \
    device const ushort* bp = biases + (size_t)row * (size_t)n_groups;                     \
    float acc[NK];                                                                        \
    _Pragma("unroll") for (int t = 0; t < NK; ++t) acc[t] = 0.0f;                          \
    for (int g = (int)lane; g < n_groups; g += 32) {                                      \
        const half sh = (half)as_type<float>((uint)sp[g] << 16);                           \
        const half bh = (half)as_type<float>((uint)bp[g] << 16);                           \
        device const uint4* gw = wp + (size_t)g * (Q4_WORDS_PER_GROUP / 4);                \
        _Pragma("unroll") for (int wi = 0; wi < Q4_WORDS_PER_GROUP / 4; ++wi) {            \
            const uint4 w4 = gw[wi];                                                       \
            const uint wds[4] = {w4.x, w4.y, w4.z, w4.w};                                  \
            _Pragma("unroll") for (int c = 0; c < 4; ++c) {                                \
                const uint word = wds[c];                                                  \
                const half4 w0 = half4((half)( word        & 0xFu),                        \
                                       (half)((word >>  4) & 0xFu),                        \
                                       (half)((word >>  8) & 0xFu),                        \
                                       (half)((word >> 12) & 0xFu)) * sh + bh;             \
                const half4 w1 = half4((half)((word >> 16) & 0xFu),                        \
                                       (half)((word >> 20) & 0xFu),                        \
                                       (half)((word >> 24) & 0xFu),                        \
                                       (half)((word >> 28) & 0xFu)) * sh + bh;             \
                _Pragma("unroll") for (int t = 0; t < NK; ++t) {                           \
                    device const half* gx = x + (size_t)t * K + g * GROUP_SIZE             \
                                          + (wi * 4 + c) * 8;                              \
                    const uint4 xv = *(device const uint4*)(gx);                           \
                    const half4 x0 = as_type<half4>(xv.xy);                                \
                    const half4 x1 = as_type<half4>(xv.zw);                                \
                    acc[t] += (float)(dot(w0, x0) + dot(w1, x1));                          \
                }                                                                          \
            }                                                                              \
        }                                                                                  \
    }                                                                                      \
    _Pragma("unroll") for (int t = 0; t < NK; ++t) {                                       \
        const float a = simd_sum(acc[t]);                                                  \
        if (lane == 0) y[(size_t)t * out_f + row] = (half)a;                               \
    }                                                                                      \
}
Q4_GEMV_KS_U4HX(q4_gemv_k3_u4hx, 3)
Q4_GEMV_KS_U4HX(q4_gemv_k4_u4hx, 4)

// The same weight-stationary kernel with eight accumulators instead of four.  The
// macro takes the token count as a compile-time parameter and the accumulator loop
// is unrolled over NK, NOT over the runtime `k` scalar - so K4_U4HX processes
// exactly four tokens whatever `k` says.  Benchmarking it with a larger `k` reports
// a speedup that is only the kernel doing less work than the timing assumes, which
// is a trap worth naming.  This instantiation makes the k=8 question answerable.
Q4_GEMV_KS_U4HX(q4_gemv_k8_u4hx, 8)

// ---------------------------------------------------------------------------
// Row-amortising GEMM for prefill.
//
// The GEMV family above is weight-stationary over at most four tokens: each
// thread keeps one accumulator per token, and registers run out at k=8 (measured
// 17-27 per cent slower than k=4, with effective bandwidth collapsing from 417 to
// 154 GB/s).  A prefill of n tokens therefore sweeps the whole 14.4 GB of weights
// n/4 times - 226 sweeps and 3.25 TB for a 905-token prompt.
//
// Tensor cores break that wall through register economy: an 8x8 simdgroup_matrix
// fragment is 64 halfs spread over 32 lanes, one register each, rather than one
// accumulator per output element per thread.  The weights are 4-bit affine, so
// they are dequantised into threadgroup memory once per tile and then reused by
// every token in the tile.
//
// Geometry: BM=32 output rows, BN=32 tokens, BK=64 K values per step, 128 threads
// in 4 simdgroups, each simdgroup owning a 16x16 quarter of the tile as four 8x8
// fragments.  Shared memory: 32*64*2 = 4 KB of dequantised weights, 64*32*2 = 4 KB
// of activations, 32*32*4 = 4 KB of output staging.
// ---------------------------------------------------------------------------
#define Q4_GEMM_BM 32
#define Q4_GEMM_BN 32
#define Q4_GEMM_BK 32
// Shared-memory leading dimensions, padded away from a multiple of the 32-bank
// (128-byte) period.  With an unpadded stride of 64 halfs, every one of the eight
// rows a simdgroup_load touches starts on the same bank and the load serialises
// eight ways; 72 halfs is 144 bytes, 36 words, and 36 mod 32 is 4, so the eight
// rows land on banks 0, 4, 8, ... 28.  Same idea for the activation tile.
#define Q4_GEMM_NT  128
#define Q4_GEMM_WLD (Q4_GEMM_BK + 8)
// Odd pad on purpose.  The x tile is written one column at a time, so with an
// even pad the bank index (kk*XLD + t) % 32 collapses to four values and the
// writes take a 4-way conflict - that is what made the earlier attempt to
// coalesce the x reads 40 per cent slower.  41 is coprime with 32, so kk*41
// cycles through every bank while the reads stay coalesced.
#define Q4_GEMM_XLD (Q4_GEMM_BN + 9)

// Split-K factor.  Weight traffic per token is W/BN and BN cannot grow past 32
// without either sixteen accumulators per simdgroup or a half-idle weight
// staging, so the only way to stop re-reading the weights four times per 128-row
// pass is to cut K instead.  Each z-slice reads K/4 of the weights, and the four
// partials are summed by `q4_gemm_reduce`.  Traffic per pass falls from 4W to W
// while grid.z restores the threadgroup count to the 640 that the BN sweep found
// to be the bandwidth sweet spot.  Fixed rather than a buffer so that every
// existing dispatch keeps working untouched; `mode == 4` selects the path.
#define Q4_GEMM_S 4

kernel void q4_gemm_tile(
    device const uint*   w      [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const half*   x      [[buffer(3)]],   // [k][K]
    device half*         y      [[buffer(4)]],   // [k][out_f]
    constant int&        K      [[buffer(5)]],
    constant int&        k      [[buffer(6)]],
    constant int&        out_f  [[buffer(7)]],
    constant int&        mode   [[buffer(8)]],   // diagnostic: 0 full, 1 skip dequant, 2 skip MACs
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
    const int row0 = (int)tg.x * Q4_GEMM_BM;
    const int tok0 = (int)tg.y * Q4_GEMM_BN;
    const int n_groups = K / GROUP_SIZE;
    // This slice's K range.  Rounded up to a whole number of BK steps so that
    // every step stages a full tile; a slice past the end simply loops zero times
    // and contributes zeros.
    // ONLY under mode 4.  Applying the slice unconditionally truncates the K loop
    // for every ordinary dispatch - grid.z is 1 there, so k_hi would come out at
    // K/4 and the GEMM would quietly compute a quarter of each dot product.  That
    // is exactly what happened on the first build: the server emitted all
    // newlines, the A/B looked 27 per cent faster, and the "win" was three
    // quarters of the work going missing.
    const bool sk   = (mode == 4);
    const int kstep = sk ? (((K + Q4_GEMM_S - 1) / Q4_GEMM_S + Q4_GEMM_BK - 1) / Q4_GEMM_BK * Q4_GEMM_BK) : K;
    const int k_lo  = sk ? ((int)tg.z * kstep) : 0;
    const int k_hi  = sk ? min(K, k_lo + kstep) : K;

    threadgroup half wsh[Q4_GEMM_BM * Q4_GEMM_WLD];
    threadgroup half xsh[Q4_GEMM_BK * Q4_GEMM_XLD];
    threadgroup float osh[Q4_GEMM_BM * Q4_GEMM_BN];

    const int r_off = (int)(sg >> 1) * 16;
    const int t_off = (int)(sg & 1u) * 16;

    simdgroup_matrix<float, 8, 8> a00 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> a01 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> a10 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> a11 = simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = k_lo; k0 < k_hi; k0 += Q4_GEMM_BK) {
        // One uint per thread, expanded to its eight nibbles.  The obvious loop - one
        // iteration per (row, K) element - loads the whole uint for every nibble in it,
        // so each uint is fetched eight times and the useful bandwidth lands at an
        // eighth of what the memory system is actually asked for.  That was the whole
        // reason the GEMM ran at 33.6 GB/s against the GEMV's 417.
        const int nwords = Q4_GEMM_BK / 8;
        for (int idx = (int)tid; idx < Q4_GEMM_BM * nwords; idx += Q4_GEMM_NT) {
            // mode 5 probes a K-major weight layout: there the tile's 32 output
            // columns sit contiguously inside one K-row, so consecutive lanes must
            // take consecutive COLUMNS to read 128 contiguous bytes.  Swapping which
            // index varies fastest is what makes the probe honest - changing only the
            // address would leave the lanes striding across out_f, which is worse, not
            // better, and would refute the hypothesis for the wrong reason.
            const int r  = (mode == 5) ? (idx - (idx / Q4_GEMM_BM) * Q4_GEMM_BM) : (idx / nwords);
            const int wd = (mode == 5) ? (idx / Q4_GEMM_BM) : (idx - r * nwords);
            const int row = row0 + r;
            const int g0  = k0 + wd * 8;
            half v[8];
            if (row < out_f && g0 < k_hi) {
                // mode 5 is a bandwidth probe, not a computation: it reads the weight
                // at the address a K-major (transposed) layout would put it at.  The
                // results are wrong on purpose.  The row-major layout gives this tile
                // only 32 contiguous bytes per row, so if the memory system fetches
                // 128-byte lines then three quarters of every line is wasted; a
                // K-major layout would give 128 contiguous bytes and no waste.  The
                // probe says which of the two the 56 GB/s is.
                const uint word = (mode == 5)
                    ? w[(size_t)(k0 / 8 + wd) * (size_t)out_f + (size_t)row]
                    : w[(size_t)row * (size_t)(K / 8) + (size_t)(g0 >> 3)];
                const float s  = as_type<float>((uint)scales[(size_t)row * (size_t)n_groups + (size_t)(g0 / GROUP_SIZE)] << 16);
                const float bb = as_type<float>((uint)biases[(size_t)row * (size_t)n_groups + (size_t)(g0 / GROUP_SIZE)] << 16);
                _Pragma("unroll") for (int i = 0; i < 8; ++i) {
                    v[i] = (half)((float)((word >> (4 * i)) & 0xFu) * s + bb);
                }
            } else {
                _Pragma("unroll") for (int i = 0; i < 8; ++i) v[i] = (half)0;
            }
            // mode 1 replaces the dequantised value with a constant so the timing
            // separates the dequantisation from the staging and the MACs.
            _Pragma("unroll") for (int i = 0; i < 8; ++i) {
                wsh[r * Q4_GEMM_WLD + wd * 8 + i] = (mode == 1) ? (half)0.01 : v[i];
            }
        }
        // mode 3 skips the activation staging, which is the read that repeats once
        // per BM-row block of the output - with large out_f that traffic is several
        // times the weight traffic, so the timing says whether it is the bottleneck.
        if (mode != 3) {
            // Consecutive threads take consecutive TOKENS.  This looks badly
            // uncoalesced - neighbouring lanes are K*2 bytes apart - and I "fixed" it
            // once by swapping the two indices, which made the kernel 40 per cent
            // SLOWER.  The activation tile is only a few KB and every row-block of the
            // output re-reads it, so it is L2 resident and the uncoalesced pattern
            // costs almost nothing, while the coalesced form makes the shared-memory
            // writes conflict instead.  Left as measured, not as it looks.
            for (int idx = (int)tid; idx < Q4_GEMM_BK * Q4_GEMM_BN; idx += Q4_GEMM_NT) {
                // Lanes take consecutive K (not consecutive tokens), so the device
                // read is a 64-byte run instead of a 10240-byte stride; the odd XLD
                // keeps the shared write conflict-free.
                const int t   = idx / Q4_GEMM_BK;
                const int kk  = idx - t * Q4_GEMM_BK;
                const int g   = k0 + kk;
                const int tok = tok0 + t;
                xsh[kk * Q4_GEMM_XLD + t] = (g < k_hi && tok < k) ? x[(size_t)tok * (size_t)K + (size_t)g] : (half)0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (mode == 2) { threadgroup_barrier(mem_flags::mem_threadgroup); continue; }
        for (int kk = 0; kk < Q4_GEMM_BK; kk += 8) {
            simdgroup_matrix<half, 8, 8> w0, w1, x0, x1;
            simdgroup_load(w0, wsh + (r_off + 0) * Q4_GEMM_WLD + kk, Q4_GEMM_WLD);
            simdgroup_load(w1, wsh + (r_off + 8) * Q4_GEMM_WLD + kk, Q4_GEMM_WLD);
            simdgroup_load(x0, xsh + kk * Q4_GEMM_XLD + t_off,     Q4_GEMM_XLD);
            simdgroup_load(x1, xsh + kk * Q4_GEMM_XLD + t_off + 8, Q4_GEMM_XLD);
            simdgroup_multiply_accumulate(a00, w0, x0, a00);
            simdgroup_multiply_accumulate(a01, w0, x1, a01);
            simdgroup_multiply_accumulate(a10, w1, x0, a10);
            simdgroup_multiply_accumulate(a11, w1, x1, a11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(a00, osh + (r_off + 0) * Q4_GEMM_BN + t_off,     Q4_GEMM_BN);
    simdgroup_store(a01, osh + (r_off + 0) * Q4_GEMM_BN + t_off + 8, Q4_GEMM_BN);
    simdgroup_store(a10, osh + (r_off + 8) * Q4_GEMM_BN + t_off,     Q4_GEMM_BN);
    simdgroup_store(a11, osh + (r_off + 8) * Q4_GEMM_BN + t_off + 8, Q4_GEMM_BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = (int)tid; idx < Q4_GEMM_BM * Q4_GEMM_BN; idx += Q4_GEMM_NT) {
        const int r   = idx / Q4_GEMM_BN;
        const int t   = idx - r * Q4_GEMM_BN;
        const int row = row0 + r;
        const int tok = tok0 + t;
        if (row < out_f && tok < k) {
            if (mode == 4) {
                // Split-K partial: y is a scratch here, read as f32 by
                // `q4_gemm_reduce`, which writes the real half result.
                ((device float*)y)[(size_t)tg.z * (size_t)k * (size_t)out_f
                                   + (size_t)tok * (size_t)out_f + (size_t)row] = osh[idx];
            } else {
                y[(size_t)tok * (size_t)out_f + (size_t)row] = (half)osh[idx];
            }
        }
    }
}

// Sums the Q4_GEMM_S partial products left by the split-K path.  One thread per
// output element: read S floats, add them in a fixed order, write one half.
kernel void q4_gemm_reduce(
    device const float* part [[buffer(0)]],   // [S][n]
    device half*        y    [[buffer(1)]],   // [n]
    constant int&       n    [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i >= n) return;
    float acc = 0.0f;
    for (int s = 0; s < Q4_GEMM_S; ++s) acc += part[(size_t)s * (size_t)n + (size_t)i];
    y[i] = (half)acc;
}

// 16 independent activations per weight read: the whole point of batch-16
// serving.  Each threadgroup still computes ONE output row, but consumes 16
// input rows, so the 14.4 GB weight stream is amortised over 16 tokens instead
// of 1.  `acc[16]` plus the 2x half4 temporaries is the register cost.
Q4_GEMV_KS_U4HX(q4_gemv_k16_u4hx, 16)
Q4_GEMV_KS_U4H(q4_gemv_k6_u4h, 6)

// The half4 form still spends a horizontal reduction per dot: `dot(half4,half4)`
// has to collapse four lanes to a scalar, twice per word per row.  But a lane sees
// only n_groups/32 iterations of the group loop (K=5120 is 80 groups over 32 lanes,
// so two or three), so the accumulator can stay a half4 and be collapsed once at the
// end instead of 16 times per group.  Same weights, same x, same order of products -
// only the order of the final additions changes, and the partial sums stay far below
// the half range because a lane only owns a couple of groups.
#define Q4_GEMV_KS_U4H4(NAME, NK)                                                         \
kernel void NAME(                                                                         \
    device const uint*   w      [[buffer(0)]],                                            \
    device const ushort* scales [[buffer(1)]],                                            \
    device const ushort* biases [[buffer(2)]],                                            \
    device const half*   x      [[buffer(3)]],                                            \
    device half*         y      [[buffer(4)]],                                            \
    constant int&        K      [[buffer(5)]],                                            \
    constant int&        k      [[buffer(6)]],                                            \
    constant int&        out_f  [[buffer(7)]],                                            \
    constant int&        R      [[buffer(8)]],                                            \
    uint row  [[threadgroup_position_in_grid]],                                            \
    uint lane [[thread_index_in_threadgroup]])                                            \
{                                                                                         \
    (void)k;                                                                              \
    (void)R;                                                                              \
    const int n_groups = K / GROUP_SIZE;                                                  \
    device const uint4*  wp = (device const uint4*)(w + (size_t)row * (size_t)(K / 8));    \
    device const ushort* sp = scales + (size_t)row * (size_t)n_groups;                     \
    device const ushort* bp = biases + (size_t)row * (size_t)n_groups;                     \
    half4 a4[NK];                                                                         \
    _Pragma("unroll") for (int t = 0; t < NK; ++t) a4[t] = half4(0.0h);                    \
    for (int g = (int)lane; g < n_groups; g += 32) {                                      \
        const half sh = (half)as_type<float>((uint)sp[g] << 16);                           \
        const half bh = (half)as_type<float>((uint)bp[g] << 16);                           \
        device const uint4* gw = wp + (size_t)g * (Q4_WORDS_PER_GROUP / 4);                \
        _Pragma("unroll") for (int wi = 0; wi < Q4_WORDS_PER_GROUP / 4; ++wi) {            \
            const uint4 w4 = gw[wi];                                                       \
            const uint wds[4] = {w4.x, w4.y, w4.z, w4.w};                                  \
            _Pragma("unroll") for (int c = 0; c < 4; ++c) {                                \
                const uint word = wds[c];                                                  \
                const half4 w0 = half4((half)( word        & 0xFu),                        \
                                       (half)((word >>  4) & 0xFu),                        \
                                       (half)((word >>  8) & 0xFu),                        \
                                       (half)((word >> 12) & 0xFu)) * sh + bh;             \
                const half4 w1 = half4((half)((word >> 16) & 0xFu),                        \
                                       (half)((word >> 20) & 0xFu),                        \
                                       (half)((word >> 24) & 0xFu),                        \
                                       (half)((word >> 28) & 0xFu)) * sh + bh;             \
                _Pragma("unroll") for (int t = 0; t < NK; ++t) {                           \
                    device const half* gx = x + (size_t)t * K + g * GROUP_SIZE             \
                                          + (wi * 4 + c) * 8;                              \
                    const half4 x0 = *(device const half4*)(gx);                           \
                    const half4 x1 = *(device const half4*)(gx + 4);                       \
                    a4[t] += w0 * x0 + w1 * x1;                                            \
                }                                                                          \
            }                                                                              \
        }                                                                                  \
    }                                                                                      \
    _Pragma("unroll") for (int t = 0; t < NK; ++t) {                                       \
        const float4 a = float4(a4[t]);                                                    \
        const float ssum = simd_sum((a.x + a.y) + (a.z + a.w));                            \
        if (lane == 0) y[(size_t)t * out_f + row] = (half)ssum;                            \
    }                                                                                      \
}
Q4_GEMV_KS_U4H4(q4_gemv_k3_u4h4, 3)

// A lane owns only n_groups/32 groups (K=5120 is 80 groups over 32 lanes, so two or
// three), and the group loop walks them one at a time: load a group, consume it, load
// the next.  Unrolling it by two puts both groups' loads in flight before either is
// consumed, which is the only knob left for the 24.8 ms of ALU work that the 69.6 ms
// weight stream does not manage to hide at the throttled clock.  Two independent fp32
// accumulators keep the add chain from serialising them; they are summed at the end.
#define Q4_GEMV_GROUP_BODY(G, AC)                                                          \
    do {                                                                                   \
        const half sh = (half)as_type<float>((uint)sp[(G)] << 16);                          \
        const half bh = (half)as_type<float>((uint)bp[(G)] << 16);                          \
        device const uint4* gw = wp + (size_t)(G) * (Q4_WORDS_PER_GROUP / 4);               \
        _Pragma("unroll") for (int wi = 0; wi < Q4_WORDS_PER_GROUP / 4; ++wi) {             \
            const uint4 w4 = gw[wi];                                                        \
            const uint wds[4] = {w4.x, w4.y, w4.z, w4.w};                                   \
            _Pragma("unroll") for (int c = 0; c < 4; ++c) {                                 \
                const uint word = wds[c];                                                   \
                const half4 w0 = half4((half)( word        & 0xFu),                         \
                                       (half)((word >>  4) & 0xFu),                         \
                                       (half)((word >>  8) & 0xFu),                         \
                                       (half)((word >> 12) & 0xFu)) * sh + bh;              \
                const half4 w1 = half4((half)((word >> 16) & 0xFu),                         \
                                       (half)((word >> 20) & 0xFu),                         \
                                       (half)((word >> 24) & 0xFu),                         \
                                       (half)((word >> 28) & 0xFu)) * sh + bh;              \
                _Pragma("unroll") for (int t = 0; t < NK_U2; ++t) {                         \
                    device const half* gx = x + (size_t)t * K + (G) * GROUP_SIZE            \
                                          + (wi * 4 + c) * 8;                               \
                    const half4 x0 = *(device const half4*)(gx);                            \
                    const half4 x1 = *(device const half4*)(gx + 4);                        \
                    AC[t] += (float)(dot(w0, x0) + dot(w1, x1));                            \
                }                                                                           \
            }                                                                               \
        }                                                                                   \
    } while (0)

#define Q4_GEMV_KS_U4HU2(NAME, NK)                                                        \
kernel void NAME(                                                                         \
    device const uint*   w      [[buffer(0)]],                                            \
    device const ushort* scales [[buffer(1)]],                                            \
    device const ushort* biases [[buffer(2)]],                                            \
    device const half*   x      [[buffer(3)]],                                            \
    device half*         y      [[buffer(4)]],                                            \
    constant int&        K      [[buffer(5)]],                                            \
    constant int&        k      [[buffer(6)]],                                            \
    constant int&        out_f  [[buffer(7)]],                                            \
    constant int&        R      [[buffer(8)]],                                            \
    uint row  [[threadgroup_position_in_grid]],                                            \
    uint lane [[thread_index_in_threadgroup]])                                            \
{                                                                                         \
    (void)k;                                                                              \
    (void)R;                                                                              \
    const int n_groups = K / GROUP_SIZE;                                                  \
    device const uint4*  wp = (device const uint4*)(w + (size_t)row * (size_t)(K / 8));    \
    device const ushort* sp = scales + (size_t)row * (size_t)n_groups;                     \
    device const ushort* bp = biases + (size_t)row * (size_t)n_groups;                     \
    enum { NK_U2 = NK };                                                                  \
    float acc[NK], acc2[NK];                                                              \
    _Pragma("unroll") for (int t = 0; t < NK; ++t) { acc[t] = 0.0f; acc2[t] = 0.0f; }      \
    int g = (int)lane;                                                                    \
    for (; g + 32 < n_groups; g += 64) {                                                  \
        Q4_GEMV_GROUP_BODY(g, acc);                                                       \
        Q4_GEMV_GROUP_BODY(g + 32, acc2);                                                 \
    }                                                                                     \
    if (g < n_groups) { Q4_GEMV_GROUP_BODY(g, acc); }                                     \
    _Pragma("unroll") for (int t = 0; t < NK; ++t) {                                       \
        const float a = simd_sum(acc[t] + acc2[t]);                                        \
        if (lane == 0) y[(size_t)t * out_f + row] = (half)a;                               \
    }                                                                                     \
}
Q4_GEMV_KS_U4HU2(q4_gemv_k3_u4hu2, 3)

// R output rows per threadgroup.  Every output row of the sweep loads the whole
// input row, so x traffic is out_dim times the input size - far more L1/L2 load
// traffic than the 4-bit weights themselves.  Loading x once per group and reusing
// it across R rows divides that traffic by R while still reading each row's weights
// exactly once.  The x values are held for the whole `wi` step so the reuse is
// real and not re-loaded per row.
#define Q4_GEMV_KS_R(NAME, NK, R)                                                        \
kernel void NAME(                                                                        \
    device const uint*   w      [[buffer(0)]],                                           \
    device const ushort* scales [[buffer(1)]],                                           \
    device const ushort* biases [[buffer(2)]],                                           \
    device const half*   x      [[buffer(3)]],                                           \
    device half*         y      [[buffer(4)]],                                           \
    constant int&        K      [[buffer(5)]],                                           \
    constant int&        k      [[buffer(6)]],                                           \
    constant int&        out_f  [[buffer(7)]],                                           \
    constant int&        Rr     [[buffer(8)]],                                           \
    uint tg   [[threadgroup_position_in_grid]],                                          \
    uint lane [[thread_index_in_threadgroup]])                                           \
{                                                                                        \
    (void)k;                                                                             \
    (void)Rr;                                                                            \
    const int n_groups = K / GROUP_SIZE;                                                 \
    const int r0 = (int)tg * R;                                                          \
    float acc[R][NK];                                                                    \
    _Pragma("unroll") for (int r = 0; r < R; ++r)                                        \
        _Pragma("unroll") for (int t = 0; t < NK; ++t) acc[r][t] = 0.0f;                 \
    for (int g = (int)lane; g < n_groups; g += 32) {                                     \
        _Pragma("unroll") for (int wi = 0; wi < Q4_WORDS_PER_GROUP; ++wi) {               \
            float4 x0[NK], x1[NK];                                                       \
            _Pragma("unroll") for (int t = 0; t < NK; ++t) {                              \
                device const half* gx = x + (size_t)t * K + g * GROUP_SIZE + wi * 8;      \
                x0[t] = float4(*(device const half4*)(gx));                               \
                x1[t] = float4(*(device const half4*)(gx + 4));                           \
            }                                                                             \
            _Pragma("unroll") for (int r = 0; r < R; ++r) {                               \
                /* clamp, do not branch: a partial last threadgroup must not read */    \
                /* scales/biases past the end of the row range.                    */    \
                const int row = (r0 + r < out_f) ? (r0 + r) : (out_f - 1);                 \
                const float s  = as_type<float>((uint)scales[(size_t)row * n_groups + g] << 16); \
                const float bb = as_type<float>((uint)biases[(size_t)row * n_groups + g] << 16); \
                const uint word = w[(size_t)row * (size_t)(K / 8)                        \
                                    + (size_t)g * Q4_WORDS_PER_GROUP + wi];               \
                const float4 w0 = float4((float)( word        & 0xFu),                    \
                                         (float)((word >>  4) & 0xFu),                    \
                                         (float)((word >>  8) & 0xFu),                    \
                                         (float)((word >> 12) & 0xFu)) * s + bb;          \
                const float4 w1 = float4((float)((word >> 16) & 0xFu),                    \
                                         (float)((word >> 20) & 0xFu),                    \
                                         (float)((word >> 24) & 0xFu),                    \
                                         (float)((word >> 28) & 0xFu)) * s + bb;          \
                _Pragma("unroll") for (int t = 0; t < NK; ++t)                            \
                    acc[r][t] += dot(w0, x0[t]) + dot(w1, x1[t]);                         \
            }                                                                             \
        }                                                                                 \
    }                                                                                     \
    _Pragma("unroll") for (int r = 0; r < R; ++r) {                                       \
        if (r0 + r < out_f) {                                                             \
            _Pragma("unroll") for (int t = 0; t < NK; ++t) {                              \
                const float a = simd_sum(acc[r][t]);                                      \
                if (lane == 0) y[(size_t)t * out_f + r0 + r] = (half)a;                   \
            }                                                                             \
        }                                                                                 \
    }                                                                                     \
}
Q4_GEMV_KS_R(q4_gemv_k3_r2, 3, 2)
Q4_GEMV_KS_R(q4_gemv_k3_r3, 3, 3)
Q4_GEMV_KS_R(q4_gemv_k3_r4, 3, 4)
Q4_GEMV_KS_R(q4_gemv_k3_r2u, 3, 2)
Q4_GEMV_KS_R(q4_gemv_k3_r6, 3, 6)
Q4_GEMV_KS_R(q4_gemv_k3_r8, 3, 8)

// RMSNorm over the last dim, one threadgroup per row.
kernel void rmsnorm(
    device const half* x [[buffer(0)]],
    device const half* w [[buffer(1)]],
    device half*       y [[buffer(2)]],
    constant int&      N [[buffer(3)]],
    constant float&  eps [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint nt   [[threads_per_threadgroup]])
{
    device const half* xr = x + (size_t)row * (size_t)N;
    device half*       yr = y + (size_t)row * (size_t)N;
    float ss = 0.0f;
    for (int i = (int)lane; i < N; i += (int)nt) {
        const float v = (float)xr[i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    // broadcast the per-simdgroup partial sums through threadgroup memory
    threadgroup float partial[32];
    const uint sg = lane / 32;
    const uint sl = lane % 32;
    if (sl == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (nt + 31) / 32;
    for (uint i = 0; i < nsg; ++i) total += partial[i];
    const float rstd = rsqrt(total / (float)N + eps);
    for (int i = (int)lane; i < N; i += (int)nt) {
        yr[i] = (half)((float)xr[i] * rstd * (float)w[i]);
    }
}

// elementwise add (used as a bring-up sanity kernel + residual plumbing)
kernel void ewise_add(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half*       o [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    o[i] = (half)((float)a[i] + (float)b[i]);
}
"#;

/// Kernel entry names.
pub const K_Q4_GEMV: &str = "q4_gemv";
pub const K_Q4_GEMV_H: &str = "q4_gemv_h";
pub const K_Q4_GEMV_HX: &str = "q4_gemv_hx";
pub const K_Q4_GEMV_K: &str = "q4_gemv_k";
pub const K_Q4_GEMV_KR: &str = "q4_gemv_kr";
pub const K_Q4_GEMV_K2: &str = "q4_gemv_k2";
pub const K_Q4_GEMV_K3: &str = "q4_gemv_k3";
pub const K_Q4_GEMV_K4: &str = "q4_gemv_k4";
pub const K_Q4_GEMV_K3_U4: &str = "q4_gemv_k3_u4";
pub const K_Q4_GEMV_K3_U4H: &str = "q4_gemv_k3_u4h";
pub const K_Q4_GEMV_K4_U4H: &str = "q4_gemv_k4_u4h";
pub const K_Q4_GEMV_K6_U4H: &str = "q4_gemv_k6_u4h";
pub const K_Q4_GEMV_K3_U4HH: &str = "q4_gemv_k3_u4hh";
pub const K_Q4_GEMV_K4_U4HH: &str = "q4_gemv_k4_u4hh";
pub const K_Q4_GEMV_K3_U4HX: &str = "q4_gemv_k3_u4hx";
pub const K_Q4_GEMV_K4_U4HX: &str = "q4_gemv_k4_u4hx";
pub const K_Q4_GEMV_K8_U4HX: &str = "q4_gemv_k8_u4hx";
pub const K_Q4_GEMM_TILE: &str = "q4_gemm_tile";
pub const K_Q4_GEMM_REDUCE: &str = "q4_gemm_reduce";
pub const K_Q4_GEMV_K16_U4HX: &str = "q4_gemv_k16_u4hx";
pub const K_Q4_GEMV_B16: &str = "q4_gemv_b16";
pub const K_Q4_GEMV_K3_U4H4: &str = "q4_gemv_k3_u4h4";
pub const K_Q4_GEMV_K3_U4HU2: &str = "q4_gemv_k3_u4hu2";
pub const K_Q4_GEMV_K3_R2: &str = "q4_gemv_k3_r2";
pub const K_Q4_GEMV_K3_R3: &str = "q4_gemv_k3_r3";
pub const K_Q4_GEMV_K3_R2U: &str = "q4_gemv_k3_r2u";
pub const K_Q4_GEMV_K3_R6: &str = "q4_gemv_k3_r6";
pub const K_Q4_GEMV_K3_R8: &str = "q4_gemv_k3_r8";
pub const K_Q4_GEMV_K3_R4: &str = "q4_gemv_k3_r4";
pub const K_RMSNORM: &str = "rmsnorm";
pub const K_EWISE_ADD: &str = "ewise_add";

/// Extra kernels: MLP activation and partial RoPE.
pub const FUSED: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// SwiGLU with the reference's fp32 intermediate precision:
//   out = silu(gate) * up     (silu and the product both in fp32)
kernel void silu_mul(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    const float g = (float)gate[i];
    const float s = g / (1.0f + exp(-g));
    out[i] = (half)(s * (float)up[i]);
}

// Partial RoPE, non-traditional (half-split) pairing:
//   pairs are (i, i + rot_dim/2), angle = pos * base^(-2i/rot_dim)
//   the tail [rot_dim, head_dim) is copied through untouched.
kernel void rope_partial_rows(
    device const half* x    [[buffer(0)]],
    device half*       y    [[buffer(1)]],
    constant int&      n_heads   [[buffer(2)]],
    constant int&      head_dim  [[buffer(3)]],
    constant int&      rot_dim   [[buffer(4)]],
    constant float&    base      [[buffer(5)]],
    constant int&      pos0      [[buffer(6)]],
    constant int&      stride_bytes [[buffer(7)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint nt   [[threads_per_threadgroup]])
{
    const uint head  = tg % (uint)n_heads;
    const uint token = tg / (uint)n_heads;
    device const half* xr = (device const half*)((device char*)x + (size_t)token * stride_bytes)
                            + (size_t)head * head_dim;
    device half*       yr = (device half*)((device char*)y + (size_t)token * stride_bytes)
                            + (size_t)head * head_dim;
    const int pos = pos0 + (int)token;
    const int half_rot = rot_dim / 2;
    for (int i = (int)lane; i < head_dim; i += (int)nt) {
        yr[i] = xr[i];
    }
    for (int i = (int)lane; i < half_rot; i += (int)nt) {
        const float theta = pow((float)base, -2.0f * (float)i / (float)rot_dim);
        const float angle = (float)pos * theta;
        const float c = cos(angle);
        const float s = sin(angle);
        const float a = (float)xr[i];
        const float b = (float)xr[i + half_rot];
        yr[i]            = (half)(a * c - b * s);
        yr[i + half_rot] = (half)(a * s + b * c);
    }
}

kernel void rope_partial(
    device const half* x    [[buffer(0)]],
    device half*       y    [[buffer(1)]],
    constant int&      n_heads   [[buffer(2)]],
    constant int&      head_dim  [[buffer(3)]],
    constant int&      rot_dim   [[buffer(4)]],
    constant float&    base      [[buffer(5)]],
    constant int&      pos       [[buffer(6)]],
    uint h    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint nt   [[threads_per_threadgroup]])
{
    const int half_rot = rot_dim / 2;
    device const half* xr = x + (size_t)h * (size_t)head_dim;
    device half*       yr = y + (size_t)h * (size_t)head_dim;

    for (int i = (int)lane; i < head_dim; i += (int)nt) {
        yr[i] = xr[i];
    }
    for (int i = (int)lane; i < half_rot; i += (int)nt) {
        const float theta = pow((float)base, -2.0f * (float)i / (float)rot_dim);
        const float angle = (float)pos * theta;
        const float c = cos(angle);
        const float s = sin(angle);
        const float a = (float)xr[i];
        const float b = (float)xr[i + half_rot];
        yr[i]            = (half)(a * c - b * s);
        yr[i + half_rot] = (half)(a * s + b * c);
    }
}
"#;

pub const K_SILU_MUL: &str = "silu_mul";
pub const K_ROPE_PARTIAL_ROWS: &str = "rope_partial_rows";
pub const K_ROPE_PARTIAL: &str = "rope_partial";

/// MetalPerformancePrimitives tensor-op probe.
///
/// Everything the simdgroup_matrix GEMM does by hand - dequantise, stage the
/// weights and the activations through threadgroup memory, feed 8x8 fragments -
/// this API does directly from `device` memory, and its type table includes
/// `half x uint4b_format -> half/float`.  That combination is exactly our
/// quantised linear: the 4-bit weights can go to the tensor units as they sit in
/// memory, with no staging and no dequantisation.  Sixteen attempts at tuning the
/// staging loop have failed, so this is the way out.
///
/// The probe deliberately computes only `A x q` and ignores the per-group scale
/// and bias, so its OUTPUT IS WRONG; it exists to prove the kernel compiles and
/// to measure what the tensor op costs on our shapes.
pub const MPP: &str = r#"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

kernel void q4_mpp_probe(
    tensor<device half, dextents<int32_t, 2>> A,           // [rows, K]
    tensor<device uint4b_format, dextents<int32_t, 2>> B,  // [out_f, K]
    tensor<device half, dextents<int32_t, 2>> C,           // [rows, out_f]
    uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        64, 32, static_cast<int>(dynamic_extent), false, true, false);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(tgid.y * 64, 0);
    auto mB = B.slice(tgid.x * 32, 0);
    auto mC = C.slice(tgid.y * 64, tgid.x * 32);
    op.run(mA, mB, mC);
}
"#;
