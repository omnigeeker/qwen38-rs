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
pub const K_ROPE_PARTIAL: &str = "rope_partial";
