//! Attention + gated-delta-net kernels.
//!
//! Conventions (from the mlx-lm reference, see `docs/M1_NOTES.md`):
//!   * KV cache / state are stored **head-major**: `[head][token][dim]`, so a
//!     head's history is contiguous and the decode kernels read it linearly.
//!   * gated delta net state is fp32 `[Hv][Dv][Dk]`, gating computed in-kernel
//!     from `a`, `b`, `A_log`, `dt_bias`.
//!   * `scores`/`probs` scratch is fp32 `[H][maxT]`.

pub const ATTN: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Scores + softmax for one query token against the cached K.
// Threadgroup layout: 256 threads = 8 simdgroups; warp sg handles score t=sg
// (stride 8), each lane covers 8 of the D dims (D = 256).
kernel void attn_scores_softmax(
    device const half*  q      [[buffer(0)]],
    device const half*  kcache [[buffer(1)]],
    device float*       scores [[buffer(2)]],
    constant int&       T      [[buffer(3)]],
    constant int&       maxT   [[buffer(4)]],
    constant int&       H      [[buffer(5)]],
    constant int&       Hkv    [[buffer(6)]],
    constant int&       D      [[buffer(7)]],
    constant float&     scale  [[buffer(8)]],
    uint h    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint nt   [[threads_per_threadgroup]])
{
    const int reps = H / Hkv;
    const int hk = (int)h / reps;
    const uint sg = lane / 32;
    const uint sl = lane % 32;
    const int nwarp = (int)(nt / 32);
    device const half* qh = q + (size_t)h * D;
    device const half* kbase = kcache + (size_t)hk * maxT * D;
    device float* sc = scores + (size_t)h * maxT;

    // ---- pass 1: dot products ----
    for (int t = (int)sg; t < T; t += nwarp) {
        device const half* kt = kbase + (size_t)t * D;
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = (int)sl * 8 + j;
            if (d < D) acc += (float)qh[d] * (float)kt[d];
        }
        acc = simd_sum(acc);
        if (sl == 0) sc[t] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ---- pass 2: max ----
    threadgroup float red[32];
    const uint nsg = nt / 32;
    float lmax = -INFINITY;
    for (int t = (int)lane; t < T; t += (int)nt) lmax = max(lmax, sc[t]);
    lmax = simd_max(lmax);
    if (sl == 0) red[sg] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = -INFINITY;
    for (uint i = 0; i < nsg; ++i) gmax = max(gmax, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- pass 3: exp + sum ----
    float lsum = 0.0f;
    for (int t = (int)lane; t < T; t += (int)nt) {
        const float e = exp(sc[t] - gmax);
        sc[t] = e;
        lsum += e;
    }
    lsum = simd_sum(lsum);
    if (sl == 0) red[sg] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gsum = 0.0f;
    for (uint i = 0; i < nsg; ++i) gsum += red[i];
    const float inv = 1.0f / gsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- pass 4: normalise ----
    for (int t = (int)lane; t < T; t += (int)nt) sc[t] *= inv;
}

// Weighted sum of cached V with the probabilities from the kernel above.
// One threadgroup per head, one thread per dim (nt == D).
kernel void attn_out(
    device const float* probs  [[buffer(0)]],
    device const half*  vcache [[buffer(1)]],
    device half*        out    [[buffer(2)]],
    constant int&       T      [[buffer(3)]],
    constant int&       maxT   [[buffer(4)]],
    constant int&       H      [[buffer(5)]],
    constant int&       Hkv    [[buffer(6)]],
    constant int&       D      [[buffer(7)]],
    uint h    [[threadgroup_position_in_grid]],
    uint d    [[thread_index_in_threadgroup]])
{
    const int reps = H / Hkv;
    const int hk = (int)h / reps;
    device const float* p = probs + (size_t)h * maxT;
    device const half* vbase = vcache + (size_t)hk * maxT * D + d;
    float acc = 0.0f;
    for (int t = 0; t < T; ++t) acc += p[t] * (float)vbase[(size_t)t * D];
    out[(size_t)h * D + d] = (half)acc;
}

// Append one token's K/V into the head-major cache.
kernel void kv_append(
    device const half*  k      [[buffer(0)]],
    device const half*  v      [[buffer(1)]],
    device half*        kcache [[buffer(2)]],
    device half*        vcache [[buffer(3)]],
    constant int&       pos    [[buffer(4)]],
    constant int&       maxT   [[buffer(5)]],
    constant int&       Hkv    [[buffer(6)]],
    constant int&       D      [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
    const int per_head = D;
    const int hk = (int)(i / (uint)per_head);
    const int d  = (int)(i % (uint)per_head);
    const size_t dst = ((size_t)hk * maxT + pos) * D + d;
    kcache[dst] = k[i];
    vcache[dst] = v[i];
}
"#;

pub const GDN: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Depth-wise causal conv1d (kernel = 4) followed by SiLU.
// window is [4][conv_dim]: 3 carried rows followed by the current token's row.
kernel void conv1d_silu(
    device const half*  window   [[buffer(0)]],
    device const half*  w        [[buffer(1)]],
    device half*        out      [[buffer(2)]],
    constant int&       conv_dim [[buffer(3)]],
    uint c [[thread_position_in_grid]])
{
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc += (float)w[c * 4 + j] * (float)window[(size_t)j * conv_dim + c];
    }
    out[c] = (half)(acc / (1.0f + exp(-acc)));
}

// Ring-buffer form of conv1d_silu.  The layer keeps its own four-row window and
// the current position's qkv row is already written at `slot`, so the convolution
// reads the slots in ring order instead of the caller copying a three-row history
// in and back out on every row - two dispatches per row per linear-attention
// layer, all of it pure overhead.
kernel void conv1d_silu_ring(
    device const half*  window   [[buffer(0)]],
    device const half*  w        [[buffer(1)]],
    device half*        out      [[buffer(2)]],
    constant int&       conv_dim [[buffer(3)]],
    constant int&       slot     [[buffer(4)]],
    constant int&       ring     [[buffer(5)]],
    uint c [[thread_position_in_grid]])
{
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        // oldest first; j == 3 is `slot`
        // ring is a power of two, so the mask is exact and free
        const int r = (slot + ring - 3 + j) & (ring - 1);
        acc += (float)w[c * 4 + j] * (float)window[(size_t)r * conv_dim + c];
    }
    out[c] = (half)(acc / (1.0f + exp(-acc)));
}

// The same convolution for a whole tile of rows at once: row `r` uses slot
// (slot0 + r).  The slots are consecutive because TILE <= ring, so one dispatch
// covers what used to take TILE of them.
kernel void conv1d_silu_ring_tile(
    device const half*  window   [[buffer(0)]],
    device const half*  w        [[buffer(1)]],
    device half*        out      [[buffer(2)]],
    constant int&       conv_dim [[buffer(3)]],
    constant int&       slot0    [[buffer(4)]],
    constant int&       ring     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const int c   = (int)(gid % (uint)conv_dim);
    const int row = (int)(gid / (uint)conv_dim);
    const int slot = (slot0 + row) & (ring - 1);
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int r = (slot + ring - 3 + j) & (ring - 1);
        acc += (float)w[c * 4 + j] * (float)window[(size_t)r * conv_dim + c];
    }
    out[(size_t)row * conv_dim + c] = (half)(acc / (1.0f + exp(-acc)));
}

// Weightless rms norm for a whole tile: threadgroup (head, tile row).  The
// per-row form needs a dispatch per row; this needs one for the tile.
kernel void rmsnorm_nw_tile(
    device const half* x [[buffer(0)]],
    device half*       y [[buffer(1)]],
    constant int&      D             [[buffer(2)]],
    constant int&      in_head_stride [[buffer(3)]],
    constant int&      in_row_stride  [[buffer(4)]],
    constant int&      out_row_stride [[buffer(5)]],
    constant float&    eps   [[buffer(6)]],
    constant float&    scale [[buffer(7)]],
    constant int&      HK    [[buffer(8)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint nt   [[threads_per_threadgroup]])
{
    // flattened (row, head): a 2-D grid cannot mix with scalar bindings in MSL
    const int row  = (int)(tg / (uint)HK);
    const int head = (int)(tg % (uint)HK);
    device const half* xr = x + (size_t)row * (size_t)in_row_stride
                              + (size_t)head * (size_t)in_head_stride;
    device half*       yr = y + (size_t)row * (size_t)out_row_stride
                              + (size_t)head * (size_t)D;
    float ss = 0.0f;
    for (int i = (int)lane; i < D; i += (int)nt) {
        const float v = (float)xr[i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    threadgroup float red[32];
    const uint sg = lane / 32, sl = lane % 32;
    if (sl == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (nt + 31) / 32;
    for (uint i = 0; i < nsg; ++i) total += red[i];
    const float rstd = rsqrt(total / (float)D + eps) * scale;
    for (int i = (int)lane; i < D; i += (int)nt) {
        yr[i] = (half)((float)xr[i] * rstd);
    }
}

// One thread owns one (v-head, v-dim) row of the fp32 state matrix; the Dk loop
// is fully serial per thread, which keeps the whole recurrence in registers.
kernel void gdn_step(
    device const half*  q       [[buffer(0)]],   // [Hk * Dk]
    device const half*  k       [[buffer(1)]],   // [Hk * Dk]
    device const half*  v       [[buffer(2)]],   // [Hv * Dv]
    device const half*  a       [[buffer(3)]],   // [Hv]
    device const half*  b       [[buffer(4)]],   // [Hv]
    device const float* A_log   [[buffer(5)]],   // [Hv]
    device const float* dt_bias [[buffer(6)]],   // [Hv]
    device float*       state   [[buffer(7)]],   // [Hv * Dv * Dk] fp32
    device half*        y       [[buffer(8)]],   // [Hv * Dv]
    constant int&       Hk      [[buffer(9)]],
    constant int&       Hv      [[buffer(10)]],
    constant int&       Dk      [[buffer(11)]],
    constant int&       Dv      [[buffer(12)]],
    uint hv [[threadgroup_position_in_grid]],
    uint dv [[thread_index_in_threadgroup]])
{
    const int reps = Hv / Hk;
    const int hk = (int)hv / reps;

    // g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
    const float x = (float)a[hv] + dt_bias[hv];
    const float sp = max(x, 0.0f) + log(1.0f + exp(-fabs(x)));
    const float g = exp(-exp(A_log[hv]) * sp);
    const float beta = 1.0f / (1.0f + exp(-(float)b[hv]));

    device const half* kp = k + (size_t)hk * Dk;
    device const half* qp = q + (size_t)hk * Dk;
    device float* S = state + ((size_t)hv * Dv + dv) * Dk;

    float kv = 0.0f;
    for (int d = 0; d < Dk; ++d) {
        S[d] *= g;
        kv += S[d] * (float)kp[d];
    }
    const float delta = ((float)v[(size_t)hv * Dv + dv] - kv) * beta;
    float acc = 0.0f;
    for (int d = 0; d < Dk; ++d) {
        S[d] += delta * (float)kp[d];
        acc += S[d] * (float)qp[d];
    }
    y[(size_t)hv * Dv + dv] = (half)acc;
}

// RMSNorm over `D` with an explicit input row stride and an output scale.
// `has_weight == 1` multiplies by `w` first. Used for
//   * delta-net q/k:  no weight, scale = inv or inv^2
//   * attention q/k:  with weight, q read at stride 2*head_dim (the projection
//     emits [q | gate] per head)
kernel void rmsnorm_s(
    device const half* x [[buffer(0)]],
    device const half* w [[buffer(1)]],
    device half*       y [[buffer(2)]],
    constant int&      D [[buffer(3)]],
    constant int&      in_stride [[buffer(4)]],
    constant float&  eps [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant int& has_weight [[buffer(7)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint nt   [[threads_per_threadgroup]])
{
    device const half* xr = x + (size_t)row * (size_t)in_stride;
    device half*       yr = y + (size_t)row * (size_t)D;
    float ss = 0.0f;
    for (int i = (int)lane; i < D; i += (int)nt) {
        const float v = (float)xr[i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    threadgroup float red[32];
    const uint sg = lane / 32, sl = lane % 32;
    if (sl == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (nt + 31) / 32;
    for (uint i = 0; i < nsg; ++i) total += red[i];
    const float rstd = rsqrt(total / (float)D + eps) * scale;
    for (int i = (int)lane; i < D; i += (int)nt) {
        const float v = (float)xr[i] * rstd;
        yr[i] = (half)(has_weight != 0 ? v * (float)w[i] : v);
    }
}

// out[i] = attn_out[i] * sigmoid(qg[head * 2D + D + d])  (attention output gate)
kernel void gate_mul(
    device const half* a   [[buffer(0)]],
    device const half* qg  [[buffer(1)]],
    device half*       out [[buffer(2)]],
    constant int&      D   [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    const int head = (int)i / D;
    const int d = (int)i % D;
    const float g = (float)qg[head * 2 * D + D + d];
    out[i] = (half)((float)a[i] / (1.0f + exp(-g)));
}

// Round-through-bfloat16: the reference (mlx-lm) runs this checkpoint in bf16,
// so its residual stream carries ~8 mantissa bits.  Used to reproduce that
// numerics exactly; a no-op for pure fp16 arithmetic.
kernel void round_bf16(
    device const half* x [[buffer(0)]],
    device half*       y [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    uint bits = as_type<uint>((float)x[i]);
    bits += 0x8000u;
    y[i] = (half)as_type<float>(bits & 0xFFFF0000u);
}

// dst[dst_off + i] = src[src_off + i]
kernel void copy_off(
    device const half* src [[buffer(0)]],
    device half*       dst [[buffer(1)]],
    constant int&      n [[buffer(2)]],
    constant int&      src_off [[buffer(3)]],
    constant int&      dst_off [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    dst[dst_off + i] = src[src_off + i];
}

// RMSNorm with weight, multiplied by silu(gate): the delta net's output norm.
kernel void rmsnorm_gated(
    device const half* x      [[buffer(0)]],
    device const half* w      [[buffer(1)]],
    device const half* gate   [[buffer(2)]],
    device half*       y      [[buffer(3)]],
    constant int&      D      [[buffer(4)]],
    constant float&    eps    [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint nt   [[threads_per_threadgroup]])
{
    device const half* xr = x + (size_t)row * D;
    device const half* gr = gate + (size_t)row * D;
    device half*       yr = y + (size_t)row * D;
    float ss = 0.0f;
    for (int i = (int)lane; i < D; i += (int)nt) {
        const float v = (float)xr[i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    threadgroup float red[32];
    const uint sg = lane / 32, sl = lane % 32;
    if (sl == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (nt + 31) / 32;
    for (uint i = 0; i < nsg; ++i) total += red[i];
    const float rstd = rsqrt(total / (float)D + eps);
    for (int i = (int)lane; i < D; i += (int)nt) {
        const float g = (float)gr[i];
        const float s = g / (1.0f + exp(-g));
        yr[i] = (half)(((float)xr[i] * rstd) * (float)w[i] * s);
    }
}

// out = a * sigmoid(b)   (attention output gate)
kernel void sigmoid_mul(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device half*       out [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    out[i] = (half)((float)a[i] / (1.0f + exp(-(float)b[i])));
}
"#;

pub const K_ATTN_SCORES_SOFTMAX: &str = "attn_scores_softmax";
pub const K_ATTN_OUT: &str = "attn_out";
pub const K_KV_APPEND: &str = "kv_append";
pub const K_CONV1D_SILU_RING: &str = "conv1d_silu_ring";
pub const K_CONV1D_SILU_RING_TILE: &str = "conv1d_silu_ring_tile";
pub const K_RMSNORM_TILE: &str = "rmsnorm_nw_tile";
pub const K_CONV1D_SILU: &str = "conv1d_silu";
pub const K_GDN_STEP: &str = "gdn_step";
pub const K_RMSNORM_WS: &str = "rmsnorm_s";
pub const K_RMSNORM_NW: &str = "rmsnorm_s";
pub const K_GATE_MUL: &str = "gate_mul";
pub const K_COPY: &str = "copy_off";
pub const K_ROUND_BF16: &str = "round_bf16";
pub const K_RMSNORM_GATED: &str = "rmsnorm_gated";
pub const K_SIGMOID_MUL: &str = "sigmoid_mul";
