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
    // Eight independent accumulators rather than one.  A single `acc` makes the whole
    // T sweep one dependent FMA chain, so only one load is ever in flight per thread
    // and the kernel runs latency-bound: measured 1.236 ms at T = 6000 for ~72 MB,
    // about 58 GB/s, far under what L2 can deliver.  Splitting the chain lets eight
    // V loads overlap.  The summation order changes, which moves the fp32 result by
    // far less than the bf16 the residual stream is rounded to anyway; the parity
    // gates are what decide whether that is acceptable.
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    int t = 0;
    for (; t + 8 <= T; t += 8) {
        acc0 += p[t + 0] * (float)vbase[(size_t)(t + 0) * D];
        acc1 += p[t + 1] * (float)vbase[(size_t)(t + 1) * D];
        acc2 += p[t + 2] * (float)vbase[(size_t)(t + 2) * D];
        acc3 += p[t + 3] * (float)vbase[(size_t)(t + 3) * D];
        acc4 += p[t + 4] * (float)vbase[(size_t)(t + 4) * D];
        acc5 += p[t + 5] * (float)vbase[(size_t)(t + 5) * D];
        acc6 += p[t + 6] * (float)vbase[(size_t)(t + 6) * D];
        acc7 += p[t + 7] * (float)vbase[(size_t)(t + 7) * D];
    }
    for (; t < T; ++t) acc0 += p[t] * (float)vbase[(size_t)t * D];
    out[(size_t)h * D + d] =
        (half)(((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7)));
}

// Append one token's K/V into the head-major cache.
kernel void kv_append_rows(
    device const half*  k      [[buffer(0)]],
    device const half*  v      [[buffer(1)]],
    device half*        kcache [[buffer(2)]],
    device half*        vcache [[buffer(3)]],
    constant int&       pos0   [[buffer(4)]],
    constant int&       maxT   [[buffer(5)]],
    constant int&       Hkv    [[buffer(6)]],
    constant int&       D      [[buffer(7)]],
    // The k and v scratch rows are NOT the same length: scratch.k is shared with
    // the delta-net path and is laid out with key_dim = Hk*Dk = 2048 elements per
    // token, while the attention k this kernel copies is only Hkv*D = 1024 of
    // them; scratch.v is exactly Hkv*D.  Reading k[i] with the v row length
    // silently walks into the next token, which is what broke the gates.
    constant int&       krow   [[buffer(8)]],
    constant int&       vrow   [[buffer(9)]],
    uint i [[thread_position_in_grid]])
{
    const int per_row = Hkv * D;
    const int token = (int)i / per_row;
    const int j = (int)i - token * per_row;
    const int hk = j / D;
    const int d  = j - hk * D;
    const size_t dst = ((size_t)hk * maxT + pos0 + token) * D + d;
    kcache[dst] = k[(size_t)token * krow + j];
    vcache[dst] = v[(size_t)token * vrow + j];
}

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
// ---------------------------------------------------------------------------
// Row-batched attention.
//
// The prefill path emitted one `attn_scores_softmax` and one `attn_out` per
// query token per full-attention layer: 2 x 16 x 598 = 19,136 of the 20,290
// dispatches in a 602-token pass, 94% of it.  These two kernels do the whole
// pass in one dispatch each by giving every threadgroup its own (query row,
// head) pair.
//
// Row `r` attends to keys `0 .. pos0+r`, so the causal mask is nothing more
// than a shorter loop bound - no mask value and no wasted work.  That is only
// valid when the pass carries one sequence with consecutive positions, which is
// exactly the case the caller checks for before choosing this path.
// ---------------------------------------------------------------------------
kernel void attn_scores_softmax_rows(
    device const half*  q      [[buffer(0)]],   // [rows][H*D]
    device const half*  kcache [[buffer(1)]],   // [Hkv][maxT][D]
    device float*       scores [[buffer(2)]],   // [rows][H][maxT]
    constant int&       maxT   [[buffer(3)]],
    constant int&       H      [[buffer(4)]],
    constant int&       Hkv    [[buffer(5)]],
    constant int&       D      [[buffer(6)]],
    constant float&     scale  [[buffer(7)]],
    constant int&       pos0   [[buffer(8)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint nt   [[threads_per_threadgroup]])
{
    const int row = (int)gid / H;
    const int h   = (int)gid - row * H;
    const int T   = pos0 + row + 1;
    const int reps = H / Hkv;
    const int hk = h / reps;
    const uint sg = lane / 32;
    const uint sl = lane % 32;
    const int nwarp = (int)(nt / 32);
    device const half* qh = q + ((size_t)row * H + (size_t)h) * D;
    device const half* kbase = kcache + (size_t)hk * maxT * D;
    device float* sc = scores + ((size_t)row * H + (size_t)h) * maxT;

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

// Row-batched `attn_out`: one threadgroup per (query row, head), one thread per
// dim.  Eight accumulators for the same reason as the single-row kernel - one
// chain leaves a single V load in flight and the sweep runs latency-bound.
kernel void attn_out_rows(
    device const float* probs  [[buffer(0)]],   // [rows][H][maxT]
    device const half*  vcache [[buffer(1)]],   // [Hkv][maxT][D]
    device half*        out    [[buffer(2)]],   // [rows][H*D]
    constant int&       maxT   [[buffer(3)]],
    constant int&       H      [[buffer(4)]],
    constant int&       Hkv    [[buffer(5)]],
    constant int&       D      [[buffer(6)]],
    constant int&       pos0   [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint d   [[thread_index_in_threadgroup]])
{
    const int row = (int)gid / H;
    const int h   = (int)gid - row * H;
    const int T   = pos0 + row + 1;
    const int reps = H / Hkv;
    const int hk = h / reps;
    device const float* p = probs + ((size_t)row * H + (size_t)h) * maxT;
    device const half* vbase = vcache + (size_t)hk * maxT * D + d;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    int t = 0;
    for (; t + 8 <= T; t += 8) {
        acc0 += p[t + 0] * (float)vbase[(size_t)(t + 0) * D];
        acc1 += p[t + 1] * (float)vbase[(size_t)(t + 1) * D];
        acc2 += p[t + 2] * (float)vbase[(size_t)(t + 2) * D];
        acc3 += p[t + 3] * (float)vbase[(size_t)(t + 3) * D];
        acc4 += p[t + 4] * (float)vbase[(size_t)(t + 4) * D];
        acc5 += p[t + 5] * (float)vbase[(size_t)(t + 5) * D];
        acc6 += p[t + 6] * (float)vbase[(size_t)(t + 6) * D];
        acc7 += p[t + 7] * (float)vbase[(size_t)(t + 7) * D];
    }
    for (; t < T; ++t) acc0 += p[t] * (float)vbase[(size_t)t * D];
    out[((size_t)row * H + (size_t)h) * D + d] =
        (half)(((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7)));
}

"#;

pub const GDN: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Register-resident GDN scan tile.  A threadgroup covers DVQ delta-net values
// and splits each value's Dk reduction across Q lanes, so one thread owns
// Dk/Q state elements in registers for the whole token loop.  Rewritten from
// the environment by `gdn_src()` so a tile sweep costs one process start.
#define QW_GDN_Q   8
#define QW_GDN_DVQ 16
#define QW_GDN_DKS 16

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
    device half*        window   [[buffer(0)]],
    device const half*  w        [[buffer(1)]],
    device half*        out      [[buffer(2)]],
    constant int&       conv_dim [[buffer(3)]],
    constant int&       slot0    [[buffer(4)]],
    constant int&       ring     [[buffer(5)]],
    device const half*  cur      [[buffer(6)]],
    constant int&       pos0     [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    const int c   = (int)(gid % (uint)conv_dim);
    const int row = (int)(gid / (uint)conv_dim);
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        // Rows of this pass are still in the staging buffer; older ones are in the
        // ring.  Reading the current row from `cur` is what lets the projection be
        // a single tiled launch instead of one launch per row.
        const int pos = pos0 + row - 3 + j;
        device const half* src = (pos >= pos0)
            ? cur + (size_t)(pos - pos0) * conv_dim + c
            : window + (size_t)(pos & (ring - 1)) * conv_dim + c;
        acc += (float)w[c * 4 + j] * (float)(*src);
    }
    out[(size_t)row * conv_dim + c] = (half)(acc / (1.0f + exp(-acc)));
    // Fold the ring update in: the row of this pass that this thread just convolved
    // is exactly the raw row the next pass needs in the ring, so writing it here
    // removes a copy launch per row (and the read that copy would have done).  No
    // thread can read the slot it writes - every window read in this dispatch is for
    // pos < pos0, i.e. at least three slots behind the ones written here.
    window[(size_t)((slot0 + row) & (ring - 1)) * conv_dim + c] =
        cur[(size_t)row * conv_dim + c];
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
    device float*       snap    [[buffer(13)]],  // [TILE][Hv * Dv * Dk] fp32
    constant int&       snap_on [[buffer(14)]],
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
    // The speculative rewind needs the state as of the end of this row.  Those
    // values are already in registers here, so writing them out costs one store per
    // element and saves a separate whole-state copy launch (plus the read it did).
    if (snap_on) {
        device float* SP = snap + ((size_t)hv * Dv + dv) * Dk;
        for (int d = 0; d < Dk; ++d) SP[d] = S[d];
    }
}

// The gdn recurrence with the token loop fused into the kernel.  Every thread
// owns one (head, value column) state slice and touches nothing another thread
// touches, so consecutive tokens need no barrier at all - and the caller stops
// paying one dispatch plus one command-buffer barrier per token.  A 128-token
// prefill pass went from 6,144 launches and 6,144 barriers to 48 launches and
// none.  The per-token strides are in `half` elements; the snapshot stride is in
// bytes because it is applied through a char* so its unit cannot be misread.
kernel void gdn_step_seq(
    device const half*  q       [[buffer(0)]],
    device const half*  k       [[buffer(1)]],
    device const half*  v       [[buffer(2)]],
    device const half*  a       [[buffer(3)]],
    device const half*  b       [[buffer(4)]],
    device const float* A_log   [[buffer(5)]],
    device const float* dt_bias [[buffer(6)]],
    device float*       state   [[buffer(7)]],
    device half*        y       [[buffer(8)]],
    constant int&       Hk      [[buffer(9)]],
    constant int&       Hv      [[buffer(10)]],
    constant int&       Dk      [[buffer(11)]],
    constant int&       Dv      [[buffer(12)]],
    device float*       snap    [[buffer(13)]],
    constant int&       snap_on [[buffer(14)]],
    constant int&       n       [[buffer(15)]],
    constant int&       s_q     [[buffer(16)]],
    constant int&       s_k     [[buffer(17)]],
    constant int&       s_v     [[buffer(18)]],
    constant int&       s_ab    [[buffer(19)]],
    constant int&       s_y     [[buffer(20)]],
    constant int&       s_snap  [[buffer(21)]],
    uint hv [[threadgroup_position_in_grid]],
    uint dv [[thread_index_in_threadgroup]])
{
    const int reps = Hv / Hk;
    const int hk = (int)hv / reps;
    device float* S = state + ((size_t)hv * Dv + dv) * Dk;
    const float alog = exp(A_log[hv]);
    const float bias = dt_bias[hv];
    for (int t = 0; t < n; ++t) {
        device const half* kt = k + (size_t)t * s_k;
        device const half* qt = q + (size_t)t * s_q;
        device const half* vt = v + (size_t)t * s_v;
        const float x = (float)a[(size_t)t * s_ab + hv] + bias;
        const float sp = max(x, 0.0f) + log(1.0f + exp(-fabs(x)));
        const float g = exp(-alog * sp);
        const float beta = 1.0f / (1.0f + exp(-(float)b[(size_t)t * s_ab + hv]));
        device const half* kp = kt + (size_t)hk * Dk;
        device const half* qp = qt + (size_t)hk * Dk;
        float kv = 0.0f;
        for (int d = 0; d < Dk; ++d) {
            S[d] *= g;
            kv += S[d] * (float)kp[d];
        }
        const float delta = ((float)vt[(size_t)hv * Dv + dv] - kv) * beta;
        float acc = 0.0f;
        for (int d = 0; d < Dk; ++d) {
            S[d] += delta * (float)kp[d];
            acc += S[d] * (float)qp[d];
        }
        y[(size_t)t * s_y + (size_t)hv * Dv + dv] = (half)acc;
        if (snap_on) {
            device float* SP = (device float*)((device char*)snap + (size_t)t * (size_t)s_snap)
                               + ((size_t)hv * Dv + dv) * Dk;
            for (int d = 0; d < Dk; ++d) SP[d] = S[d];
        }
    }
}

// Same recurrence as `gdn_step_seq`, but with the state in registers.
//
// The old kernel gave every (hv, dv) pair its own thread and kept S in device
// memory, so each token read the state twice and wrote it twice: 384 bytes read
// and 256 written per thread per token.  Over a 602-token pass that is 303 GB,
// and at 481 GB/s it accounts for the entire 0.63 s the scan costs - which is
// 43% of everything the pass spends outside the matmul.
//
// Here a threadgroup covers QW_GDN_DVQ values and splits each value's Dk
// reduction across QW_GDN_Q lanes, so a thread holds Dk/Q state elements in
// registers and never touches device memory inside the loop.  The two dot
// products become butterfly reductions across the Q lanes, which sit in one
// aligned group of consecutive lanes, so three `simd_shuffle_xor` steps do it.
kernel void gdn_step_seq4(
    device const half*  q       [[buffer(0)]],
    device const half*  k       [[buffer(1)]],
    device const half*  v       [[buffer(2)]],
    device const half*  a       [[buffer(3)]],
    device const half*  b       [[buffer(4)]],
    device const float* A_log   [[buffer(5)]],
    device const float* dt_bias [[buffer(6)]],
    device float*       state   [[buffer(7)]],
    device half*        y       [[buffer(8)]],
    constant int&       Hk      [[buffer(9)]],
    constant int&       Hv      [[buffer(10)]],
    constant int&       Dk      [[buffer(11)]],
    constant int&       Dv      [[buffer(12)]],
    device float*       snap    [[buffer(13)]],
    constant int&       snap_on [[buffer(14)]],
    constant int&       n       [[buffer(15)]],
    constant int&       s_q     [[buffer(16)]],
    constant int&       s_k     [[buffer(17)]],
    constant int&       s_v     [[buffer(18)]],
    constant int&       s_ab    [[buffer(19)]],
    constant int&       s_y     [[buffer(20)]],
    constant int&       s_snap  [[buffer(21)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    constexpr int Q   = QW_GDN_Q;
    constexpr int DVQ = QW_GDN_DVQ;
    // Dk/Q has to be a compile-time array size, so it arrives as a define.
    constexpr int DKS = QW_GDN_DKS;
    const int dvb = (int)tg % (Dv / DVQ);
    const int hv  = (int)tg / (Dv / DVQ);
    const int dv  = dvb * DVQ + (int)(lane / (uint)Q);
    const int qq  = (int)(lane % (uint)Q);
    const int d0  = qq * DKS;
    const int reps = Hv / Hk;
    const int hk = hv / reps;

    device float* Sd = state + ((size_t)hv * Dv + dv) * Dk + d0;
    float S[DKS];
    for (int i = 0; i < DKS; ++i) S[i] = Sd[i];

    const float alog = exp(A_log[hv]);
    const float bias = dt_bias[hv];
    for (int t = 0; t < n; ++t) {
        const float x = (float)a[(size_t)t * s_ab + hv] + bias;
        const float sp = max(x, 0.0f) + log(1.0f + exp(-fabs(x)));
        const float g = exp(-alog * sp);
        const float beta = 1.0f / (1.0f + exp(-(float)b[(size_t)t * s_ab + hv]));
        device const half* kp = k + (size_t)t * s_k + (size_t)hk * Dk + d0;
        device const half* qp = q + (size_t)t * s_q + (size_t)hk * Dk + d0;
        float kv = 0.0f;
        for (int i = 0; i < DKS; ++i) {
            S[i] *= g;
            kv += S[i] * (float)kp[i];
        }
        // butterfly reduction across the Q lanes of this value
        for (int m = 1; m < Q; m <<= 1) kv += simd_shuffle_xor(kv, (ushort)m);
        const float delta = ((float)v[(size_t)t * s_v + (size_t)hv * Dv + dv] - kv) * beta;
        float acc = 0.0f;
        for (int i = 0; i < DKS; ++i) {
            S[i] += delta * (float)kp[i];
            acc += S[i] * (float)qp[i];
        }
        for (int m = 1; m < Q; m <<= 1) acc += simd_shuffle_xor(acc, (ushort)m);
        if (qq == 0) y[(size_t)t * s_y + (size_t)hv * Dv + dv] = (half)acc;
        if (snap_on) {
            device float* SP = (device float*)((device char*)snap + (size_t)t * (size_t)s_snap)
                               + ((size_t)hv * Dv + dv) * Dk + d0;
            for (int i = 0; i < DKS; ++i) SP[i] = S[i];
        }
    }
    for (int i = 0; i < DKS; ++i) Sd[i] = S[i];
}

// RMSNorm over `D` with an explicit input row stride and an output scale.
// `has_weight == 1` multiplies by `w` first. Used for
//   * delta-net q/k:  no weight, scale = inv or inv^2
//   * attention q/k:  with weight, q read at stride 2*head_dim (the projection
//     emits [q | gate] per head)
kernel void rmsnorm_s_rows(
    device const half* x [[buffer(0)]],
    device const half* w [[buffer(1)]],
    device half*       y [[buffer(2)]],
    constant int&      D [[buffer(3)]],
    constant int&      in_stride_bytes [[buffer(4)]],
    constant float&  eps [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant int& has_weight [[buffer(7)]],
    constant int& heads [[buffer(8)]],
    constant int& sx_bytes [[buffer(9)]],
    constant int& sy_bytes [[buffer(10)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint nt   [[threads_per_threadgroup]])
{
    const uint head  = tg % (uint)heads;
    const uint token = tg / (uint)heads;
    device const half* xr = (device const half*)((device char*)x + (size_t)token * sx_bytes
                            + (size_t)head * in_stride_bytes);
    device half*       yr = (device half*)((device char*)y + (size_t)token * sy_bytes
                            + (size_t)head * D * 2);
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
        const float v = (float)xr[i];
        yr[i] = has_weight ? (half)(v * rstd * (float)w[i]) : (half)(v * rstd);
    }
}

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
kernel void gate_mul_rows(
    device const half* a   [[buffer(0)]],
    device const half* qg  [[buffer(1)]],
    device half*       out [[buffer(2)]],
    constant int&      D   [[buffer(3)]],
    constant int&      row_elems [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    const int token = (int)i / row_elems;
    const int j = (int)i - token * row_elems;
    const int head = j / D;
    const int d = j - head * D;
    const float g = (float)qg[token * row_elems * 2 + head * 2 * D + D + d];
    out[i] = (half)((float)a[i] / (1.0f + exp(-g)));
}

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
    // Elements between consecutive rows.  Zero for every caller that launches one
    // row per dispatch, which makes the second term vanish and leaves the address
    // exactly as it was; the prefill path sets it to the row length and puts the
    // row count in grid.y, which turns 6,144 launches per pass into 48.
    constant int&      stride [[buffer(6)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint nt   [[threads_per_threadgroup]])
{
    // MSL insists that every input declaration be the same vector width, so the
    // grid stays one-dimensional and the row index is decomposed here instead:
    // `stride` is the elements between tokens and D the elements per tile, so
    // tiles = stride/D and tg splits into (token, tile).  stride == 0 reproduces
    // the original one-row-per-dispatch mapping exactly.
    size_t row;
    if (stride > 0) {
        const uint tiles = (uint)stride / (uint)D;
        row = (size_t)(tg / tiles) * (size_t)stride + (size_t)(tg % tiles) * (size_t)D;
    } else {
        row = (size_t)tg * (size_t)D;
    }
    device const half* xr = x + row;
    device const half* gr = gate + row;
    device half*       yr = y + row;
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
pub const K_ATTN_SCORES_SOFTMAX_ROWS: &str = "attn_scores_softmax_rows";
pub const K_ATTN_OUT_ROWS: &str = "attn_out_rows";
pub const K_ATTN_OUT: &str = "attn_out";
pub const K_KV_APPEND: &str = "kv_append";
pub const K_KV_APPEND_ROWS: &str = "kv_append_rows";
pub const K_CONV1D_SILU_RING: &str = "conv1d_silu_ring";
pub const K_CONV1D_SILU_RING_TILE: &str = "conv1d_silu_ring_tile";
pub const K_RMSNORM_TILE: &str = "rmsnorm_nw_tile";
pub const K_CONV1D_SILU: &str = "conv1d_silu";
pub const K_GDN_STEP: &str = "gdn_step";
pub const K_GDN_STEP_SEQ: &str = "gdn_step_seq";
pub const K_GDN_STEP_SEQ4: &str = "gdn_step_seq4";
pub const K_RMSNORM_WS: &str = "rmsnorm_s";
pub const K_RMSNORM_WS_ROWS: &str = "rmsnorm_s_rows";
pub const K_RMSNORM_NW: &str = "rmsnorm_s";
pub const K_GATE_MUL: &str = "gate_mul";
pub const K_GATE_MUL_ROWS: &str = "gate_mul_rows";
pub const K_COPY: &str = "copy_off";
pub const K_ROUND_BF16: &str = "round_bf16";
pub const K_RMSNORM_GATED: &str = "rmsnorm_gated";
pub const K_SIGMOID_MUL: &str = "sigmoid_mul";

/// (Q, DVQ) for `gdn_step_seq4`, overridable from the environment so the tile
/// can be swept without a rebuild.
pub fn gdn_tiles() -> [i32; 2] {
    static T: std::sync::OnceLock<[i32; 2]> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        fn one(k: &str, d: i32) -> i32 {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        }
        [one("QW_GDN_Q", 8), one("QW_GDN_DVQ", 16)]
    })
}

/// The GDN source with the `gdn_step_seq4` tile substituted in.  The pipeline
/// cache is keyed on the source text, so each tile is its own pipeline.
pub fn gdn_src(dks: i32) -> String {
    static S: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        let [q, dvq] = gdn_tiles();
        GDN.replace("#define QW_GDN_Q   8", &format!("#define QW_GDN_Q   {q}"))
            .replace("#define QW_GDN_DVQ 16", &format!("#define QW_GDN_DVQ {dvq}"))
            .replace("#define QW_GDN_DKS 16", &format!("#define QW_GDN_DKS {dks}"))
    })
    .clone()
}
