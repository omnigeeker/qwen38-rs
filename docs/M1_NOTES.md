# M1 implementation notes — exact reference math

Transcribed from `mlx_lm/models/qwen3_5.py` + `qwen3_next.py` + `gated_delta.py`
(the correctness oracle). Kernel work must follow this literally.

## Decoder layer (all 64)

```
is_linear(i) = (i + 1) % 4 != 0            -> 48 gated-delta-net layers
                                            16 full-attention layers (i = 3, 7, ... 63)

h = x + sublayer(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))
final: norm(h) -> lm_head
```

Tensors: `input_layernorm.weight`, `post_attention_layernorm.weight`,
`mlp.{gate,up,down}_proj.*`, `linear_attn.*` or `self_attn.*`.

### Full attention (`Qwen3NextAttention`)

```
q_gate = W_q x            # [L, 24, 512]  (num_attention_heads * head_dim * 2 = 12288 total)
queries, gate = split(q_gate, 2, axis=-1)     # both [L, 24, 256]
gate = gate.reshape(L, 24*256)                # 3072
keys  = W_k x             # [L, 4, 256]   (4 KV heads)
values= W_v x             # [L, 4, 256]

queries = q_norm(queries)     # RMSNorm over head_dim=256, eps 1e-6, weight +1 shifted
keys    = k_norm(keys)
queries = rope(queries, offset=pos)   # dims = head_dim * partial_rotary_factor = 64
keys    = rope(keys,    offset=pos)   # traditional = False -> half-split pairs
out = SDPA(queries, keys, values, scale = 256^-0.5 = 1/16)   # GQA 24 q-heads : 4 kv-heads
out = out.reshape(L, 24*256)
r   = W_o (out * sigmoid(gate))
```

**Gotcha**: `q_proj` is twice as wide as the query count because of the output
gate; verified against the checkpoint (`q_proj` shape is `[12288, 5120]` for the
full-attention layers).

### MLP

```
r = W_down( swiglu(W_gate x, W_up x) )
swiglu(g, u) = silu_fp32(g) * fp32(u)   # reference computes this in fp32 ("_precise_swiglu")
```

### Gated delta net (`GatedDeltaNet`, 48 layers)

```
qkv = W_qkv x            # in_proj_qkv -> key_dim*2 + value_dim
                           key_dim   = 16 * 128 = 2048
                           value_dim = 48 * 128 = 6144
                           conv_dim  = 4096 + 6144 = 10240
z   = W_z x              # in_proj_z -> value_dim (6144), reshaped [L, 48, 128]
b   = W_b x              # -> 48
a   = W_a x              # -> 48

# depthwise causal conv1d over the LAST conv_kernel_dim=4 positions, silu after
conv_state = last 3 rows of qkv (carried in cache)
conv_in    = concat(conv_state, qkv)                 # [L+3, 10240]
conv_out   = silu(conv1d(conv_in))                   # groups = 10240, kernel = 4

q, k, v = split(conv_out, [2048, 4096], -1) reshaped
          q:[L,16,128]  k:[L,16,128]  v:[L,48,128]
inv_scale = 128^-0.5
q = inv_scale^2 * rms_norm(q, weight=None, eps=1e-6)   # note: no learned weight
k = inv_scale   * rms_norm(k, weight=None, eps=1e-6)

beta = sigmoid(b)                                       # [L, 48]
g    = exp(-exp(A_log) * softplus(a + dt_bias))         # [L, 48], fp32
(y, state) = gated_delta_update(q, k, v, g, beta, state)   # state fp32 [48,128,128]

y = out_proj( RMSNormGated(y, z) )   # rms_norm(y, weight=norm.weight) then * silu_fp32(z)
```

Recurrence per timestep (state `S` is `[48, 128(v-dim), 128(k-dim)]`, fp32):

```
S      = S * g_t
kv_mem = S @ k_t                      # over k-dim
delta  = (v_t - kv_mem) * beta_t
S      = S + outer(delta, k_t)
y_t    = S @ q_t                      # over k-dim
```

`A_log` and `dt_bias` are fp32 in the checkpoint (`cast_predicate` keeps `A_log`
uncast); `dt_bias` is initialised to ones.

### MTP block (1 layer, bf16 in the official release)

```
h_prev, emb_next = ...
x = W_fc( concat( pre_fc_norm_hidden(h_prev), pre_fc_norm_embedding(emb_next) ) )   # 2*hidden -> hidden
x = decoder_layer_full_attn(x)          # mtp.layers.0.*, weights +1 shifted norms
logits = lm_head(mtp.norm(x))
```

## Norm convention — measured, not assumed

Metric check on the **actual** shards (mean / min / max of each norm vector):

| tensor | repo | mean | min | verdict |
|---|---|---|---|---|
| `model.norm.weight` | 4-bit | +1.944 | 0.715 | use as stored |
| `layers.0.input_layernorm.weight` | 4-bit | +0.967 | 0.867 | use as stored |
| `layers.0.post_attention_layernorm.weight` | 4-bit | +0.783 | 0.004 | use as stored |
| `layers.0.linear_attn.norm.weight` | 4-bit | +0.869 | 0.785 | use as stored |
| `layers.3.self_attn.q_norm.weight` | 4-bit | +1.230 | 0.824 | use as stored |
| `mtp.layers.0.input_layernorm.weight` | bf16 | +0.036 | -0.226 | **needs +1.0** |
| `mtp.layers.0.post_attention_layernorm.weight` | bf16 | +0.206 | -0.162 | **needs +1.0** |
| `mtp.layers.0.self_attn.q_norm.weight` | bf16 | +0.791 | -0.555 | **needs +1.0** |
| `mtp.norm.weight` | bf16 | +1.252 | -0.225 | use as stored |

Why: `sanitize()` shifts norm weights only when the checkpoint carries `mtp.*`
keys or an un-sanitised conv1d. The MLX 4-bit export already ran sanitize (its
`conv1d.weight` is `[10240, 4, 1]`, i.e. moved-axis, and `mtp.*` was stripped), so
re-loading it shifts **nothing** — confirmed against the measured means above.
The official bf16 repo still carries `mtp.*`, so its `input_layernorm`,
`post_attention_layernorm` and `q_norm` **are** shifted by +1 on load, while
`mtp.norm.weight` is *not* (it does not match any suffix in the reference's
pattern list). We replicate the reference exactly; MTP only affects acceptance
rate, never output correctness.


## Kernel status for M1

| kernel | state |
|---|---|
| `q4_gemv` (all projections) | done, validated on real weights |
| `rmsnorm` (weighted) | done |
| `silu_mul` (fp32 intermediate) | done, tested |
| `rope_partial` (half-split, pos offset) | done, tested |
| `rmsnorm_noweight` (gdn q/k) | next |
| `conv1d_depthwise_silu` (k=4, groups=10240) | next |
| `gdn_step` (fp32 state recurrence) | next |
| `attn_decode` (GQA + softmax over cached K/V) | next |
| `rmsnorm_gated` (gdn output) | next |
| `sigmoid_gate_mul` (attention output gate) | next |
| `add_residual` | use `ewise_add` |
