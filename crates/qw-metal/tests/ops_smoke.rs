//! Attention / gated-delta-net kernel tests against CPU references.
//!
//! These kernels carry the model's sequential state, so a mistake here shows up
//! as slow drift rather than an obvious crash — every one of them is checked
//! against a literal transcription of the mlx-lm reference.

use half::f16;
use qw_metal::{msl_ops, Dispatch, GpuDevice};

fn f(v: f32) -> f16 {
    f16::from_f32(v)
}

/// deterministic pseudo-random stream
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

#[test]
fn conv1d_silu_matches_reference() {
    let mut dev = GpuDevice::new().unwrap();
    let conv_dim = 64usize;
    let mut rng = Rng::new(0x1234);
    // window rows: 3 carried + current
    let window: Vec<f16> = (0..4 * conv_dim).map(|_| f(rng.next_f32())).collect();
    let w: Vec<f16> = (0..conv_dim * 4).map(|_| f(rng.next_f32() * 0.5)).collect();

    let bw = dev.buffer_from_bytes(&window);
    let bwt = dev.buffer_from_bytes(&w);
    let bo = dev.buffer(conv_dim * 2);
    {
        let mut b = dev.batch();
        let k = b.kernel(msl_ops::GDN, msl_ops::K_CONV1D_SILU).unwrap();
        b.encode(
            Dispatch::new(&k, (conv_dim, 1, 1), (64, 1, 1))
                .buf(0, &bw)
                .buf(1, &bwt)
                .buf(2, &bo)
                .scalar(3, conv_dim as i32),
        );
        b.finish(true);
    }
    let got: Vec<f16> = bo.to_vec(0, conv_dim);
    for c in 0..conv_dim {
        let mut acc = 0f32;
        for j in 0..4 {
            acc += w[c * 4 + j].to_f32() * window[j * conv_dim + c].to_f32();
        }
        let want = acc / (1.0 + (-acc as f64).exp() as f32);
        assert!(
            (got[c].to_f32() - want).abs() <= 2e-2 * want.abs().max(1.0),
            "c={c}: {} vs {want}",
            got[c].to_f32()
        );
    }
}

#[test]
fn gdn_step_matches_reference_recurrence() {
    let mut dev = GpuDevice::new().unwrap();
    let (hk, hv, dk, dv) = (2usize, 6usize, 8usize, 4usize); // hv/hk = 3
    let mut rng = Rng::new(0xbeef);

    let q: Vec<f16> = (0..hk * dk).map(|_| f(rng.next_f32() * 0.3)).collect();
    let k: Vec<f16> = (0..hk * dk).map(|_| f(rng.next_f32() * 0.3)).collect();
    let a: Vec<f16> = (0..hv).map(|_| f(rng.next_f32())).collect();
    let b: Vec<f16> = (0..hv).map(|_| f(rng.next_f32())).collect();
    let a_log: Vec<f32> = (0..hv).map(|_| rng.next_f32() * 3.0).collect();
    let dt_bias: Vec<f32> = (0..hv).map(|_| rng.next_f32()).collect();

    // Buffers live across steps so the kernel owns the recurrent state, exactly
    // as it will during decoding (no state round-trip through the CPU).
    let bq = dev.buffer_from_bytes(&q);
    let bk = dev.buffer_from_bytes(&k);
    let ba = dev.buffer_from_bytes(&a);
    let bb = dev.buffer_from_bytes(&b);
    let bal = dev.buffer_from_bytes(&a_log);
    let bdt = dev.buffer_from_bytes(&dt_bias);
    let bstate = dev.buffer(hv * dv * dk * 4);
    let by = dev.buffer(hv * dv * 2);

    let mut cpu_state = vec![0f32; hv * dv * dk];
    let steps = 4usize;

    for step in 0..steps {
        let v: Vec<f16> = (0..hv * dv).map(|_| f(rng.next_f32())).collect();
        let bv = dev.buffer_from_bytes(&v);

        // ---- CPU reference ----
        let mut cpu_y = vec![0f32; hv * dv];
        for h in 0..hv {
            let rep = hv / hk;
            let kh = h / rep;
            let x = a[h].to_f32() + dt_bias[h];
            let sp = x.max(0.0) + (1.0f32 + (-x.abs()).exp()).ln();
            let g = (-(a_log[h].exp() * sp)).exp();
            let beta = 1.0 / (1.0 + (-b[h].to_f32()).exp());
            for dvv in 0..dv {
                let base = (h * dv + dvv) * dk;
                for d in 0..dk {
                    cpu_state[base + d] *= g;
                }
                let mut kv = 0f32;
                for d in 0..dk {
                    kv += cpu_state[base + d] * k[kh * dk + d].to_f32();
                }
                let delta = (v[h * dv + dvv].to_f32() - kv) * beta;
                let mut acc = 0f32;
                for d in 0..dk {
                    cpu_state[base + d] += delta * k[kh * dk + d].to_f32();
                    acc += cpu_state[base + d] * q[kh * dk + d].to_f32();
                }
                cpu_y[h * dv + dvv] = acc;
            }
        }

        // ---- GPU step ----
        {
            let mut batch = dev.batch();
            let kern = batch.kernel(msl_ops::GDN, msl_ops::K_GDN_STEP).unwrap();
            batch.encode(
                Dispatch::new(&kern, (hv * dv, 1, 1), (dv, 1, 1))
                    .buf(0, &bq)
                    .buf(1, &bk)
                    .buf(2, &bv)
                    .buf(3, &ba)
                    .buf(4, &bb)
                    .buf(5, &bal)
                    .buf(6, &bdt)
                    .buf(7, &bstate)
                    .buf(8, &by)
                    .scalar(9, hk as i32)
                    .scalar(10, hv as i32)
                    .scalar(11, dk as i32)
                    .scalar(12, dv as i32),
            );
            batch.finish(true);
        }
        let y: Vec<f16> = by.to_vec(0, hv * dv);
        for i in 0..hv * dv {
            let want = cpu_y[i];
            assert!(
                (y[i].to_f32() - want).abs() <= 5e-2 * want.abs().max(1.0),
                "step {step} elem {i}: gpu={} cpu={want}",
                y[i].to_f32()
            );
        }
    }

    // The recurrent state itself must match after all steps.
    let gpu_state: Vec<f32> = bstate.to_vec(0, hv * dv * dk);
    let mx = cpu_state.iter().fold(0f32, |a, b| a.max(b.abs()));
    for i in 0..hv * dv * dk {
        assert!(
            (gpu_state[i] - cpu_state[i]).abs() <= 5e-2 * mx.max(1.0),
            "state elem {i}: gpu={} cpu={}",
            gpu_state[i],
            cpu_state[i]
        );
    }
}

#[test]
fn attention_scores_softmax_and_out_match_reference() {
    let mut dev = GpuDevice::new().unwrap();
    let (h, hkv, d, max_t) = (6usize, 2usize, 8usize, 5usize);
    let t = 4usize; // 4 cached tokens
    let scale = 1.0 / (d as f32).sqrt();
    let mut rng = Rng::new(0xfeed);

    let q: Vec<f16> = (0..h * d).map(|_| f(rng.next_f32())).collect();
    let mut kcache = vec![f16::ZERO; hkv * max_t * d];
    let mut vcache = vec![f16::ZERO; hkv * max_t * d];
    for hk in 0..hkv {
        for ti in 0..t {
            for dd in 0..d {
                kcache[(hk * max_t + ti) * d + dd] = f(rng.next_f32());
                vcache[(hk * max_t + ti) * d + dd] = f(rng.next_f32());
            }
        }
    }

    let bq = dev.buffer_from_bytes(&q);
    let bk = dev.buffer_from_bytes(&kcache);
    let bv = dev.buffer_from_bytes(&vcache);
    let bscores = dev.buffer(h * max_t * 4);
    let bout = dev.buffer(h * d * 2);
    {
        let mut b = dev.batch();
        let k1 = b
            .kernel(msl_ops::ATTN, msl_ops::K_ATTN_SCORES_SOFTMAX)
            .unwrap();
        b.encode(
            Dispatch::new(&k1, (h * 256, 1, 1), (256, 1, 1))
                .buf(0, &bq)
                .buf(1, &bk)
                .buf(2, &bscores)
                .scalar(3, t as i32)
                .scalar(4, max_t as i32)
                .scalar(5, h as i32)
                .scalar(6, hkv as i32)
                .scalar(7, d as i32)
                .scalar(8, scale),
        );
        b.barrier();
        let k2 = b.kernel(msl_ops::ATTN, msl_ops::K_ATTN_OUT).unwrap();
        b.encode(
            Dispatch::new(&k2, (h * d, 1, 1), (d, 1, 1))
                .buf(0, &bscores)
                .buf(1, &bv)
                .buf(2, &bout)
                .scalar(3, t as i32)
                .scalar(4, max_t as i32)
                .scalar(5, h as i32)
                .scalar(6, hkv as i32)
                .scalar(7, d as i32),
        );
        b.finish(true);
    }

    let probs: Vec<f32> = bscores.to_vec(0, h * max_t);
    let out: Vec<f16> = bout.to_vec(0, h * d);

    let reps = h / hkv;
    for head in 0..h {
        let hk = head / reps;
        // reference scores + softmax
        let mut sc = vec![0f32; t];
        for ti in 0..t {
            let mut acc = 0f32;
            for dd in 0..d {
                acc += q[head * d + dd].to_f32() * kcache[(hk * max_t + ti) * d + dd].to_f32();
            }
            sc[ti] = acc * scale;
        }
        let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ex: Vec<f32> = sc.iter().map(|v| (v - mx).exp()).collect();
        let sum: f32 = ex.iter().sum();
        let p: Vec<f32> = ex.iter().map(|v| v / sum).collect();

        for ti in 0..t {
            let got = probs[head * max_t + ti];
            assert!(
                (got - p[ti]).abs() < 2e-3,
                "head {head} t {ti}: prob {got} vs {}",
                p[ti]
            );
        }
        for dd in 0..d {
            let mut want = 0f32;
            for ti in 0..t {
                want += p[ti] * vcache[(hk * max_t + ti) * d + dd].to_f32();
            }
            let got = out[head * d + dd].to_f32();
            assert!(
                (got - want).abs() <= 5e-2 * want.abs().max(1.0),
                "head {head} d {dd}: out {got} vs {want}"
            );
        }
    }
}
