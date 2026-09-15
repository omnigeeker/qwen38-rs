//! GPU bring-up tests: runtime MSL JIT must work without the Xcode Metal
//! toolchain, and the 4-bit affine GEMV must match a CPU reference.

use half::f16;
use qw_metal::{msl, Dispatch, GpuDevice};

/// MLX-style packing: 8 unsigned 4-bit values per u32, little-endian nibble order.
fn pack_q4(values: &[u8]) -> Vec<u32> {
    assert!(values.len().is_multiple_of(8));
    values
        .chunks(8)
        .map(|c| {
            let mut w = 0u32;
            for (i, v) in c.iter().enumerate() {
                w |= ((*v as u32) & 0xF) << (4 * i);
            }
            w
        })
        .collect()
}

#[test]
fn gpu_is_available_and_jit_compiles() {
    let dev = GpuDevice::new().expect("Metal device");
    assert!(dev.max_threads_per_threadgroup >= 32);
}

#[test]
fn ewise_add_matches_cpu() {
    let mut dev = GpuDevice::new().unwrap();
    let n = 1024usize;
    let a: Vec<f16> = (0..n).map(|i| f16::from_f32(i as f32 * 0.5)).collect();
    let b: Vec<f16> = (0..n).map(|i| f16::from_f32(i as f32 * 0.25)).collect();
    let ba = dev.buffer_from_bytes(&a);
    let bb = dev.buffer_from_bytes(&b);
    let bo = dev.buffer(n * 2);
    let k = {
        let mut batch = dev.batch();
        batch.kernel(msl::COMMON, msl::K_EWISE_ADD).unwrap()
    };
    {
        let mut batch = dev.batch();
        batch.encode(
            Dispatch::new(&k, (n, 1, 1), (64, 1, 1))
                .buf(0, &ba)
                .buf(1, &bb)
                .buf(2, &bo),
        );
        batch.finish(true);
    }
    let out: Vec<f16> = bo.to_vec(0, n);
    for i in 0..n {
        // compare against the correctly-rounded fp16 result: magnitudes here
        // exceed 512, where the fp16 ULP is 0.5, so an absolute 1e-3 tolerance
        // would be wrong on principle.
        let want = f16::from_f32(a[i].to_f32() + b[i].to_f32()).to_f32();
        assert!(
            (out[i].to_f32() - want).abs() <= 1e-2 * want.abs().max(1.0),
            "mismatch at {i}: {} vs {want}",
            out[i].to_f32()
        );
    }
}

#[test]
fn q4_gemv_matches_dequantized_cpu_reference() {
    let mut dev = GpuDevice::new().unwrap();
    let (rows, k) = (64usize, 512usize); // 8 groups of 64
    let n_groups = k / 64;

    // random 4-bit weights + per-group scales/biases
    let mut rng: u64 = 0x1234_5678_9abc_def0;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f32 / (1u64 << 53) as f32
    };

    let mut q = vec![0u8; rows * k];
    // scales/biases are bf16 in the real checkpoint; store the high half of the
    // f32 so the CPU side can reproduce the exact same value.
    let mut scales = vec![0u16; rows * n_groups];
    let mut biases = vec![0u16; rows * n_groups];
    for r in 0..rows {
        for g in 0..n_groups {
            let s = 0.01 + next() * 0.05;
            let b = -0.5 + next();
            scales[r * n_groups + g] = (s.to_bits() >> 16) as u16;
            biases[r * n_groups + g] = (b.to_bits() >> 16) as u16;
            for i in 0..64 {
                q[r * k + g * 64 + i] = (next() * 16.0) as u8 & 0xF;
            }
        }
    }
    let x: Vec<f16> = (0..k).map(|_| f16::from_f32(-1.0 + next() * 2.0)).collect();
    let packed = pack_q4(&q);

    // CPU reference
    let mut want = vec![0f32; rows];
    for r in 0..rows {
        let mut acc = 0f32;
        for i in 0..k {
            let g = i / 64;
            let sf = f32::from_bits((scales[r * n_groups + g] as u32) << 16);
            let bf = f32::from_bits((biases[r * n_groups + g] as u32) << 16);
            let w = q[r * k + i] as f32 * sf + bf;
            acc += w * x[i].to_f32();
        }
        want[r] = acc;
    }

    let bw = dev.buffer_from_bytes(&packed);
    let bs = dev.buffer_from_bytes(&scales);
    let bb = dev.buffer_from_bytes(&biases);
    let bx = dev.buffer_from_bytes(&x);
    let by = dev.buffer(rows * 2);
    {
        let mut batch = dev.batch();
        let kern = batch.kernel(msl::COMMON, msl::K_Q4_GEMV).unwrap();
        batch.encode(
            Dispatch::new(&kern, (rows * 32, 1, 1), (32, 1, 1))
                .buf(0, &bw)
                .buf(1, &bs)
                .buf(2, &bb)
                .buf(3, &bx)
                .buf(4, &by)
                .scalar(5, k as i32),
        );
        batch.finish(true);
    }
    let got: Vec<f16> = by.to_vec(0, rows);
    for r in 0..rows {
        let (g, w) = (got[r].to_f32(), want[r]);
        let tol = 2e-2 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "row {r}: gpu={g} cpu={w}");
    }
}

/// The engine must be able to hold the whole 4-bit model in unified memory.
#[test]
fn unified_memory_budget_check() {
    let dev = GpuDevice::new().unwrap();
    let d = dev.metal_device();
    let recommended = d.recommended_max_working_set_size();
    println!(
        "device={} recommendedMaxWorkingSet={:.1} GB",
        d.name(),
        recommended as f64 / 1e9
    );
    assert!(recommended > 40 * 1024 * 1024 * 1024);
}

#[test]
fn silu_mul_matches_fp32_reference() {
    let mut dev = GpuDevice::new().unwrap();
    let n = 512usize;
    let gate: Vec<f16> = (0..n)
        .map(|i| f16::from_f32((i as f32 - 256.0) * 0.05))
        .collect();
    let up: Vec<f16> = (0..n)
        .map(|i| f16::from_f32((i as f32 * 0.01).sin()))
        .collect();
    let bg = dev.buffer_from_bytes(&gate);
    let bu = dev.buffer_from_bytes(&up);
    let bo = dev.buffer(n * 2);
    let k = {
        let mut b = dev.batch();
        b.kernel(msl::FUSED, msl::K_SILU_MUL).unwrap()
    };
    {
        let mut b = dev.batch();
        b.encode(
            Dispatch::new(&k, (n, 1, 1), (64, 1, 1))
                .buf(0, &bg)
                .buf(1, &bu)
                .buf(2, &bo),
        );
        b.finish(true);
    }
    let out: Vec<f16> = bo.to_vec(0, n);
    for i in 0..n {
        let g = gate[i].to_f32();
        let want = f16::from_f32((g / (1.0 + (-g as f64).exp() as f32)) * up[i].to_f32()).to_f32();
        assert!(
            (out[i].to_f32() - want).abs() <= 1e-2 * want.abs().max(1.0),
            "i={i} {} vs {want}",
            out[i].to_f32()
        );
    }
}

#[test]
fn rope_partial_matches_reference() {
    let mut dev = GpuDevice::new().unwrap();
    let (heads, hd, rot) = (4usize, 16usize, 8usize);
    let base = 1e7f32;
    let pos = 37i32;
    let x: Vec<f16> = (0..heads * hd)
        .map(|i| f16::from_f32(((i * 7) % 23) as f32 * 0.1 - 1.0))
        .collect();
    let bx = dev.buffer_from_bytes(&x);
    let by = dev.buffer(heads * hd * 2);
    let k = {
        let mut b = dev.batch();
        b.kernel(msl::FUSED, msl::K_ROPE_PARTIAL).unwrap()
    };
    {
        let mut b = dev.batch();
        b.encode(
            Dispatch::new(&k, (heads * 32, 1, 1), (32, 1, 1))
                .buf(0, &bx)
                .buf(1, &by)
                .scalar(2, heads as i32)
                .scalar(3, hd as i32)
                .scalar(4, rot as i32)
                .scalar(5, base)
                .scalar(6, pos),
        );
        b.finish(true);
    }
    let out: Vec<f16> = by.to_vec(0, heads * hd);
    let half_rot = rot / 2;
    for h in 0..heads {
        for i in 0..hd {
            let got = out[h * hd + i].to_f32();
            if i >= rot {
                assert!(
                    (got - x[h * hd + i].to_f32()).abs() < 1e-3,
                    "tail must pass through"
                );
                continue;
            }
            let (a, bq) = if i < half_rot {
                (x[h * hd + i].to_f32(), x[h * hd + i + half_rot].to_f32())
            } else {
                (x[h * hd + i - half_rot].to_f32(), x[h * hd + i].to_f32())
            };
            let theta = base.powf(-2.0 * ((i % half_rot) as f32) / rot as f32);
            let ang = pos as f32 * theta;
            let want = if i < half_rot {
                a * ang.cos() - bq * ang.sin()
            } else {
                a * ang.sin() + bq * ang.cos()
            };
            assert!(
                (got - want).abs() <= 2e-2 * want.abs().max(1.0),
                "h={h} i={i}: {got} vs {want}"
            );
        }
    }
}

/// The multi-token GEMV must produce, for every token, exactly what the
/// single-token kernel produces — it only changes how often weights are read.
#[test]
fn q4_gemv_k_matches_single_token_kernel() {
    let mut dev = GpuDevice::new().unwrap();
    let (rows, k_in, k) = (48usize, 512usize, 3usize);
    let n_groups = k_in / 64;
    let mut rng: u64 = 0xdead_beef_1234_5678;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f32 / (1u64 << 53) as f32
    };

    let mut q = vec![0u8; rows * k_in];
    let mut scales = vec![0u16; rows * n_groups];
    let mut biases = vec![0u16; rows * n_groups];
    for r in 0..rows {
        for g in 0..n_groups {
            scales[r * n_groups + g] = ((0.01 + next() * 0.05).to_bits() >> 16) as u16;
            biases[r * n_groups + g] = ((-0.5 + next()).to_bits() >> 16) as u16;
            for i in 0..64 {
                q[r * k_in + g * 64 + i] = (next() * 16.0) as u8 & 0xF;
            }
        }
    }
    let packed = pack_q4(&q);
    let xs: Vec<f16> = (0..k * k_in)
        .map(|_| f16::from_f32(-1.0 + next() * 2.0))
        .collect();

    // single-token kernel, one call per token
    let bw = dev.buffer_from_bytes(&packed);
    let bs = dev.buffer_from_bytes(&scales);
    let bb = dev.buffer_from_bytes(&biases);
    let mut want = Vec::new();
    for t in 0..k {
        let bx = dev.buffer_from_bytes(&xs[t * k_in..(t + 1) * k_in]);
        let by = dev.buffer(rows * 2);
        let mut batch = dev.batch();
        let kern = batch.kernel(msl::COMMON, msl::K_Q4_GEMV).unwrap();
        batch.encode(
            Dispatch::new(&kern, (rows * 32, 1, 1), (32, 1, 1))
                .buf(0, &bw)
                .buf(1, &bs)
                .buf(2, &bb)
                .buf(3, &bx)
                .buf(4, &by)
                .scalar(5, k_in as i32),
        );
        batch.finish(true);
        let got: Vec<f16> = by.to_vec(0, rows);
        want.push(got);
    }

    // multi-token kernel
    let bx = dev.buffer_from_bytes(&xs);
    let by = dev.buffer(k * rows * 2);
    let mut batch = dev.batch();
    let kern = batch.kernel(msl::COMMON, msl::K_Q4_GEMV_K).unwrap();
    batch.encode(
        Dispatch::new(&kern, (rows * 32, 1, 1), (32, 1, 1))
            .buf(0, &bw)
            .buf(1, &bs)
            .buf(2, &bb)
            .buf(3, &bx)
            .buf(4, &by)
            .scalar(5, k_in as i32)
            .scalar(6, k as i32)
            .scalar(7, rows as i32),
    );
    batch.finish(true);
    let got: Vec<f16> = by.to_vec(0, k * rows);

    for t in 0..k {
        for r in 0..rows {
            let a = got[t * rows + r].to_f32();
            let b = want[t][r].to_f32();
            assert!(
                (a - b).abs() <= 1e-3 * b.abs().max(1.0),
                "token {t} row {r}: multi={a} single={b}"
            );
        }
    }
}
