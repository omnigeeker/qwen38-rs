//! GPU bring-up tests: runtime MSL JIT must work without the Xcode Metal
//! toolchain, and the 4-bit affine GEMV must match a CPU reference.

use half::f16;
use qw_metal::{msl, Dispatch, GpuBuffer, GpuDevice};

/// MLX-style packing: 8 unsigned 4-bit values per u32, little-endian nibble order.
fn pack_q4(values: &[u8]) -> Vec<u32> {
    assert!(values.len() % 8 == 0);
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
    let mut scales = vec![f16::ZERO; rows * n_groups];
    let mut biases = vec![f16::ZERO; rows * n_groups];
    for r in 0..rows {
        for g in 0..n_groups {
            let s = 0.01 + next() * 0.05;
            let b = -0.5 + next();
            scales[r * n_groups + g] = f16::from_f32(s);
            biases[r * n_groups + g] = f16::from_f32(b);
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
            let w = q[r * k + i] as f32 * scales[r * n_groups + g].to_f32()
                + biases[r * n_groups + g].to_f32();
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
