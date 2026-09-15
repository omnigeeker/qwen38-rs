//! Weight loading and quantisation metadata.

pub mod safetensors;

pub use safetensors::{Shard, TensorHandle, TensorInfo, WeightStore};

/// MLX-style affine quantisation parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantSpec {
    pub group_size: usize,
    pub bits: usize,
}

impl QuantSpec {
    pub const MLX_4BIT: Self = Self {
        group_size: 64,
        bits: 4,
    };

    pub fn values_per_word(&self) -> usize {
        32 / self.bits
    }

    pub fn bytes_for(&self, numel: usize) -> usize {
        numel * self.bits / 8
    }

    pub fn groups(&self, k: usize) -> usize {
        k / self.group_size
    }
}

/// Dequantise one row of affine 4-bit weights on the CPU (test oracle).
pub fn dequantize_row(q: &[u8], scales: &[half::f16], biases: &[half::f16], spec: QuantSpec) -> Vec<f32> {
    let mut out = Vec::with_capacity(q.len());
    for (i, v) in q.iter().enumerate() {
        let g = i / spec.group_size;
        let s = scales[g].to_f32();
        let b = biases[g].to_f32();
        out.push(*v as f32 * s + b);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn dequant_matches_affine_formula() {
        let spec = QuantSpec::MLX_4BIT;
        let q: Vec<u8> = (0..128).map(|i| (i % 16) as u8).collect();
        let scales = vec![f16::from_f32(0.5); 2];
        let biases = vec![f16::from_f32(-1.0); 2];
        let d = dequantize_row(&q, &scales, &biases, spec);
        assert_eq!(d.len(), 128);
        assert!((d[0] - -1.0).abs() < 1e-6);
        assert!((d[15] - (15.0 * 0.5 - 1.0)).abs() < 1e-6);
        assert!((d[64] - -1.0).abs() < 1e-6);
    }

    #[test]
    fn spec_sizes() {
        let s = QuantSpec::MLX_4BIT;
        assert_eq!(s.values_per_word(), 8);
        assert_eq!(s.bytes_for(1024), 512);
        assert_eq!(s.groups(5120), 80);
    }
}
