//! Quantised linear layers bound directly to the zero-copy weight store.

use anyhow::{anyhow, bail, Result};
use qw_metal::{msl, CommandBatch, Dispatch, GpuBuffer, Kernel};
use qw_weights::{QuantSpec, TensorHandle, WeightStore};

/// An affine 4-bit `nn.Linear` (no bias), resolved to GPU buffer offsets.
pub struct QLinear<'a> {
    pub name: String,
    pub weight: TensorHandle<'a>,
    pub scales: TensorHandle<'a>,
    pub biases: TensorHandle<'a>,
    pub out_f: usize,
    pub in_f: usize,
    pub spec: QuantSpec,
}

impl<'a> QLinear<'a> {
    /// `weight_name` is the `.weight` key; `scales`/`biases` are derived.
    pub fn from_store(store: &'a WeightStore, weight_name: &str) -> Result<Self> {
        if !weight_name.ends_with(".weight") {
            bail!("expected a .weight name, got {weight_name}");
        }
        let stem = &weight_name[..weight_name.len() - ".weight".len()];
        let weight = store.handle(weight_name)?;
        let scales = store.handle(&format!("{stem}.scales"))?;
        let biases = store.handle(&format!("{stem}.biases"))?;

        let wshape = weight.shape();
        if wshape.len() != 2 {
            bail!("{weight_name}: expected 2-D packed weight, got {wshape:?}");
        }
        let out_f = wshape[0];
        let in_f = wshape[1] * QuantSpec::MLX_4BIT.values_per_word();
        let sshape = scales.shape();
        if sshape != [out_f, in_f / QuantSpec::MLX_4BIT.group_size] {
            bail!(
                "{weight_name}: scales shape {sshape:?} != [{out_f}, {}]",
                in_f / QuantSpec::MLX_4BIT.group_size
            );
        }
        Ok(Self {
            name: stem.to_string(),
            weight,
            scales,
            biases,
            out_f,
            in_f,
            spec: QuantSpec::MLX_4BIT,
        })
    }

    pub fn weight_bytes(&self) -> usize {
        self.spec.bytes_for(self.out_f * self.in_f)
            + self.out_f * (self.in_f / self.spec.group_size) * 4
    }

    /// Check that every pointer the kernel will load from is suitably aligned
    /// for 16-byte `half4` / 4-byte `uint` vector loads.
    pub fn alignment_ok(&self) -> bool {
        self.weight.offset.is_multiple_of(4)
            && self.scales.offset.is_multiple_of(8)
            && self.biases.offset.is_multiple_of(8)
    }

    /// Encode `y = W x` for a single token (`x` is `[in_f]` fp16, `y` is `[out_f]`).
    pub fn encode(&self, batch: &mut CommandBatch, kernel: &Kernel, x: &GpuBuffer, y: &GpuBuffer) {
        let d = Dispatch::new(kernel, (self.out_f * 32, 1, 1), (32, 1, 1))
            .buf_offset(0, self.weight.buf, self.weight.offset)
            .buf_offset(1, self.scales.buf, self.scales.offset)
            .buf_offset(2, self.biases.buf, self.biases.offset)
            .buf(3, x)
            .buf(4, y)
            .scalar(5, self.in_f as i32);
        batch.encode(d);
    }

    /// Compile (or fetch) the GEMV entry point.
    pub fn kernel(batch: &mut CommandBatch) -> Result<Kernel> {
        batch.kernel(msl::COMMON, msl::K_Q4_GEMV)
    }

    /// CPU reference: dequantise this layer's real weights and multiply.
    pub fn cpu_reference(&self, x: &[half::f16]) -> Result<Vec<f32>> {
        if x.len() != self.in_f {
            bail!("x has {} elements, expected {}", x.len(), self.in_f);
        }
        let packed: &[u8] = self.weight.bytes();
        let scales: &[half::f16] = self.scales.as_f16();
        let biases: &[half::f16] = self.biases.as_f16();
        let words: &[u32] = unsafe {
            std::slice::from_raw_parts(
                packed.as_ptr() as *const u32,
                packed.len() / std::mem::size_of::<u32>(),
            )
        };
        if words.len() < self.out_f * (self.in_f / 8) {
            return Err(anyhow!(
                "{}: packed weight too small ({} words)",
                self.name,
                words.len()
            ));
        }
        let xf: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
        let mut out = vec![0f32; self.out_f];
        let groups = self.in_f / self.spec.group_size;
        for (row, o) in out.iter_mut().enumerate() {
            let wrow = &words[row * (self.in_f / 8)..(row + 1) * (self.in_f / 8)];
            let srow = &scales[row * groups..(row + 1) * groups];
            let brow = &biases[row * groups..(row + 1) * groups];
            let mut acc = 0f32;
            for (wi, word) in wrow.iter().enumerate() {
                for v in 0..8 {
                    let idx = wi * 8 + v;
                    let q = ((word >> (4 * v)) & 0xF) as f32;
                    let w = q * srow[idx / self.spec.group_size].to_f32()
                        + brow[idx / self.spec.group_size].to_f32();
                    acc += w * xf[idx];
                }
            }
            *o = acc;
        }
        Ok(out)
    }
}
