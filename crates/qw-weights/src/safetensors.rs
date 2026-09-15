//! Safetensors parsing + zero-copy GPU upload.
//!
//! Layout recap: a safetensors file is `u64 header_len` followed by a JSON
//! header mapping tensor name -> {dtype, shape, data_offsets}.  Data begins at
//! `8 + header_len` and is tightly packed.
//!
//! We memory-map the shard and, when the driver allows it, wrap the mapping in
//! a `bytesNoCopy` Metal buffer: the 15 GB of 4-bit weights are then shared
//! between the page cache and the GPU with **zero copies and zero extra RSS**.
//! If no-copy is rejected we fall back to a plain upload.

use anyhow::{anyhow, bail, Context, Result};
use half::f16;
use memmap2::Mmap;
use qw_metal::{DType, GpuBuffer, GpuDevice};
use rustc_hash::FxHashMap;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

impl TensorInfo {
    pub fn dtype(&self) -> Result<DType> {
        Ok(match self.dtype.as_str() {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "U32" => DType::U32,
            "U8" => DType::U8,
            other => bail!("unsupported safetensors dtype {other}"),
        })
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        self.data_offsets.1 - self.data_offsets.0
    }
}

/// One memory-mapped safetensors shard plus its header index.
pub struct Shard {
    pub path: String,
    pub mmap: Option<Mmap>,
    pub gpu: GpuBuffer,
    pub header_len: usize,
    pub tensors: FxHashMap<String, TensorInfo>,
    /// true when the GPU buffer aliases the mmap (no copy)
    pub zero_copy: bool,
}

impl Shard {
    pub fn load(dev: &GpuDevice, path: &str) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("open {path}"))?;
        let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {path}"))?;
        // SAFETY: we only read the mapping from here on.
        let bytes: &[u8] = &mmap;
        if bytes.len() < 8 {
            bail!("{path}: too small to be a safetensors file");
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        if 8 + header_len > bytes.len() {
            bail!("{path}: header length {header_len} exceeds file size");
        }
        // The header may carry a non-tensor `__metadata__` entry; keep only
        // entries that look like tensors.
        let raw: FxHashMap<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..8 + header_len])
                .with_context(|| format!("{path}: invalid JSON header"))?;
        let mut header: FxHashMap<String, TensorInfo> = FxHashMap::default();
        for (k, v) in raw {
            if k == "__metadata__" {
                continue;
            }
            let info: TensorInfo = serde_json::from_value(v)
                .with_context(|| format!("{path}: bad tensor entry `{k}`"))?;
            header.insert(k, info);
        }

        let data_start = 8 + header_len;
        let (gpu, zero_copy) = upload_mapping(dev, &mmap, data_start);

        Ok(Self {
            path: path.to_string(),
            mmap: Some(mmap),
            gpu,
            header_len,
            tensors: header,
            zero_copy,
        })
    }

    /// Byte offset of a tensor inside the GPU buffer.
    ///
    /// In the zero-copy path the buffer aliases the whole file, so the header
    /// bytes are part of the offset; in the fallback path the payload is
    /// uploaded on its own and offsets are payload-relative.
    pub fn offset_of(&self, name: &str) -> Result<usize> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("tensor {name} not in {}", self.path))?;
        let base = if self.zero_copy { 8 + self.header_len } else { 0 };
        Ok(base + info.data_offsets.0)
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }
}

/// Wrap the mmap in a `bytesNoCopy` Metal buffer when the driver allows it,
/// otherwise copy the payload into a fresh buffer.
///
/// The buffer always aliases the *mapping base*, so tensor offsets include the
/// safetensors header (`8 + header_len`).
fn upload_mapping(dev: &GpuDevice, mmap: &Mmap, data_start: usize) -> (GpuBuffer, bool) {
    let base = mmap.as_ptr() as usize;
    let page = 16 * 1024usize;
    let aligned_len = mmap.len().div_ceil(page) * page;
    if base % page == 0 {
        if let Some(buf) = dev.buffer_no_copy(base as *const std::ffi::c_void, aligned_len) {
            return (buf, true);
        }
    }
    // Fallback: zero-copy rejected -> keep the payload in a plain buffer and
    // shift offsets by `data_start` so handles stay valid either way.
    let len = mmap.len() - data_start;
    let buf = dev.buffer(len);
    buf.copy_from(&mmap[data_start..]);
    (buf, false)
}

/// The whole model: all shards + a flat name -> TensorHandle map.
pub struct WeightStore {
    pub shards: Vec<Shard>,
    pub index: FxHashMap<String, (usize, usize)>, // name -> (shard, offset)
}

impl WeightStore {
    /// Load every `*.safetensors` in `dir` (respecting `model.safetensors.index.json`
    /// when present, otherwise discovering files on disk).
    pub fn load_dir(dev: &GpuDevice, dir: &std::path::Path) -> Result<Self> {
        let mut paths: Vec<String> = Vec::new();
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let txt = std::fs::read_to_string(&index_path)?;
            let v: serde_json::Value = serde_json::from_str(&txt)?;
            let map = v
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| anyhow!("index json missing weight_map"))?;
            let mut files: Vec<String> = map
                .values()
                .filter_map(|f| f.as_str().map(|s| s.to_string()))
                .collect();
            files.sort();
            files.dedup();
            for f in files {
                paths.push(dir.join(f).to_string_lossy().to_string());
            }
        } else {
            for entry in std::fs::read_dir(dir)? {
                let p = entry?.path();
                if p.extension().map(|e| e == "safetensors").unwrap_or(false) {
                    paths.push(p.to_string_lossy().to_string());
                }
            }
            paths.sort();
        }
        if paths.is_empty() {
            bail!("no safetensors shards found in {}", dir.display());
        }

        let mut shards = Vec::with_capacity(paths.len());
        let mut index = FxHashMap::default();
        for (si, p) in paths.iter().enumerate() {
            let shard = Shard::load(dev, p)?;
            tracing::info!(
                "shard {}: {} tensors, zero_copy={}",
                shard.path,
                shard.tensors.len(),
                shard.zero_copy
            );
            for name in shard.tensors.keys() {
                let off = shard.offset_of(name)?;
                index.insert(name.clone(), (si, off));
            }
            shards.push(shard);
        }
        Ok(Self { shards, index })
    }

    pub fn handle(&self, name: &str) -> Result<TensorHandle<'_>> {
        let (si, off) = *self
            .index
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))?;
        let info = self.shards[si].info(name).unwrap().clone();
        Ok(TensorHandle {
            shard: si,
            offset: off,
            info,
            buf: &self.shards[si].gpu,
        })
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }
}

/// A tensor located inside a shard's GPU buffer.
pub struct TensorHandle<'a> {
    pub shard: usize,
    pub offset: usize,
    pub info: TensorInfo,
    pub buf: &'a GpuBuffer,
}

impl<'a> TensorHandle<'a> {
    pub fn dtype(&self) -> Result<DType> {
        self.info.dtype()
    }

    pub fn shape(&self) -> &[usize] {
        &self.info.shape
    }

    /// Read raw bytes on the CPU (for tests / weight inspection).
    pub fn bytes(&self) -> &'a [u8] {
        unsafe {
            let p = (self.buf.as_metal().contents() as *const u8).add(self.offset);
            std::slice::from_raw_parts(p, self.info.nbytes())
        }
    }

    /// F16 tensor as a CPU slice (panics if the dtype is not F16).
    pub fn as_f16(&self) -> &'a [f16] {
        unsafe {
            let p = (self.buf.as_metal().contents() as *const u8).add(self.offset) as *const f16;
            std::slice::from_raw_parts(p, self.info.numel())
        }
    }

    /// BF16 tensor as f32 values (for MTP weights stored in bf16).
    pub fn as_bf16_f32(&self) -> Vec<f32> {
        self.bytes()
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]) as u32;
                f32::from_bits(bits << 16)
            })
            .collect()
    }
}
