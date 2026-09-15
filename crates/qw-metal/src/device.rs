//! Device + pipeline cache.

use crate::buffer::GpuBuffer;
use anyhow::{anyhow, Result};
use metal::{CompileOptions, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize};
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Thin wrapper around a Metal device, its command queue and a hot pipeline cache.
pub struct GpuDevice {
    device: Device,
    queue: metal::CommandQueue,
    /// Compiled MSL libraries keyed by a hash of their source.
    libraries: FxHashMap<u64, Library>,
    pipelines: FxHashMap<(u64, String), Arc<ComputePipelineState>>,
    pub max_threads_per_threadgroup: usize,
}

impl GpuDevice {
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow!("no Metal device found"))?;
        let queue = device.new_command_queue();
        let max_threads = device.max_threads_per_threadgroup().width as usize;
        let _ = &max_threads;
        tracing::info!(
            "metal device: {} (unified memory: {}, max threads/tg: {})",
            device.name(),
            device.has_unified_memory(),
            max_threads
        );
        Ok(Self {
            device,
            queue,
            libraries: FxHashMap::default(),
            pipelines: FxHashMap::default(),
            max_threads_per_threadgroup: max_threads,
        })
    }

    pub fn metal_device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &metal::CommandQueue {
        &self.queue
    }

    /// Start a new command batch (one command buffer, one compute encoder).
    pub fn batch(&mut self) -> crate::kernel::CommandBatch<'_> {
        crate::kernel::CommandBatch::new(self)
    }

    /// JIT-compile an MSL source string (cached by content hash).
    pub fn library(&mut self, source: &str) -> Result<Library> {
        let key = fxhash(source.as_bytes());
        if let Some(lib) = self.libraries.get(&key) {
            return Ok(lib.clone());
        }
        let opts = CompileOptions::new();
        let lib = self
            .device
            .new_library_with_source(source, &opts)
            .map_err(|e| anyhow!("MSL compile failed: {e}"))?;
        self.libraries.insert(key, lib.clone());
        Ok(lib)
    }

    /// Compile `source` and return a ready-to-dispatch pipeline for `entry`.
    pub fn pipeline(&mut self, source: &str, entry: &str) -> Result<Arc<ComputePipelineState>> {
        let key = (fxhash(source.as_bytes()), entry.to_string());
        if let Some(p) = self.pipelines.get(&key) {
            return Ok(p.clone());
        }
        let lib = self.library(source)?;
        let func = lib
            .get_function(entry, None)
            .map_err(|e| anyhow!("kernel `{entry}` not found in MSL source: {e}"))?;
        let state = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow!("failed to build pipeline for `{entry}`: {e}"))?;
        let state = Arc::new(state);
        self.pipelines.insert(key, state.clone());
        Ok(state)
    }

    /// Allocate a shared-memory buffer of `len` bytes.
    pub fn buffer(&self, len: usize) -> GpuBuffer {
        let buf = self
            .device
            .new_buffer(len.max(16) as u64, MTLResourceOptions::StorageModeShared);
        GpuBuffer::from_metal(buf)
    }

    /// Wrap an existing (page-aligned) host pointer as a Metal buffer without
    /// copying.  Returns `None` if the driver rejects the pointer.
    pub fn buffer_no_copy(&self, ptr: *const std::ffi::c_void, len: usize) -> Option<GpuBuffer> {
        let buf = self.device.new_buffer_with_bytes_no_copy(
            ptr,
            len as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        if buf.length() == 0 {
            None
        } else {
            Some(GpuBuffer::from_metal(buf))
        }
    }

    /// Allocate a buffer and copy `data` into it.
    pub fn buffer_from_bytes<T: Copy>(&self, data: &[T]) -> GpuBuffer {
        let bytes = std::mem::size_of_val(data);
        let buf = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            bytes.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        GpuBuffer::from_metal(buf)
    }
}

/// FNV-1a, good enough for content-addressed kernel caching.
pub fn fxhash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// Convenience: dispatch size helper.
pub fn grid(width: usize, height: usize, depth: usize) -> MTLSize {
    MTLSize {
        width: width as u64,
        height: height as u64,
        depth: depth as u64,
    }
}
