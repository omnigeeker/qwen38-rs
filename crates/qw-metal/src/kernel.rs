//! Kernel dispatch: one command buffer per forward step, one compute encoder,
//! explicit encoder boundaries where a kernel consumes another kernel's output.
//!
//! Metal serialises dispatches inside one compute encoder, but the memory model
//! only guarantees ordering across encoder boundaries.  Dependencies therefore
//! go through `barrier()`, which closes the current encoder and opens a fresh
//! one (cheap: no command buffer round-trip).

use crate::device::GpuDevice;
use crate::GpuBuffer;
use anyhow::Result;
use metal::{ComputeCommandEncoder, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::sync::Arc;

/// A compiled entry point, identified by (source, name). Cheap to clone.
#[derive(Clone)]
pub struct Kernel {
    pub name: String,
    pub pipeline: Arc<ComputePipelineState>,
}

impl Kernel {
    pub fn thread_execution_width(&self) -> usize {
        self.pipeline.thread_execution_width() as usize
    }
    pub fn max_total_threads(&self) -> usize {
        self.pipeline.max_total_threads_per_threadgroup() as usize
    }
}

/// Parameters for a single dispatch.
pub struct Dispatch<'a> {
    pub kernel: &'a Kernel,
    pub buffers: Vec<(usize, &'a GpuBuffer, usize)>,
    pub constants: Vec<(usize, Vec<u8>)>,
    pub grid: MTLSize,
    pub threadgroup: MTLSize,
}

fn size3(t: (usize, usize, usize)) -> MTLSize {
    MTLSize {
        width: t.0 as u64,
        height: t.1 as u64,
        depth: t.2 as u64,
    }
}

impl<'a> Dispatch<'a> {
    pub fn new(kernel: &'a Kernel, grid: (usize, usize, usize), tg: (usize, usize, usize)) -> Self {
        Self {
            kernel,
            buffers: Vec::new(),
            constants: Vec::new(),
            grid: size3(grid),
            threadgroup: size3(tg),
        }
    }

    pub fn buf(mut self, index: usize, b: &'a GpuBuffer) -> Self {
        self.buffers.push((index, b, 0));
        self
    }

    /// Bind a buffer at a byte offset (used by the zero-copy weight store, where
    /// every tensor lives inside a shard-sized buffer).
    pub fn buf_offset(mut self, index: usize, b: &'a GpuBuffer, offset: usize) -> Self {
        self.buffers.push((index, b, offset));
        self
    }

    /// Scalar argument (`constant T&` in MSL).
    pub fn scalar<T: Copy>(mut self, index: usize, v: T) -> Self {
        let bytes = unsafe {
            std::slice::from_raw_parts(&v as *const T as *const u8, std::mem::size_of::<T>())
        };
        self.constants.push((index, bytes.to_vec()));
        self
    }
}

/// A batch of dispatches encoded into one command buffer.
pub struct CommandBatch<'d> {
    dev: &'d mut GpuDevice,
    cb: metal::CommandBuffer,
    enc: Option<ComputeCommandEncoder>,
    count: usize,
    /// Dispatches per entry point, printed when QW_DISPATCH_HIST is set.
    hist: Vec<(String, usize)>,
}

impl<'d> CommandBatch<'d> {
    pub fn new(dev: &'d mut GpuDevice) -> Self {
        let cb = dev.queue().new_command_buffer().to_owned();
        Self {
            dev,
            cb,
            enc: None,
            count: 0,
            hist: Vec::new(),
        }
    }

    /// Compile `source`/`entry` (cached) and return a Kernel handle.
    pub fn kernel(&mut self, source: &str, entry: &str) -> Result<Kernel> {
        let pipeline = self.dev.pipeline(source, entry)?;
        Ok(Kernel {
            name: entry.to_string(),
            pipeline,
        })
    }

    fn encoder(&mut self) -> &ComputeCommandEncoderRef {
        if self.enc.is_none() {
            self.enc = Some(self.cb.new_compute_command_encoder().to_owned());
        }
        self.enc.as_ref().unwrap()
    }

    pub fn encode(&mut self, d: Dispatch<'_>) -> &mut Self {
        let enc = self.encoder();
        enc.set_compute_pipeline_state(&d.kernel.pipeline);
        for (i, b, off) in &d.buffers {
            enc.set_buffer(*i as u64, Some(b.as_metal()), *off as u64);
        }
        for (i, bytes) in &d.constants {
            enc.set_bytes(*i as u64, bytes.len() as u64, bytes.as_ptr() as *const _);
        }
        enc.dispatch_threads(d.grid, d.threadgroup);
        match self.hist.iter_mut().find(|(n, _)| n == &d.kernel.name) {
            Some((_, c)) => *c += 1,
            None => self.hist.push((d.kernel.name.clone(), 1)),
        }
        self.count += 1;
        self
    }

    /// Close the current encoder so the next dispatch observes prior writes.
    pub fn barrier(&mut self) -> &mut Self {
        if let Some(enc) = self.enc.take() {
            enc.end_encoding();
        }
        self
    }

    pub fn dispatches(&self) -> usize {
        self.count
    }

    /// Commit and (optionally) block until the GPU is done.
    pub fn finish(mut self, wait: bool) {
        // A full-model pass is a chain of ~1300 dispatches and the small
        // elementwise ops dominate the count, so knowing who they are is the
        // first step to fusing them.
        if std::env::var_os("QW_DISPATCH_HIST").is_some() && self.count > 400 {
            let mut v = std::mem::take(&mut self.hist);
            v.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
            eprintln!("dispatch histogram, {} total:", self.count);
            for (n, c) in v.iter().take(24) {
                eprintln!(
                    "  {n:22} {c:5}  ({:.0}%)",
                    100.0 * *c as f64 / self.count as f64
                );
            }
        }
        if let Some(enc) = self.enc.take() {
            enc.end_encoding();
        }
        self.cb.commit();
        if wait {
            self.cb.wait_until_completed();
        }
    }
}
