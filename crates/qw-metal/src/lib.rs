//! Runtime Metal compute layer for qwen38.
//!
//! Design notes
//! ------------
//! * All kernels are written in MSL and compiled **at runtime** through
//!   `MTLDevice::newLibraryWithSource`.  This is deliberately the only way we
//!   touch the GPU: the machine this engine targets only has the Command Line
//!   Tools installed, so `xcrun metal` (offline shader compilation) is *not*
//!   available.  Runtime JIT is the same mechanism `mlx.fast.metal_kernel`
//!   uses, and it is verified to work on macOS 26.6.
//! * One `MTLCommandQueue` for the whole process; one command buffer per
//!   forward step (per token during decode) so we do not pay per-op sync.
//! * Buffers are storage-mode shared so the CPU can read results without a
//!   blit, but the fast path never round-trips: sampling happens on the GPU.

pub mod buffer;
pub mod device;
pub mod kernel;
pub mod msl;

pub use buffer::{DType, GpuBuffer};
pub use device::GpuDevice;
pub use kernel::{CommandBatch, Dispatch, Kernel};
