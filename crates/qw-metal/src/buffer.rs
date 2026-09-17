//! GPU buffer helpers.

use metal::Buffer;
use std::sync::Arc;

/// Reference-counted Metal buffer with typed, checked CPU views.
#[derive(Clone)]
pub struct GpuBuffer {
    pub(crate) buf: Buffer,
    pub len: usize,
    pub label: Option<String>,
}

impl GpuBuffer {
    pub fn from_metal(buf: Buffer) -> Self {
        let len = buf.length() as usize;
        Self {
            buf,
            len,
            label: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        let l = label.into();
        self.buf.set_label(&l);
        self.label = Some(l);
        self
    }

    pub fn as_metal(&self) -> &Buffer {
        &self.buf
    }

    pub fn len_bytes(&self) -> usize {
        self.len
    }

    /// Zero-copy CPU slice (storage mode shared).
    ///
    /// # Safety
    /// Caller must guarantee the GPU is not concurrently writing this buffer
    /// (i.e. the last command buffer touching it has completed).
    pub unsafe fn as_slice<T: Copy>(&self) -> &[T] {
        let n = self.len / std::mem::size_of::<T>();
        std::slice::from_raw_parts(self.buf.contents() as *const T, n)
    }

    /// Copy `src` into this buffer at byte offset 0.
    pub fn copy_from<T: Copy>(&self, src: &[T]) {
        let n = std::mem::size_of_val(src).min(self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr() as *const u8,
                self.buf.contents() as *mut u8,
                n,
            );
        }
    }

    /// Copy `bytes` into this buffer starting at `byte_offset`.
    ///
    /// `copy_from` can only write from offset 0, which is useless for restoring
    /// one slot's slice of a batched buffer.  Metal backing here is CPU-visible
    /// (`to_vec` already reads it directly), so this is a plain memcpy.
    pub fn write_at(&self, byte_offset: usize, bytes: &[u8]) {
        if byte_offset >= self.len {
            return;
        }
        let n = bytes.len().min(self.len - byte_offset);
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (self.buf.contents() as *mut u8).add(byte_offset),
                n,
            );
        }
    }

    /// Copy `n` bytes out starting at `byte_offset`.
    pub fn read_at(&self, byte_offset: usize, n: usize) -> Vec<u8> {
        let n = n.min(self.len.saturating_sub(byte_offset));
        let mut v = vec![0u8; n];
        if n > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (self.buf.contents() as *const u8).add(byte_offset),
                    v.as_mut_ptr(),
                    n,
                );
            }
        }
        v
    }

    /// Copy out `n` elements of `T` starting at `offset` elements.
    pub fn to_vec<T: Copy>(&self, offset: usize, n: usize) -> Vec<T> {
        let mut v = vec![unsafe { std::mem::zeroed() }; n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (self.buf.contents() as *const u8).add(offset * std::mem::size_of::<T>()),
                v.as_mut_ptr() as *mut u8,
                n * std::mem::size_of::<T>(),
            );
        }
        v
    }
}

/// Shared wrapper used by the weight store: keeps mmap'd pages alive.
pub type SharedBuffer = Arc<GpuBuffer>;

/// Bytes-per-element helper for the dtypes we care about.
pub fn bytes_of(dtype: DType) -> usize {
    match dtype {
        DType::F32 => 4,
        DType::F16 | DType::BF16 => 2,
        DType::U32 => 4,
        DType::U8 => 1,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    U32,
    U8,
}
