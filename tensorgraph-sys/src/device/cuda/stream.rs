use std::{marker::PhantomData, ops::Deref, ptr::NonNull};

use cust::error::CudaResult;
use cust_raw::{CUstream, CUstream_st};

use super::{SharedContext, ToCudaResult};

/// Represents an owned CUDA Stream.
pub struct Stream<'ctx> {
    inner: NonNull<CUstream_st>,
    _marker: PhantomData<&'ctx SharedContext>,
}

impl<'ctx> Stream<'ctx> {
    /// Create a new CUDA Stream
    pub fn new(_ctx: &'ctx SharedContext) -> CudaResult<Self> {
        let mut stream = std::ptr::null_mut();

        unsafe {
            cust_raw::cuStreamCreate(&mut stream, 0).to_cuda_result()?;

            Ok(Self {
                inner: NonNull::new_unchecked(stream),
                _marker: PhantomData,
            })
        }
    }
}

impl<'ctx> Deref for Stream<'ctx> {
    type Target = SharedStream;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self.inner.as_ptr() as *const _) }
    }
}

impl<'ctx> Drop for Stream<'ctx> {
    fn drop(&mut self) {
        unsafe {
            cust_raw::cuStreamDestroy_v2(self.inner.as_mut());
        }
    }
}

/// A Shared CUDA Stream. Created through [`Deref`] from [`Stream`].
/// Is a DeviceAllocator for Cuda
pub struct SharedStream(CUstream_st);

impl SharedStream {
    pub fn inner(&self) -> CUstream {
        self as *const _ as *mut _
    }
}
