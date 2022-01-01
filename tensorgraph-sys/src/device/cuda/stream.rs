use std::{marker::PhantomData, ptr::NonNull};

use cust::error::CudaResult;
use cust_raw::{CUstream, CUstream_st};

use super::ToCudaResult;

pub struct Stream {
    inner: NonNull<CUstream_st>,
}

impl Stream {
    pub fn new() -> CudaResult<Self> {
        let mut stream = std::ptr::null_mut();

        unsafe {
            cust_raw::cuStreamCreateWithPriority(&mut stream, 0, 0).to_cuda_result()?;

            Ok(Self {
                inner: NonNull::new_unchecked(stream),
            })
        }
    }

    pub fn share(&self) -> SharedStream {
        SharedStream {
            inner: self.inner,
            _marker: PhantomData,
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            cust_raw::cuStreamDestroy_v2(self.inner.as_mut());
        }
    }
}

#[derive(Clone, Copy)]
pub struct SharedStream<'a> {
    pub(crate) inner: NonNull<CUstream_st>,
    _marker: PhantomData<&'a Stream>,
}

impl<'a> SharedStream<'a> {
    pub(crate) unsafe fn inner(mut self) -> CUstream {
        self.inner.as_mut()
    }
}
