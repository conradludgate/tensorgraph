use std::marker::PhantomData;

use cust::error::CudaResult;
use cust_raw::CUstream;

use super::ToCudaResult;

pub struct Stream {
    inner: CUstream,
}

impl Stream {
    pub fn new() -> CudaResult<Self> {
        let mut stream = std::ptr::null_mut();

        unsafe { cust_raw::cuStreamCreateWithPriority(&mut stream, 0, 0).to_cuda_result()? }

        Ok(Self { inner: stream })
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
        if self.inner.is_null() {
            return;
        }

        unsafe {
            let inner = std::mem::replace(&mut self.inner, std::ptr::null_mut());
            cust_raw::cuStreamDestroy_v2(inner);
        }
    }
}

#[derive(Clone, Copy)]
pub struct SharedStream<'a> {
    pub(crate) inner: CUstream,
    _marker: PhantomData<&'a Stream>,
}
