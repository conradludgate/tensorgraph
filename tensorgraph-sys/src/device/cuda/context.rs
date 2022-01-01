use std::{marker::PhantomData, ptr::NonNull};

use cust::error::CudaResult;
use cust_raw::CUcontext;

use crate::device::cuda::ToCudaResult;

pub struct Context {
    inner: NonNull<cust_raw::CUctx_st>,
}

impl Context {
    /// Shortcut for initializing the CUDA Driver API and creating a CUDA context with default settings
    /// for the first device.
    #[must_use = "The CUDA Context must be kept alive or errors will be issued for any CUDA function that is run"]
    pub fn quick_init() -> CudaResult<Self> {
        use cust::context::ContextFlags;

        cust::init(cust::CudaFlags::empty())?;
        let device = cust::device::Device::get_device(0)?;

        let flags = ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO;

        unsafe {
            let mut ctx: CUcontext = std::ptr::null_mut();
            cust_raw::cuCtxCreate_v2(&mut ctx as *mut CUcontext, flags.bits(), device.as_raw())
                .to_cuda_result()?;
            Ok(Self {
                inner: NonNull::new_unchecked(ctx),
            })
        }
    }

    pub fn share(&self) -> SharedContext {
        SharedContext {
            _inner: self.inner,
            _marker: PhantomData,
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            cust_raw::cuCtxDestroy_v2(self.inner.as_mut());
        }
    }
}

#[derive(Clone, Copy)]
pub struct SharedContext<'a> {
    _inner: NonNull<cust_raw::CUctx_st>,
    _marker: PhantomData<&'a Context>,
}
