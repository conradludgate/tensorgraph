use std::marker::PhantomData;

use cust::error::CudaResult;
use cust_raw::CUcontext;

use crate::device::cuda::ToCudaResult;

pub struct Context {
    inner: CUcontext,
}

impl Context {
    /// Shortcut for initializing the CUDA Driver API and creating a CUDA context with default settings
    /// for the first device.
    ///
    /// **You must keep this context alive while you do further operations or you will get an InvalidContext
    /// error**. e.g. using `let _ctx = quick_init()?;`.
    ///
    /// This is useful for testing or just setting up a basic CUDA context quickly. Users with more
    /// complex needs (multiple devices, custom flags, etc.) should use `init` and create their own
    /// context.
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
            Ok(Self { inner: ctx })
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
        if self.inner.is_null() {
            return;
        }

        unsafe {
            let inner = std::mem::replace(&mut self.inner, std::ptr::null_mut());
            cust_raw::cuCtxDestroy_v2(inner);
        }
    }
}

#[derive(Clone, Copy)]
pub struct SharedContext<'a> {
    _inner: CUcontext,
    _marker: PhantomData<&'a Context>,
}
