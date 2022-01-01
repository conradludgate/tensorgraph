use std::{ops::Deref, ptr::NonNull};

use cust::error::CudaResult;
use cust_raw::CUcontext;

use crate::device::cuda::ToCudaResult;

pub struct Context {
    inner: NonNull<cust_raw::CUctx_st>,
}

// impl Clone for Context {
//     fn clone(&self) -> Self {
//         let mut ctx = self.inner.as_ptr();
//         unsafe {
//             cust_raw::cuCtxAttach(&mut ctx, 0).to_cuda_result().unwrap();
//             Self {
//                 inner: NonNull::new_unchecked(ctx),
//             }
//         }
//     }
// }

impl Context {
    /// Shortcut for initializing the CUDA Driver API and creating a CUDA context with default settings
    /// for the first device.
    #[must_use = "The CUDA Context must be kept alive or errors will be issued for any CUDA function that is run"]
    pub fn new(device: cust::device::Device) -> CudaResult<Self> {
        unsafe {
            let mut ctx: CUcontext = std::ptr::null_mut();
            cust_raw::cuCtxCreate_v2(&mut ctx as *mut CUcontext, 0, device.as_raw())
                .to_cuda_result()?;
            Ok(Self {
                inner: NonNull::new_unchecked(ctx),
            })
        }
    }
}

impl Deref for Context {
    type Target = SharedContext;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self.inner.as_ptr() as *const _) }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            cust_raw::cuCtxDestroy_v2(self.inner.as_ptr())
                .to_cuda_result()
                .unwrap();
        }
    }
}

pub struct SharedContext(cust_raw::CUctx_st);
