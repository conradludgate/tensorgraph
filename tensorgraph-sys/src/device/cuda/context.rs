use std::{mem::ManuallyDrop, ops::Deref, ptr::NonNull};

use cust::{error::CudaResult, CudaFlags};
use cust_raw::CUcontext;

use crate::device::cuda::ToCudaResult;

/// Represents an owned CUDA context
pub struct Context {
    inner: NonNull<cust_raw::CUctx_st>,
}

/// Represents an attached CUDA context
pub struct AttachedContext<'a> {
    inner: &'a SharedContext,
}

impl<'a> AttachedContext<'a> {
    /// Attaches to the context
    ///
    /// # Errors
    /// If attaching to the context is not possible
    pub fn attach_to(ctx: &'a SharedContext) -> CudaResult<Self> {
        let mut ptr = ctx as *const _ as *mut _;
        unsafe {
            cust_raw::cuCtxAttach(&mut ptr, 0).to_cuda_result()?;

            Ok(Self {
                inner: &*(ptr as *const _),
            })
        }
    }
}

impl<'a> Drop for AttachedContext<'a> {
    fn drop(&mut self) {
        unsafe {
            cust_raw::cuCtxDetach(self.inner as *const _ as *mut _)
                .to_cuda_result()
                .unwrap();
        }
    }
}

impl<'a> Deref for AttachedContext<'a> {
    type Target = SharedContext;
    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl Context {
    /// Shortcut for initializing the CUDA Driver API and creating a CUDA context with default settings
    /// for the first device.
    #[must_use = "The CUDA Context must be kept alive or errors will be issued for any CUDA function that is run"]
    /// Creates a new cuda context. Cuda must be initialised, see [`cust::init`].
    ///
    /// # Errors
    /// If cuda fails to return a context
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

    /// Initialises the cuda context on the default device.
    ///
    /// # Errors
    /// If cuda fails to return a context
    pub fn quick_init() -> CudaResult<Self> {
        cust::init(CudaFlags::empty())?;
        Self::new(cust::device::Device::get_device(0)?)
    }

    #[must_use]
    /// Pops the context from the current thread-stack.
    ///
    /// # Panics
    /// If the context cannot be popped
    pub fn pop(self) -> FloatingContext {
        let this = ManuallyDrop::new(self);
        unsafe {
            let mut ptr = this.inner.as_ptr();
            cust_raw::cuCtxPopCurrent_v2(&mut ptr as *mut CUcontext)
                .to_cuda_result()
                .unwrap();
            FloatingContext {
                inner: NonNull::new_unchecked(ptr),
            }
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

/// Represents a shared CUDA context
pub struct SharedContext(cust_raw::CUctx_st);

impl SharedContext {
    /// Attaches to the context
    ///
    /// # Errors
    /// If attaching to the context is not possible
    pub fn attach(&self) -> CudaResult<AttachedContext> {
        AttachedContext::attach_to(self)
    }
}

pub struct FloatingContext {
    inner: NonNull<cust_raw::CUctx_st>,
}

impl FloatingContext {
    #[must_use]
    /// Pushes the context onto the current thread stack
    ///
    /// # Panics
    /// If cuda returns an error
    pub fn push(self) -> Context {
        let this = ManuallyDrop::new(self);
        unsafe {
            let ptr = this.inner.as_ptr();
            cust_raw::cuCtxPushCurrent_v2(ptr.cast())
                .to_cuda_result()
                .unwrap();
            Context {
                inner: NonNull::new_unchecked(ptr),
            }
        }
    }
}

impl Drop for FloatingContext {
    fn drop(&mut self) {
        unsafe {
            cust_raw::cuCtxDestroy_v2(self.inner.as_ptr())
                .to_cuda_result()
                .unwrap();
        }
    }
}
