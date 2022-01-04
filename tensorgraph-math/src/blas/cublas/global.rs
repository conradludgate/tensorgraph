use std::{cell::RefCell, lazy::Lazy};

use rcublas_sys::cublasContext;
use tensorgraph_sys::device::cuda::{Cuda, CudaUnified, Unified};

use crate::blas::DefaultBLASContext;

use super::SharedCublasContext;

#[thread_local]
static GLOBAL: Lazy<RefCell<Option<std::ptr::NonNull<cublasContext>>>> =
    Lazy::new(|| RefCell::new(None));

impl SharedCublasContext {
    /// Sets the cublas context as the global thread-local context
    #[must_use]
    pub fn as_global(&self) -> CublasContextHandle {
        CublasContextHandle(GLOBAL.replace(Some(unsafe {
            std::ptr::NonNull::new_unchecked(self.handle())
        })))
    }
}

pub struct CublasContextHandle(Option<std::ptr::NonNull<cublasContext>>);

impl Drop for CublasContextHandle {
    fn drop(&mut self) {
        let _ctx = GLOBAL.replace(self.0);
    }
}

/// Get the global stream set via [`SharedCublasContext::as_global`]
#[must_use]
pub fn get_cublas() -> Option<&'static SharedCublasContext> {
    GLOBAL
        .borrow()
        .map(|p| unsafe { &*(p.as_ptr() as *const _) })
}

impl DefaultBLASContext for Cuda {
    type Context = &'static SharedCublasContext;
}

impl DefaultBLASContext for CudaUnified {
    type Context = &'static Unified<SharedCublasContext>;
}

impl Default for &'static SharedCublasContext {
    fn default() -> Self {
        get_cublas().unwrap()
    }
}
