use std::{cell::RefCell, lazy::Lazy, ops::Deref};

use rcublas_sys::cublasContext;
use tensorgraph_sys::device::cuda::Cuda;

use crate::blas::DefaultBLASContext;

use super::SharedCublasContext;

#[thread_local]
static GLOBAL: Lazy<RefCell<Option<std::ptr::NonNull<cublasContext>>>> =
    Lazy::new(|| RefCell::new(None));

impl SharedCublasContext {
    /// Runs the given closure with the cublas context as the global thread-local context
    pub fn global_over<R, F: FnOnce(&Self) -> R>(&self, f: F) -> R {
        let pointer = GLOBAL.deref();

        let old = pointer.replace(Some(unsafe {
            std::ptr::NonNull::new_unchecked(self.handle())
        }));

        let out = f(self);

        let _ctx = pointer.replace(old);

        out
    }
}

/// Get the global stream set via [`with_stream`]
pub fn get_cublas() -> Option<&'static SharedCublasContext> {
    GLOBAL
        .borrow()
        .map(|p| unsafe { &*(p.as_ptr() as *const _) })
}

impl DefaultBLASContext for Cuda {
    type Context = &'static SharedCublasContext;
}

impl Default for &'static SharedCublasContext {
    fn default() -> Self {
        get_cublas().unwrap()
    }
}
