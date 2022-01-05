use std::{ops::Deref, ptr::NonNull};

use rcublas_sys::{
    cublasContext, cublasCreate_v2, cublasDestroy_v2, cublasHandle_t, cublasSetStream_v2,
    cublasStatus_t,
};

use tensorgraph_sys::device::cuda::{Cuda, CudaUnified, SharedStream, Unified};

use super::{BLASContext};

mod gemm;
mod global;
pub use global::get_cublas;

pub struct CublasContext {
    inner: NonNull<cublasContext>,
}

impl CublasContext {
    /// Creates a new cublas context.
    ///
    /// # Panics
    /// If cublas can't create a new context
    #[must_use]
    pub fn new() -> Self {
        unsafe {
            let mut handle = std::ptr::null_mut();
            cublasCreate_v2(&mut handle).to_cublas_result().unwrap();

            Self {
                inner: NonNull::new_unchecked(handle),
            }
        }
    }

    /// Sets the stream for the cublas context.
    ///
    /// # Panics
    /// If cuda returns an error when configuring
    #[must_use]
    pub fn with_stream<'a>(&'a self, stream: Option<&'a SharedStream>) -> &'a SharedCublasContext {
        unsafe {
            let ptr = self.inner.as_ptr();

            cublasSetStream_v2(
                ptr.cast(),
                stream
                    .map_or_else(std::ptr::null_mut, SharedStream::inner)
                    .cast(),
            )
            .to_cublas_result()
            .unwrap();

            &*(ptr as *const _)
        }
    }
}

impl Default for CublasContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CublasContext {
    fn drop(&mut self) {
        unsafe {
            cublasDestroy_v2(self.inner.as_ptr())
                .to_cublas_result()
                .unwrap();
        }
    }
}

impl Deref for CublasContext {
    type Target = SharedCublasContext;
    fn deref(&self) -> &SharedCublasContext {
        unsafe { &*(self.inner.as_ptr().cast()) }
    }
}

pub struct SharedCublasContext(cublasContext);

impl SharedCublasContext {
    const fn handle(&self) -> cublasHandle_t {
        self as *const _ as *mut _
    }
}

impl<'a> BLASContext for &'a SharedCublasContext {
    type Device = Cuda;
}

impl<'a> BLASContext for &'a Unified<SharedCublasContext> {
    type Device = CudaUnified;
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum CublasError {
    NotInitialized,
    AllocFailed,
    InvalidValue,
    ArchMismatch,
    MappingError,
    ExecutionFailed,
    InternalError,
    NotSupported,
    LicenseError,
    UnexpectedError,
}

pub(crate) type CublasResult<T, E = CublasError> = Result<T, E>;

pub(crate) trait ToCublasResult {
    fn to_cublas_result(self) -> CublasResult<()>;
}
impl ToCublasResult for cublasStatus_t {
    fn to_cublas_result(self) -> CublasResult<()> {
        #[allow(clippy::enum_glob_use)]
        use cublasStatus_t::*;
        match self {
            CUBLAS_STATUS_SUCCESS => Ok(()),
            CUBLAS_STATUS_NOT_INITIALIZED => Err(CublasError::NotInitialized),
            CUBLAS_STATUS_ALLOC_FAILED => Err(CublasError::AllocFailed),
            CUBLAS_STATUS_INVALID_VALUE => Err(CublasError::InvalidValue),
            CUBLAS_STATUS_ARCH_MISMATCH => Err(CublasError::ArchMismatch),
            CUBLAS_STATUS_MAPPING_ERROR => Err(CublasError::MappingError),
            CUBLAS_STATUS_EXECUTION_FAILED => Err(CublasError::ExecutionFailed),
            CUBLAS_STATUS_INTERNAL_ERROR => Err(CublasError::InternalError),
            CUBLAS_STATUS_NOT_SUPPORTED => Err(CublasError::NotSupported),
            CUBLAS_STATUS_LICENSE_ERROR => Err(CublasError::LicenseError),
            _ => Err(CublasError::UnexpectedError),
        }
    }
}
