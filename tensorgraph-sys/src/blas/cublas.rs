use std::{ops::Deref, ptr::NonNull};

use cust::memory::DevicePointer;
use rcublas_sys::{
    cublasContext, cublasCreate_v2, cublasDestroy_v2, cublasDgemm_v2, cublasSgemm_v2,
    cublasStatus_t, cublasSetStream_v2, cublasHandle_t,
};

use crate::device::cuda::{Cuda, SharedStream};

use super::{BLASDevice, GEMM};

pub struct CublasContext {
    inner: NonNull<cublasContext>,
}

impl CublasContext {
    pub fn new() -> Self {
        unsafe {
            let mut handle = std::ptr::null_mut();
            cublasCreate_v2(&mut handle).to_cublas_result().unwrap();

            Self {
                inner: NonNull::new_unchecked(handle),
            }
        }
    }

    pub fn with_stream<'a>(&'a self, stream: Option<&'a SharedStream>) -> &'a SharedCublasContext {
        unsafe {
            let ptr = self.inner.as_ptr();

            cublasSetStream_v2(
                ptr as *mut _,
                stream.map_or_else(std::ptr::null_mut, SharedStream::inner) as *mut _,
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
                .unwrap()
        }
    }
}

impl Deref for CublasContext {
    type Target = SharedCublasContext;
    fn deref(&self) -> &SharedCublasContext {
        unsafe { &*(self.inner.as_ptr() as *mut _) }
    }
}

pub struct SharedCublasContext(cublasContext);

impl SharedCublasContext {
    fn handle(&self) -> cublasHandle_t {
        self as *const _ as *mut _
    }
}

impl BLASDevice for Cuda {
    type Context = SharedCublasContext;
}

impl From<super::MatrixOp> for rcublas_sys::cublasOperation_t {
    fn from(op: super::MatrixOp) -> Self {
        match op {
            super::MatrixOp::NoTrans => rcublas_sys::cublasOperation_t::CUBLAS_OP_N,
            super::MatrixOp::Trans => rcublas_sys::cublasOperation_t::CUBLAS_OP_T,
            // super::MatrixOp::ConjTrans => rcublas_sys::cublasOperation_t::CUBLAS_OP_HERMITAN,
        }
    }
}

impl GEMM<Cuda> for f32 {
    unsafe fn gemm(
        handle: &SharedCublasContext,
        transa: super::MatrixOp,
        transb: super::MatrixOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: DevicePointer<f32>,
        lda: i32,
        b: DevicePointer<f32>,
        ldb: i32,
        beta: f32,
        mut c: DevicePointer<f32>,
        ldc: i32,
    ) {
        cublasSgemm_v2(
            handle.handle(),
            transa.into(),
            transb.into(),
            m,
            n,
            k,
            &alpha,
            a.as_raw(),
            lda,
            b.as_raw(),
            ldb,
            &beta,
            c.as_raw_mut(),
            ldc,
        )
        .to_cublas_result()
        .unwrap();
    }
}

impl GEMM<Cuda> for f64 {
    unsafe fn gemm(
        handle: &SharedCublasContext,
        transa: super::MatrixOp,
        transb: super::MatrixOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: DevicePointer<f64>,
        lda: i32,
        b: DevicePointer<f64>,
        ldb: i32,
        beta: f64,
        mut c: DevicePointer<f64>,
        ldc: i32,
    ) {
        cublasDgemm_v2(
            handle.handle(),
            transa.into(),
            transb.into(),
            m,
            n,
            k,
            &alpha,
            a.as_raw(),
            lda,
            b.as_raw(),
            ldb,
            &beta,
            c.as_raw_mut(),
            ldc,
        )
        .to_cublas_result()
        .unwrap();
    }
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
