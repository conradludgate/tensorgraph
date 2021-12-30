use cust::memory::DevicePointer;

use crate::device::cuda::Cuda;

use super::{BLASDevice, GEMM};

impl BLASDevice for Cuda {
    type Context = rcublas_sys::cublasHandle_t;
}

impl From<super::MatrixOp> for rcublas_sys::cublasOperation_t {
    fn from(op: super::MatrixOp) -> Self {
        match op {
            super::MatrixOp::NoTrans => rcublas_sys::cublasOperation_t::CUBLAS_OP_N,
            super::MatrixOp::Trans => rcublas_sys::cublasOperation_t::CUBLAS_OP_T,
            super::MatrixOp::ConjTrans => rcublas_sys::cublasOperation_t::CUBLAS_OP_HERMITAN,
        }
    }
}

impl GEMM<Cuda> for f32 {
    unsafe fn gemm(
        handle: rcublas_sys::cublasHandle_t,
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
        rcublas_sys::cublasSgemm_v2(
            handle,
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
        ).to_cublas_result().unwrap();
    }
}

impl GEMM<Cuda> for f64 {
    unsafe fn gemm(
        handle: rcublas_sys::cublasHandle_t,
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
        rcublas_sys::cublasDgemm_v2(
            handle,
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
        ).to_cublas_result().unwrap();
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
impl ToCublasResult for rcublas_sys::cublasStatus_t {
    fn to_cublas_result(self) -> CublasResult<()> {
        match self {
            rcublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            rcublas_sys::cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => Err(CublasError::NotInitialized),
            rcublas_sys::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED => Err(CublasError::AllocFailed),
            rcublas_sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => Err(CublasError::InvalidValue),
            rcublas_sys::cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => Err(CublasError::ArchMismatch),
            rcublas_sys::cublasStatus_t::CUBLAS_STATUS_MAPPING_ERROR => Err(CublasError::MappingError),
            rcublas_sys::cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => Err(CublasError::ExecutionFailed),
            rcublas_sys::cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR => Err(CublasError::InternalError),
            rcublas_sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED => Err(CublasError::NotSupported),
            rcublas_sys::cublasStatus_t::CUBLAS_STATUS_LICENSE_ERROR => Err(CublasError::LicenseError),
            _ => Err(CublasError::UnexpectedError)
        }
    }
}
