use cust::memory::DevicePointer;
use rcublas_sys::{
    cublasCreate_v2, cublasDgemm_v2, cublasSetStream_v2, cublasSgemm_v2, cublasStatus_t,
    cudaStream_t,
};

use crate::device::cuda::Cuda;

use super::{BLASDevice, GEMM};

#[derive(Clone, Copy)]
pub struct CublasContext(rcublas_sys::cublasHandle_t);

impl CublasContext {
    pub(crate) unsafe fn new(stream: cudaStream_t) -> Self {
        let mut handle = std::ptr::null_mut();
        cublasCreate_v2(&mut handle).to_cublas_result().unwrap();

        cublasSetStream_v2(handle, stream)
            .to_cublas_result()
            .unwrap();
        Self(handle)
    }
}

impl<'a> BLASDevice for Cuda<'a> {
    type Context = CublasContext;
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

impl<'a> GEMM<Cuda<'a>> for f32 {
    unsafe fn gemm(
        handle: CublasContext,
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
            handle.0,
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

impl<'a> GEMM<Cuda<'a>> for f64 {
    unsafe fn gemm(
        handle: CublasContext,
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
            handle.0,
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
