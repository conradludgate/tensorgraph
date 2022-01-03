use rcublas_sys::{cublasDgemm_v2, cublasSgemm_v2};

use tensorgraph_sys::device::{
    cuda::{Cuda, CudaUnified},
    Device,
};

use crate::blas::MatrixOp;

use super::{SharedCublasContext, ToCublasResult, GEMM};

impl From<MatrixOp> for rcublas_sys::cublasOperation_t {
    fn from(op: MatrixOp) -> Self {
        match op {
            MatrixOp::NoTrans => rcublas_sys::cublasOperation_t::CUBLAS_OP_N,
            MatrixOp::Trans => rcublas_sys::cublasOperation_t::CUBLAS_OP_T,
            // MatrixOp::ConjTrans => rcublas_sys::cublasOperation_t::CUBLAS_OP_HERMITAN,
        }
    }
}

type DevicePointer<T> = <Cuda as Device>::Ptr<T>;

impl<'a> GEMM<&'a SharedCublasContext, Cuda> for f32 {
    unsafe fn gemm(
        handle: &SharedCublasContext,
        transa: MatrixOp,
        transb: MatrixOp,
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

impl<'a> GEMM<&'a SharedCublasContext, Cuda> for f64 {
    unsafe fn gemm(
        handle: &SharedCublasContext,
        transa: MatrixOp,
        transb: MatrixOp,
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

impl<'a> GEMM<&'a SharedCublasContext, CudaUnified> for f32 {
    unsafe fn gemm(
        handle: &SharedCublasContext,
        transa: MatrixOp,
        transb: MatrixOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *mut f32,
        lda: i32,
        b: *mut f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
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
            a,
            lda,
            b,
            ldb,
            &beta,
            c,
            ldc,
        )
        .to_cublas_result()
        .unwrap();
    }
}

impl<'a> GEMM<&'a SharedCublasContext, CudaUnified> for f64 {
    unsafe fn gemm(
        handle: &SharedCublasContext,
        transa: MatrixOp,
        transb: MatrixOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *mut f64,
        lda: i32,
        b: *mut f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
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
            a,
            lda,
            b,
            ldb,
            &beta,
            c,
            ldc,
        )
        .to_cublas_result()
        .unwrap();
    }
}
