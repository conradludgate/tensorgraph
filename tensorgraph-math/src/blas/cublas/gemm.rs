use rcublas_sys::{cublasDgemm_v2, cublasSgemm_v2};

use tensorgraph_sys::device::{
    cuda::{Cuda, Unified},
    Device,
};

use crate::blas::MatrixOp;

use super::{SharedCublasContext, ToCublasResult, GEMM};

impl From<MatrixOp> for rcublas_sys::cublasOperation_t {
    fn from(op: MatrixOp) -> Self {
        match op {
            MatrixOp::NoTrans => Self::CUBLAS_OP_N,
            MatrixOp::Trans => Self::CUBLAS_OP_T,
            // MatrixOp::ConjTrans => Self::CUBLAS_OP_HERMITAN,
        }
    }
}

type DevicePointer<T> = <Cuda as Device>::Ptr<T>;

impl<'a> GEMM<&'a SharedCublasContext> for f32 {
    unsafe fn gemm(
        handle: &SharedCublasContext,
        transa: MatrixOp,
        transb: MatrixOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: Self,
        a: DevicePointer<Self>,
        lda: i32,
        b: DevicePointer<Self>,
        ldb: i32,
        beta: Self,
        mut c: DevicePointer<Self>,
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

impl<'a> GEMM<&'a SharedCublasContext> for f64 {
    unsafe fn gemm(
        handle: &SharedCublasContext,
        transa: MatrixOp,
        transb: MatrixOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: Self,
        a: DevicePointer<Self>,
        lda: i32,
        b: DevicePointer<Self>,
        ldb: i32,
        beta: Self,
        mut c: DevicePointer<Self>,
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

impl<'a> GEMM<&'a Unified<SharedCublasContext>> for f32 {
    unsafe fn gemm(
        handle: &Unified<SharedCublasContext>,
        transa: MatrixOp,
        transb: MatrixOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: Self,
        a: *mut Self,
        lda: i32,
        b: *mut Self,
        ldb: i32,
        beta: Self,
        c: *mut Self,
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

impl<'a> GEMM<&'a Unified<SharedCublasContext>> for f64 {
    unsafe fn gemm(
        handle: &Unified<SharedCublasContext>,
        transa: MatrixOp,
        transb: MatrixOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: Self,
        a: *mut Self,
        lda: i32,
        b: *mut Self,
        ldb: i32,
        beta: Self,
        c: *mut Self,
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
