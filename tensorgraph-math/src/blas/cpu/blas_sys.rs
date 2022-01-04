extern crate blas_src;
extern crate blas_sys;

use crate::blas::{MatrixOp, GEMM};

impl GEMM<()> for f32 {
    unsafe fn gemm(
        _ctx: (),
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
        blas_sys::sgemm_(
            &(transa as i8),
            &(transb as i8),
            &m,
            &n,
            &k,
            &alpha,
            a,
            &lda,
            b,
            &ldb,
            &beta,
            c,
            &ldc,
        );
    }
}

impl GEMM<()> for f64 {
    unsafe fn gemm(
        _ctx: (),
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
        blas_sys::dgemm_(
            &(transa as i8),
            &(transb as i8),
            &m,
            &n,
            &k,
            &alpha,
            a,
            &lda,
            b,
            &ldb,
            &beta,
            c,
            &ldc,
        );
    }
}
