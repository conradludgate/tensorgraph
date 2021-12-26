extern crate blas_src;
extern crate blas_sys;

use crate::device::cpu::Cpu;
use std::alloc::Allocator;

use super::{BLASDevice, GEMM};

impl<A: Allocator> BLASDevice for Cpu<A> {
    type Context = ();
}

impl<A: Allocator> GEMM<Cpu<A>> for f32 {
    unsafe fn gemm(
        _ctx: (),
        transa: super::MatrixOp,
        transb: super::MatrixOp,
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
        )
    }
}

impl<A: Allocator> GEMM<Cpu<A>> for f64 {
    unsafe fn gemm(
        _ctx: (),
        transa: super::MatrixOp,
        transb: super::MatrixOp,
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
        )
    }
}
