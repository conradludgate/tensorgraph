use matrixmultiply::{dgemm, sgemm};

use crate::blas::{MatrixOp, GEMM};

use super::CpuContext;

impl GEMM<CpuContext> for f32 {
    unsafe fn gemm(
        _ctx: CpuContext,
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
        let mut sa = [1, lda as isize];
        let mut sb = [1, ldb as isize];
        let sc = [1, ldc as isize];

        match transa {
            MatrixOp::NoTrans => (),
            MatrixOp::Trans => {
                sa.rotate_left(1);
            }
        }
        match transb {
            MatrixOp::NoTrans => (),
            MatrixOp::Trans => {
                sb.rotate_left(1);
            }
        }

        sgemm(
            m as usize, k as usize, n as usize, alpha, a, sa[0], sa[1], b, sb[0], sb[1], beta, c,
            sc[0], sc[1],
        )
    }
}

impl GEMM<CpuContext> for f64 {
    unsafe fn gemm(
        _ctx: CpuContext,
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
        let mut sa = [1, lda as isize];
        let mut sb = [1, ldb as isize];
        let sc = [1, ldc as isize];

        match transa {
            MatrixOp::NoTrans => (),
            MatrixOp::Trans => {
                sa.rotate_left(1);
            }
        }
        match transb {
            MatrixOp::NoTrans => (),
            MatrixOp::Trans => {
                sb.rotate_left(1);
            }
        }

        dgemm(
            m as usize, k as usize, n as usize, alpha, a, sa[0], sa[1], b, sb[0], sb[1], beta, c,
            sc[0], sc[1],
        )
    }
}
