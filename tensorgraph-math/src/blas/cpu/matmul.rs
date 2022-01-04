use matrixmultiply::{dgemm, sgemm};

use crate::blas::{MatrixOp, GEMM};

#[allow(clippy::cast_sign_loss)]
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
        );
    }
}

#[allow(clippy::cast_sign_loss)]
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
        );
    }
}
