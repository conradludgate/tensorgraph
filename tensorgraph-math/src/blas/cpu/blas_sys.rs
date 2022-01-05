extern crate blas_src;
extern crate blas_sys;

use crate::blas::{MatrixOp, BLAS1, BLAS2, BLAS3};

macro_rules! impl_blas1 {
    ($float:ident =>
        axpy: $axpy:path,
        dot: $dot:path,
    ) => {
        impl BLAS1<()> for $float {
            unsafe fn axpy(
                _ctx: (),
                n: i32,
                alpha: Self,
                x: *mut Self,
                incx: i32,
                y: *mut Self,
                incy: i32,
            ) {
                $axpy(&n, &alpha, x, &incx, y, &incy);
            }

            unsafe fn dot(
                _ctx: (),
                n: i32,
                x: *mut Self,
                incx: i32,
                y: *mut Self,
                incy: i32,
            ) -> Self {
                $dot(&n, x, &incx, y, &incy)
            }
        }
    };
}

macro_rules! impl_blas2 {
    ($float:ident =>
    ) => {
        impl BLAS2<()> for $float {}
    };
}

macro_rules! impl_blas3 {
    ($float:ident =>
        gemm: $gemm:path,
    ) => {
        impl BLAS3<()> for $float {
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
                $gemm(
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
    };
}

impl_blas1!(f32 =>
    axpy: blas_sys::saxpy_,
    dot: blas_sys::sdot_,
);
impl_blas2!(f32 =>
);
impl_blas3!(f32 =>
    gemm: blas_sys::sgemm_,
);

impl_blas1!(f64 =>
    axpy: blas_sys::daxpy_,
    dot: blas_sys::ddot_,
);
impl_blas2!(f64 =>
);
impl_blas3!(f64 =>
    gemm: blas_sys::dgemm_,
);
