use rcublas_sys::{
    cublasDaxpy_v2, cublasDdot_v2, cublasDgemm_v2, cublasSaxpy_v2, cublasSdot_v2, cublasSgemm_v2,
};

use tensorgraph_sys::{
    device::{cuda::Unified, DevicePtr},
    ptr::DPtr,
};

use crate::blas::{MatrixOp, BLAS1, BLAS2, BLAS3};

use super::{BLASContext, SharedCublasContext, ToCublasResult};

impl From<MatrixOp> for rcublas_sys::cublasOperation_t {
    fn from(op: MatrixOp) -> Self {
        match op {
            MatrixOp::NoTrans => Self::CUBLAS_OP_N,
            MatrixOp::Trans => Self::CUBLAS_OP_T,
            // MatrixOp::ConjTrans => Self::CUBLAS_OP_HERMITAN,
        }
    }
}

macro_rules! impl_cublas1 {
    ($float:ident<$ctx:ty> =>
        axpy: $axpy:path,
        dot: $dot:path,
    ) => {
        impl<'a> BLAS1<&'a $ctx> for $float {
            unsafe fn axpy(
                handle: &$ctx,
                n: i32,
                alpha: Self,
                x: DPtr<Self, <&'a $ctx as BLASContext>::Device>,
                incx: i32,
                y: DPtr<Self, <&'a $ctx as BLASContext>::Device>,
                incy: i32,
            ) {
                $axpy(
                    handle.handle(),
                    n,
                    &alpha,
                    DevicePtr::as_raw(x),
                    incx,
                    DevicePtr::as_raw(y),
                    incy,
                )
                .to_cublas_result()
                .unwrap();
            }

            unsafe fn dot(
                handle: &$ctx,
                n: i32,
                x: DPtr<Self, <&'a $ctx as BLASContext>::Device>,
                incx: i32,
                y: DPtr<Self, <&'a $ctx as BLASContext>::Device>,
                incy: i32,
            ) -> Self {
                let mut res = 0.;
                $dot(
                    handle.handle(),
                    n,
                    DevicePtr::as_raw(x),
                    incx,
                    DevicePtr::as_raw(y),
                    incy,
                    &mut res,
                )
                .to_cublas_result()
                .unwrap();
                res
            }
        }
    };
}

macro_rules! impl_cublas2 {
    ($float:ident<$ctx:ty> =>
    ) => {
        impl<'a> BLAS2<&'a $ctx> for $float {}
    };
}

macro_rules! impl_cublas3 {
    ($float:ident<$ctx:ty> =>
        gemm: $gemm:path,
    ) => {
        impl<'a> BLAS3<&'a $ctx> for $float {
            unsafe fn gemm(
                handle: &$ctx,
                transa: MatrixOp,
                transb: MatrixOp,
                m: i32,
                n: i32,
                k: i32,
                alpha: Self,
                a: DPtr<Self, <&'a $ctx as BLASContext>::Device>,
                lda: i32,
                b: DPtr<Self, <&'a $ctx as BLASContext>::Device>,
                ldb: i32,
                beta: Self,
                c: DPtr<Self, <&'a $ctx as BLASContext>::Device>,
                ldc: i32,
            ) {
                $gemm(
                    handle.handle(),
                    transa.into(),
                    transb.into(),
                    m,
                    n,
                    k,
                    &alpha,
                    DevicePtr::as_raw(a),
                    lda,
                    DevicePtr::as_raw(b),
                    ldb,
                    &beta,
                    DevicePtr::as_raw(c),
                    ldc,
                )
                .to_cublas_result()
                .unwrap();
            }
        }
    };
}

impl_cublas1!(f32<SharedCublasContext> =>
    axpy: cublasSaxpy_v2,
    dot: cublasSdot_v2,
);
impl_cublas2!(f32<SharedCublasContext> =>
);
impl_cublas3!(f32<SharedCublasContext> =>
    gemm: cublasSgemm_v2,
);

impl_cublas1!(f64<SharedCublasContext> =>
    axpy: cublasDaxpy_v2,
    dot: cublasDdot_v2,
);
impl_cublas2!(f64<SharedCublasContext> =>
);
impl_cublas3!(f64<SharedCublasContext> =>
    gemm: cublasDgemm_v2,
);

impl_cublas1!(f32<Unified<SharedCublasContext>> =>
    axpy: cublasSaxpy_v2,
    dot: cublasSdot_v2,
);
impl_cublas2!(f32<Unified<SharedCublasContext>> =>
);
impl_cublas3!(f32<Unified<SharedCublasContext>> =>
    gemm: cublasSgemm_v2,
);

impl_cublas1!(f64<Unified<SharedCublasContext>> =>
    axpy: cublasDaxpy_v2,
    dot: cublasDdot_v2,
);
impl_cublas2!(f64<Unified<SharedCublasContext>> =>
);
impl_cublas3!(f64<Unified<SharedCublasContext>> =>
    gemm: cublasDgemm_v2,
);
