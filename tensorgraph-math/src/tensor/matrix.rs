use std::mem::MaybeUninit;

use num_traits::{One, Zero};
use tensorgraph_sys::{
    device::{DefaultDeviceAllocator, Device, DeviceAllocator},
    DefaultVec, Vec, View,
};

use crate::{
    blas::{BLASContext, DefaultBLASContext, MatrixOp, BLAS2, BLAS3},
    storage::Storage,
};

use super::{Slice, Tensor, UninitVector, Vector, VectorView, VectorViewMut, ViewOf};

/// A 2-dimensional tensor
pub type Matrix<S> = Tensor<S, [usize; 2]>;

/// A 'view' of a matrix, Like `&[T]` is to `Vec<T>`
pub type MatrixView<'a, T, D> = Matrix<&'a Slice<T, D>>;

/// A 'mut view' of a matrix, Like `&mut [T]` is to `Vec<T>`
pub type MatrixViewMut<'a, T, D> = Matrix<&'a mut Slice<T, D>>;

/// An uninit matrix. Contents are mutable and specified as [`MaybeUninit`].
pub type UninitMatrix<'a, T, D> = MatrixViewMut<'a, MaybeUninit<T>, D>;

impl<S: Storage> Matrix<S> {
    /// Matrix-vector multiplication
    pub fn dot(&self, rhs: Vector<&ViewOf<S>>) -> Vector<DefaultVec<S::T, S::Device>>
    where
        S::Device: DefaultDeviceAllocator + DefaultBLASContext,
        S::T: Zero + One + BLAS2<<S::Device as DefaultBLASContext>::Context>,
    {
        self.dot_using(rhs, Default::default())
    }

    /// Matrix-vector multiplication, using the specified [`BLASContext`]
    pub fn dot_using<C: BLASContext<Device = S::Device>>(
        &self,
        rhs: Vector<&ViewOf<S>>,
        ctx: C,
    ) -> Vector<DefaultVec<S::T, S::Device>>
    where
        S::Device: DefaultDeviceAllocator,
        S::T: Zero + One + BLAS2<C>,
    {
        self.dot_into(rhs, ctx, Default::default())
    }

    /// Matrix-vector multiplication, using the provided [`DeviceAllocator`], using the specified [`BLASContext`]
    pub fn dot_into<C: BLASContext<Device = S::Device>, A: DeviceAllocator<Device = S::Device>>(
        &self,
        rhs: Vector<&ViewOf<S>>,
        ctx: C,
        alloc: A,
    ) -> Vector<Vec<S::T, A>>
    where
        S::T: Zero + One + BLAS2<C>,
    {
        let rows = self.shape[0];
        let mut v = Vec::with_capacity_in(rows, alloc);
        unsafe {
            let uninit = Vector::from_shape([rows], &mut v.space_capacity_mut()[..rows]);

            gemv_uninit_ctx(ctx, S::T::one(), self.view(), rhs, uninit);

            v.set_len(rows);
        }
        Vector::from_shape([rows], v)
    }
}

impl<S: Storage> Matrix<S> {
    /// Multiply two matricies together.
    pub fn matmul(&self, rhs: Matrix<&ViewOf<S>>) -> Matrix<DefaultVec<S::T, S::Device>>
    where
        S::Device: DefaultDeviceAllocator + DefaultBLASContext,
        S::T: Zero + One + BLAS3<<S::Device as DefaultBLASContext>::Context>,
    {
        self.matmul_using(rhs, Default::default())
    }

    /// Multiply two matricies together, using the specified [`BLASContext`]
    pub fn matmul_using<C: BLASContext<Device = S::Device>>(
        &self,
        rhs: Matrix<&ViewOf<S>>,
        ctx: C,
    ) -> Matrix<DefaultVec<S::T, S::Device>>
    where
        S::Device: DefaultDeviceAllocator,
        S::T: Zero + One + BLAS3<C>,
    {
        self.matmul_into(rhs, ctx, Default::default())
    }

    /// Multiply two matricies together, using the provided [`DeviceAllocator`], using the specified [`BLASContext`]
    pub fn matmul_into<C: BLASContext<Device = S::Device>, A: DeviceAllocator<Device = S::Device>>(
        &self,
        rhs: Matrix<&ViewOf<S>>,
        ctx: C,
        alloc: A,
    ) -> Matrix<Vec<S::T, A>>
    where
        S::T: Zero + One + BLAS3<C>,
    {
        let rows = self.shape[0];
        let cols = rhs.shape[1];
        let mut v = Vec::with_capacity_in(rows * cols, alloc);
        unsafe {
            let uninit =
                Matrix::from_shape([rows, cols], &mut v.space_capacity_mut()[..rows * cols]);

            gemm_uninit_ctx(ctx, S::T::one(), self.view(), rhs, uninit);

            v.set_len(rows * cols);
        }
        Matrix::from_shape([rows, cols], v)
    }
}

/// Performs a matrix-vector multiplication operation into an uninit vector.
/// > y = alpha * Ax.
///
/// Uses the default [`BLASContext`] for the device.
///
/// # Panics
/// If the shapes of the matricies do not match the following pattern:
/// * A = (M, N)
/// * X = (N)
/// * Y = (M)
pub fn gemv_uninit<F: BLAS2<D::Context> + Zero, D: DefaultBLASContext>(
    alpha: F,
    a: MatrixView<F, D>,
    x: VectorView<F, D>,
    y: UninitVector<F, D>,
) {
    gemv_uninit_ctx(D::Context::default(), alpha, a, x, y);
}

/// Performs a matrix-vector multiplication operation into an uninit vector.
/// > y = alpha * Ax.
///
/// # Panics
/// If the shapes of the matricies do not match the following pattern:
/// * A = (M, N)
/// * X = (N)
/// * Y = (M)
pub fn gemv_uninit_ctx<F: BLAS2<C> + Zero, C: BLASContext<Device = D>, D: Device>(
    ctx: C,
    alpha: F,
    a: MatrixView<F, D>,
    x: VectorView<F, D>,
    y: UninitVector<F, D>,
) {
    // Safety:
    // Specifying beta == 0.0 should allow c to be safely read while uninitialised
    unsafe { gemv_ctx(ctx, alpha, a, x, F::zero(), y.assume_init()) }
}

/// Performs a matrix-vector multiplication operation.
/// > y = alpha * Ax + beta * y.
///
/// Uses the default [`BLASContext`] for the device.
///
/// # Panics
/// If the shapes of the matricies do not match the following pattern:
/// * A = (M, N)
/// * X = (N)
/// * Y = (M)
pub fn gemv<F: BLAS2<D::Context> + Zero, D: DefaultBLASContext>(
    alpha: F,
    a: MatrixView<F, D>,
    x: VectorView<F, D>,
    beta: F,
    y: VectorViewMut<F, D>,
) {
    gemv_ctx(D::Context::default(), alpha, a, x, beta, y);
}

/// Performs a matrix-vector multiplication operation.
/// > y = alpha * Ax + beta * y.
///
/// # Panics
/// If the shapes of the matricies do not match the following pattern:
/// * A = (M, N)
/// * X = (N)
/// * Y = (M)
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::needless_pass_by_value
)]
pub fn gemv_ctx<F: BLAS2<C> + Zero, C: BLASContext<Device = D>, D: Device>(
    ctx: C,
    alpha: F,
    a: MatrixView<F, D>,
    x: VectorView<F, D>,
    beta: F,
    y: VectorViewMut<F, D>,
) {
    let [rowsa, colsa] = a.shape;
    let [rowsx] = x.shape;
    let [rowsy] = y.shape;
    assert_eq!(rowsa, rowsy);
    assert_eq!(colsa, rowsx);

    let m = rowsa as i32;
    let n = colsa as i32;

    let (trans, lda) = lead(a.strides);
    let incx = x.strides[0] as i32;
    let incy = y.strides[0] as i32;

    unsafe {
        F::gemv(
            ctx,
            trans,
            m,
            n,
            alpha,
            a.data.as_ref().as_ptr(),
            lda,
            x.data.as_ref().as_ptr(),
            incx,
            beta,
            y.data.as_ptr(),
            incy,
        );
    }
}

/// Performs the matrix-matrix multiplication operation into an uninit matrix.
/// > C = alpha * AB.
///
/// Uses the default [`BLASContext`] for the device.
///
/// # Panics
/// If the shapes of the matricies do not match the following pattern:
/// * A = (M, K)
/// * B = (K, N)
/// * C = (M, N)
pub fn gemm_uninit<F: BLAS3<D::Context> + Zero, D: DefaultBLASContext>(
    alpha: F,
    a: MatrixView<F, D>,
    b: MatrixView<F, D>,
    c: UninitMatrix<F, D>,
) {
    gemm_uninit_ctx(D::Context::default(), alpha, a, b, c);
}

/// Performs the matrix-matrix multiplication operation into an uninit matrix.
/// > C = alpha * AB.
///
/// # Panics
/// If the shapes of the matricies do not match the following pattern:
/// * A = (M, K)
/// * B = (K, N)
/// * C = (M, N)
pub fn gemm_uninit_ctx<F: BLAS3<C> + Zero, C: BLASContext<Device = D>, D: Device>(
    ctx: C,
    alpha: F,
    a: MatrixView<F, D>,
    b: MatrixView<F, D>,
    c: UninitMatrix<F, D>,
) {
    // Safety:
    // Specifying beta == 0.0 should allow c to be safely read while uninitialised
    unsafe { gemm_ctx(ctx, alpha, a, b, F::zero(), c.assume_init()) }
}

/// Performs the matrix-matrix multiplication operation.
/// > C = alpha * AB + beta * C.
///
/// Uses the default [`BLASContext`] for the device.
///
/// # Panics
/// If the shapes of the matricies do not match the following pattern:
/// * A = (M, K)
/// * B = (K, N)
/// * C = (M, N)
pub fn gemm<F: BLAS3<D::Context> + Zero, D: DefaultBLASContext>(
    alpha: F,
    a: MatrixView<F, D>,
    b: MatrixView<F, D>,
    beta: F,
    c: MatrixViewMut<F, D>,
) {
    gemm_ctx(D::Context::default(), alpha, a, b, beta, c);
}

/// Performs the matrix-matrix multiplication operation.
/// > C = alpha * AB + beta * C.
///
/// # Panics
/// If the shapes of the matricies do not match the following pattern:
/// * A = (M, K)
/// * B = (K, N)
/// * C = (M, N)
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::needless_pass_by_value
)]
pub fn gemm_ctx<F: BLAS3<C> + Zero, C: BLASContext<Device = D>, D: Device>(
    ctx: C,
    alpha: F,
    a: MatrixView<F, D>,
    b: MatrixView<F, D>,
    beta: F,
    c: MatrixViewMut<F, D>,
) {
    let [rowsa, colsa] = a.shape;
    let [rowsb, colsb] = b.shape;
    let [rowsc, colsc] = c.shape;
    assert_eq!(rowsa, rowsc);
    assert_eq!(colsb, colsc);
    assert_eq!(colsa, rowsb);

    let m = rowsa as i32;
    let k = rowsb as i32;
    let n = colsb as i32;

    let (transa, lda) = lead(a.strides);
    let (transb, ldb) = lead(b.strides);

    // C must not be transposed
    assert_eq!(c.strides[0], 1);
    let ldc = c.strides[1] as i32;

    unsafe {
        F::gemm(
            ctx,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a.data.as_ref().as_ptr(),
            lda,
            b.data.as_ref().as_ptr(),
            ldb,
            beta,
            c.data.as_ptr(),
            ldc,
        );
    }
}

#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn lead(s: [usize; 2]) -> (MatrixOp, i32) {
    if s[0] == 1 {
        (MatrixOp::NoTrans, s[1] as i32)
    } else if s[1] == 1 {
        (MatrixOp::Trans, s[0] as i32)
    } else {
        panic!("one of the strides must be 1 (contiguous)")
    }
}
