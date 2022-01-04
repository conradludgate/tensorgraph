use std::mem::MaybeUninit;

use num_traits::{One, Zero};
use tensorgraph_sys::{
    device::{DefaultDeviceAllocator, Device, DeviceAllocator},
    DefaultVec, Vec, View,
};

use crate::{
    blas::{BLASContext, DefaultBLASContext, MatrixOp, GEMM},
    dims::Dimension,
    storage::Storage,
};

use super::{Slice, Tensor, TensorView, TensorViewMut, UninitTensor, ViewOf};

/// A 2-dimensional tensor
pub type Matrix<S> = Tensor<S, [usize; 2]>;

/// A 'view' of a matrix, Like `&[T]` is to `Vec<T>`
pub type MatrixView<'a, T, D> = Matrix<&'a Slice<T, D>>;

/// A 'mut view' of a matrix, Like `&mut [T]` is to `Vec<T>`
pub type MatrixViewMut<'a, T, D> = Matrix<&'a mut Slice<T, D>>;

/// An uninit matrix. Contents are mutable and specified as [`MaybeUninit`].
pub type UninitMatrix<'a, T, D> = MatrixViewMut<'a, MaybeUninit<T>, D>;

impl<S: Storage> Matrix<S> {
    /// Multiply two matricies together.
    pub fn dot(&self, rhs: Matrix<&ViewOf<S>>) -> Matrix<DefaultVec<S::T, S::Device>>
    where
        S::T: Zero + One,
        S::Device: DefaultDeviceAllocator + DefaultBLASContext,
        S::T: GEMM<<S::Device as DefaultBLASContext>::Context>,
    {
        self.dot_using(rhs, Default::default())
    }

    /// Multiply two matricies together, using the specified [`BLASContext`]
    pub fn dot_using<C: BLASContext<Device = S::Device>>(
        &self,
        rhs: Matrix<&ViewOf<S>>,
        ctx: C,
    ) -> Matrix<DefaultVec<S::T, S::Device>>
    where
        S::T: Zero + One,
        S::Device: DefaultDeviceAllocator,
        S::T: GEMM<C>,
    {
        self.dot_into(rhs, ctx, Default::default())
    }

    /// Multiply two matricies together, using the provided [`DeviceAllocator`], using the specified [`BLASContext`]
    pub fn dot_into<C: BLASContext<Device = S::Device>, A: DeviceAllocator<Device = S::Device>>(
        &self,
        rhs: Matrix<&ViewOf<S>>,
        ctx: C,
        alloc: A,
    ) -> Matrix<Vec<S::T, A>>
    where
        S::T: Zero + One,
        S::T: GEMM<C>,
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

impl<'a, T, D: Device, Dim: Dimension> UninitTensor<'a, T, D, Dim> {
    /// # Safety
    /// Contents must be initialised
    pub unsafe fn assume_init(self) -> TensorViewMut<'a, T, D, Dim> {
        Tensor {
            shape: self.shape,
            strides: self.strides,
            data: self.data.assume_init_mut(),
        }
    }
}

impl<'a, T, D: Device, Dim: Dimension> TensorView<'a, MaybeUninit<T>, D, Dim> {
    /// # Safety
    /// Contents must be initialised
    pub unsafe fn assume_init(self) -> TensorView<'a, T, D, Dim> {
        Tensor {
            shape: self.shape,
            strides: self.strides,
            data: self.data.assume_init(),
        }
    }
}

/// Performs the basic matmul operation.
/// C = alpha * A * B.
///
/// Uses the default [`BLASContext`] for the device.
pub fn gemm_uninit<F: GEMM<D::Context> + Zero, D: DefaultBLASContext>(
    alpha: F,
    a: Matrix<impl Storage<T = F, Device = D>>,
    b: Matrix<impl Storage<T = F, Device = D>>,
    c: UninitMatrix<F, D>,
) {
    gemm_uninit_ctx(D::Context::default(), alpha, a, b, c)
}

/// Performs the basic matmul operation.
/// C = alpha * A * B.
pub fn gemm_uninit_ctx<F: GEMM<C> + Zero, C: BLASContext<Device = D>, D: Device>(
    ctx: C,
    alpha: F,
    a: Matrix<impl Storage<T = F, Device = D>>,
    b: Matrix<impl Storage<T = F, Device = D>>,
    c: UninitMatrix<F, D>,
) {
    // Safety:
    // Specifying beta == 0.0 should allow c to be safely read while uninitialised
    unsafe { gemm_ctx(ctx, alpha, a, b, F::zero(), c.assume_init()) }
}

/// Performs the basic matmul operation.
/// C = alpha * A * B + beta * C.
///
/// Uses the default [`BLASContext`] for the device.
pub fn gemm<F: GEMM<D::Context> + Zero, D: DefaultBLASContext>(
    alpha: F,
    a: Matrix<impl Storage<T = F, Device = D>>,
    b: Matrix<impl Storage<T = F, Device = D>>,
    beta: F,
    c: MatrixViewMut<F, D>,
) {
    gemm_ctx(D::Context::default(), alpha, a, b, beta, c)
}

/// Performs the basic matmul operation.
/// C = alpha * A * B + beta * C.
pub fn gemm_ctx<F: GEMM<C> + Zero, C: BLASContext<Device = D>, D: Device>(
    ctx: C,
    alpha: F,
    a: Matrix<impl Storage<T = F, Device = D>>,
    b: Matrix<impl Storage<T = F, Device = D>>,
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
        )
    }
}

fn lead(s: [usize; 2]) -> (MatrixOp, i32) {
    if s[0] == 1 {
        (MatrixOp::NoTrans, s[1] as i32)
    } else if s[1] == 1 {
        (MatrixOp::Trans, s[0] as i32)
    } else {
        panic!("one of the strides must be 1 (contiguous)")
    }
}
