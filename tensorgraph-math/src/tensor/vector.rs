use std::{
    mem::MaybeUninit,
    ops::{AddAssign, Mul, MulAssign},
};

use num_traits::One;
use tensorgraph_sys::{
    device::{DefaultDeviceAllocator, Device},
    ViewMut,
};

use crate::{
    blas::{BLASContext, DefaultBLASContext, BLAS1},
    storage::{IntoOwned, Storage, StorageMut},
};

use super::{Slice, Tensor, ViewOf};

/// A 1-dimensional tensor
pub type Vector<S> = Tensor<S, [usize; 1]>;

/// A 'view' of a vector, Like `&[T]` is to `Vec<T>`
pub type VectorView<'a, T, D> = Vector<&'a Slice<T, D>>;

/// A 'mut view' of a vector, Like `&mut [T]` is to `Vec<T>`
pub type VectorViewMut<'a, T, D> = Vector<&'a mut Slice<T, D>>;

/// An uninit vector. Contents are mutable and specified as [`MaybeUninit`].
pub type UninitVector<'a, T, D> = VectorViewMut<'a, MaybeUninit<T>, D>;

impl<S: Storage> Vector<S> {
    /// Vector dot product
    pub fn dot(&self, rhs: Vector<&ViewOf<S>>) -> S::T
    where
        S::Device: DefaultBLASContext,
        S::T: BLAS1<<S::Device as DefaultBLASContext>::Context>,
    {
        self.dot_using(rhs, Default::default())
    }

    /// Vector dot product, using the specified [`BLASContext`]
    ///
    /// # Panics
    /// If the vectors do not have the same length
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    pub fn dot_using<C: BLASContext<Device = S::Device>>(
        &self,
        rhs: Vector<&ViewOf<S>>,
        ctx: C,
    ) -> S::T
    where
        S::T: BLAS1<C>,
    {
        let x = self;
        let y = rhs;
        let [n] = x.shape;
        let [m] = y.shape;
        assert_eq!(n, m);

        let incx = x.strides[0] as i32;
        let incy = y.strides[0] as i32;

        unsafe {
            <S::T as BLAS1<C>>::dot(
                ctx,
                n as i32,
                x.data.as_ref().as_ptr(),
                incx,
                y.data.as_ptr(),
                incy,
            )
        }
    }
}

impl<'a, S: StorageMut> AddAssign<Vector<&'a ViewOf<S>>> for Vector<S>
where
    S::Device: DefaultBLASContext,
    S::T: One + BLAS1<<S::Device as DefaultBLASContext>::Context>,
{
    fn add_assign(&mut self, rhs: Vector<&'a ViewOf<S>>) {
        axpy_ctx(Default::default(), One::one(), rhs, self.view_mut());
    }
}

impl<S: StorageMut> MulAssign<S::T> for Vector<S>
where
    S::Device: DefaultBLASContext,
    S::T: BLAS1<<S::Device as DefaultBLASContext>::Context>,
{
    fn mul_assign(&mut self, rhs: S::T) {
        self.scale_using(rhs, Default::default());
    }
}

impl<S: Storage + IntoOwned> Mul<S::T> for Vector<S>
where
    S::Device: DefaultBLASContext + DefaultDeviceAllocator,
    S::T: BLAS1<<S::Device as DefaultBLASContext>::Context>,
    S::Owned: Storage<T = S::T, Device = S::Device> + StorageMut,
{
    type Output = Vector<S::Owned>;
    fn mul(self, rhs: S::T) -> Self::Output {
        let mut x = self.into_owned();
        x *= rhs;
        x
    }
}

impl<S: StorageMut> Vector<S> {
    /// Vector scaling, using the specified [`BLASContext`]
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    pub fn scale_using<C: BLASContext<Device = S::Device>>(&mut self, alpha: S::T, ctx: C)
    where
        S::T: BLAS1<C>,
    {
        scal_ctx(ctx, alpha, self.view_mut());
    }
}

/// Performs the vector scale and add operation.
/// > y = alpha * x + y.
///
/// # Panics
/// If the vectors do not have the same length
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::needless_pass_by_value
)]
pub fn axpy_ctx<F: BLAS1<C>, C: BLASContext<Device = D>, D: Device>(
    ctx: C,
    alpha: F,
    x: VectorView<F, D>,
    y: VectorViewMut<F, D>,
) {
    let [n] = x.shape;
    let [m] = y.shape;
    assert_eq!(n, m);

    let incx = x.strides[0] as i32;
    let incy = y.strides[0] as i32;

    unsafe {
        F::axpy(
            ctx,
            n as i32,
            alpha,
            x.data.as_ref().as_ptr(),
            incx,
            y.data.as_ptr(),
            incy,
        );
    }
}

/// Performs the vector scale operation.
/// > x = alpha * x.
///
/// # Panics
/// If the vectors do not have the same length
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::needless_pass_by_value
)]
pub fn scal_ctx<F: BLAS1<C>, C: BLASContext<Device = D>, D: Device>(
    ctx: C,
    alpha: F,
    x: VectorViewMut<F, D>,
) {
    let [n] = x.shape;
    let incx = x.strides[0] as i32;

    unsafe {
        F::scal(ctx, n as i32, alpha, x.data.as_ref().as_ptr(), incx);
    }
}
