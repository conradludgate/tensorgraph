use std::mem::MaybeUninit;

use crate::{
    blas::{BLASContext, DefaultBLASContext, BLAS1},
    storage::Storage,
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
    /// If vectors are not the same size
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
    )]
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
