use std::mem::MaybeUninit;

use tensorgraph_sys::{ptr::Ref, View, ViewMut, device::Device};

use crate::{
    dims::{Dimension, RemoveDim},
    storage::{IntoOwned, Storage, StorageMut},
};

mod dot;
mod matrix;
mod vector;
pub use dot::Dot;
pub use matrix::*;
pub use vector::*;

/// A representation of a slice
pub type Slice<T, D> = Ref<[T], D>;

/// Gets the view repr of the provided storage
pub type ViewOf<S> = Slice<<S as Storage>::T, <S as Storage>::Device>;

/// A 'view' of a tensor, Like `&[T]` is to `Vec<T>`
pub type TensorView<'a, T, D, Dim> = Tensor<&'a Slice<T, D>, Dim>;

/// A 'mut view' of a tensor, Like `&mut [T]` is to `Vec<T>`
pub type TensorViewMut<'a, T, D, Dim> = Tensor<&'a mut Slice<T, D>, Dim>;

/// An uninit tensor. Contents are mutable and specified as [`MaybeUninit`].
pub type UninitTensor<'a, T, D, Dim> = TensorViewMut<'a, MaybeUninit<T>, D, Dim>;

/// A multidimensional data structure not unlike [`ndarray::ArrayBase`](https://docs.rs/ndarray/0.15.4/ndarray/struct.ArrayBase.html).
#[derive(Copy, Clone)]
pub struct Tensor<S: Storage, Dim: Dimension> {
    shape: Dim,
    strides: Dim,
    data: S,
}

impl<S: Storage, Dim: Dimension> View for Tensor<S, Dim> {
    type Ref<'a>
    where
        Self: 'a,
    = Tensor<&'a ViewOf<S>, Dim>;

    fn view(&self) -> TensorView<S::T, S::Device, Dim> {
        Tensor {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: self.data.as_ref(),
        }
    }
}

impl<S: StorageMut, Dim: Dimension> ViewMut for Tensor<S, Dim> {
    type Mut<'a>
    where
        Self: 'a,
    = Tensor<&'a mut ViewOf<S>, Dim>;

    fn view_mut(&mut self) -> TensorViewMut<S::T, S::Device, Dim> {
        Tensor {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: self.data.as_mut(),
        }
    }
}

impl<S: Storage, Dim: Dimension> Tensor<S, Dim> {
    /// Creates a new tensor using the shape and the raw data.
    ///
    /// # Panics
    /// The length of the data structure must match the size of the dimensions
    pub fn from_shape(shape: Dim, data: S) -> Self {
        assert_eq!(data.as_ref().len(), shape.size());
        let strides = shape.column_major_strides();
        Self {
            shape,
            strides,
            data,
        }
    }

    /// Consumes the tensor, returning the underlying data
    pub fn into_inner(self) -> S {
        self.data
    }

    /// Reverses the axes of the tensor. An inplace transpose
    pub fn reverse_axes(&mut self) {
        self.shape.as_mut().reverse();
        self.strides.as_mut().reverse();
    }

    pub fn swap_axes(&mut self, i: usize, j: usize) {
        self.shape.as_mut().swap(i, j);
        self.strides.as_mut().swap(i, j);
    }

    /// Returns a view of the tensor with the contents transposed.
    /// This operation happens without mutating or cloning any data
    pub fn t(&self) -> Tensor<&ViewOf<S>, Dim> {
        let mut view = self.view();
        view.reverse_axes();
        view
    }

    /// Creates a new owned version of the tensor.
    /// Will only clone the contents if needed
    pub fn into_owned(self) -> Tensor<S::Owned, Dim>
    where
        S: IntoOwned,
        S::Owned: Storage<T = S::T, Device = S::Device>,
    {
        Tensor {
            shape: self.shape,
            strides: self.strides,
            data: self.data.into_owned(),
        }
    }

    /// Slices the tensor over a specific axis. The resulting tensor will be a dimension smaller
    ///
    /// # Panics
    /// If the axis is outside of the length of the dimensions
    pub fn slice_axis(&self, axis: usize, n: usize) -> Tensor<&ViewOf<S>, Dim::Smaller>
    where
        Dim: RemoveDim,
    {
        assert!(axis < self.shape.as_ref().len());

        let (shape, m) = self.shape.remove(axis);
        let (strides, s) = self.strides.remove(axis);

        assert!(n < m);

        Tensor {
            shape,
            strides,
            data: &self.data.as_ref()[s * n..],
        }
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

#[cfg(test)]
mod tests {
    use tensorgraph_sys::{View, ViewMut};

    use crate::tensor::{gemm, Tensor};

    #[test]
    fn matmul() {
        //     0 1
        // A = 2 3
        //     4 5

        // B = 0 1
        //     2 3

        // column major (read each column first)
        let a = [0., 2., 4., 1., 3., 5.];
        let b = [0., 2., 1., 3.];
        let a = Tensor::from_shape([3, 2], a); // 3 rows x 2 cols
        let b = Tensor::from_shape([2, 2], b); // 2 rows x 2 cols

        //           2  3
        // C = AB =  6 11
        //          10 19

        let c = a.matmul(b.view());
        assert_eq!(c.into_inner().into_std(), [2., 6., 10., 3., 11., 19.]);
    }

    #[test]
    fn matmul_t() {
        // A = 1 3
        //     2 4

        // B = 5 7
        //     6 8

        let a = [1., 2., 3., 4.];
        let b = [5., 6., 7., 8.];
        let a = Tensor::from_shape([2, 2], a);
        let b = Tensor::from_shape([2, 2], b);

        // C1 = A^B^ = 19 22
        //             43 50

        let c1 = a.t().matmul(b.t());
        assert_eq!(c1.into_inner().into_std(), [19.0, 43.0, 22.0, 50.0]);

        // C2 = AB^ = 26 30
        //            38 44

        let c2 = a.matmul(b.t());
        assert_eq!(c2.into_inner().into_std(), [26.0, 38.0, 30.0, 44.0]);

        // C3 = A^B = 17 23
        //            39 53

        let c3 = a.t().matmul(b.view());
        assert_eq!(c3.into_inner().into_std(), [17.0, 39.0, 23.0, 53.0]);
    }

    #[test]
    fn slice() {
        //     0 2 4
        // A = 1 3 5

        // column major
        let a = [0., 1., 2., 3., 4., 5.];
        let a = Tensor::from_shape([2, 3], a);

        // axis 0 (columns)
        let a00 = a.slice_axis(0, 0);
        assert_eq!(&**a00.into_inner(), [0., 1., 2., 3., 4., 5.]); // represents 0, 2, 4
        assert_eq!(a00.shape, [3]);
        assert_eq!(a00.strides, [2]);

        let a01 = a.slice_axis(0, 1);
        assert_eq!(&**a01.into_inner(), [1., 2., 3., 4., 5.]); // represents 1, 3, 5
        assert_eq!(a01.shape, [3]);
        assert_eq!(a01.strides, [2]);

        // axis 1 (rows)
        let a10 = a.slice_axis(1, 0);
        assert_eq!(&**a10.into_inner(), [0., 1., 2., 3., 4., 5.]); // represents 0, 1
        assert_eq!(a10.shape, [2]);
        assert_eq!(a10.strides, [1]);

        let a11 = a.slice_axis(1, 1);
        assert_eq!(&**a11.into_inner(), [2., 3., 4., 5.]); // represents 2, 3
        assert_eq!(a11.shape, [2]);
        assert_eq!(a11.strides, [1]);

        let a12 = a.slice_axis(1, 2);
        assert_eq!(&**a12.into_inner(), [4., 5.]); // represents 4, 5
        assert_eq!(a12.shape, [2]);
        assert_eq!(a12.strides, [1]);
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn matmul_cuda() {
        use crate::blas::cublas::CublasContext;
        use tensorgraph_sys::{
            device::cuda::{Context, Stream},
            Vec,
        };

        let ctx = Context::quick_init().unwrap();
        let cuda = Stream::new(&ctx).unwrap();
        let cuda = &*cuda;

        // column major
        let a = Vec::copy_from_host_in(&[0., 2., 4., 1., 3., 5.], cuda);
        let b = Vec::copy_from_host_in(&[0., 2., 1., 3.], cuda);

        let ctx = CublasContext::new();
        let ctx = ctx.with_stream(Some(cuda));

        let a = Tensor::from_shape([3, 2], a);
        let b = Tensor::from_shape([2, 2], b);

        let c = a.matmul_into(b.view(), ctx, cuda);

        let mut out = vec![0.0_f32; 6];
        c.data.copy_to_host(&mut out);

        assert_eq!(out, vec![2., 6., 10., 3., 11., 19.]);
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn matmul_cuda_global() {
        use crate::blas::cublas::CublasContext;
        use tensorgraph_sys::{
            device::cuda::{Context, Cuda, Stream},
            DefaultVec,
        };

        let ctx = Context::quick_init().unwrap();
        let cuda = Stream::new(&ctx).unwrap();
        let _handle = cuda.as_global();

        // column major
        let a = DefaultVec::<f32, Cuda>::copy_from_host(&[0., 2., 4., 1., 3., 5.]);
        let b = DefaultVec::<f32, Cuda>::copy_from_host(&[0., 2., 1., 3.]);

        let ctx = CublasContext::new();
        let _handle = ctx.with_stream(Some(&cuda)).as_global();

        let a = Tensor::from_shape([3, 2], a);
        let b = Tensor::from_shape([2, 2], b);

        let c = a.matmul(b.view());

        let mut out = vec![0.0_f32; 6];
        c.data.copy_to_host(&mut out);

        assert_eq!(out, vec![2., 6., 10., 3., 11., 19.]);
    }

    #[test]
    fn matmul2() {
        // column major
        let a = [0.001, 1.0, 1.0, 0.];
        let b = a;
        let c = [0.; 4];

        let mut a = Tensor::from_shape([2, 2], a);
        let b = Tensor::from_shape([2, 2], b);
        let mut c = Tensor::from_shape([2, 2], c);

        for _ in 0..1000 {
            gemm(1., a.view(), b.view(), 0., c.view_mut());
            std::mem::swap(&mut a, &mut c);
        }

        let out = c.into_inner();
        let expected = [
            1.1278865019586632,
            0.5210952168646452,
            0.5210952168646452,
            1.1273654067417986,
        ];

        assert_eq!(out[0], expected[0]);
        assert_eq!(out[1], expected[1]);
        assert_eq!(out[2], expected[2]);
        assert_eq!(out[3], expected[3]);
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn matmul_cuda2() {
        use crate::{blas::cublas::CublasContext, tensor::gemm_ctx};
        use tensorgraph_sys::{
            device::cuda::{Context, Stream},
            Vec,
        };

        let ctx = Context::quick_init().unwrap();
        let cuda = Stream::new(&ctx).unwrap();
        let cuda = &*cuda;

        // column major
        let a = Vec::copy_from_host_in(&[0.001, 1.0, 1.0, 0.], cuda);
        let b = a.clone();
        let c = b.clone();

        let ctx = CublasContext::new();
        let ctx = ctx.with_stream(Some(cuda));

        let mut a = Tensor::from_shape([2, 2], a);
        let b = Tensor::from_shape([2, 2], b);
        let mut c = Tensor::from_shape([2, 2], c);

        for _ in 0..1000 {
            gemm_ctx(ctx, 1., a.view(), b.view(), 0., c.view_mut());
            std::mem::swap(&mut a, &mut c);
        }

        let mut out = vec![0.; 4];
        c.data.copy_to_host(&mut out);

        let expected = [
            1.1278865019586632,
            0.5210952168646452,
            0.5210952168646452,
            1.1273654067417986,
        ];

        assert_eq!(out[0], expected[0]);
        assert_eq!(out[1], expected[1]);
        assert_eq!(out[2], expected[2]);
        assert_eq!(out[3], expected[3]);
    }
}
