use std::mem::MaybeUninit;

use num_traits::{One, Zero};

use tensorgraph_sys::{
    device::{DefaultDeviceAllocator, DeviceAllocator},
    ptr::reef::Ref,
    vec::Vec,
    Share, ShareMut,
};

use crate::{
    blas::{BLASContext, DefaultBLASContext, MatrixOp, GEMM},
    dims::{Dimension, RemoveDim},
    storage::{IntoOwned, Storage, StorageMut},
};

/// A multidimensional data structure not unlike [`ndarray::ArrayBase`].
pub struct Tensor<S: Storage, C: BLASContext<Device = S::Device>, Dim: Dimension> {
    shape: Dim,
    strides: Dim,
    data: S,
    ctx: C,
}

impl<S: Storage, Dim: Dimension> Tensor<S, <S::Device as DefaultBLASContext>::Context, Dim>
where
    S::Device: DefaultBLASContext,
{
    pub fn from_shape(shape: Dim, data: S) -> Self {
        Self::from_shape_in(S::Device::default_ctx(), shape, data)
    }
}

impl<S: Storage, C: BLASContext<Device = S::Device>, Dim: Dimension> Share for Tensor<S, C, Dim> {
    type Ref<'a>
    where
        Self: 'a,
    = TensorView<'a, S::T, S::Device, C, Dim>;

    fn share(&self) -> TensorView<S::T, S::Device, C, Dim> {
        Tensor {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: self.data.as_ref(),
            ctx: self.ctx.clone(),
        }
    }
}

impl<S: StorageMut, C: BLASContext<Device = S::Device>, Dim: Dimension> ShareMut
    for Tensor<S, C, Dim>
{
    type Mut<'a>
    where
        Self: 'a,
    = TensorViewMut<'a, S::T, S::Device, C, Dim>;

    fn share_mut(&mut self) -> TensorViewMut<S::T, S::Device, C, Dim> {
        Tensor {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: self.data.as_mut(),
            ctx: self.ctx.clone(),
        }
    }
}

impl<S: Storage, C: BLASContext<Device = S::Device>, Dim: Dimension> Tensor<S, C, Dim> {
    pub fn from_shape_in(ctx: C, shape: Dim, data: S) -> Self {
        assert_eq!(data.as_ref().len(), shape.len());
        let strides = shape.column_major_strides();
        Self {
            shape,
            strides,
            data,
            ctx,
        }
    }

    pub fn into_inner(self) -> S {
        self.data
    }

    pub fn reverse_axes(&mut self) {
        self.shape.as_mut().reverse();
        self.strides.as_mut().reverse();
    }

    pub fn swap_axes(&mut self, i: usize, j: usize) {
        self.shape.as_mut().swap(i, j);
        self.strides.as_mut().swap(i, j);
    }

    pub fn t(&self) -> TensorView<S::T, S::Device, C, Dim> {
        let mut view = self.share();
        view.reverse_axes();
        view
    }

    pub fn into_owned(self) -> Tensor<S::Owned, C, Dim>
    where
        S: IntoOwned,
        S::Owned: Storage<T = S::T, Device = S::Device>,
    {
        Tensor {
            shape: self.shape,
            strides: self.strides,
            data: self.data.into_owned(),
            ctx: self.ctx,
        }
    }

    pub fn slice_axis(&self, axis: usize, n: usize) -> TensorView<S::T, S::Device, C, Dim::Smaller>
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
            ctx: self.ctx.clone(),
        }
    }
}

pub type DefaultVec<T, D> = Vec<T, <D as DefaultDeviceAllocator>::Alloc>;
pub type TensorView<'a, T, D, C, Dim> = Tensor<&'a Ref<[T], D>, C, Dim>;
pub type TensorViewMut<'a, T, D, C, Dim> = Tensor<&'a mut Ref<[T], D>, C, Dim>;
pub type UninitTensor<'a, T, D, C, Dim> = TensorViewMut<'a, MaybeUninit<T>, D, C, Dim>;

impl<S: Storage, C: BLASContext<Device = S::Device>> Tensor<S, C, [usize; 2]> {
    pub fn dot(
        &self,
        rhs: Tensor<impl Storage<T = S::T, Device = S::Device>, C, [usize; 2]>,
    ) -> Tensor<DefaultVec<S::T, S::Device>, C, [usize; 2]>
    where
        S::T: Zero + One,
        S::Device: DefaultDeviceAllocator,
        S::T: GEMM<C>,
    {
        self.dot_in(rhs, S::Device::default_alloc())
    }

    pub fn dot_in<A: DeviceAllocator<Device = S::Device>>(
        &self,
        rhs: Tensor<impl Storage<T = S::T, Device = S::Device>, C, [usize; 2]>,
        alloc: A,
    ) -> Tensor<Vec<S::T, A>, C, [usize; 2]>
    where
        S::T: Zero + One,
        S::T: GEMM<C>,
    {
        let rows = self.shape[0];
        let cols = rhs.shape[1];
        let mut v = Vec::with_capacity_in(rows * cols, alloc);
        unsafe {
            let uninit = Tensor::from_shape_in(
                self.ctx.clone(),
                [rows, cols],
                &mut v.space_capacity_mut()[..rows * cols],
            );

            gemm_uninit(S::T::one(), self.share(), rhs, uninit);

            v.set_len(rows * cols);
        }
        Tensor::from_shape_in(self.ctx.clone(), [rows, cols], v)
    }
}

impl<'a, T: Copy, C: BLASContext, Dim: Dimension> UninitTensor<'a, T, C::Device, C, Dim> {
    /// # Safety
    /// Contents must be initialised
    pub unsafe fn assume_init(self) -> TensorViewMut<'a, T, C::Device, C, Dim> {
        Tensor {
            shape: self.shape,
            strides: self.strides,
            data: self.data.assume_init_mut(),
            ctx: self.ctx,
        }
    }
}

impl<'a, T: Copy, C: BLASContext, Dim: Dimension>
    TensorView<'a, MaybeUninit<T>, C::Device, C, Dim>
{
    /// # Safety
    /// Contents must be initialised
    pub unsafe fn assume_init(self) -> TensorView<'a, T, C::Device, C, Dim> {
        Tensor {
            shape: self.shape,
            strides: self.strides,
            data: self.data.assume_init(),
            ctx: self.ctx,
        }
    }
}

pub fn gemm_uninit<F: GEMM<C> + Zero, C: BLASContext>(
    alpha: F,
    a: Tensor<impl Storage<T = F, Device = C::Device>, C, [usize; 2]>,
    b: Tensor<impl Storage<T = F, Device = C::Device>, C, [usize; 2]>,
    c: UninitTensor<F, C::Device, C, [usize; 2]>,
) {
    // Safety:
    // Specifying beta == 0.0 should allow c to be safely read while uninitialised
    unsafe { gemm(alpha, a, b, F::zero(), c.assume_init()) }
}

pub fn gemm<F: GEMM<C> + Zero, C: BLASContext>(
    alpha: F,
    a: Tensor<impl Storage<T = F, Device = C::Device>, C, [usize; 2]>,
    b: Tensor<impl Storage<T = F, Device = C::Device>, C, [usize; 2]>,
    beta: F,
    c: TensorViewMut<F, C::Device, C, [usize; 2]>,
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
            a.ctx,
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

#[cfg(test)]
mod tests {
    use std::ops::Deref;

    use tensorgraph_sys::{vec::{vec_from_host, Vec}, Share, ShareMut};

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

        let c = a.dot(b);
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

        let c1 = a.t().dot(b.t());
        assert_eq!(c1.into_inner().into_std(), [19.0, 43.0, 22.0, 50.0]);

        // C2 = AB^ = 26 30
        //            38 44

        let c2 = a.dot(b.t());
        assert_eq!(c2.into_inner().into_std(), [26.0, 38.0, 30.0, 44.0]);

        // C3 = A^B = 17 23
        //            39 53

        let c3 = a.t().dot(b);
        assert_eq!(c3.into_inner().into_std(), [17.0, 39.0, 23.0, 53.0]);
    }

    #[test]
    fn slice() {
        // column major
        let a = [0., 1., 2., 3., 4., 5.];
        let a = Tensor::from_shape([2, 3], a);

        // axis 0 (columns)
        let a00 = a.slice_axis(0, 0);
        assert_eq!(a00.data.deref(), [0., 1., 2., 3., 4., 5.]); // represents 0, 2, 4
        assert_eq!(a00.shape, [3]);
        assert_eq!(a00.strides, [2]);

        let a01 = a.slice_axis(0, 1);
        assert_eq!(a01.data.deref(), [1., 2., 3., 4., 5.]); // represents 1, 3, 5
        assert_eq!(a01.shape, [3]);
        assert_eq!(a01.strides, [2]);

        // axis 1 (rows)
        let a10 = a.slice_axis(1, 0);
        assert_eq!(a10.data.deref(), [0., 1., 2., 3., 4., 5.]); // represents 0, 1
        assert_eq!(a10.shape, [2]);
        assert_eq!(a10.strides, [1]);

        let a11 = a.slice_axis(1, 1);
        assert_eq!(a11.data.deref(), [2., 3., 4., 5.]); // represents 2, 3
        assert_eq!(a11.shape, [2]);
        assert_eq!(a11.strides, [1]);

        let a12 = a.slice_axis(1, 2);
        assert_eq!(a12.data.deref(), [4., 5.]); // represents 4, 5
        assert_eq!(a12.shape, [2]);
        assert_eq!(a12.strides, [1]);
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn matmul_cuda() {
        use crate::blas::cublas::CublasContext;
        use tensorgraph_sys::device::cuda::{Context, Stream};

        let _ctx = Context::quick_init().unwrap();
        let cuda = Stream::new().unwrap();
        let cuda = cuda.deref();

        // column major
        let a = Vec::copy_from_host_in(&[0., 2., 4., 1., 3., 5.], cuda);
        let b = Vec::copy_from_host_in(&[0., 2., 1., 3.], cuda);

        let ctx = CublasContext::new();
        let ctx = ctx.with_stream(Some(cuda));

        let a = Tensor::from_shape_in(ctx, [3, 2], a);
        let b = Tensor::from_shape_in(ctx, [2, 2], b);

        let c = a.dot_in(b, cuda);

        let mut out = vec![0.0_f32; 6];
        c.data.copy_to_host(&mut out);

        assert_eq!(out, vec![2., 6., 10., 3., 11., 19.]);
    }

    #[test]
    #[cfg(feature = "cublas")]
    fn matmul_cuda_global() {
        use crate::blas::cublas::CublasContext;
        use tensorgraph_sys::device::cuda::{with_stream, Context, Cuda, Stream};

        let _ctx = Context::quick_init().unwrap();
        let cuda = Stream::new().unwrap();

        let out = with_stream(&cuda, |cuda| {
            // column major
            let a = vec_from_host::<f32, Cuda>(&[0., 2., 4., 1., 3., 5.]);
            let b = vec_from_host::<f32, Cuda>(&[0., 2., 1., 3.]);

            let ctx = CublasContext::new();
            let ctx = ctx.with_stream(Some(cuda));

            let a = Tensor::from_shape_in(ctx, [3, 2], a);
            let b = Tensor::from_shape_in(ctx, [2, 2], b);

            let c = a.dot(b);

            let mut out = vec![0.0_f32; 6];
            c.data.copy_to_host(&mut out);

            out
        });

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
            gemm(1., a.share(), b.share(), 0., c.share_mut());
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
        use crate::blas::cublas::CublasContext;
        use tensorgraph_sys::device::cuda::{Context, Stream};

        let _ctx = Context::quick_init().unwrap();
        let cuda = Stream::new().unwrap();
        let cuda = cuda.deref();

        // column major
        let a = Vec::copy_from_host_in(&[0.001, 1.0, 1.0, 0.], cuda);
        let b = a.clone();
        let c = b.clone();

        let ctx = CublasContext::new();
        let ctx = ctx.with_stream(Some(cuda));

        let mut a = Tensor::from_shape_in(ctx, [2, 2], a);
        let b = Tensor::from_shape_in(ctx, [2, 2], b);
        let mut c = Tensor::from_shape_in(ctx, [2, 2], c);

        for _ in 0..1000 {
            gemm(1., a.view(), b.view(), 0., c.view_mut());
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
