use std::mem::MaybeUninit;

use num_traits::{One, Zero};

use crate::{
    blas::{BLASDevice, MatrixOp, GEMM},
    dims::{Dimension, ReduceDim},
    ptr::slice::Slice,
    storage::{IntoOwned, Storage, StorageMut},
    vec::Vec,
};

pub struct Tensor<S: Storage, Dim: Dimension>
where
    S::Device: BLASDevice,
{
    shape: Dim,
    strides: Dim,
    data: S,

    ctx: <S::Device as BLASDevice>::Context,
}

impl<S: Storage, Dim: Dimension> Tensor<S, Dim>
where
    S::Device: BLASDevice,
{
    pub fn from_shape_in(ctx: <S::Device as BLASDevice>::Context, shape: Dim, data: S) -> Self {
        assert_eq!(data.as_ref().len(), shape.len());
        let strides = shape.column_major_strides();
        Self {
            shape,
            strides,
            data,
            ctx,
        }
    }

    pub fn from_shape(shape: Dim, data: S) -> Self
    where
        <S::Device as BLASDevice>::Context: Default,
    {
        Self::from_shape_in(Default::default(), shape, data)
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

    pub fn view(&self) -> Tensor<&Slice<S::T, S::Device>, Dim> {
        Tensor {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: self.data.as_ref(),
            ctx: self.ctx.clone(),
        }
    }

    pub fn view_mut(&mut self) -> Tensor<&mut Slice<S::T, S::Device>, Dim>
    where
        S: StorageMut,
    {
        Tensor {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: self.data.as_mut(),
            ctx: self.ctx.clone(),
        }
    }

    pub fn t(&self) -> Tensor<&Slice<S::T, S::Device>, Dim> {
        let mut view = self.view();
        view.reverse_axes();
        view
    }

    pub fn into_owned(self) -> Tensor<S::Owned, Dim>
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

    pub fn slice_axis(&self, axis: usize, n: usize) -> Tensor<&Slice<S::T, S::Device>, Dim::Smaller>
    where
        Dim: ReduceDim,
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

impl<S: Storage> Tensor<S, [usize; 2]>
where
    S::Device: BLASDevice,
{
    pub fn dot(
        &self,
        rhs: Tensor<impl Storage<T = S::T, Device = S::Device>, [usize; 2]>,
    ) -> Tensor<Vec<S::T, S::Device>, [usize; 2]>
    where
        S::T: Zero + One,
        S::Device: Default,
        S::T: GEMM<S::Device>,
    {
        self.dot_in(rhs, S::Device::default())
    }

    pub fn dot_in(
        &self,
        rhs: Tensor<impl Storage<T = S::T, Device = S::Device>, [usize; 2]>,
        device: S::Device,
    ) -> Tensor<Vec<S::T, S::Device>, [usize; 2]>
    where
        S::T: Zero + One,
        S::T: GEMM<S::Device>,
    {
        let rows = self.shape[0];
        let cols = rhs.shape[1];
        let mut v = Vec::with_capacity_in(rows * cols, device);
        unsafe {
            let uninit = Tensor::from_shape_in(
                self.ctx.clone(),
                [rows, cols],
                &mut v.space_capacity_mut()[..rows * cols],
            );

            gemm_uninit(S::T::one(), self.view(), rhs, uninit);

            v.set_len(rows * cols);
        }
        Tensor::from_shape_in(self.ctx.clone(), [rows, cols], v)
    }
}

impl<'a, T: Copy, D: BLASDevice, Dim: Dimension> Tensor<&'a mut Slice<MaybeUninit<T>, D>, Dim> {
    /// # Safety
    /// Contents must be initialised
    pub unsafe fn assume_init(self) -> Tensor<&'a mut Slice<T, D>, Dim> {
        Tensor {
            shape: self.shape,
            strides: self.strides,
            data: self.data.assume_init_mut(),
            ctx: self.ctx,
        }
    }
}

impl<'a, T: Copy, D: BLASDevice + Default, Dim: Dimension>
    Tensor<&'a Slice<MaybeUninit<T>, D>, Dim>
{
    /// # Safety
    /// Contents must be initialised
    pub unsafe fn assume_init(self) -> Tensor<&'a Slice<T, D>, Dim> {
        Tensor {
            shape: self.shape,
            strides: self.strides,
            data: self.data.assume_init(),
            ctx: self.ctx,
        }
    }
}

pub fn gemm_uninit<F: GEMM<D> + Zero, D: BLASDevice>(
    alpha: F,
    a: Tensor<impl Storage<T = F, Device = D>, [usize; 2]>,
    b: Tensor<impl Storage<T = F, Device = D>, [usize; 2]>,
    c: Tensor<&mut Slice<MaybeUninit<F>, D>, [usize; 2]>,
) {
    // Safety:
    // Specifying beta == 0.0 should allow c to be safely read while uninitialised
    unsafe { gemm(alpha, a, b, F::zero(), c.assume_init()) }
}

pub fn gemm<F: GEMM<D>, D: BLASDevice>(
    alpha: F,
    a: Tensor<impl Storage<T = F, Device = D>, [usize; 2]>,
    b: Tensor<impl Storage<T = F, Device = D>, [usize; 2]>,
    beta: F,
    c: Tensor<&mut Slice<F, D>, [usize; 2]>,
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

    use crate::{
        device::Device,
        tensor::{gemm, Tensor},
        vec::Vec,
    };

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
    fn matmul_cuda() {
        use crate::device::cuda::Cuda;

        let cuda_ctx = cust::quick_init().unwrap();

        {
            let cuda = Cuda::new(cuda_ctx.get_unowned());

            // column major
            let a = Vec::copy_from_host_in(&[0., 2., 4., 1., 3., 5.], cuda.clone());
            let b = Vec::copy_from_host_in(&[0., 2., 1., 3.], cuda.clone());

            let ctx = cuda.init_cublas();

            let a = Tensor::from_shape_in(ctx, [3, 2], a);
            let b = Tensor::from_shape_in(ctx, [2, 2], b);

            let c = a.dot_in(b, cuda);

            let mut out = vec![0.0_f32; 6];
            Cuda::copy_to_host(c.data.deref(), &mut out);

            assert_eq!(out, vec![2., 6., 10., 3., 11., 19.]);
        }
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

        approx::assert_relative_eq!(out[0], expected[0]);
        approx::assert_relative_eq!(out[1], expected[1]);
        approx::assert_relative_eq!(out[2], expected[2]);
        approx::assert_relative_eq!(out[3], expected[3]);
    }

    #[test]
    fn matmul_cuda2() {
        use crate::device::cuda::Cuda;

        let ctx = cust::quick_init().unwrap();

        {
            let cuda = Cuda::new(ctx.get_unowned());

            // column major
            let a = Vec::copy_from_host_in(&[0.001, 1.0, 1.0, 0.], cuda.clone());
            let b = a.clone();
            let c = b.clone();

            let handle = cuda.init_cublas();

            let mut a = Tensor::from_shape_in(handle, [2, 2], a);
            let b = Tensor::from_shape_in(handle, [2, 2], b);
            let mut c = Tensor::from_shape_in(handle, [2, 2], c);

            for _ in 0..1000 {
                gemm(1., a.view(), b.view(), 0., c.view_mut());
                std::mem::swap(&mut a, &mut c);
            }

            let mut out = vec![0.; 4];
            Cuda::copy_to_host(c.data.deref(), &mut out);

            let expected = [
                1.1278865019586632,
                0.5210952168646452,
                0.5210952168646452,
                1.1273654067417986,
            ];

            approx::assert_relative_eq!(out[0], expected[0]);
            approx::assert_relative_eq!(out[1], expected[1]);
            approx::assert_relative_eq!(out[2], expected[2]);
            approx::assert_relative_eq!(out[3], expected[3]);
        }

        cust::context::Context::drop(ctx).unwrap();
    }
}
