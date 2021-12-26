#![feature(
    generic_associated_types,
    allocator_api,
    alloc_layout_extra,
    nonnull_slice_from_raw_parts,
    slice_ptr_len,
    ptr_metadata,
    maybe_uninit_slice
)]

use std::mem::MaybeUninit;

use blas::{MatrixOp, BLAS, BLASDevice};
use device::Device;
use num_traits::{One, Zero};
use ptr::slice::Slice;
use vec::Vec;

pub mod blas;
pub mod device;
pub mod ptr;
pub mod vec;

/// Low level matrix
pub struct Matrix<S: Storage> where S::Device: BLASDevice {
    rows: usize,
    cols: usize,
    op: MatrixOp,
    data: S,

    ctx: <S::Device as BLASDevice>::Context,
}

impl<S: Storage> Matrix<S> where S::Device: BLASDevice + Default {
    pub fn from_shape_in(ctx: <S::Device as BLASDevice>::Context, rows: usize, cols: usize, data: S) -> Self {
        assert_eq!(data.as_ref().len(), rows * cols);
        Self {
            rows,
            cols,
            op: MatrixOp::NoTrans,
            data,
            ctx
        }
    }

    pub fn view(&self) -> Matrix<&Slice<S::T, S::Device>> where S::T: Copy {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            op: self.op,
            data: self.data.as_ref(),
            ctx: self.ctx.clone(),
        }
    }

    pub fn view_mut(&mut self) -> Matrix<&mut Slice<S::T, S::Device>>
    where
        S: StorageMut,
        S::T: Copy,
    {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            op: self.op,
            data: self.data.as_mut(),
            ctx: self.ctx.clone(),
        }
    }

    pub fn t(&mut self) {
        let op = match self.op {
            MatrixOp::NoTrans => MatrixOp::Trans,
            MatrixOp::Trans => MatrixOp::NoTrans,
            MatrixOp::ConjTrans => panic!("can't yet deal with conjugate transposed matricies"),
        };
        std::mem::swap(&mut self.rows, &mut self.cols);
        self.op = op;
    }

    pub fn dot(
        &self,
        rhs: Matrix<impl Storage<T = S::T, Device = S::Device>>,
    ) -> Matrix<Vec<S::T, S::Device>>
    where
        S::T: Zero + One,
        S::Device: Default,
        S::T: BLAS<S::Device>
    {
        let rows = self.rows;
        let cols = self.cols;
        let mut v = Vec::with_capacity(rows * cols);
        unsafe {
            let uninit = Matrix::from_shape_in(self.ctx.clone(), rows, cols, &mut v.space_capacity_mut()[..rows * cols]);

            gemm_uninit(S::T::one(), self.view(), rhs, uninit);

            v.set_len(rows * cols);
        }
        Matrix::from_shape_in(self.ctx.clone(), rows, cols, v)
    }
}

impl<'a, T: Copy, D: BLASDevice + Default> Matrix<&'a mut Slice<MaybeUninit<T>, D>> {
    pub unsafe fn assume_init(self) -> Matrix<&'a mut Slice<T, D>> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            op: self.op,
            data: self.data.assume_init_mut(),
            ctx: self.ctx,
        }
    }
}

impl<'a, T: Copy, D: BLASDevice + Default> Matrix<&'a Slice<MaybeUninit<T>, D>> {
    pub unsafe fn assume_init(self) -> Matrix<&'a Slice<T, D>> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            op: self.op,
            data: self.data.assume_init(),
            ctx: self.ctx,
        }
    }
}

pub fn gemm_uninit<F: BLAS<D> + Zero, D: BLASDevice + Default>(
    alpha: F,
    a: Matrix<impl Storage<T = F, Device = D>>,
    b: Matrix<impl Storage<T = F, Device = D>>,
    c: Matrix<&mut Slice<MaybeUninit<F>, D>>,
) {
    // Safety:
    // Specifying beta == 0.0 should allow c to be safely read while uninitialised
    unsafe { gemm(alpha, a, b, F::zero(), c.assume_init()) }
}

pub fn gemm<F: BLAS<D>, D: BLASDevice + Default>(
    alpha: F,
    a: Matrix<impl Storage<T = F, Device = D>>,
    b: Matrix<impl Storage<T = F, Device = D>>,
    beta: F,
    c: Matrix<&mut Slice<F, D>>,
) {
    assert_eq!(a.cols, b.rows);
    assert_eq!(a.rows, c.rows);
    assert_eq!(b.cols, c.cols);

    let m = a.rows as i32;
    let n = b.cols as i32;
    let k = a.cols as i32;

    let transa = a.op;
    let transb = b.op;

    // currently hard coding the strides
    let lda = m;
    let ldb = k;
    let ldc = m;

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

pub trait Storage {
    type T;
    type Device: Device;
    fn into_owned(self) -> Vec<Self::T, Self::Device>;
    fn as_ref(&self) -> &Slice<Self::T, Self::Device>;
}

pub trait StorageMut: Storage {
    fn as_mut(&mut self) -> &mut Slice<Self::T, Self::Device>;
}

impl<T, D: device::Device> Storage for Vec<T, D> {
    type T = T;
    type Device = D;

    fn into_owned(self) -> Vec<T, D> {
        self
    }

    fn as_ref(&self) -> &Slice<T, D> {
        self
    }
}

impl<'a, T: Copy, D: device::Device + Default> Storage for &'a Slice<T, D> {
    type T = T;
    type Device = D;
    fn into_owned(self) -> Vec<T, D> {
        self.to_owned()
    }

    fn as_ref(&self) -> &Slice<T, D> {
        self
    }
}

impl<'a, T: Copy, D: device::Device + Default> Storage for &'a mut Slice<T, D> {
    type T = T;
    type Device = D;
    fn into_owned(self) -> Vec<T, D> {
        self.to_owned()
    }

    fn as_ref(&self) -> &Slice<T, D> {
        self
    }
}

impl<'a, T, D: device::Device> StorageMut for Vec<T, D> {
    fn as_mut(&mut self) -> &mut Slice<T, D> {
        self
    }
}

impl<'a, T: Copy, D: device::Device + Default> StorageMut for &'a mut Slice<T, D> {
    fn as_mut(&mut self) -> &mut Slice<T, D> {
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::{vec::Vec, Matrix};

    #[test]
    fn matmul() {
        let a: Vec<f32, _> = Vec::from(vec![1., 2., 3., 4.]);
        let b: Vec<f32, _> = Vec::from(vec![5., 6., 7., 8.]);
        let a = Matrix::from_shape_in((), 2, 2, a);
        let b = Matrix::from_shape_in((), 2, 2, b);

        let c = a.dot(b);

        // result of column major dot
        assert_eq!(std::vec::Vec::from(c.data), vec![23., 34., 31., 46.]);
    }
}
