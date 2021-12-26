
use std::mem::MaybeUninit;

use num_traits::{Zero, One};

use crate::{blas::{MatrixOp, BLASDevice, BLAS}, ptr::slice::Slice, device::Device, vec::Vec};

/// Low level matrix
pub struct Matrix<S: Storage> where S::Device: BLASDevice {
    shape: [usize; 2],
    strides: [usize; 2],
    data: S,

    ctx: <S::Device as BLASDevice>::Context,
}

impl<S: Storage> Matrix<S> where S::Device: BLASDevice + Default {
    pub fn from_shape_in(ctx: <S::Device as BLASDevice>::Context, rows: usize, cols: usize, data: S) -> Self {
        assert_eq!(data.as_ref().len(), rows * cols);
        Self {
            shape: [rows, cols],
            strides: [1, rows], // column major
            data,
            ctx
        }
    }

    pub fn view(&self) -> Matrix<&Slice<S::T, S::Device>> where S::T: Copy {
        Matrix {
            shape: self.shape,
            strides: self.strides,
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
            shape: self.shape,
            strides: self.strides,
            data: self.data.as_mut(),
            ctx: self.ctx.clone(),
        }
    }

    pub fn t(&self) -> Matrix<&Slice<S::T, S::Device>> where S::T: Copy {
        let mut view = self.view();
        view.swap_axes();
        view
    }

    pub fn swap_axes(&mut self) {
        self.shape.reverse();
        self.strides.reverse();
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
        let rows = self.shape[0];
        let cols = rhs.shape[1];
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
            shape: self.shape,
            strides: self.strides,
            data: self.data.assume_init_mut(),
            ctx: self.ctx,
        }
    }
}

impl<'a, T: Copy, D: BLASDevice + Default> Matrix<&'a Slice<MaybeUninit<T>, D>> {
    pub unsafe fn assume_init(self) -> Matrix<&'a Slice<T, D>> {
        Matrix {
            shape: self.shape,
            strides: self.strides,
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
    assert_eq!(a.shape[1], b.shape[0]);
    assert_eq!(a.shape[0], c.shape[0]);
    assert_eq!(b.shape[1], c.shape[1]);

    let m = a.shape[0] as i32;
    let n = b.shape[1] as i32;
    let k = a.shape[1] as i32;

    let (transa, lda) = if a.strides[0] == 1 {
        (MatrixOp::NoTrans, a.strides[1] as i32)
    } else if a.strides[1] == 1{
        (MatrixOp::Trans, a.strides[0] as i32)
    } else {
        panic!("one of the strides must be 1 (contiguous)")
    };
    let (transb, ldb) = if b.strides[0] == 1 {
        (MatrixOp::NoTrans, b.strides[1] as i32)
    } else if b.strides[1] == 1{
        (MatrixOp::Trans, b.strides[0] as i32)
    } else {
        panic!("one of the strides must be 1 (contiguous)")
    };

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

pub trait Storage {
    type T;
    type Device: Device;
    fn into_owned(self) -> Vec<Self::T, Self::Device>;
    fn as_ref(&self) -> &Slice<Self::T, Self::Device>;
}

pub trait StorageMut: Storage {
    fn as_mut(&mut self) -> &mut Slice<Self::T, Self::Device>;
}

impl<T, D: Device> Storage for Vec<T, D> {
    type T = T;
    type Device = D;

    fn into_owned(self) -> Vec<T, D> {
        self
    }

    fn as_ref(&self) -> &Slice<T, D> {
        self
    }
}

impl<'a, T: Copy, D: Device + Default> Storage for &'a Slice<T, D> {
    type T = T;
    type Device = D;
    fn into_owned(self) -> Vec<T, D> {
        self.to_owned()
    }

    fn as_ref(&self) -> &Slice<T, D> {
        self
    }
}

impl<'a, T: Copy, D: Device + Default> Storage for &'a mut Slice<T, D> {
    type T = T;
    type Device = D;
    fn into_owned(self) -> Vec<T, D> {
        self.to_owned()
    }

    fn as_ref(&self) -> &Slice<T, D> {
        self
    }
}

impl<'a, T, D: Device> StorageMut for Vec<T, D> {
    fn as_mut(&mut self) -> &mut Slice<T, D> {
        self
    }
}

impl<'a, T: Copy, D: Device + Default> StorageMut for &'a mut Slice<T, D> {
    fn as_mut(&mut self) -> &mut Slice<T, D> {
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::{vec::Vec, matrix::Matrix};

    #[test]
    fn matmul() {
        let a: Vec<f32, _> = Vec::from(vec![0., 2., 4., 1., 3., 5.]);
        let b: Vec<f32, _> = Vec::from(vec![0., 2., 1., 3.]);
        let a = Matrix::from_shape_in((), 3, 2, a);
        let b = Matrix::from_shape_in((), 2, 2, b);

        let c = a.dot(b);

        // result of column major dot
        assert_eq!(std::vec::Vec::from(c.data), vec![2., 6., 10., 3., 11., 19.]);
    }

    #[test]
    fn matmul_t() {
        let a: Vec<f32, _> = Vec::from(vec![1., 2., 3., 4.]);
        let b: Vec<f32, _> = Vec::from(vec![5., 6., 7., 8.]);
        let a = Matrix::from_shape_in((), 2, 2, a);
        let b = Matrix::from_shape_in((), 2, 2, b);

        let c = a.t().dot(b.t());

        // result of row major dot
        assert_eq!(std::vec::Vec::from(c.data), vec![19.0, 43.0, 22.0, 50.0]);
    }
}
