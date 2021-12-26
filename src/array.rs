use std::mem::MaybeUninit;

use num_traits::{One, Zero};

use crate::{
    blas::{BLASDevice, MatrixOp, GEMM},
    device::Device,
    ptr::slice::Slice,
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
    S::Device: BLASDevice + Default,
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

    pub fn reverse_axes(&mut self) {
        self.shape.as_mut().reverse();
        self.strides.as_mut().reverse();
    }

    pub fn swap_axes(&mut self, i: usize, j: usize) {
        self.shape.as_mut().swap(i, j);
        self.strides.as_mut().swap(i, j);
    }
}

impl<S: Storage, Dim: Dimension + Clone> Tensor<S, Dim>
where
    S::Device: BLASDevice + Default,
{
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
}

impl<S: Storage> Tensor<S, [usize; 2]>
where
    S::Device: BLASDevice + Default,
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
        let rows = self.shape[0];
        let cols = rhs.shape[1];
        let mut v = Vec::with_capacity(rows * cols);
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

impl<'a, T: Copy, D: BLASDevice + Default, Dim: Dimension>
    Tensor<&'a mut Slice<MaybeUninit<T>, D>, Dim>
{
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

pub fn gemm_uninit<F: GEMM<D> + Zero, D: BLASDevice + Default>(
    alpha: F,
    a: Tensor<impl Storage<T = F, Device = D>, [usize; 2]>,
    b: Tensor<impl Storage<T = F, Device = D>, [usize; 2]>,
    c: Tensor<&mut Slice<MaybeUninit<F>, D>, [usize; 2]>,
) {
    // Safety:
    // Specifying beta == 0.0 should allow c to be safely read while uninitialised
    unsafe { gemm(alpha, a, b, F::zero(), c.assume_init()) }
}

pub fn gemm<F: GEMM<D>, D: BLASDevice + Default>(
    alpha: F,
    a: Tensor<impl Storage<T = F, Device = D>, [usize; 2]>,
    b: Tensor<impl Storage<T = F, Device = D>, [usize; 2]>,
    beta: F,
    c: Tensor<&mut Slice<F, D>, [usize; 2]>,
) {
    assert_eq!(a.shape[1], b.shape[0]);
    assert_eq!(a.shape[0], c.shape[0]);
    assert_eq!(b.shape[1], c.shape[1]);

    let m = a.shape[0] as i32;
    let n = b.shape[1] as i32;
    let k = a.shape[1] as i32;

    let (transa, lda) = if a.strides[0] == 1 {
        (MatrixOp::NoTrans, a.strides[1] as i32)
    } else if a.strides[1] == 1 {
        (MatrixOp::Trans, a.strides[0] as i32)
    } else {
        panic!("one of the strides must be 1 (contiguous)")
    };
    let (transb, ldb) = if b.strides[0] == 1 {
        (MatrixOp::NoTrans, b.strides[1] as i32)
    } else if b.strides[1] == 1 {
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

pub trait IntoOwned {
    type Owned;
    fn into_owned(self) -> Self::Owned;
}

pub trait Storage: AsRef<Slice<Self::T, Self::Device>> {
    type T;
    type Device: Device;
}

pub trait StorageMut: Storage + AsMut<Slice<Self::T, Self::Device>> {}

impl<T, D: Device> IntoOwned for Vec<T, D> {
    type Owned = Self;
    fn into_owned(self) -> Self::Owned {
        self
    }
}

impl<'a, T: Copy, D: Device + Default> IntoOwned for &'a Slice<T, D> {
    type Owned = Vec<T, D>;
    fn into_owned(self) -> Self::Owned {
        self.to_owned()
    }
}

impl<'a, T: Copy, D: Device + Default> IntoOwned for &'a mut Slice<T, D> {
    type Owned = Vec<T, D>;
    fn into_owned(self) -> Self::Owned {
        self.to_owned()
    }
}

impl<T, D: Device> Storage for Vec<T, D> {
    type T = T;
    type Device = D;
}

impl<'a, T, D: Device> Storage for &'a Slice<T, D> {
    type T = T;
    type Device = D;
}

impl<'a, T, D: Device> Storage for &'a mut Slice<T, D> {
    type T = T;
    type Device = D;
}

impl<'a, T, D: Device> StorageMut for Vec<T, D> {}

impl<'a, T, D: Device> StorageMut for &'a mut Slice<T, D> {}

pub trait Dimension: AsRef<[usize]> + AsMut<[usize]> {
    fn len(&self) -> usize;
    fn column_major_strides(&self) -> Self;
}

impl<const N: usize> Dimension for [usize; N] {
    fn len(&self) -> usize {
        self.iter().product()
    }
    fn column_major_strides(&self) -> Self {
        let mut strides = *self;
        strides[0] = 1;

        for i in 1..N {
            strides[i] = strides[i - 1] * self[i - 1];
        }

        strides
    }
}

impl Dimension for std::vec::Vec<usize> {
    fn len(&self) -> usize {
        self.iter().product()
    }
    fn column_major_strides(&self) -> Self {
        let mut strides = self.clone();
        strides[0] = 1;

        for i in 1..self.len() {
            strides[i] = strides[i - 1] * self[i - 1];
        }

        strides
    }
}

#[cfg(test)]
mod tests {
    use crate::{array::Tensor, vec::Vec};

    #[test]
    fn matmul() {
        // column major
        let a: Vec<f32, _> = Vec::from(vec![0., 2., 4., 1., 3., 5.]);
        let b: Vec<f32, _> = Vec::from(vec![0., 2., 1., 3.]);
        let a = Tensor::from_shape_in((), [3, 2], a);
        let b = Tensor::from_shape_in((), [2, 2], b);

        let c = a.dot(b);

        assert_eq!(std::vec::Vec::from(c.data), vec![2., 6., 10., 3., 11., 19.]);
    }

    #[test]
    fn matmul_t() {
        let a: Vec<f32, _> = Vec::from(vec![1., 2., 3., 4.]);
        let b: Vec<f32, _> = Vec::from(vec![5., 6., 7., 8.]);
        let a = Tensor::from_shape_in((), [2, 2], a);
        let b = Tensor::from_shape_in((), [2, 2], b);

        let c1 = a.t().dot(b.t());
        assert_eq!(std::vec::Vec::from(c1.data), vec![19.0, 43.0, 22.0, 50.0]);

        let c2 = a.dot(b.t());
        assert_eq!(std::vec::Vec::from(c2.data), vec![26.0, 38.0, 30.0, 44.0]);

        let c3 = a.t().dot(b);
        assert_eq!(std::vec::Vec::from(c3.data), vec![17.0, 39.0, 23.0, 53.0]);
    }
}
