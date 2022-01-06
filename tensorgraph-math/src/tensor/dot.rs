use num_traits::{Zero, One};
use tensorgraph_sys::{device::DefaultDeviceAllocator, DefaultVec};

use crate::{blas::{DefaultBLASContext, BLAS3, BLAS2, BLAS1}, storage::Storage};

use super::{Matrix, ViewOf, Vector};

/// Trait for the dot product.
pub trait Dot<Rhs> {
    type Output;
    fn dot(&self, rhs: Rhs) -> Self::Output;
}

impl<S: Storage> Dot<Matrix<&ViewOf<S>>> for Matrix<S>
where
    S::Device: DefaultDeviceAllocator + DefaultBLASContext,
    S::T: Zero + One + BLAS3<<S::Device as DefaultBLASContext>::Context>,
{
    type Output = Matrix<DefaultVec<S::T, S::Device>>;

    fn dot(&self, rhs: Matrix<&ViewOf<S>>) -> Self::Output {
        self.matmul(rhs)
    }
}

impl<S: Storage> Dot<Vector<&ViewOf<S>>> for Matrix<S>
where
    S::Device: DefaultDeviceAllocator + DefaultBLASContext,
    S::T: Zero + One + BLAS2<<S::Device as DefaultBLASContext>::Context>,
{
    type Output = Vector<DefaultVec<S::T, S::Device>>;

    fn dot(&self, rhs: Vector<&ViewOf<S>>) -> Self::Output {
        self.dot(rhs)
    }
}

impl<S: Storage> Dot<Vector<&ViewOf<S>>> for Vector<S>
where
    S::Device: DefaultDeviceAllocator + DefaultBLASContext,
    S::T: Zero + One + BLAS1<<S::Device as DefaultBLASContext>::Context>,
{
    type Output = S::T;

    fn dot(&self, rhs: Vector<&ViewOf<S>>) -> Self::Output {
        self.dot(rhs)
    }
}
