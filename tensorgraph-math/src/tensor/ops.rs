use std::ops::AddAssign;

use num_traits::One;
use tensorgraph_sys::ViewMut;

use crate::{
    blas::{DefaultBLASContext, BLAS1},
    storage::StorageMut,
};

use super::{Tensor, ViewOf};

impl<'a, S: StorageMut> AddAssign<Tensor<&'a ViewOf<S>, Vec<usize>>> for Tensor<S, Vec<usize>>
where
    S::Device: DefaultBLASContext,
    S::T: One + BLAS1<<S::Device as DefaultBLASContext>::Context>,
{
    fn add_assign(&mut self, rhs: Tensor<&'a ViewOf<S>, Vec<usize>>) {
        match (&*self.shape, &*rhs.shape) {
            ([], []) => {
                // this is a but meh. We have no general purpose scalar add.
                // this is a niche use though, so we just up the dimension to [1]
                let mut a = self
                    .view_mut()
                    .try_into_dim::<[usize; 0]>()
                    .unwrap()
                    .insert_axis(0);
                let b = rhs.try_into_dim::<[usize; 0]>().unwrap().insert_axis(0);
                a += b;
            }
            ([_], [_]) => {
                let mut a = self.view_mut().try_into_dim::<[usize; 1]>().unwrap();
                let b = rhs.try_into_dim::<[usize; 1]>().unwrap();
                a += b;
            }
            ([a, ar @ ..], [b, br @ ..]) if ar.len() == br.len() => {
                assert_eq!(a, b, "tensors should have the same shape");
                for i in 0..*a {
                    let mut a = self.slice_axis_mut(0, i);
                    let b = rhs.slice_axis(0, i);
                    a += b;
                }
            }
            _ => panic!("dimensions should have the same length - maybe try broadcasting first"),
        }
    }
}

#[cfg(test)]
mod tests {
    use tensorgraph_sys::View;

    use crate::tensor::Tensor;

    #[test]
    fn add() {
        let a = (0..60).map(|i| i as f32).collect::<Vec<_>>();
        let b = (0..60).rev().map(|i| i as f32).collect::<Vec<_>>();
        let dim = vec![3, 4, 5];

        let mut a = Tensor::from_shape(dim.clone(), a);
        let b = Tensor::from_shape(dim, b);

        a += b.view();

        assert_eq!(a.into_inner(), vec![59.0; 60]);
    }
}
