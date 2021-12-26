use crate::{
    blas::{BLASDevice, BLAS},
    matrix::{Storage, StorageMut},
    ptr::slice::Slice,
    vec::Vec,
};

pub struct Tensor<S: Storage>
where
    S::Device: BLASDevice,
{
    shape: std::vec::Vec<usize>,
    data: S,
    ctx: <S::Device as BLASDevice>::Context,
}

impl<S: Storage> Tensor<S>
where
    S::Device: BLASDevice + Default,
{
    pub fn from_shape_in(
        ctx: <S::Device as BLASDevice>::Context,
        shape: std::vec::Vec<usize>,
        data: S,
    ) -> Self {
        assert_eq!(data.as_ref().len(), shape.iter().product());
        Self { shape, data, ctx }
    }

    pub fn view(&self) -> Tensor<&Slice<S::T, S::Device>>
    where
        S::T: Copy,
    {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.as_ref(),
            ctx: self.ctx.clone(),
        }
    }

    pub fn view_mut(&mut self) -> Tensor<&mut Slice<S::T, S::Device>>
    where
        S: StorageMut,
        S::T: Copy,
    {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.as_mut(),
            ctx: self.ctx.clone(),
        }
    }

    pub fn tensordot(
        &self,
        rhs: Tensor<impl Storage<T = S::T, Device = S::Device>>,
        axis0: &[usize],
        axis1: &[usize],
    ) -> Tensor<Vec<S::T, S::Device>>
    where
        S::T: BLAS<S::Device>,
    {
        assert_eq!(axis0.len(), axis1.len());
        let mut sh0 = self.shape.clone();
        let mut sh1 = rhs.shape;
        let mut out = vec![];
        for i in 0..axis0.len() {
            let ax0 = axis0[i];
            let ax1 = axis1[i];

            assert_eq!(sh0[ax0], sh1[ax1]);
            out.push(sh0[ax0]);
            sh0[ax0] = usize::MAX;
            sh1[ax1] = usize::MAX;
        }
        sh0.retain(|&n| n != usize::MAX);
        sh1.retain(|&n| n != usize::MAX);

        sh0.extend_from_slice(&sh1);
        let size = sh0.iter().product();

        let mut v = Vec::with_capacity(size);

        unsafe {
            // let uninit =
            //     Tensor::from_shape_in(self.ctx.clone(), sh0, &mut v.space_capacity_mut()[..size]);



            // // S::T::gemm(self.ctx.clone(), )

            // // gemm_uninit(S::T::one(), self.view(), rhs, uninit);

            // v.set_len(rows * cols);
        }
        Tensor::from_shape_in(self.ctx.clone(), vec![], v)
    }
}

#[cfg(test)]
mod tests {
    use crate::vec::Vec;

    use super::Tensor;

    #[test]
    fn tensordot() {
        let a = std::vec::Vec::from_iter((0..60).map(|x| x as f32));
        let b = std::vec::Vec::from_iter((0..24).map(|x| x as f32));
        let a = Vec::from(a);
        let b = Vec::from(b);
        let a = Tensor::from_shape_in((), vec![3, 4, 5], a);
        let b = Tensor::from_shape_in((), vec![4, 3, 2], b);
        a.tensordot(b, &[0, 1], &[1, 0]);
    }
}
