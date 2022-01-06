use ndarray::{Axis, Dimension};

use super::Tensor;

impl<F> From<ndarray::Array1<F>> for Tensor<Vec<F>, [usize; 1]> {
    fn from(a: ndarray::Array1<F>) -> Self {
        #[allow(clippy::cast_sign_loss)]
        let s = a.stride_of(Axis(0)) as usize;
        let d = a.raw_dim().into_pattern();
        let data = a.into_raw_vec();
        Self {
            shape: [d],
            strides: [s],
            data,
        }
    }
}

#[allow(clippy::fallible_impl_from)]
impl<F> From<ndarray::Array2<F>> for Tensor<Vec<F>, [usize; 2]> {
    fn from(a: ndarray::Array2<F>) -> Self {
        // reinterpret signed integer as unsigned signed
        let s = a.strides();
        let s = unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<usize>(), s.len()) };
        let s = s.try_into().unwrap();

        let (d1, d2) = a.raw_dim().into_pattern();
        let data = a.into_raw_vec();
        Self {
            shape: [d1, d2],
            strides: s,
            data,
        }
    }
}

#[allow(clippy::fallible_impl_from)]
impl<F> From<ndarray::Array3<F>> for Tensor<Vec<F>, [usize; 3]> {
    fn from(a: ndarray::Array3<F>) -> Self {
        // reinterpret signed integer as unsigned signed
        let s = a.strides();
        let s = unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<usize>(), s.len()) };
        let s = s.try_into().unwrap();

        let (d1, d2, d3) = a.raw_dim().into_pattern();
        let data = a.into_raw_vec();
        Self {
            shape: [d1, d2, d3],
            strides: s,
            data,
        }
    }
}

#[allow(clippy::fallible_impl_from)]
impl<F> From<ndarray::Array4<F>> for Tensor<Vec<F>, [usize; 4]> {
    fn from(a: ndarray::Array4<F>) -> Self {
        // reinterpret signed integer as unsigned signed
        let s = a.strides();
        let s = unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<usize>(), s.len()) };
        let s = s.try_into().unwrap();

        let (d1, d2, d3, d4) = a.raw_dim().into_pattern();
        let data = a.into_raw_vec();
        Self {
            shape: [d1, d2, d3, d4],
            strides: s,
            data,
        }
    }
}

#[allow(clippy::fallible_impl_from)]
impl<F> From<ndarray::Array5<F>> for Tensor<Vec<F>, [usize; 5]> {
    fn from(a: ndarray::Array5<F>) -> Self {
        // reinterpret signed integer as unsigned signed
        let s = a.strides();
        let s = unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<usize>(), s.len()) };
        let s = s.try_into().unwrap();

        let (d1, d2, d3, d4, d5) = a.raw_dim().into_pattern();
        let data = a.into_raw_vec();
        Self {
            shape: [d1, d2, d3, d4, d5],
            strides: s,
            data,
        }
    }
}

#[allow(clippy::fallible_impl_from)]
impl<F> From<ndarray::Array6<F>> for Tensor<Vec<F>, [usize; 6]> {
    fn from(a: ndarray::Array6<F>) -> Self {
        // reinterpret signed integer as unsigned signed
        let s = a.strides();
        let s = unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<usize>(), s.len()) };
        let s = s.try_into().unwrap();

        let (d1, d2, d3, d4, d5, d6) = a.raw_dim().into_pattern();
        let data = a.into_raw_vec();
        Self {
            shape: [d1, d2, d3, d4, d5, d6],
            strides: s,
            data,
        }
    }
}

#[allow(clippy::fallible_impl_from)]
impl<F> From<ndarray::ArrayD<F>> for Tensor<Vec<F>, Vec<usize>> {
    fn from(a: ndarray::ArrayD<F>) -> Self {
        // reinterpret signed integer as unsigned signed
        let s = a.strides();
        let s = unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<usize>(), s.len()) };
        let s = s.to_owned();

        let shape = a.shape().to_owned();
        let data = a.into_raw_vec();
        Self {
            shape,
            strides: s,
            data,
        }
    }
}
