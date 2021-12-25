#![feature(
    generic_associated_types,
    allocator_api,
    alloc_layout_extra,
    nonnull_slice_from_raw_parts,
    slice_ptr_len,
    ptr_metadata
)]

use device::Device;
use ptr::slice::Slice;
use vec::Vec;

pub mod blas;
pub mod device;
pub mod ptr;
pub mod vec;

/// Low level matrix
pub struct Matrix<S> {
    rows: usize,
    cols: usize,
    data: S,
}

impl<S: Storage> Matrix<S> {
    pub fn with_shape(rows: usize, cols: usize, data: S) -> Self {
        assert_eq!(data.as_ref().len(), rows * cols);
        Self { rows, cols, data }
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

impl<'a, T, D: device::Device> Storage for &'a Slice<T, D> {
    type T = T;
    type Device = D;
    fn into_owned(self) -> Vec<T, D> {
        self.to_owned()
    }

    fn as_ref(&self) -> &Slice<T, D> {
        self
    }
}

impl<'a, T, D: device::Device> Storage for &'a mut Slice<T, D> {
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

impl<'a, T, D: device::Device> StorageMut for &'a mut Slice<T, D> {
    fn as_mut(&mut self) -> &mut Slice<T, D> {
        self
    }
}
