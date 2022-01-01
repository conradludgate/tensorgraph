use std::alloc::Allocator;

use crate::{
    device::{cpu::Cpu, Device},
    ptr::slice::Slice,
    vec::Vec,
};

pub trait IntoOwned {
    type Owned;
    fn into_owned(self) -> Self::Owned;
}

pub trait Storage: AsRef<Slice<Self::T, Self::Device>> {
    type T;
    type Device: Device;
}

pub trait StorageMut: Storage + AsMut<Slice<Self::T, Self::Device>> {}

// Vec

impl<T, D: Device> Storage for Vec<T, D> {
    type T = T;
    type Device = D;
}

impl<T, D: Device> StorageMut for Vec<T, D> {}

impl<T, D: Device> IntoOwned for Vec<T, D> {
    type Owned = Self;
    fn into_owned(self) -> Self::Owned {
        self
    }
}

// Shared Slice

impl<'a, T, D: Device> Storage for &'a Slice<T, D> {
    type T = T;
    type Device = D;
}

impl<'a, T: Copy, D: Device + Default> IntoOwned for &'a Slice<T, D> {
    type Owned = Vec<T, D>;
    fn into_owned(self) -> Self::Owned {
        self.to_owned()
    }
}

// Mut Slice

impl<'a, T, D: Device> Storage for &'a mut Slice<T, D> {
    type T = T;
    type Device = D;
}

impl<'a, T, D: Device> StorageMut for &'a mut Slice<T, D> {}

impl<'a, T: Copy, D: Device + Default> IntoOwned for &'a mut Slice<T, D> {
    type Owned = Vec<T, D>;
    fn into_owned(self) -> Self::Owned {
        self.to_owned()
    }
}

// Array (CPU Only)

impl<T, const N: usize> Storage for [T; N] {
    type T = T;
    type Device = Cpu;
}

impl<T, const N: usize> StorageMut for [T; N] {}

impl<T, const N: usize> IntoOwned for [T; N] {
    type Owned = Self;
    fn into_owned(self) -> Self::Owned {
        self
    }
}

// std::vec::Vec (CPU Only)

impl<T, A: Allocator> Storage for std::vec::Vec<T, A> {
    type T = T;
    type Device = Cpu<A>;
}

impl<T, A: Allocator> StorageMut for std::vec::Vec<T, A> {}

impl<T, A: Allocator> IntoOwned for std::vec::Vec<T, A> {
    type Owned = Self;
    fn into_owned(self) -> Self::Owned {
        self
    }
}
