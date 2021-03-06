use std::alloc::Allocator;

use tensorgraph_sys::{
    device::{cpu::Cpu, DefaultDeviceAllocator, Device, DeviceAllocator},
    ptr::Ref,
    DefaultVec, Vec,
};

/// Convert a value into it's owned representation.
pub trait IntoOwned {
    type Owned;
    fn into_owned(self) -> Self::Owned;
}

/// Represents a storage for [`crate::tensor::Tensor`]
pub trait Storage: AsRef<Ref<[Self::T], Self::Device>> {
    type T;
    type Device: Device;
}

/// Represents a mutable storage for [`crate::tensor::Tensor`]
pub trait StorageMut: Storage + AsMut<Ref<[Self::T], Self::Device>> {}

// Vec

impl<T, A: DeviceAllocator> Storage for Vec<T, A> {
    type T = T;
    type Device = A::Device;
}

impl<T, A: DeviceAllocator> StorageMut for Vec<T, A> {}

impl<T, A: DeviceAllocator> IntoOwned for Vec<T, A> {
    type Owned = Self;
    fn into_owned(self) -> Self::Owned {
        self
    }
}

// Shared Slice

impl<'a, T, D: Device> Storage for &'a Ref<[T], D> {
    type T = T;
    type Device = D;
}

impl<'a, T: Copy, D: DefaultDeviceAllocator> IntoOwned for &'a Ref<[T], D> {
    type Owned = DefaultVec<T, D>;
    fn into_owned(self) -> Self::Owned {
        self.to_owned()
    }
}

// Mut Slice

impl<'a, T, D: Device> Storage for &'a mut Ref<[T], D> {
    type T = T;
    type Device = D;
}

impl<'a, T, D: Device> StorageMut for &'a mut Ref<[T], D> {}

impl<'a, T: Copy, D: DefaultDeviceAllocator> IntoOwned for &'a mut Ref<[T], D> {
    type Owned = DefaultVec<T, D>;
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
    type Device = Cpu;
}

impl<T, A: Allocator> StorageMut for std::vec::Vec<T, A> {}

impl<T, A: Allocator> IntoOwned for std::vec::Vec<T, A> {
    type Owned = Self;
    fn into_owned(self) -> Self::Owned {
        self
    }
}

// Shared Slice (CPU Only)

impl<'a, T> Storage for &'a [T] {
    type T = T;
    type Device = Cpu;
}

impl<'a, T: Copy> IntoOwned for &'a [T] {
    type Owned = std::vec::Vec<T>;
    fn into_owned(self) -> Self::Owned {
        self.to_owned()
    }
}

// Mut Slice (CPU Only)

impl<'a, T> Storage for &'a mut [T] {
    type T = T;
    type Device = Cpu;
}

impl<'a, T> StorageMut for &'a mut [T] {}

impl<'a, T: Copy> IntoOwned for &'a mut [T] {
    type Owned = std::vec::Vec<T>;
    fn into_owned(self) -> Self::Owned {
        self.to_owned()
    }
}
