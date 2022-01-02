use std::alloc::Layout;

use crate::ptr::{non_null::NonNull, reef::Ref};

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;

// pub mod global;

pub trait Device {
    #![allow(clippy::missing_safety_doc)]

    type Ptr<T: ?Sized>: DevicePtr<T>;
    const IS_CPU: bool = false;

    fn copy_from_host<T: Copy>(from: &[T], to: &mut Ref<[T], Self>);
    fn copy_to_host<T: Copy>(from: &Ref<[T], Self>, to: &mut [T]);
    fn copy<T: Copy>(from: &Ref<[T], Self>, to: &mut Ref<[T], Self>);
}

pub trait DefaultDeviceAllocator: Device {
    type Alloc: DeviceAllocator<Device = Self>;
    fn default_alloc() -> Self::Alloc;
}

pub trait DevicePtr<T: ?Sized>: Copy {
    fn as_raw(self) -> *mut T;
    fn from_raw(ptr: *mut T) -> Self;

    /// # Safety
    /// Pointer must be valid and aligned
    unsafe fn write(self, val: T)
    where
        T: Sized;

    /// # Safety
    /// Offset should not overflow isize.
    /// Resulting pointer should not overflow usize.
    /// Resulting pointer must be in bounds of an allocated buffer.
    #[must_use]
    unsafe fn add(self, count: usize) -> Self
    where
        T: Sized,
    {
        Self::from_raw(self.as_raw().add(count))
    }

    /// # Safety
    /// Resulting pointer should not underflow usize.
    /// Resulting pointer must be in bounds of an allocated buffer.
    #[must_use]
    unsafe fn sub(self, count: usize) -> Self
    where
        T: Sized,
    {
        Self::from_raw(self.as_raw().sub(count))
    }

    /// # Safety
    /// Resulting pointer should not overflow usize.
    /// Resulting pointer must be in bounds of an allocated buffer.
    #[must_use]
    unsafe fn offset(self, count: isize) -> Self
    where
        T: Sized,
    {
        Self::from_raw(self.as_raw().offset(count))
    }
}

pub trait DeviceAllocator {
    #![allow(clippy::missing_safety_doc)]

    type AllocError: std::error::Error;
    type Device: Device;

    // copied from the Allocator trait
    unsafe fn allocate(
        &self,
        layout: Layout,
    ) -> Result<NonNull<[u8], Self::Device>, Self::AllocError>;
    unsafe fn allocate_zeroed(
        &self,
        layout: Layout,
    ) -> Result<NonNull<[u8], Self::Device>, Self::AllocError>;
    unsafe fn deallocate(&self, ptr: NonNull<u8, Self::Device>, layout: Layout);
    unsafe fn grow(
        &self,
        ptr: NonNull<u8, Self::Device>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8], Self::Device>, Self::AllocError>;
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8, Self::Device>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8], Self::Device>, Self::AllocError>;
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8, Self::Device>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8], Self::Device>, Self::AllocError>;
}

// // impl is conflicting :(
// impl<'a, A: DeviceAllocator> DeviceAllocator for &'a A {
//     type AllocError = A::AllocError;

//     type Device = A::Device;

//     unsafe fn allocate(
//         &self,
//         layout: Layout,
//     ) -> Result<NonNull<[u8], Self::Device>, Self::AllocError> {
//         A::allocate(self, layout)
//     }

//     unsafe fn allocate_zeroed(
//         &self,
//         layout: Layout,
//     ) -> Result<NonNull<[u8], Self::Device>, Self::AllocError> {
//         A::allocate_zeroed(self, layout)
//     }

//     unsafe fn deallocate(&self, ptr: NonNull<u8, Self::Device>, layout: Layout) {
//         A::deallocate(self, ptr, layout)
//     }

//     unsafe fn grow(
//         &self,
//         ptr: NonNull<u8, Self::Device>,
//         old_layout: Layout,
//         new_layout: Layout,
//     ) -> Result<NonNull<[u8], Self::Device>, Self::AllocError> {
//         A::grow(self, ptr, old_layout, new_layout)
//     }

//     unsafe fn grow_zeroed(
//         &self,
//         ptr: NonNull<u8, Self::Device>,
//         old_layout: Layout,
//         new_layout: Layout,
//     ) -> Result<NonNull<[u8], Self::Device>, Self::AllocError> {
//         A::grow_zeroed(self, ptr, old_layout, new_layout)
//     }

//     unsafe fn shrink(
//         &self,
//         ptr: NonNull<u8, Self::Device>,
//         old_layout: Layout,
//         new_layout: Layout,
//     ) -> Result<NonNull<[u8], Self::Device>, Self::AllocError> {
//         A::shrink(self, ptr, old_layout, new_layout)
//     }
// }
