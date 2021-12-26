use std::alloc::Layout;

use crate::ptr::{non_null::NonNull, slice::Slice};

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;

pub trait Device {
    #![allow(clippy::missing_safety_doc)]

    type Ptr<T: ?Sized>: DevicePtr<T>;
    type AllocError: std::error::Error;

    // copied from the Allocator trait
    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<[u8], Self>, Self::AllocError>;
    unsafe fn allocate_zeroed(
        &self,
        layout: Layout,
    ) -> Result<NonNull<[u8], Self>, Self::AllocError>;
    unsafe fn deallocate(&self, ptr: NonNull<u8, Self>, layout: Layout);
    unsafe fn grow(
        &self,
        ptr: NonNull<u8, Self>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8], Self>, Self::AllocError>;
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8, Self>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8], Self>, Self::AllocError>;
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8, Self>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8], Self>, Self::AllocError>;

    fn copy_from_host<T: Copy>(from: &[T], to: &mut Slice<T, Self>);
    fn copy_to_host<T: Copy>(from: &Slice<T, Self>, to: &mut [T]);
    fn copy<T: Copy>(from: &Slice<T, Self>, to: &mut Slice<T, Self>);
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
    unsafe fn add(self, count: usize) -> Self
    where
        T: Sized;
    /// # Safety
    /// Resulting pointer should not underflow usize.
    /// Resulting pointer must be in bounds of an allocated buffer.
    unsafe fn sub(self, count: usize) -> Self
    where
        T: Sized;
    /// # Safety
    /// Resulting pointer should not overflow usize.
    /// Resulting pointer must be in bounds of an allocated buffer.
    unsafe fn offset(self, count: isize) -> Self
    where
        T: Sized;
}
