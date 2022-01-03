use std::alloc::Layout;

use crate::ptr::{NonNull, Ref};

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

/// Represents a physical device that can host memory.
/// For example: [`cpu::Cpu`], [`cuda::Cuda`]
pub trait Device {
    #![allow(clippy::missing_safety_doc)]

    type Ptr<T: ?Sized>: DevicePtr<T>;
    const IS_CPU: bool = false;

    fn copy_from_host<T: Copy>(from: &[T], to: &mut Ref<[T], Self>);
    fn copy_to_host<T: Copy>(from: &Ref<[T], Self>, to: &mut [T]);
    fn copy<T: Copy>(from: &Ref<[T], Self>, to: &mut Ref<[T], Self>);
}

/// Defines the default allocator for a device.
/// For instance, The default allocator for [`cpu::Cpu`] is [`std::alloc::Global`]
pub trait DefaultDeviceAllocator: Device {
    type Alloc: DeviceAllocator<Self> + Default;
}

/// Represents a type safe device-based pointer.
/// For the CPU, this will be just `*mut T`.
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

/// An Allocator in a specific device.
/// All [`std::alloc::Allocator`]s are [`DeviceAllocator<Device=cpu::Cpu>`]
pub trait DeviceAllocator<D: Device + ?Sized> {
    #![allow(clippy::missing_safety_doc)]

    /// Error returned when failing to allocate
    type AllocError: std::error::Error;

    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8], D>, Self::AllocError>;

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8], D>, Self::AllocError>;

    unsafe fn deallocate(&self, ptr: NonNull<u8, D>, layout: Layout);

    unsafe fn grow(
        &self,
        ptr: NonNull<u8, D>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8], D>, Self::AllocError>;

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8, D>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8], D>, Self::AllocError>;

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8, D>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8], D>, Self::AllocError>;
}
