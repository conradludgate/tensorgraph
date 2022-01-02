use std::{
    alloc::{AllocError, Allocator, Global, Layout},
    ops::{Deref, DerefMut},
};

use crate::ptr::reef::Ref;

use super::{DefaultDeviceAllocator, Device, DeviceAllocator, DevicePtr, NonNull};

#[derive(Clone, Copy)]
pub struct Cpu;

impl Default for Cpu {
    fn default() -> Self {
        Self
    }
}

impl Device for Cpu {
    type Ptr<T: ?Sized> = *mut T;
    const IS_CPU: bool = true;

    fn copy_from_host<T: Copy>(from: &[T], to: &mut Ref<[T], Self>) {
        to.deref_mut().copy_from_slice(from)
    }

    fn copy_to_host<T: Copy>(from: &Ref<[T], Self>, to: &mut [T]) {
        to.copy_from_slice(from.deref())
    }

    fn copy<T: Copy>(from: &Ref<[T], Self>, to: &mut Ref<[T], Self>) {
        to.deref_mut().copy_from_slice(from.deref())
    }
}

impl DefaultDeviceAllocator for Cpu {
    type Alloc = Global;
    fn default_alloc() -> Global {
        Global
    }
}

impl<T: ?Sized> DevicePtr<T> for *mut T {
    fn as_raw(self) -> *mut T {
        self
    }

    fn from_raw(ptr: *mut T) -> Self {
        ptr
    }

    unsafe fn write(self, val: T)
    where
        T: Sized,
    {
        self.write(val)
    }
}

impl<T: ?Sized> From<std::ptr::NonNull<T>> for NonNull<T, Cpu> {
    fn from(ptr: std::ptr::NonNull<T>) -> Self {
        unsafe { NonNull::new_unchecked(ptr.as_ptr()) }
    }
}

impl<T: ?Sized> From<NonNull<T, Cpu>> for std::ptr::NonNull<T> {
    fn from(ptr: NonNull<T, Cpu>) -> Self {
        unsafe { std::ptr::NonNull::new_unchecked(ptr.as_ptr()) }
    }
}

impl<A: Allocator> DeviceAllocator for A {
    type AllocError = AllocError;
    type Device = Cpu;

    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<[u8], Cpu>, AllocError> {
        Ok(self.allocate(layout)?.into())
    }

    unsafe fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8], Cpu>, AllocError> {
        Ok(self.allocate_zeroed(layout)?.into())
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8, Cpu>, layout: Layout) {
        self.deallocate(ptr.into(), layout)
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8, Cpu>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8], Cpu>, AllocError> {
        Ok(self.grow(ptr.into(), old_layout, new_layout)?.into())
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8, Cpu>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8], Cpu>, AllocError> {
        Ok(self.grow_zeroed(ptr.into(), old_layout, new_layout)?.into())
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8, Cpu>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8], Cpu>, AllocError> {
        Ok(self.shrink(ptr.into(), old_layout, new_layout)?.into())
    }
}
