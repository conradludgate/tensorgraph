use std::{
    alloc::{AllocError, Allocator, Global, Layout},
    ops::{Deref, DerefMut},
};

use crate::ptr::slice::Slice;

use super::{Device, DevicePtr, NonNull};

#[derive(Clone, Copy)]
pub struct Cpu<A: Allocator> {
    alloc: A,
}

impl Default for Cpu<Global> {
    fn default() -> Self {
        Self { alloc: Global }
    }
}

impl<A: Allocator> Cpu<A> {
    pub fn new(alloc: A) -> Self {
        Self { alloc }
    }
    pub fn alloc(self) -> A {
        self.alloc
    }
}

impl<A: Allocator> Device for Cpu<A> {
    type Ptr<T: ?Sized> = *mut T;
    type AllocError = AllocError;

    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<[u8], Self>, AllocError> {
        Ok(self.alloc.allocate(layout)?.into())
    }

    unsafe fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8], Self>, AllocError> {
        Ok(self.alloc.allocate_zeroed(layout)?.into())
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8, Self>, layout: Layout) {
        self.alloc.deallocate(ptr.into(), layout)
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8, Self>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8], Self>, AllocError> {
        Ok(self.alloc.grow(ptr.into(), old_layout, new_layout)?.into())
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8, Self>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8], Self>, AllocError> {
        Ok(self
            .alloc
            .grow_zeroed(ptr.into(), old_layout, new_layout)?
            .into())
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8, Self>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8], Self>, AllocError> {
        Ok(self
            .alloc
            .shrink(ptr.into(), old_layout, new_layout)?
            .into())
    }

    fn copy_from_host<T: Copy>(from: &[T], to: &mut Slice<T, Self>) {
        to.deref_mut().copy_from_slice(from)
    }

    fn copy_to_host<T: Copy>(from: &Slice<T, Self>, to: &mut [T]) {
        to.copy_from_slice(from.deref())
    }

    fn copy<T: Copy>(from: &Slice<T, Self>, to: &mut Slice<T, Self>) {
        to.deref_mut().copy_from_slice(from.deref())
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

impl<T: ?Sized, A: Allocator> From<std::ptr::NonNull<T>> for NonNull<T, Cpu<A>> {
    fn from(ptr: std::ptr::NonNull<T>) -> Self {
        unsafe { NonNull::new_unchecked(ptr.as_ptr()) }
    }
}

impl<T: ?Sized, A: Allocator> From<NonNull<T, Cpu<A>>> for std::ptr::NonNull<T> {
    fn from(ptr: NonNull<T, Cpu<A>>) -> Self {
        unsafe { std::ptr::NonNull::new_unchecked(ptr.as_ptr()) }
    }
}
