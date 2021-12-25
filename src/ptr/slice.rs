use std::{
    alloc::Allocator,
    marker::PhantomData,
    ops::{Deref, DerefMut}, mem::MaybeUninit,
};

use crate::{
    device::{cpu::Cpu, Device},
    vec::Vec,
};

pub struct Slice<T, D: Device + ?Sized> {
    _device: PhantomData<D>,
    inner: [T],
}

impl<T, D: Device> Slice<T, D> {
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<T: Copy, D: Device> Slice<T, D> {
    pub fn copy_from_slice(&mut self, from: &Self) {
        D::copy(from, self)
    }

    pub fn copy_from_host(&mut self, from: &[T]) {
        D::copy_from_host(from, self)
    }
}

impl<T: Copy, D: Device> Slice<MaybeUninit<T>, D> {
    pub fn init_from_slice(&mut self, from: &Slice<T, D>) {
        unsafe {
            let ptr = self as *mut _ as *mut Slice<T, D>;
            (*ptr).copy_from_slice(from)
        }
    }

    pub fn init_from_host(&mut self, from: &[T]) {
        unsafe {
            let ptr = self as *mut _ as *mut Slice<T, D>;
            (*ptr).copy_from_host(from)
        }
    }
}

impl<T, D: Device> ToOwned for Slice<T, D> {
    type Owned = Vec<T, D>;

    fn to_owned(&self) -> Self::Owned {
        todo!()
    }
}

impl<T, A: Allocator> Deref for Slice<T, Cpu<A>> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T, A: Allocator> DerefMut for Slice<T, Cpu<A>> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
