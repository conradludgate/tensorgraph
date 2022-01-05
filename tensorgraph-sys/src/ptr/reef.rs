use std::{
    alloc::Allocator,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use crate::{
    device::{cpu::Cpu, DefaultDeviceAllocator, Device, DevicePtr},
    vec::Vec,
};

/// A reference type for devices. Should have the same representation as `&T` but
/// is not safely transmutable. Device references can not be read directly safely since
/// the host is not guaranteed to be on that device.
pub struct Ref<T: ?Sized, D: Device> {
    _device: PhantomData<D>,
    inner: T,
}
impl<T: ?Sized, D: Device> Ref<T, D> {
    /// # Safety
    /// Refs should always point to properly allocated and initialised memory.
    /// If any of the slice's values are not initialised or not in an allocated region,
    /// it may result in undefined behaviour
    pub unsafe fn from_ptr<'a>(ptr: D::Ptr<T>) -> &'a Self {
        &*(ptr.as_raw() as *const Self)
    }

    /// # Safety
    /// Refs should always point to properly allocated and initialised memory.
    /// If any of the slice's values are not initialised or not in an allocated region,
    /// it may result in undefined behaviour
    ///
    /// Also, you must make sure that only one mut reference is active to this pointer
    pub unsafe fn from_ptr_mut<'a>(ptr: D::Ptr<T>) -> &'a mut Self {
        &mut *(ptr.as_raw() as *mut Self)
    }
}

impl<T, D: Device> Ref<[T], D> {
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn as_ptr(&self) -> D::Ptr<T> {
        D::Ptr::from_raw(&self.inner as *const [T] as *mut T)
    }

    pub fn as_slice_ptr(&self) -> D::Ptr<[T]> {
        D::Ptr::from_raw(&self.inner as *const [T] as *mut [T])
    }
}
impl<'a, T, D: Device> Ref<[MaybeUninit<T>], D> {
    /// # Safety
    /// Contents must be initialised
    pub unsafe fn assume_init(&self) -> &Ref<[T], D> {
        &*(MaybeUninit::slice_assume_init_ref(&self.inner) as *const [T] as *const _)
    }
    /// # Safety
    /// Contents must be initialised
    pub unsafe fn assume_init_mut(&mut self) -> &mut Ref<[T], D> {
        &mut *(MaybeUninit::slice_assume_init_mut(&mut self.inner) as *mut [T] as *mut _)
    }
}

impl<T: Copy, D: Device> Ref<[T], D> {
    pub fn copy_from_slice(&mut self, from: &Self) {
        D::copy(from, self);
    }

    pub fn copy_from_host(&mut self, from: &[T]) {
        D::copy_from_host(from, self);
    }

    pub fn copy_to_host(&self, to: &mut [T]) {
        D::copy_to_host(self, to);
    }
}

impl<T: Copy, D: Device> Ref<[MaybeUninit<T>], D> {
    pub fn init_from_slice(&mut self, from: &Ref<[T], D>) {
        unsafe {
            let ptr = self as *mut _ as *mut Ref<[T], D>;
            (*ptr).copy_from_slice(from);
        }
    }

    pub fn init_from_host(&mut self, from: &[T]) {
        unsafe {
            let ptr = self as *mut _ as *mut Ref<[T], D>;
            (*ptr).copy_from_host(from);
        }
    }
}

impl<T: Copy, D: DefaultDeviceAllocator> ToOwned for Ref<[T], D> {
    type Owned = Vec<T, D::Alloc>;

    fn to_owned(&self) -> Self::Owned {
        unsafe {
            let mut v = Vec::with_capacity_in(self.len(), D::Alloc::default());
            let buf = &mut v.space_capacity_mut()[..self.len()];
            buf.init_from_slice(self);
            v.set_len(self.len());
            v
        }
    }
}

impl<T> Deref for Ref<[T], Cpu> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for Ref<[T], Cpu> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'a, T, D: Device> AsRef<Ref<[T], D>> for &'a Ref<[T], D> {
    fn as_ref(&self) -> &Ref<[T], D> {
        self
    }
}

impl<'a, T, D: Device> AsRef<Ref<[T], D>> for &'a mut Ref<[T], D> {
    fn as_ref(&self) -> &Ref<[T], D> {
        self
    }
}

impl<'a, T, D: Device> AsMut<Ref<[T], D>> for &'a mut Ref<[T], D> {
    fn as_mut(&mut self) -> &mut Ref<[T], D> {
        self
    }
}

impl<T, D: Device, S> Index<S> for Ref<[T], D>
where
    [T]: Index<S, Output = [T]>,
{
    type Output = Self;

    fn index(&self, index: S) -> &Self::Output {
        unsafe { &*(&self.inner[index] as *const [T] as *const Self) }
    }
}

impl<T, D: Device, S> IndexMut<S> for Ref<[T], D>
where
    [T]: IndexMut<S, Output = [T]>,
{
    fn index_mut(&mut self, index: S) -> &mut Self::Output {
        unsafe { &mut *(&mut self.inner[index] as *mut [T] as *mut Self) }
    }
}

impl<T, const N: usize> AsRef<Ref<[T], Cpu>> for [T; N] {
    fn as_ref(&self) -> &Ref<[T], Cpu> {
        unsafe { &*(self.as_slice() as *const [T] as *const Ref<[T], Cpu>) }
    }
}

impl<T, const N: usize> AsMut<Ref<[T], Cpu>> for [T; N] {
    fn as_mut(&mut self) -> &mut Ref<[T], Cpu> {
        unsafe { &mut *(self.as_mut_slice() as *mut [T] as *mut Ref<[T], Cpu>) }
    }
}

impl<T, A: Allocator> AsRef<Ref<[T], Cpu>> for std::vec::Vec<T, A> {
    fn as_ref(&self) -> &Ref<[T], Cpu> {
        unsafe { &*(self.as_slice() as *const [T] as *const Ref<[T], Cpu>) }
    }
}

impl<T, A: Allocator> AsMut<Ref<[T], Cpu>> for std::vec::Vec<T, A> {
    fn as_mut(&mut self) -> &mut Ref<[T], Cpu> {
        unsafe { &mut *(self.as_mut_slice() as *mut [T] as *mut Ref<[T], Cpu>) }
    }
}

impl<T> AsRef<Ref<[T], Cpu>> for [T] {
    fn as_ref(&self) -> &Ref<[T], Cpu> {
        unsafe { &*(self as *const [T] as *const _) }
    }
}

impl<T> AsMut<Ref<[T], Cpu>> for [T] {
    fn as_mut(&mut self) -> &mut Ref<[T], Cpu> {
        unsafe { &mut *(self as *mut [T] as *mut _) }
    }
}
