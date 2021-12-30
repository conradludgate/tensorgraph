use std::{
    alloc::Allocator,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use crate::{
    device::{cpu::Cpu, Device, DevicePtr},
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

    pub fn as_ptr(&self) -> D::Ptr<T> {
        D::Ptr::from_raw(&self.inner as *const [T] as *mut T)
    }

    pub fn as_slice_ptr(&self) -> D::Ptr<[T]> {
        D::Ptr::from_raw(&self.inner as *const [T] as *mut [T])
    }

    /// # Safety
    /// Slices should always point to properly allocated and initialised memory.
    /// If any of the slice's values are not initialised or not in an allocated region,
    /// it may result in undefined behaviour
    pub unsafe fn from_slice_ptr<'a>(ptr: D::Ptr<[T]>) -> &'a Self {
        &*(ptr.as_raw() as *const Self)
    }

    /// # Safety
    /// Slices should always point to properly allocated and initialised memory.
    /// If any of the slice's values are not initialised or not in an allocated region,
    /// it may result in undefined behaviour
    ///
    /// Also, you must make sure that only one mut reference is active to this pointer
    pub unsafe fn from_slice_ptr_mut<'a>(ptr: D::Ptr<[T]>) -> &'a mut Self {
        &mut *(ptr.as_raw() as *mut Self)
    }
}
impl<'a, T, D: Device> Slice<MaybeUninit<T>, D> {
    /// # Safety
    /// Contents must be initialised
    pub unsafe fn assume_init(&self) -> &Slice<T, D> {
        &*(MaybeUninit::slice_assume_init_ref(&self.inner) as *const [T] as *const _)
    }
    /// # Safety
    /// Contents must be initialised
    pub unsafe fn assume_init_mut(&mut self) -> &mut Slice<T, D> {
        &mut *(MaybeUninit::slice_assume_init_mut(&mut self.inner) as *mut [T] as *mut _)
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

impl<T: Copy, D: Device + Default> ToOwned for Slice<T, D> {
    type Owned = Vec<T, D>;

    fn to_owned(&self) -> Self::Owned {
        unsafe {
            let mut v = Vec::with_capacity(self.len());
            let buf = &mut v.space_capacity_mut()[..self.len()];
            buf.init_from_slice(self);
            v.set_len(self.len());
            v
        }
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

impl<'a, T, D: Device> AsRef<Slice<T, D>> for &'a Slice<T, D> {
    fn as_ref(&self) -> &Slice<T, D> {
        self
    }
}

impl<'a, T, D: Device> AsRef<Slice<T, D>> for &'a mut Slice<T, D> {
    fn as_ref(&self) -> &Slice<T, D> {
        self
    }
}

impl<'a, T, D: Device> AsMut<Slice<T, D>> for &'a mut Slice<T, D> {
    fn as_mut(&mut self) -> &mut Slice<T, D> {
        self
    }
}

impl<T, D: Device, S> Index<S> for Slice<T, D>
where
    [T]: Index<S, Output = [T]>,
{
    type Output = Self;

    fn index(&self, index: S) -> &Self::Output {
        unsafe { &*(&self.inner[index] as *const [T] as *const Self) }
    }
}

impl<T, D: Device, S> IndexMut<S> for Slice<T, D>
where
    [T]: IndexMut<S, Output = [T]>,
{
    fn index_mut(&mut self, index: S) -> &mut Self::Output {
        unsafe { &mut *(&mut self.inner[index] as *mut [T] as *mut Self) }
    }
}
