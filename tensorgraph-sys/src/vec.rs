use std::{
    alloc::{Allocator, Global},
    borrow::Borrow,
    mem::{ManuallyDrop, MaybeUninit},
    ops::{Deref, DerefMut},
};

use crate::{
    boxed::Box,
    device::{cpu::Cpu, DefaultDeviceAllocator, Device, DeviceAllocator, DevicePtr},
    ptr::{NonNull, Ref},
    zero::Zero,
};

/// Same as [`std::vec::Vec`] but using device allocators rather than host allocators.
/// This allows you to have owned buffers on GPUs and CPUs using a single data structure.
pub struct Vec<T, A: DeviceAllocator = Global> {
    buf: Box<[MaybeUninit<T>], A>,
    len: usize,
}

/// A [`Vec`] that uses the default allocator for the device
pub type DefaultVec<T, D = Cpu> = Vec<T, <D as DefaultDeviceAllocator>::Alloc>;

impl<T, A: DeviceAllocator> Drop for Vec<T, A> {
    fn drop(&mut self) {
        unsafe {
            // drop the data
            if std::mem::needs_drop::<T>() {
                // we are on the CPU
                if A::Device::IS_CPU {
                    let slice = &mut *(self.buf.ptr.as_ptr().as_raw());
                    let slice = &mut slice[..self.len];
                    for i in slice {
                        std::ptr::drop_in_place(i);
                    }
                } else {
                    panic!("drop types should not be initialised outside of the CPU")
                }
            }
        }
    }
}

impl<T: Copy, A: DeviceAllocator + Clone> Clone for Vec<T, A> {
    fn clone(&self) -> Self {
        let slice = &*self;
        unsafe {
            let mut vec = Self::with_capacity_in(slice.len(), self.buf.allocator().clone());
            vec.space_capacity_mut().init_from_slice(slice);
            vec.set_len(slice.len());
            vec
        }
    }
}

impl<T, A: DeviceAllocator> Vec<T, A> {
    pub fn from_box(b: Box<[T], A>) -> Self {
        let len = b.len();
        unsafe { Self::from_raw_parts(b.into_uninit(), len) }
    }

    pub fn zeroed_in(len: usize, alloc: A) -> Self
    where
        T: Zero,
    {
        Self::from_box(Box::zeroed(len, alloc))
    }

    #[must_use]
    pub fn zeroed(len: usize) -> Self
    where
        T: Zero,
        A: Default,
    {
        Self::zeroed_in(len, A::default())
    }

    pub fn copy_from_host_in(slice: &[T], alloc: A) -> Self
    where
        T: Copy,
    {
        unsafe {
            let mut vec = Self::with_capacity_in(slice.len(), alloc);
            vec.space_capacity_mut().init_from_host(slice);
            vec.set_len(slice.len());
            vec
        }
    }

    pub fn copy_from_host(slice: &[T]) -> Self
    where
        T: Copy,
        A: Default,
    {
        Self::copy_from_host_in(slice, A::default())
    }

    /// # Safety
    /// `buf` must be a valid allocation in `device`, and `len` items must be initialised
    pub unsafe fn from_raw_parts(buf: Box<[MaybeUninit<T>], A>, len: usize) -> Self {
        Self { buf, len }
    }

    pub fn into_raw_parts(self) -> (Box<[MaybeUninit<T>], A>, usize) {
        let v = ManuallyDrop::new(self);
        unsafe { (std::ptr::read(&v.buf), v.len) }
    }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self
    where
        A: Default,
    {
        Self::with_capacity_in(capacity, A::default())
    }

    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        let buf = Box::with_capacity(capacity, alloc);
        unsafe { Self::from_raw_parts(buf, 0) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    pub fn space_capacity_mut(&mut self) -> &mut Ref<[MaybeUninit<T>], A::Device> {
        &mut self.buf.deref_mut()[self.len..]
    }

    /// # Safety
    /// If len is smaller than the current length, the caller must ensure they drop the values.
    /// If the len is greater than the current length, the caller must ensure they have initialised those values
    pub unsafe fn set_len(&mut self, len: usize) {
        self.len = len;
    }

    unsafe fn ensure(&mut self, capacity: usize) {
        let old = self.capacity();
        if capacity > old {
            let new = match capacity {
                1..=4 => 4,
                n => n.next_power_of_two(),
            };

            self.buf.resize(new);
        }
    }

    pub fn push(&mut self, val: T) {
        unsafe {
            self.ensure(self.len + 1);
            self.buf.ptr.cast::<T>().as_ptr().add(self.len).write(val);
            self.len += 1;
        }
    }
}

impl<T, A: Allocator> From<std::vec::Vec<T, A>> for Vec<T, A> {
    fn from(v: std::vec::Vec<T, A>) -> Self {
        unsafe {
            let (ptr, len, cap, alloc) = v.into_raw_parts_with_alloc();
            let data = NonNull::new_unchecked(ptr.cast());
            let ptr = NonNull::slice_from_raw_parts(data, cap);
            let buf = Box::from_raw_parts(ptr, alloc);
            Self::from_raw_parts(buf, len)
        }
    }
}

impl<T, A: Allocator> From<Vec<T, A>> for std::vec::Vec<T, A> {
    fn from(v: Vec<T, A>) -> Self {
        unsafe {
            let (buf, len) = v.into_raw_parts();
            let (ptr, alloc) = buf.into_raw_parts();
            let (ptr, cap) = ptr.as_ptr().to_raw_parts();
            Self::from_raw_parts_in(ptr.cast(), len, cap, alloc)
        }
    }
}

impl<T, A: Allocator> Vec<T, A> {
    pub fn into_std(self) -> std::vec::Vec<T, A> {
        self.into()
    }
}

impl<T, A: DeviceAllocator> Deref for Vec<T, A> {
    type Target = Ref<[T], A::Device>;

    fn deref(&self) -> &Self::Target {
        unsafe { self.buf.deref()[..self.len()].assume_init() }
    }
}

impl<T, A: DeviceAllocator> DerefMut for Vec<T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.buf.deref_mut()[..self.len].assume_init_mut() }
    }
}

impl<T, A: DeviceAllocator> Borrow<Ref<[T], A::Device>> for Vec<T, A> {
    fn borrow(&self) -> &Ref<[T], A::Device> {
        self
    }
}

impl<T, A: DeviceAllocator> AsRef<Ref<[T], A::Device>> for Vec<T, A> {
    fn as_ref(&self) -> &Ref<[T], A::Device> {
        self
    }
}

impl<T, A: DeviceAllocator<Device = Cpu>> AsRef<[T]> for Vec<T, A> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T, A: DeviceAllocator> AsMut<Ref<[T], A::Device>> for Vec<T, A> {
    fn as_mut(&mut self) -> &mut Ref<[T], A::Device> {
        self
    }
}

impl<T, A: DeviceAllocator<Device = Cpu>> AsMut<[T]> for Vec<T, A> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::Vec;

    #[test]
    fn push() {
        let mut v = Vec::<_>::with_capacity(0);

        assert_eq!(v.capacity(), 0);
        v.push(0);
        assert_eq!(v.capacity(), 4);
        v.push(1);
        assert_eq!(v.capacity(), 4);
        v.push(2);
        assert_eq!(v.capacity(), 4);
        v.push(3);
        assert_eq!(v.capacity(), 4);
        v.push(4);
        assert_eq!(v.capacity(), 8);
    }

    #[test]
    fn convert() {
        let mut v1 = Vec::with_capacity(0);

        v1.push(0);
        v1.push(1);
        v1.push(2);
        v1.push(3);
        v1.push(4);

        let v2 = vec![0, 1, 2, 3, 4];

        assert_eq!(&**v1, v2.as_slice());
        assert_eq!(v1.into_std(), v2);
    }
}
