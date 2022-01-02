use std::{
    alloc::{Allocator, Global, Layout},
    borrow::Borrow,
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    ops::{Deref, DerefMut},
};

use crate::{
    device::{Device, DeviceAllocator, DevicePtr, DefaultDeviceAllocator},
    ptr::{non_null::NonNull, slice::Slice},
    zero::Zero,
};

pub struct Vec<T, A: DeviceAllocator = Global> {
    alloc: A,
    buf: NonNull<[T], A::Device>,
    len: usize,

    // marks that Vec owned the T values
    _marker: PhantomData<T>,
}

impl<T, A: DeviceAllocator> Drop for Vec<T, A> {
    fn drop(&mut self) {
        unsafe {
            // drop the data
            if std::mem::needs_drop::<T>() {
                // we are on the CPU
                if A::Device::IS_CPU {
                    let slice = &mut *(self.buf.as_ptr().as_raw());
                    let slice = &mut slice[..self.len];
                    for i in slice {
                        std::ptr::drop_in_place(i);
                    }
                } else {
                    panic!("drop types should not be initialised outside of the CPU")
                }
            }
            let (layout, _) = Layout::new::<T>().repeat(self.capacity()).unwrap();
            self.alloc.deallocate(self.buf.cast(), layout)
        }
    }
}

impl<T: Copy, A: DeviceAllocator + Clone> Clone for Vec<T, A> {
    fn clone(&self) -> Self {
        let slice = self.deref();
        unsafe {
            let mut vec = Self::with_capacity_in(slice.len(), self.alloc.clone());
            vec.space_capacity_mut().init_from_slice(slice);
            vec.set_len(slice.len());
            vec
        }
    }
}

pub fn vec_from_host<T: Copy, D: DefaultDeviceAllocator>(slice: &[T]) -> Vec<T, D::Alloc> {
    Vec::copy_from_host_in(slice, D::default_alloc())
}

impl<T, A: DeviceAllocator> Vec<T, A> {
    pub fn zeroed_in(len: usize, alloc: A) -> Self
    where
        T: Zero,
    {
        unsafe {
            let (layout, _) = Layout::new::<T>().repeat(len).unwrap();
            let data = alloc.allocate_zeroed(layout).unwrap().cast();
            let buf = NonNull::slice_from_raw_parts(data, len);
            Self::from_raw_parts_in(buf, len, alloc)
        }
    }

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
    pub unsafe fn from_raw_parts_in(buf: NonNull<[T], A::Device>, len: usize, alloc: A) -> Self {
        Self {
            alloc,
            buf,
            len,
            _marker: PhantomData,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self
    where
        A: Default,
    {
        Self::with_capacity_in(capacity, A::default())
    }

    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        unsafe {
            let (layout, _) = Layout::new::<T>().repeat(capacity).unwrap();
            let data = alloc.allocate(layout).unwrap().cast();
            let buf = NonNull::slice_from_raw_parts(data, capacity);
            Self::from_raw_parts_in(buf, 0, alloc)
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        let buf: *mut [T] = self.buf.as_ptr().as_raw();
        buf.len()
    }

    pub fn space_capacity_mut(&mut self) -> &mut Slice<MaybeUninit<T>, A::Device> {
        unsafe {
            let ptr: *mut [T] = self.buf.as_ptr().as_raw();
            let (ptr, cap) = ptr.to_raw_parts();
            let ptr = ptr as *mut T;
            let ptr = ptr.add(self.len);
            let ptr = std::ptr::from_raw_parts_mut(ptr as *mut _, cap - self.len);
            &mut *(ptr as *mut _)
        }
    }

    /// # Safety
    /// If len is smaller than the current length, the caller must ensure they drop the values.
    /// If the len is greater than the current length, the caller must ensure they have initialised those values
    pub unsafe fn set_len(&mut self, len: usize) {
        self.len = len
    }

    unsafe fn ensure(&mut self, capacity: usize) {
        let old = self.capacity();
        if capacity > old {
            let new = match capacity {
                1..=4 => 4,
                n => n.next_power_of_two(),
            };

            let layout = Layout::new::<T>();
            let old_layout = layout.repeat(old).unwrap().0;
            let new_layout = layout.repeat(new).unwrap().0;

            let data = self
                .alloc
                .grow(self.buf.cast(), old_layout, new_layout)
                .unwrap()
                .cast();
            self.buf = NonNull::slice_from_raw_parts(data, new);
        }
    }

    pub fn push(&mut self, val: T) {
        unsafe {
            self.ensure(self.len + 1);
            self.buf.cast::<T>().as_ptr().add(self.len).write(val);
            self.len += 1;
        }
    }
}

impl<T, A: Allocator> From<std::vec::Vec<T, A>> for Vec<T, A> {
    fn from(v: std::vec::Vec<T, A>) -> Self {
        unsafe {
            let (ptr, len, cap, alloc) = v.into_raw_parts_with_alloc();
            let data = NonNull::new_unchecked(ptr);
            let buf = NonNull::slice_from_raw_parts(data, cap);
            Self::from_raw_parts_in(buf, len, alloc)
        }
    }
}

impl<T, A: Allocator> From<Vec<T, A>> for std::vec::Vec<T, A> {
    fn from(v: Vec<T, A>) -> Self {
        unsafe {
            let v = ManuallyDrop::new(v);
            let alloc = std::ptr::read(&v.alloc);
            let (ptr, cap) = v.buf.as_ptr().to_raw_parts();
            let ptr = ptr.cast();
            Self::from_raw_parts_in(ptr, v.len, cap, alloc)
        }
    }
}

impl<T, A: Allocator> Vec<T, A> {
    pub fn into_std(self) -> std::vec::Vec<T, A> {
        self.into()
    }
}

impl<T, A: DeviceAllocator> Deref for Vec<T, A> {
    type Target = Slice<T, A::Device>;

    fn deref(&self) -> &Self::Target {
        unsafe {
            let ptr: *mut [T] = self.buf.as_ptr().as_raw();
            let (ptr, _) = ptr.to_raw_parts();
            let ptr = std::ptr::from_raw_parts(ptr, self.len);
            &*(ptr as *const _)
        }
    }
}

impl<T, A: DeviceAllocator> DerefMut for Vec<T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            let ptr: *mut [T] = self.buf.as_ptr().as_raw();
            let (ptr, _) = ptr.to_raw_parts();
            let ptr = std::ptr::from_raw_parts_mut(ptr, self.len);
            &mut *(ptr as *mut _)
        }
    }
}

impl<T, A: DeviceAllocator> Borrow<Slice<T, A::Device>> for Vec<T, A> {
    fn borrow(&self) -> &Slice<T, A::Device> {
        self.deref()
    }
}

impl<T, A: DeviceAllocator> AsRef<Slice<T, A::Device>> for Vec<T, A> {
    fn as_ref(&self) -> &Slice<T, A::Device> {
        self.deref()
    }
}

impl<T, A: DeviceAllocator> AsMut<Slice<T, A::Device>> for Vec<T, A> {
    fn as_mut(&mut self) -> &mut Slice<T, A::Device> {
        self.deref_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::Vec;
    use std::ops::Deref;

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

        assert_eq!(v1.deref().deref(), v2.as_slice());
        assert_eq!(std::vec::Vec::from(v1), v2);
    }
}
