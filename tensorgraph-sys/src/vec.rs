use std::{
    alloc::{Allocator, Layout},
    borrow::Borrow,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
};

use crate::{
    device::{cpu::Cpu, Device, DevicePtr},
    ptr::{non_null::NonNull, slice::Slice},
};

pub struct Vec<T, D: Device> {
    device: D,
    buf: NonNull<[T], D>,
    len: usize,

    // marks that Vec owned the T values
    _marker: PhantomData<T>,
}

impl<T, D: Device> Vec<T, D> {
    /// # Safety
    /// `buf` must be a valid allocation in `device`, and `len` items must be initialised
    pub unsafe fn from_raw_parts_in(buf: NonNull<[T], D>, len: usize, device: D) -> Self {
        Self {
            device,
            buf,
            len,
            _marker: PhantomData,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self
    where
        D: Default,
    {
        Self::with_capacity_in(capacity, D::default())
    }

    pub fn with_capacity_in(capacity: usize, device: D) -> Self {
        unsafe {
            let (layout, _) = Layout::new::<T>().repeat(capacity).unwrap();
            let data = device.allocate(layout).unwrap().cast();
            let buf = NonNull::slice_from_raw_parts(data, capacity);
            Self::from_raw_parts_in(buf, 0, device)
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

    pub fn space_capacity_mut(&mut self) -> &mut Slice<MaybeUninit<T>, D> {
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
                .device
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

impl<T, A: Allocator> From<std::vec::Vec<T, A>> for Vec<T, Cpu<A>> {
    fn from(v: std::vec::Vec<T, A>) -> Self {
        unsafe {
            let (ptr, len, cap, alloc) = v.into_raw_parts_with_alloc();
            let data = NonNull::new_unchecked(ptr);
            let buf = NonNull::slice_from_raw_parts(data, cap);
            let device = Cpu::new(alloc);
            Self::from_raw_parts_in(buf, len, device)
        }
    }
}

impl<T, A: Allocator> From<Vec<T, Cpu<A>>> for std::vec::Vec<T, A> {
    fn from(v: Vec<T, Cpu<A>>) -> Self {
        unsafe {
            let Vec {
                device,
                buf,
                len,
                _marker: _,
            } = v;
            let alloc = device.alloc();
            let (ptr, cap) = buf.as_ptr().to_raw_parts();
            let ptr = ptr.cast();
            Self::from_raw_parts_in(ptr, len, cap, alloc)
        }
    }
}

impl<T, D: Device> Deref for Vec<T, D> {
    type Target = Slice<T, D>;

    fn deref(&self) -> &Self::Target {
        unsafe {
            let ptr: *mut [T] = self.buf.as_ptr().as_raw();
            let (ptr, _) = ptr.to_raw_parts();
            let ptr = std::ptr::from_raw_parts(ptr, self.len);
            &*(ptr as *const _)
        }
    }
}

impl<T, D: Device> DerefMut for Vec<T, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            let ptr: *mut [T] = self.buf.as_ptr().as_raw();
            let (ptr, _) = ptr.to_raw_parts();
            let ptr = std::ptr::from_raw_parts_mut(ptr, self.len);
            &mut *(ptr as *mut _)
        }
    }
}

impl<T, D: Device> Borrow<Slice<T, D>> for Vec<T, D> {
    fn borrow(&self) -> &Slice<T, D> {
        self.deref()
    }
}

impl<T, D: Device> AsRef<Slice<T, D>> for Vec<T, D> {
    fn as_ref(&self) -> &Slice<T, D> {
        self.deref()
    }
}

impl<T, D: Device> AsMut<Slice<T, D>> for Vec<T, D> {
    fn as_mut(&mut self) -> &mut Slice<T, D> {
        self.deref_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::Vec;
    use crate::device::cpu::Cpu;
    use std::{alloc::Global, ops::Deref};

    #[test]
    fn push() {
        let mut v = Vec::with_capacity_in(0, Cpu::<Global>::default());

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
        let mut v1 = Vec::with_capacity_in(0, Cpu::<Global>::default());

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
