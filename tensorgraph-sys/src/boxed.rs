use std::{
    alloc::{Allocator, Global, Layout},
    marker::PhantomData,
    mem::{align_of_val, size_of_val, MaybeUninit},
    ops::{Deref, DerefMut},
};

use crate::{
    device::{DeviceAllocator, DevicePtr},
    ptr::{non_null::NonNull, reef::Ref},
    zero::Zero,
};

/// Similar to [`std::boxed::Box`] but on device.
pub struct Box<T: ?Sized, A: DeviceAllocator = Global> {
    pub(crate) ptr: NonNull<T, A::Device>,
    alloc: A,

    _marker: PhantomData<T>,
}

impl<T: ?Sized, A: DeviceAllocator> Box<T, A> {
    pub fn into_raw_parts(self) -> (NonNull<T, A::Device>, A) {
        let b = std::mem::ManuallyDrop::new(self);
        (b.ptr, unsafe { std::ptr::read(&b.alloc) })
    }

    pub unsafe fn from_raw_parts(ptr: NonNull<T, A::Device>, alloc: A) -> Self {
        Self {
            ptr,
            alloc,
            _marker: PhantomData,
        }
    }

    pub fn allocator(&self) -> &A {
        &self.alloc
    }
}
impl<T: ?Sized, A: Allocator> Box<T, A> {
    pub fn into_std(self) -> std::boxed::Box<T, A> {
        unsafe {
            let (ptr, alloc) = self.into_raw_parts();
            std::boxed::Box::from_raw_in(ptr.as_ptr(), alloc)
        }
    }
}

impl<T: ?Sized, A: DeviceAllocator> Deref for Box<T, A> {
    type Target = Ref<T, A::Device>;

    fn deref(&self) -> &Self::Target {
        unsafe { Ref::from_ptr(self.ptr.as_ptr()) }
    }
}

impl<T, A: DeviceAllocator> DerefMut for Box<[T], A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { Ref::from_ptr_mut(self.ptr.as_ptr()) }
    }
}

impl<T, A: DeviceAllocator> Box<[MaybeUninit<T>], A> {
    /// # Safety
    /// If this resize results in a shrink, the data that is lost must be already dropped
    pub unsafe fn resize(&mut self, capacity: usize) {
        let new = capacity;
        let old = self.len();

        let layout = Layout::new::<T>();
        let old_layout = layout.repeat(old).unwrap().0;
        let new_layout = layout.repeat(new).unwrap().0;

        let data = match new.cmp(&old) {
            std::cmp::Ordering::Greater => self
                .alloc
                .grow(self.ptr.cast(), old_layout, new_layout)
                .unwrap()
                .cast(),
            std::cmp::Ordering::Less => self
                .alloc
                .shrink(self.ptr.cast(), old_layout, new_layout)
                .unwrap()
                .cast(),
            std::cmp::Ordering::Equal => self.ptr.cast(),
        };

        self.ptr = NonNull::slice_from_raw_parts(data, new);
    }

    pub fn with_capacity(capacity: usize, alloc: A) -> Self {
        unsafe {
            let (layout, _) = Layout::new::<T>().repeat(capacity).unwrap();
            let data = alloc.allocate(layout).unwrap().cast();
            let buf = NonNull::slice_from_raw_parts(data, capacity);
            Self::from_raw_parts(buf, alloc)
        }
    }
    pub fn zeroed_uninit(capacity: usize, alloc: A) -> Self {
        unsafe {
            let (layout, _) = Layout::new::<T>().repeat(capacity).unwrap();
            let data = alloc.allocate_zeroed(layout).unwrap().cast();
            let buf = NonNull::slice_from_raw_parts(data, capacity);
            Self::from_raw_parts(buf, alloc)
        }
    }
}

impl<T, A: DeviceAllocator> Box<[T], A> {
    pub fn zeroed(capacity: usize, alloc: A) -> Self
    where
        T: Zero,
    {
        unsafe {
            let (layout, _) = Layout::new::<T>().repeat(capacity).unwrap();
            let data = alloc.allocate_zeroed(layout).unwrap().cast();
            let buf = NonNull::slice_from_raw_parts(data, capacity);
            Self::from_raw_parts(buf, alloc)
        }
    }

    pub fn into_uninit(self) -> Box<[MaybeUninit<T>], A> {
        unsafe {
            let (ptr, alloc) = self.into_raw_parts();
            let (ptr, len) = ptr.to_raw_parts();
            let ptr = NonNull::slice_from_raw_parts(ptr.cast(), len);
            Box::from_raw_parts(ptr, alloc)
        }
    }
}

impl<T: ?Sized, A: DeviceAllocator> Drop for Box<T, A> {
    fn drop(&mut self) {
        unsafe {
            let _ref = &*(self.ptr.as_ptr().as_raw());
            let size = size_of_val(_ref);
            let align = align_of_val(_ref);
            let layout = Layout::from_size_align_unchecked(size, align);
            self.alloc.deallocate(self.ptr.cast(), layout)
        }
    }
}