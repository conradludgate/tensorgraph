use std::{fmt::Debug, marker::PhantomData, ptr::Pointee};

use crate::device::{Device, DevicePtr};

pub struct NonNull<T: ?Sized, D: Device + ?Sized> {
    inner: std::ptr::NonNull<T>,
    _marker: PhantomData<D>,
}

impl<T: ?Sized, D: Device + ?Sized> Debug for NonNull<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("NonNull").field(&self.inner).finish()
    }
}

impl<T: ?Sized, D: Device + ?Sized> Clone for NonNull<T, D> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized, D: Device + ?Sized> Copy for NonNull<T, D> {}

impl<T: ?Sized, D: Device + ?Sized> NonNull<T, D> {
    pub fn new(ptr: D::Ptr<T>) -> Option<Self> {
        let inner = std::ptr::NonNull::new(ptr.as_raw())?;
        Some(Self {
            inner,
            _marker: PhantomData,
        })
    }

    /// # Safety
    /// ptr must not be null
    pub unsafe fn new_unchecked(ptr: D::Ptr<T>) -> Self {
        let inner = std::ptr::NonNull::new_unchecked(ptr.as_raw());
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    pub fn as_ptr(self) -> D::Ptr<T> {
        D::Ptr::from_raw(self.inner.as_ptr())
    }

    pub fn cast<U>(self) -> NonNull<U, D> {
        let Self { inner, _marker } = self;
        NonNull {
            inner: inner.cast(),
            _marker,
        }
    }

    pub fn to_raw_parts(self) -> (NonNull<(), D>, <T as Pointee>::Metadata) {
        let (ptr, meta) = self.inner.as_ptr().to_raw_parts();
        let ptr = D::Ptr::from_raw(ptr);
        let data = unsafe { NonNull::new_unchecked(ptr) };
        (data, meta)
    }
}

impl<T, D: Device + ?Sized> NonNull<[T], D> {
    pub fn slice_from_raw_parts(data: NonNull<T, D>, len: usize) -> Self {
        let NonNull { inner, _marker } = data;
        let inner = std::ptr::NonNull::slice_from_raw_parts(inner, len);
        Self { inner, _marker }
    }
}
