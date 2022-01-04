use std::{fmt::Debug, marker::PhantomData, ptr::Pointee};

use crate::device::{Device, DevicePtr};

/// Same as [`std::ptr::NonNull<T>`] but backed by a [`Device::Ptr`] instead of a raw pointer
pub struct NonNull<T: ?Sized, D: Device> {
    inner: std::ptr::NonNull<T>,
    marker: PhantomData<D>,
}

impl<T: ?Sized, D: Device> Debug for NonNull<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("NonNull").field(&self.inner).finish()
    }
}

impl<T: ?Sized, D: Device> Clone for NonNull<T, D> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized, D: Device> Copy for NonNull<T, D> {}

impl<T: ?Sized, D: Device> NonNull<T, D> {
    pub fn new(ptr: D::Ptr<T>) -> Option<Self> {
        let inner = std::ptr::NonNull::new(ptr.as_raw())?;
        Some(Self {
            inner,
            marker: PhantomData,
        })
    }

    /// # Safety
    /// ptr must not be null
    pub unsafe fn new_unchecked(ptr: D::Ptr<T>) -> Self {
        let inner = std::ptr::NonNull::new_unchecked(ptr.as_raw());
        Self {
            inner,
            marker: PhantomData,
        }
    }

    #[must_use]
    pub fn as_ptr(self) -> D::Ptr<T> {
        D::Ptr::from_raw(self.inner.as_ptr())
    }

    #[must_use]
    pub fn cast<U>(self) -> NonNull<U, D> {
        let Self { inner, marker } = self;
        NonNull {
            inner: inner.cast(),
            marker,
        }
    }

    #[must_use]
    pub fn to_raw_parts(self) -> (NonNull<(), D>, <T as Pointee>::Metadata) {
        let (ptr, meta) = self.inner.as_ptr().to_raw_parts();
        let ptr = D::Ptr::from_raw(ptr);
        let data = unsafe { NonNull::new_unchecked(ptr) };
        (data, meta)
    }
}

impl<T, D: Device> NonNull<[T], D> {
    #[must_use]
    pub fn slice_from_raw_parts(data: NonNull<T, D>, len: usize) -> Self {
        let NonNull { inner, marker } = data;
        let inner = std::ptr::NonNull::slice_from_raw_parts(inner, len);
        Self { inner, marker }
    }
}
