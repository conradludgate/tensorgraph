use std::{cell::RefCell, lazy::Lazy, ops::Deref};

use cust_raw::CUstream_st;

use crate::device::DefaultDeviceAllocator;

use super::{Cuda, SharedStream};

#[thread_local]
static GLOBAL: Lazy<RefCell<Option<std::ptr::NonNull<CUstream_st>>>> =
    Lazy::new(|| RefCell::new(None));

impl SharedStream {
    /// Runs the given closure with the stream as the global thread-local stream
    pub fn global_over<R, F: FnOnce(&Self) -> R>(&self, f: F) -> R {
        let pointer = GLOBAL.deref();

        let old = pointer.replace(Some(unsafe {
            std::ptr::NonNull::new_unchecked(self.inner())
        }));

        let out = f(self);

        let _stream = pointer.replace(old);

        out
    }
}

/// Get the global stream set via [`with_stream`]
pub fn get_stream() -> Option<&'static SharedStream> {
    GLOBAL
        .borrow()
        .map(|p| unsafe { &*(p.as_ptr() as *const _) })
}

impl DefaultDeviceAllocator for Cuda {
    type Alloc = &'static SharedStream;
}

impl Default for &'static SharedStream {
    fn default() -> Self {
        get_stream().unwrap()
    }
}
