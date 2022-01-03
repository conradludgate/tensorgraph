use std::{cell::RefCell, lazy::Lazy};

use cust_raw::CUstream_st;

use crate::device::DefaultDeviceAllocator;

use super::{Cuda, SharedStream};

#[thread_local]
static GLOBAL: Lazy<RefCell<Option<std::ptr::NonNull<CUstream_st>>>> =
    Lazy::new(|| RefCell::new(None));

impl SharedStream {
    /// Sets the stream as the global thread-local stream
    pub fn as_global(&self) -> StreamHandle {
        StreamHandle(GLOBAL.replace(Some(unsafe {
            std::ptr::NonNull::new_unchecked(self.inner())
        })))
    }
}

pub struct StreamHandle(Option<std::ptr::NonNull<CUstream_st>>);

impl Drop for StreamHandle {
    fn drop(&mut self) {
        let _ctx = GLOBAL.replace(self.0);
    }
}

/// Get the global stream set via [`SharedStream::as_global`]
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
