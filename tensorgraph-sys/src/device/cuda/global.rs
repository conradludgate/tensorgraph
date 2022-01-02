use std::{cell::RefCell, lazy::Lazy, ops::Deref};

use cust_raw::CUstream_st;

use crate::device::DefaultDeviceAllocator;

use super::{Cuda, SharedStream};

#[thread_local]
static GLOBAL: Lazy<RefCell<Option<std::ptr::NonNull<CUstream_st>>>> =
    Lazy::new(|| RefCell::new(None));

pub fn with_stream<R, F: FnOnce(&SharedStream) -> R>(stream: &SharedStream, f: F) -> R {
    let pointer = GLOBAL.deref();

    let old = pointer.replace(Some(unsafe {
        std::ptr::NonNull::new_unchecked(stream.inner())
    }));

    let out = f(stream);

    let _stream = pointer.replace(old);

    out
}

pub fn get_stream() -> Option<&'static SharedStream> {
    GLOBAL
        .borrow()
        .map(|p| unsafe { &*(p.as_ptr() as *const _) })
}

impl DefaultDeviceAllocator for Cuda {
    type Alloc = &'static SharedStream;

    fn default_alloc() -> Self::Alloc {
        get_stream().unwrap()
    }
}

impl Default for &'static SharedStream {
    fn default() -> Self {
        get_stream().unwrap()
    }
}

// pub struct GlobalStream {
//     pub(crate) inner: &'static SharedStream,
// }

// impl Default for GlobalStream {
//     fn default() -> Self {
//         Self {
//             inner: get_stream().unwrap(),
//         }
//     }
// }

// impl DeviceAllocator for GlobalStream {
//     type AllocError = <&'static SharedStream as DeviceAllocator>::AllocError;
//     type Device = Cuda;

//     unsafe fn allocate(&self, layout: std::alloc::Layout) -> CudaResult<NonNull<[u8], Cuda>> {
//         self.inner.allocate(layout)
//     }

//     unsafe fn allocate_zeroed(
//         &self,
//         layout: std::alloc::Layout,
//     ) -> CudaResult<NonNull<[u8], Cuda>> {
//         self.inner.allocate_zeroed(layout)
//     }

//     unsafe fn deallocate(&self, ptr: NonNull<u8, Cuda>, layout: std::alloc::Layout) {
//         self.inner.deallocate(ptr, layout)
//     }

//     unsafe fn grow(
//         &self,
//         ptr: NonNull<u8, Cuda>,
//         old_layout: std::alloc::Layout,
//         new_layout: std::alloc::Layout,
//     ) -> CudaResult<NonNull<[u8], Cuda>> {
//         self.inner.grow(ptr, old_layout, new_layout)
//     }

//     unsafe fn grow_zeroed(
//         &self,
//         ptr: NonNull<u8, Cuda>,
//         old_layout: std::alloc::Layout,
//         new_layout: std::alloc::Layout,
//     ) -> CudaResult<NonNull<[u8], Cuda>> {
//         self.inner.grow_zeroed(ptr, old_layout, new_layout)
//     }

//     unsafe fn shrink(
//         &self,
//         ptr: NonNull<u8, Cuda>,
//         old_layout: std::alloc::Layout,
//         new_layout: std::alloc::Layout,
//     ) -> CudaResult<NonNull<[u8], Cuda>> {
//         self.inner.shrink(ptr, old_layout, new_layout)
//     }
// }
