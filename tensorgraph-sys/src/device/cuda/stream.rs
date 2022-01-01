use std::{cell::RefCell, lazy::Lazy, ops::Deref, ptr::NonNull};

use cust::error::CudaResult;
use cust_raw::{CUstream, CUstream_st};

use super::ToCudaResult;

pub struct Stream {
    inner: NonNull<CUstream_st>,
}

impl Stream {
    pub fn new() -> CudaResult<Self> {
        let mut stream = std::ptr::null_mut();

        unsafe {
            cust_raw::cuStreamCreate(&mut stream, 0).to_cuda_result()?;

            Ok(Self {
                inner: NonNull::new_unchecked(stream),
            })
        }
    }
}

impl Deref for Stream {
    type Target = SharedStream;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self.inner.as_ptr() as *const _) }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            cust_raw::cuStreamDestroy_v2(self.inner.as_mut());
        }
    }
}

pub struct SharedStream(CUstream_st);

impl SharedStream {
    pub(crate) fn inner(&self) -> CUstream {
        self as *const _ as *mut _
    }
}

#[thread_local]
static GLOBAL: Lazy<RefCell<Option<NonNull<CUstream_st>>>> = Lazy::new(|| RefCell::new(None));

pub fn with_stream<R, F: FnOnce(&SharedStream) -> R>(stream: &SharedStream, f: F) -> R {
    let pointer = GLOBAL.deref();

    let old = pointer.replace(Some(unsafe { NonNull::new_unchecked(stream.inner()) }));

    let out = f(stream);

    let _stream = pointer.replace(old);

    out
}

pub fn get_stream<'a>() -> Option<&'a SharedStream> {
    GLOBAL
        .borrow()
        .map(|p| unsafe { &*(p.as_ptr() as *const _) })
}

pub struct GlobalStream<'a> {
    pub(crate) inner: &'a SharedStream,
}

impl<'a> Default for GlobalStream<'a> {
    fn default() -> Self {
        Self {
            inner: get_stream().unwrap(),
        }
    }
}
