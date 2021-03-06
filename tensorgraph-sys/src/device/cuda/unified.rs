use std::{
    ffi::c_void,
    ops::{Deref, DerefMut},
};

use cust::error::{CudaError, CudaResult};
use cust_raw::CUmemAttach_flags_enum;

use crate::{
    device::{cuda::ToCudaResult, DefaultDeviceAllocator, Device, DeviceAllocator},
    ptr::{NonNull, Ref},
};

#[derive(Debug)]
/// Device for CUDA enabled GPUs, using unified memory
pub struct CudaUnified;

impl Device for CudaUnified {
    type Ptr<T: ?Sized> = *mut T;
    const IS_CPU: bool = true;

    fn copy_from_host<T: Copy>(from: &[T], to: &mut Ref<[T], Self>) {
        assert_eq!(from.len(), to.len());
        // Safety:
        // These slices had to be made using unsafe functions. Those should ensure that these pointers are valid.
        unsafe {
            cust_raw::cuMemcpyHtoD_v2(
                to.as_slice_ptr().cast::<u8>() as u64,
                from.as_ptr().cast(),
                std::mem::size_of::<T>() * from.len(),
            )
            .to_cuda_result()
            .unwrap();
        }
    }

    fn copy_to_host<T: Copy>(from: &Ref<[T], Self>, to: &mut [T]) {
        assert_eq!(from.len(), to.len());
        // Safety:
        // These slices had to be made using unsafe functions. Those should ensure that these pointers are valid.
        unsafe {
            cust_raw::cuMemcpyDtoH_v2(
                to.as_mut_ptr().cast(),
                from.as_slice_ptr().cast::<u8>() as u64,
                std::mem::size_of::<T>() * from.len(),
            )
            .to_cuda_result()
            .unwrap();
        }
    }

    fn copy<T: Copy>(from: &Ref<[T], Self>, to: &mut Ref<[T], Self>) {
        assert_eq!(from.len(), to.len());
        // Safety:
        // These slices had to be made using unsafe functions. Those should ensure that these pointers are valid.
        unsafe {
            cust_raw::cuMemcpy(
                to.as_slice_ptr().cast::<u8>() as u64,
                from.as_slice_ptr().cast::<u8>() as u64,
                std::mem::size_of::<T>() * from.len(),
            )
            .to_cuda_result()
            .unwrap();
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct UnifiedAlloc;

impl DefaultDeviceAllocator for CudaUnified {
    type Alloc = UnifiedAlloc;
}

impl DeviceAllocator for UnifiedAlloc {
    type AllocError = CudaError;
    type Device = CudaUnified;

    fn allocate(&self, layout: std::alloc::Layout) -> CudaResult<NonNull<[u8], CudaUnified>> {
        let size = layout.size();
        if size == 0 {
            return Err(CudaError::InvalidMemoryAllocation);
        }

        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            cust_raw::cuMemAllocManaged(
                (&mut ptr as *mut *mut c_void).cast(),
                size,
                CUmemAttach_flags_enum::CU_MEM_ATTACH_GLOBAL as u32,
            )
            .to_cuda_result()?;
        }
        let ptr = std::ptr::from_raw_parts_mut(ptr.cast(), size);
        unsafe { Ok(NonNull::new_unchecked(ptr)) }
    }

    fn allocate_zeroed(
        &self,
        layout: std::alloc::Layout,
    ) -> CudaResult<NonNull<[u8], CudaUnified>> {
        let size = layout.size();
        let ptr = self.allocate(layout)?;
        unsafe {
            cust_raw::cuMemsetD8_v2(d_ptr(ptr), 0, size).to_cuda_result()?;
        }
        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8, CudaUnified>, _layout: std::alloc::Layout) {
        cust_raw::cuMemFree_v2(d_ptr1(ptr))
            .to_cuda_result()
            .unwrap();
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8, CudaUnified>,
        old_layout: std::alloc::Layout,
        new_layout: std::alloc::Layout,
    ) -> CudaResult<NonNull<[u8], CudaUnified>> {
        let new = self.allocate(new_layout)?;

        let size = old_layout.size();
        cust_raw::cuMemcpy(d_ptr(new), d_ptr1(ptr), size).to_cuda_result()?;
        cust_raw::cuMemFree_v2(d_ptr1(ptr)).to_cuda_result()?;

        Ok(new)
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8, CudaUnified>,
        old_layout: std::alloc::Layout,
        new_layout: std::alloc::Layout,
    ) -> CudaResult<NonNull<[u8], CudaUnified>> {
        let new = self.allocate_zeroed(new_layout)?;

        let size = old_layout.size();
        cust_raw::cuMemcpy(d_ptr(new), d_ptr1(ptr), size).to_cuda_result()?;
        cust_raw::cuMemFree_v2(d_ptr1(ptr)).to_cuda_result()?;

        Ok(new)
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8, CudaUnified>,
        _old_layout: std::alloc::Layout,
        new_layout: std::alloc::Layout,
    ) -> CudaResult<NonNull<[u8], CudaUnified>> {
        let size = new_layout.size();
        let new = self.allocate(new_layout)?;

        cust_raw::cuMemcpy(d_ptr(new), d_ptr1(ptr), size).to_cuda_result()?;
        cust_raw::cuMemFree_v2(d_ptr1(ptr)).to_cuda_result()?;

        Ok(new)
    }
}

fn d_ptr(ptr: NonNull<[u8], CudaUnified>) -> cust_raw::CUdeviceptr {
    d_ptr2(ptr.as_ptr())
}
fn d_ptr1(ptr: NonNull<u8, CudaUnified>) -> cust_raw::CUdeviceptr {
    d_ptr3(ptr.as_ptr())
}
fn d_ptr2(ptr: *mut [u8]) -> cust_raw::CUdeviceptr {
    ptr.cast::<u8>() as cust_raw::CUdeviceptr
}
fn d_ptr3(ptr: *mut u8) -> cust_raw::CUdeviceptr {
    ptr as cust_raw::CUdeviceptr
}

/// Newtype wrapper used in any unified contexts.
#[repr(transparent)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Unified<T: ?Sized>(T);

impl<T> From<T> for Unified<T> {
    fn from(t: T) -> Self {
        Self(t)
    }
}

impl<'a, T: ?Sized> From<&'a T> for &'a Unified<T> {
    fn from(t: &'a T) -> Self {
        unsafe { &*(t as *const _ as *const _) }
    }
}

impl<T: ?Sized> Deref for Unified<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<'a, T: ?Sized> From<&'a mut T> for &'a mut Unified<T> {
    fn from(t: &'a mut T) -> Self {
        unsafe { &mut *(t as *mut _ as *mut _) }
    }
}

impl<T: ?Sized> DerefMut for Unified<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<'a, T: ?Sized> Default for &'a Unified<T>
where
    &'a T: Default,
{
    fn default() -> Self {
        <&'a T>::default().into()
    }
}
