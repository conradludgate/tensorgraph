use cust::memory::DevicePointer;

use crate::Device;

pub struct Cuda;

impl Device for Cuda {
    type Ptr<T: ?Sized> = DevicePointer<T>;
    type AllocError = cust::error::CudaError;

    unsafe fn allocate(&self, layout: std::alloc::Layout) -> cust::Result<Self::Ptr<[u8]>> {
        let size = layout.size();
        if size == 0 {
            return Err(CudaError::InvalidMemoryAllocation);
        }

        let mut ptr: *mut c_void = ptr::null_mut();
        cust_raw::cuMemAlloc_v2(&mut ptr as *mut *mut c_void as *mut u64, size).to_result()?;
        let ptr = ptr as *mut [u8];
        Ok(DevicePointer::wrap(ptr as *mut [u8]))
    }

    fn allocate_zeroed(&self, layout: std::alloc::Layout) -> cust::Result<Self::Ptr<[u8]>> {
        todo!()
    }

    unsafe fn deallocate(&self, ptr: Self::Ptr<u8>, layout: std::alloc::Layout) {
        let ptr = p.as_raw_mut();
        cuda::cuMemFree_v2(ptr as u64).to_result()?;
    }

    unsafe fn grow(
        &self,
        ptr: Self::Ptr<u8>,
        old_layout: std::alloc::Layout,
        new_layout: std::alloc::Layout,
    ) -> cust::Result<Self::Ptr<[u8]>> {
        todo!()
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: Self::Ptr<u8>,
        old_layout: std::alloc::Layout,
        new_layout: std::alloc::Layout,
    ) -> cust::Result<Self::Ptr<[u8]>> {
        todo!()
    }

    unsafe fn shrink(
        &self,
        ptr: Self::Ptr<u8>,
        old_layout: std::alloc::Layout,
        new_layout: std::alloc::Layout,
    ) -> cust::Result<Self::Ptr<[u8]>> {
        todo!()
    }
}
