use std::ffi::c_void;

use cust::{
    error::{CudaError, CudaResult},
    memory::DevicePointer,
};

use crate::ptr::{NonNull, Ref};

use super::{Device, DeviceAllocator, DevicePtr};

mod context;
mod global;
mod stream;
mod unified;

pub use context::{AttachedContext, Context, FloatingContext, SharedContext};
pub use global::get_stream;
pub use stream::{SharedStream, Stream};
pub use unified::{CudaUnified, Unified, UnifiedAlloc};

#[derive(Debug)]
/// Device for CUDA enabled GPUs
///
/// # Note
/// Needs a [`Context`] to be created on the current thread to work
pub struct Cuda;

impl Device for Cuda {
    type Ptr<T: ?Sized> = DevicePointer<T>;

    fn copy_from_host<T: Copy>(from: &[T], to: &mut Ref<[T], Self>) {
        assert_eq!(from.len(), to.len());
        // Safety:
        // These slices had to be made using unsafe functions. Those should ensure that these pointers are valid.
        unsafe {
            cust_raw::cuMemcpyHtoD_v2(
                to.as_slice_ptr().as_raw_mut().cast::<u8>() as u64,
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
                from.as_slice_ptr().as_raw_mut().cast::<u8>() as u64,
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
                to.as_slice_ptr().as_raw_mut().cast::<u8>() as u64,
                from.as_slice_ptr().as_raw_mut().cast::<u8>() as u64,
                std::mem::size_of::<T>() * from.len(),
            )
            .to_cuda_result()
            .unwrap();
        }
    }
}

impl SharedStream {
    unsafe fn grow_impl(
        &self,
        ptr: NonNull<u8, Cuda>,
        old_layout: std::alloc::Layout,
        new_layout: std::alloc::Layout,
        zeroed: bool,
    ) -> CudaResult<NonNull<[u8], Cuda>> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        let new = if zeroed {
            self.allocate_zeroed(new_layout)
        } else {
            self.allocate(new_layout)
        }?;

        let size = old_layout.size();
        if size != 0 {
            cust_raw::cuMemcpyAsync(d_ptr(new), d_ptr1(ptr), size, self.inner())
                .to_cuda_result()?;
            cust_raw::cuMemFreeAsync(d_ptr1(ptr), self.inner()).to_cuda_result()?;
        }

        Ok(new)
    }
}

impl<'a> DeviceAllocator for &'a SharedStream {
    type AllocError = CudaError;
    type Device = Cuda;

    fn allocate(&self, layout: std::alloc::Layout) -> CudaResult<NonNull<[u8], Cuda>> {
        let size = layout.size();
        if size == 0 {
            unsafe {
                let dangling =
                    NonNull::new_unchecked(DevicePointer::from_raw(layout.dangling().as_ptr()));
                return Ok(NonNull::slice_from_raw_parts(dangling, 0));
            }
        }

        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            cust_raw::cuMemAllocAsync((&mut ptr as *mut *mut c_void).cast(), size, self.inner())
                .to_cuda_result()?;
        }
        let ptr = std::ptr::from_raw_parts_mut(ptr.cast(), size);
        unsafe { Ok(NonNull::new_unchecked(DevicePointer::wrap(ptr))) }
    }

    fn allocate_zeroed(&self, layout: std::alloc::Layout) -> CudaResult<NonNull<[u8], Cuda>> {
        let size = layout.size();
        let ptr = self.allocate(layout)?;
        unsafe {
            cust_raw::cuMemsetD8Async(d_ptr(ptr), 0, size, self.inner()).to_cuda_result()?;
        }
        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8, Cuda>, layout: std::alloc::Layout) {
        if layout.size() != 0 {
            cust_raw::cuMemFreeAsync(d_ptr1(ptr), self.inner())
                .to_cuda_result()
                .unwrap();
        }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8, Cuda>,
        old_layout: std::alloc::Layout,
        new_layout: std::alloc::Layout,
    ) -> CudaResult<NonNull<[u8], Cuda>> {
        self.grow_impl(ptr, old_layout, new_layout, false)
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8, Cuda>,
        old_layout: std::alloc::Layout,
        new_layout: std::alloc::Layout,
    ) -> CudaResult<NonNull<[u8], Cuda>> {
        self.grow_impl(ptr, old_layout, new_layout, true)
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8, Cuda>,
        old_layout: std::alloc::Layout,
        new_layout: std::alloc::Layout,
    ) -> CudaResult<NonNull<[u8], Cuda>> {
        debug_assert!(
            new_layout.size() <= old_layout.size(),
            "`new_layout.size()` must be less than or equal to `old_layout.size()`"
        );

        let size = new_layout.size();

        if size == 0 {
            let dangling =
                NonNull::new_unchecked(DevicePointer::from_raw(new_layout.dangling().as_ptr()));
            Ok(NonNull::slice_from_raw_parts(dangling, 0))
        } else {
            let new = self.allocate(new_layout)?;

            cust_raw::cuMemcpyAsync(d_ptr(new), d_ptr1(ptr), size, self.inner())
                .to_cuda_result()?;
            cust_raw::cuMemFreeAsync(d_ptr1(ptr), self.inner()).to_cuda_result()?;

            Ok(new)
        }
    }
}

impl<T: ?Sized> DevicePtr<T> for DevicePointer<T> {
    fn as_raw(mut self) -> *mut T {
        self.as_raw_mut()
    }

    fn from_raw(ptr: *mut T) -> Self {
        // creating pointers shouldn't be unsafe
        // it's reading/writing to them that's unsafe
        #![allow(clippy::not_unsafe_ptr_arg_deref)]
        unsafe { Self::wrap(ptr) }
    }

    unsafe fn write(self, val: T)
    where
        T: Sized,
    {
        // this might not be the most efficient op, but I dount this will be used much
        let host_slice: *const [u8] =
            std::ptr::from_raw_parts((&val as *const T).cast(), std::mem::size_of::<T>());
        let dev_slice: *mut [u8] =
            std::ptr::from_raw_parts_mut(self.as_raw() as *mut (), std::mem::size_of::<T>());
        let dev_slice = DevicePointer::from_raw(dev_slice);
        Cuda::copy_from_host(&*host_slice, Ref::from_ptr_mut(dev_slice));
    }
}

fn d_ptr(ptr: NonNull<[u8], Cuda>) -> cust_raw::CUdeviceptr {
    d_ptr2(ptr.as_ptr())
}
fn d_ptr1(ptr: NonNull<u8, Cuda>) -> cust_raw::CUdeviceptr {
    d_ptr3(ptr.as_ptr())
}
fn d_ptr2(mut ptr: DevicePointer<[u8]>) -> cust_raw::CUdeviceptr {
    ptr.as_raw_mut().cast::<u8>() as cust_raw::CUdeviceptr
}
fn d_ptr3(mut ptr: DevicePointer<u8>) -> cust_raw::CUdeviceptr {
    ptr.as_raw_mut() as cust_raw::CUdeviceptr
}

pub(crate) trait ToCudaResult {
    fn to_cuda_result(self) -> CudaResult<()>;
}
impl ToCudaResult for cust_raw::cudaError_enum {
    fn to_cuda_result(self) -> CudaResult<()> {
        use cust_raw::cudaError_enum;
        match self {
            cudaError_enum::CUDA_SUCCESS => Ok(()),
            cudaError_enum::CUDA_ERROR_INVALID_VALUE => Err(CudaError::InvalidValue),
            cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY => Err(CudaError::OutOfMemory),
            cudaError_enum::CUDA_ERROR_NOT_INITIALIZED => Err(CudaError::NotInitialized),
            cudaError_enum::CUDA_ERROR_DEINITIALIZED => Err(CudaError::Deinitialized),
            cudaError_enum::CUDA_ERROR_PROFILER_DISABLED => Err(CudaError::ProfilerDisabled),
            cudaError_enum::CUDA_ERROR_PROFILER_NOT_INITIALIZED => {
                Err(CudaError::ProfilerNotInitialized)
            }
            cudaError_enum::CUDA_ERROR_PROFILER_ALREADY_STARTED => {
                Err(CudaError::ProfilerAlreadyStarted)
            }
            cudaError_enum::CUDA_ERROR_PROFILER_ALREADY_STOPPED => {
                Err(CudaError::ProfilerAlreadyStopped)
            }
            cudaError_enum::CUDA_ERROR_NO_DEVICE => Err(CudaError::NoDevice),
            cudaError_enum::CUDA_ERROR_INVALID_DEVICE => Err(CudaError::InvalidDevice),
            cudaError_enum::CUDA_ERROR_INVALID_IMAGE => Err(CudaError::InvalidImage),
            cudaError_enum::CUDA_ERROR_INVALID_CONTEXT => Err(CudaError::InvalidContext),
            cudaError_enum::CUDA_ERROR_CONTEXT_ALREADY_CURRENT => {
                Err(CudaError::ContextAlreadyCurrent)
            }
            cudaError_enum::CUDA_ERROR_MAP_FAILED => Err(CudaError::MapFailed),
            cudaError_enum::CUDA_ERROR_UNMAP_FAILED => Err(CudaError::UnmapFailed),
            cudaError_enum::CUDA_ERROR_ARRAY_IS_MAPPED => Err(CudaError::ArrayIsMapped),
            cudaError_enum::CUDA_ERROR_ALREADY_MAPPED => Err(CudaError::AlreadyMapped),
            cudaError_enum::CUDA_ERROR_NO_BINARY_FOR_GPU => Err(CudaError::NoBinaryForGpu),
            cudaError_enum::CUDA_ERROR_ALREADY_ACQUIRED => Err(CudaError::AlreadyAcquired),
            cudaError_enum::CUDA_ERROR_NOT_MAPPED => Err(CudaError::NotMapped),
            cudaError_enum::CUDA_ERROR_NOT_MAPPED_AS_ARRAY => Err(CudaError::NotMappedAsArray),
            cudaError_enum::CUDA_ERROR_NOT_MAPPED_AS_POINTER => Err(CudaError::NotMappedAsPointer),
            cudaError_enum::CUDA_ERROR_ECC_UNCORRECTABLE => Err(CudaError::EccUncorrectable),
            cudaError_enum::CUDA_ERROR_UNSUPPORTED_LIMIT => Err(CudaError::UnsupportedLimit),
            cudaError_enum::CUDA_ERROR_CONTEXT_ALREADY_IN_USE => {
                Err(CudaError::ContextAlreadyInUse)
            }
            cudaError_enum::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => {
                Err(CudaError::PeerAccessUnsupported)
            }
            cudaError_enum::CUDA_ERROR_INVALID_PTX => Err(CudaError::InvalidPtx),
            cudaError_enum::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => {
                Err(CudaError::InvalidGraphicsContext)
            }
            cudaError_enum::CUDA_ERROR_NVLINK_UNCORRECTABLE => Err(CudaError::NvlinkUncorrectable),
            cudaError_enum::CUDA_ERROR_INVALID_SOURCE => Err(CudaError::InvalidSouce),
            cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND => Err(CudaError::FileNotFound),
            cudaError_enum::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => {
                Err(CudaError::SharedObjectSymbolNotFound)
            }
            cudaError_enum::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => {
                Err(CudaError::SharedObjectInitFailed)
            }
            cudaError_enum::CUDA_ERROR_OPERATING_SYSTEM => Err(CudaError::OperatingSystemError),
            cudaError_enum::CUDA_ERROR_INVALID_HANDLE => Err(CudaError::InvalidHandle),
            cudaError_enum::CUDA_ERROR_NOT_FOUND => Err(CudaError::NotFound),
            cudaError_enum::CUDA_ERROR_NOT_READY => Err(CudaError::NotReady),
            cudaError_enum::CUDA_ERROR_ILLEGAL_ADDRESS => Err(CudaError::IllegalAddress),
            cudaError_enum::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => {
                Err(CudaError::LaunchOutOfResources)
            }
            cudaError_enum::CUDA_ERROR_LAUNCH_TIMEOUT => Err(CudaError::LaunchTimeout),
            cudaError_enum::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => {
                Err(CudaError::LaunchIncompatibleTexturing)
            }
            cudaError_enum::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => {
                Err(CudaError::PeerAccessAlreadyEnabled)
            }
            cudaError_enum::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => {
                Err(CudaError::PeerAccessNotEnabled)
            }
            cudaError_enum::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => {
                Err(CudaError::PrimaryContextActive)
            }
            cudaError_enum::CUDA_ERROR_CONTEXT_IS_DESTROYED => Err(CudaError::ContextIsDestroyed),
            cudaError_enum::CUDA_ERROR_ASSERT => Err(CudaError::AssertError),
            cudaError_enum::CUDA_ERROR_TOO_MANY_PEERS => Err(CudaError::TooManyPeers),
            cudaError_enum::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => {
                Err(CudaError::HostMemoryAlreadyRegistered)
            }
            cudaError_enum::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => {
                Err(CudaError::HostMemoryNotRegistered)
            }
            cudaError_enum::CUDA_ERROR_HARDWARE_STACK_ERROR => Err(CudaError::HardwareStackError),
            cudaError_enum::CUDA_ERROR_ILLEGAL_INSTRUCTION => Err(CudaError::IllegalInstruction),
            cudaError_enum::CUDA_ERROR_MISALIGNED_ADDRESS => Err(CudaError::MisalignedAddress),
            cudaError_enum::CUDA_ERROR_INVALID_ADDRESS_SPACE => Err(CudaError::InvalidAddressSpace),
            cudaError_enum::CUDA_ERROR_INVALID_PC => Err(CudaError::InvalidProgramCounter),
            cudaError_enum::CUDA_ERROR_LAUNCH_FAILED => Err(CudaError::LaunchFailed),
            cudaError_enum::CUDA_ERROR_NOT_PERMITTED => Err(CudaError::NotPermitted),
            cudaError_enum::CUDA_ERROR_NOT_SUPPORTED => Err(CudaError::NotSupported),
            _ => Err(CudaError::UnknownError),
        }
    }
}
