use tensorgraph_sys::device::cpu::Cpu;

use super::{BLASContext, DefaultBLASContext};

#[cfg(feature = "blas-sys")]
#[allow(clippy::module_inception)]
mod blas_sys;

#[cfg(feature = "matrixmultiply")]
mod matmul;

#[derive(Clone, Copy, Default)]
pub struct CpuContext;

impl BLASContext for CpuContext {
    type Device = Cpu;
}

impl DefaultBLASContext for Cpu {
    type Context = CpuContext;
}
