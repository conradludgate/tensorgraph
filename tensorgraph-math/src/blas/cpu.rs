use tensorgraph_sys::device::cpu::Cpu;

use super::{BLASContext, DefaultBLASContext};

#[cfg(feature = "blas-sys")]
#[allow(clippy::module_inception)]
mod blas_sys;

impl BLASContext for () {
    type Device = Cpu;
}

impl DefaultBLASContext for Cpu {
    type Context = ();
}
