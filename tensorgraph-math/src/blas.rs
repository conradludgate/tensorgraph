use tensorgraph_sys::device::Device;

pub mod cpu;

#[cfg(feature = "cublas")]
pub mod cublas;

#[repr(u8)]
#[derive(Clone, Copy, PartialEq)]
pub enum MatrixOp {
    NoTrans = b'N',
    Trans = b'T',
    // ConjTrans = b'C',
}

/// A context needed for running BLAS operations
pub trait BLASContext<D: Device + ?Sized>: Clone {}

/// The default blas context for a device
pub trait DefaultBLASContext: Device {
    type Context: BLASContext<Self> + Default;
}

/// A type that can be matrix multiplied
pub trait GEMM<C: BLASContext<D>, D: Device>: Sized + Copy {
    #[allow(clippy::too_many_arguments)]
    /// # Safety
    /// This is often a call across an FFI barrier, so the links or devices need to be
    /// running and may perform UB unchecked by rust
    unsafe fn gemm(
        ctx: C,
        transa: MatrixOp,
        transb: MatrixOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: Self,
        a: D::Ptr<Self>,
        lda: i32,
        b: D::Ptr<Self>,
        ldb: i32,
        beta: Self,
        c: D::Ptr<Self>,
        ldc: i32,
    );
}
