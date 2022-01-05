use tensorgraph_sys::{device::Device, ptr::DPtr};

mod cpu;

#[cfg(feature = "cublas")]
pub mod cublas;

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Debug)]
/// Represents how a matrix can be represented internally.
pub enum MatrixOp {
    NoTrans = b'N',
    Trans = b'T',
    // ConjTrans = b'C',
}

/// A context needed for running BLAS operations
pub trait BLASContext: Clone {
    type Device: Device;
}

/// The default blas context for a device
pub trait DefaultBLASContext: Device {
    type Context: BLASContext<Device = Self> + Default;
}

/// BLAS Level 1 operations (Vector only)
#[allow(clippy::too_many_arguments)]
pub trait BLAS1<C: BLASContext>: Sized + Copy {
    /// Computes
    /// > Y = alpha * X + Y
    ///
    /// # Safety
    /// This is often a call across an FFI barrier, so the links or devices need to be
    /// running and may perform UB unchecked by rust
    unsafe fn axpy(
        ctx: C,
        n: i32,
        alpha: Self,
        x: DPtr<Self, C::Device>,
        incx: i32,
        y: DPtr<Self, C::Device>,
        incy: i32,
    );

    /// Computes the vector dot product
    ///
    /// # Safety
    /// This is often a call across an FFI barrier, so the links or devices need to be
    /// running and may perform UB unchecked by rust
    unsafe fn dot(
        ctx: C,
        n: i32,
        x: DPtr<Self, C::Device>,
        incx: i32,
        y: DPtr<Self, C::Device>,
        incy: i32,
    ) -> Self;
}

/// BLAS Level 2 operations (Matrix-Vector)
#[allow(clippy::too_many_arguments)]
pub trait BLAS2<C: BLASContext>: Sized + Copy {
}

/// BLAS Level 3 operations (Matrix-Matrix)
#[allow(clippy::too_many_arguments)]
pub trait BLAS3<C: BLASContext>: Sized + Copy {
    /// Compute the **Ge**neralised **M**atrix **M**ultiplication:
    /// > C = alpha * AB + beta * C
    ///
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
        a: DPtr<Self, C::Device>,
        lda: i32,
        b: DPtr<Self, C::Device>,
        ldb: i32,
        beta: Self,
        c: DPtr<Self, C::Device>,
        ldc: i32,
    );
}

/// A complete BLAS library, levels 1, 2 and 3
pub trait BLAS<C: BLASContext>: BLAS1<C> + BLAS2<C> + BLAS3<C> {}
impl<F, C: BLASContext> BLAS<C> for F where F: BLAS1<C> + BLAS2<C> + BLAS3<C> {}
