use crate::device::Device;

mod cpu;

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum MatrixOp {
    NoTrans = b'N',
    Trans = b'T',
    ConjTrans = b'C',
}

pub trait BLASDevice: Device {
    type Context: Clone;
}

pub trait BLAS<D: BLASDevice>: Sized + Copy {
    #[allow(clippy::too_many_arguments)]
    /// # Safety
    /// This is often a call across an FFI barrier, so the links or devices need to be
    /// running and may perform UB unchecked by rust
    unsafe fn gemm(
        ctx: D::Context,
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
