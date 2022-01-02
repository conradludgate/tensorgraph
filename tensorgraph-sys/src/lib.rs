#![allow(incomplete_features)]
#![feature(
    generic_associated_types,
    allocator_api,
    alloc_layout_extra,
    nonnull_slice_from_raw_parts,
    slice_ptr_len,
    ptr_metadata,
    maybe_uninit_slice,
    generic_const_exprs,
    thread_local,
    once_cell,
    layout_for_ptr,
)]

pub mod blas;
pub mod device;
pub mod dims;
pub mod ptr;
pub mod storage;
pub mod tensor;
pub mod vec;
pub mod zero;
pub mod boxed;

pub trait Share {
    type Ref<'a>
    where
        Self: 'a;

    fn share(&self) -> Self::Ref<'_>;
}

impl<D: std::ops::Deref> Share for D {
    type Ref<'a>
    where
        Self: 'a,
    = &'a D::Target;

    fn share(&self) -> Self::Ref<'_> {
        self
    }
}
