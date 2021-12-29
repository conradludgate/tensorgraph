#![allow(incomplete_features)]
#![feature(
    generic_associated_types,
    allocator_api,
    alloc_layout_extra,
    nonnull_slice_from_raw_parts,
    slice_ptr_len,
    ptr_metadata,
    maybe_uninit_slice,
    generic_const_exprs
)]

pub mod blas;
pub mod device;
pub mod ptr;
pub mod tensor;
pub mod vec;
