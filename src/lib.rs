#![feature(
    generic_associated_types,
    allocator_api,
    alloc_layout_extra,
    nonnull_slice_from_raw_parts,
    slice_ptr_len,
    ptr_metadata,
    maybe_uninit_slice
)]


pub mod blas;
pub mod device;
pub mod ptr;
pub mod vec;
pub mod matrix;
pub mod tensor;
