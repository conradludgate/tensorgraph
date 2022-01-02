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

pub mod dims;
pub mod storage;
pub mod tensor;
