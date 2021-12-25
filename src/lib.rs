#![feature(
    generic_associated_types,
    allocator_api,
    alloc_layout_extra,
    nonnull_slice_from_raw_parts,
    slice_ptr_len,
    ptr_metadata,
)]

pub mod device;
pub mod vec;
pub mod ptr;
pub mod blas;

/// Low level matrix
pub struct Matrix<T, D: device::Device> {
    rows: usize,
    cols: usize,

    data: vec::Vec<T, D>,
}
