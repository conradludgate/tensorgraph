//! # tensorgraph-math
//! Mathematics primitives used by tensorgraph.
//! Builds upon [tensorgraph-sys](https://docs.rs/tensorgraph-sys/latest/tensorgraph_sys/)
//! to support many BLAS backends and devices.
//!
//! ## Basic example using openblas:
//!
//! Enable features in the Cargo.toml:
//! ```toml
//! tensorgraph-math = { version = "LATEST_VERSION", features = ["openblas"] }
//! ```
//!
//! ```
//! use tensorgraph_math::{tensor::Tensor, sys::View};
//!
//! //     0 1
//! // A = 2 3
//! //     4 5
//!
//! // B = 0 1
//! //     2 3
//!
//! // column major (read each column first)
//! let a = [0., 2., 4., 1., 3., 5.];
//! let b = [0., 2., 1., 3.];
//!
//! let a = Tensor::from_shape([3, 2], a); // 3 rows x 2 cols
//! let b = Tensor::from_shape([2, 2], b); // 2 rows x 2 cols
//!
//! //           2  3
//! // C = AB =  6 11
//! //          10 19
//!
//! let c = a.dot(b.view());
//! assert_eq!(c.into_inner().into_std(), [2., 6., 10., 3., 11., 19.]);
//! ```
//!
//! ## Intermediate example using cublas globals and openblas together:
//!
//! Enable features in the Cargo.toml:
//! ```toml
//! tensorgraph-math = { version = "LATEST_VERSION", features = ["openblas", "cublas"] }
//! ```
//!
//! ```
//! use tensorgraph_math::{
//!     blas::{DefaultBLASContext, cublas::CublasContext, GEMM},
//!     sys::{
//!         device::{DefaultDeviceAllocator, cuda::{Context, Cuda, Stream}, cpu::Cpu},
//!         DefaultVec, View,
//!     },
//!     tensor::Tensor,
//! };
//!
//! fn main() {
//!     // init cuda context
//!     let cuda_ctx = Context::quick_init().unwrap();
//!
//!     // create cuda stream and configure it as the global
//!     let stream = Stream::new(&cuda_ctx).unwrap();
//!     let _handle = stream.as_global();
//!
//!     // create cublas context, with the provided stream, and configure it as the global
//!     let cublas_ctx = CublasContext::new();
//!     let _handle = cublas_ctx.with_stream(Some(&stream)).as_global();
//!
//!     // cublas is the default BLAS implementation for CUDA when the feature is enabled
//!     run::<Cuda>();
//!
//!     // openblas is the default BLAS implemenetation for CPU when the feature is enabled
//!     run::<Cpu>();
//! }
//!
//! /// Generic code that runs on the specified device
//! /// using that devices default allocator and BLAS provider
//! fn run<D: DefaultDeviceAllocator + DefaultBLASContext>()
//! where
//!     f32: GEMM<D::Context>,
//! {
//!     //     0 1
//!     // A = 2 3
//!     //     4 5
//!
//!     // B = 0 1
//!     //     2 3
//!
//!     // column major (read each column first)
//!     let a = DefaultVec::<f32, D>::copy_from_host(&[0., 2., 4., 1., 3., 5.]);
//!     let b = DefaultVec::<f32, D>::copy_from_host(&[0., 2., 1., 3.]);
//!
//!     let a = Tensor::from_shape([3, 2], a); // 3 rows x 2 cols
//!     let b = Tensor::from_shape([2, 2], b); // 2 rows x 2 cols
//!
//!     //           2  3
//!     // C = AB =  6 11
//!     //          10 19
//!
//!     let c = a.dot(b.view());
//!
//!     let mut out = [0.; 6];
//!     c.into_inner().copy_to_host(&mut out);
//!     assert_eq!(out, [2., 6., 10., 3., 11., 19.]);
//! }
//! ```
//!
//! ## Advanced example using openblas and cublas by passing blas contexts and allocators:
//!
//! Enable features in the Cargo.toml:
//! ```toml
//! tensorgraph-math = { version = "LATEST_VERSION", features = ["openblas", "cublas"] }
//! ```
//!
//! ```
//! #![feature(allocator_api)]
//! use std::{alloc::Global, ops::Deref};
//! use tensorgraph_math::{
//!     blas::{BLASContext, cublas::{CublasContext}, GEMM},
//!     sys::{
//!         device::{cuda::{Context, Cuda, Stream}, cpu::Cpu, Device, DeviceAllocator},
//!         Vec, View,
//!     },
//!     tensor::Tensor,
//! };
//!
//! fn main() {
//!     // init cuda context
//!     let cuda_ctx = Context::quick_init().unwrap();
//!
//!     // create cuda stream
//!     let stream = Stream::new(&cuda_ctx).unwrap();
//!
//!     // create cublas context, with the provided stream
//!     let cublas_ctx = CublasContext::new();
//!     let cublas_ctx = cublas_ctx.with_stream(Some(&stream));
//!
//!     // run using the CUDA stream as the allocator, and cublas
//!     // as the BLAS provider
//!     run(cublas_ctx, stream.deref());
//!
//!     // run using the CPU default BLAS and Global allocator
//!     run((), Global);
//! }
//!
//! fn run<C: BLASContext, A: DeviceAllocator<Device = C::Device> + Copy>(ctx: C, alloc: A)
//! where
//!     f32: GEMM<C>,
//! {
//!     //     0 1
//!     // A = 2 3
//!     //     4 5
//!
//!     // B = 0 1
//!     //     2 3
//!
//!     // column major (read each column first)
//!     let a = Vec::copy_from_host_in(&[0., 2., 4., 1., 3., 5.], alloc);
//!     let b = Vec::copy_from_host_in(&[0., 2., 1., 3.0_f32], alloc);
//!
//!     let a = Tensor::from_shape([3, 2], a); // 3 rows x 2 cols
//!     let b = Tensor::from_shape([2, 2], b); // 2 rows x 2 cols
//!
//!     //           2  3
//!     // C = AB =  6 11
//!     //          10 19
//!
//!     let c = a.dot_into(b.view(), ctx, alloc);
//!
//!     let mut out = [0.; 6];
//!     c.into_inner().copy_to_host(&mut out);
//!     assert_eq!(out, [2., 6., 10., 3., 11., 19.]);
//! }
//! ```

#![warn(clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::module_name_repetitions,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::unreadable_literal
)]
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
    layout_for_ptr
)]

pub use tensorgraph_sys as sys;

/// Traits and implementations of BLAS providers
pub mod blas;

/// Traits and implementations for basic dimension types
pub mod dims;

/// Traits and implementations for basic storage buffers
pub mod storage;

/// Implementations for tensor operations and structures
pub mod tensor;
