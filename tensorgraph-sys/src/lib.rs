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
#![warn(clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::module_name_repetitions,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names
)]

/// Provides implementation of a Device [`Box`]
pub mod boxed;
/// Provides trait defintion and implementations of a [`Device`]
pub mod device;

/// Provides standard pointer types
pub mod ptr;

mod vec;
mod zero;

pub use vec::{DefaultVec, Vec};
pub use zero::Zero;

/// Represents a type that can be 'viewed' (derefed).
/// Mimics the impl for [`std::ops::Deref`] but makes use of `GAT`'s in order
/// to provide non `&` refs. Useful for things like tensor views.
pub trait View {
    type Ref<'a>
    where
        Self: 'a;

    fn view(&self) -> Self::Ref<'_>;
}

/// Represents a type that can be mutably 'viewed' (derefed).
/// Mimics the impl for [`std::ops::DerefMut`] but makes use of `GAT`'s in order
/// to provide non `&mut` refs. Useful for things like tensor views.
pub trait ViewMut {
    type Mut<'a>
    where
        Self: 'a;

    fn view_mut(&mut self) -> Self::Mut<'_>;
}

impl<D: std::ops::Deref> View for D {
    type Ref<'a>
    where
        Self: 'a,
    = &'a D::Target;

    fn view(&self) -> Self::Ref<'_> {
        self
    }
}

impl<D: std::ops::DerefMut> ViewMut for D {
    type Mut<'a>
    where
        Self: 'a,
    = &'a mut D::Target;

    fn view_mut(&mut self) -> Self::Mut<'_> {
        self
    }
}
