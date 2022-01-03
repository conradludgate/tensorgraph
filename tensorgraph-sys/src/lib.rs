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

pub mod device;
pub mod ptr;
pub mod vec;
pub mod zero;
pub mod boxed;

/// Represents a type that can be shared.
/// Mimics the impl for [`std::ops::Deref`] but makes use of `GAT`'s in order
/// to provide non `&` refs. Useful for things like tensor views.
pub trait Share {
    type Ref<'a>
    where
        Self: 'a;

    fn share(&self) -> Self::Ref<'_>;
}

pub trait ShareMut {
    type Mut<'a>
    where
        Self: 'a;

    fn share_mut(&mut self) -> Self::Mut<'_>;
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

impl<D: std::ops::DerefMut> ShareMut for D {
    type Mut<'a>
    where
        Self: 'a,
    = &'a mut D::Target;

    fn share_mut(&mut self) -> Self::Mut<'_> {
        self
    }
}
