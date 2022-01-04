mod non_null;
mod reef;

pub use non_null::NonNull;
pub use reef::Ref;

use crate::device::Device;

pub type DPtr<T, D> = <D as Device>::Ptr<T>;
