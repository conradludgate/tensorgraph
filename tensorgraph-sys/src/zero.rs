/// This trait signifies that `[0; size_of::<T>]` is transmutable
/// to a valid `T` safely, and that `T` fits some concept of 'zero'.
///
/// # Safety
/// The set of zero bytes the size of Self should be a valid value
pub unsafe trait Zero: Copy {}

// at the moment, we only support floats in our tensor operations,
// so let's only define those to minimize our code
unsafe impl Zero for f32 {}
unsafe impl Zero for f64 {}
