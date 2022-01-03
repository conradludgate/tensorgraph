#[allow(clippy::len_without_is_empty)]
/// A dimension type for a tensor
pub trait Dimension: AsRef<[usize]> + AsMut<[usize]> + Clone {
    fn len(&self) -> usize {
        self.as_ref().iter().product()
    }

    #[must_use]
    fn column_major_strides(&self) -> Self {
        let mut strides = self.clone();
        let s = strides.as_mut();
        s[0] = 1;

        for i in 1..s.len() {
            s[i] = s[i - 1] * self.as_ref()[i - 1];
        }

        strides
    }
}

impl<const N: usize> Dimension for [usize; N] {}
impl Dimension for std::vec::Vec<usize> {}

/// Reduce an axis from a dimension
pub trait RemoveDim: Dimension {
    type Smaller: Dimension;
    fn remove(&self, axis: usize) -> (Self::Smaller, usize);
}

impl<const N: usize> RemoveDim for [usize; N]
where
    [(); N - 1]: Sized,
{
    type Smaller = [usize; N - 1];
    fn remove(&self, axis: usize) -> (Self::Smaller, usize) {
        assert!(axis < N);
        if N == 1 {
            return ([0_usize; N - 1], self[0]);
        }

        let mut new = [0; N - 1];
        let (lhs, rhs) = self.split_at(axis);
        let (n, rhs) = rhs.split_first().unwrap();
        new[..axis].copy_from_slice(lhs);
        new[axis..].copy_from_slice(rhs);
        (new, *n)
    }
}

impl RemoveDim for std::vec::Vec<usize> {
    type Smaller = Self;
    fn remove(&self, axis: usize) -> (Self::Smaller, usize) {
        let mut new = self.clone();
        let n = std::vec::Vec::remove(&mut new, axis);
        (new, n)
    }
}


/// Insert an axis into a dimension
pub trait InsertDim: Dimension {
    type Larger: Dimension;
    fn insert(&self, axis: usize, n: usize) -> Self::Larger;
}

impl<const N: usize> InsertDim for [usize; N]
where
    [(); N + 1]: Sized,
{
    type Larger = [usize; N + 1];
    fn insert(&self, axis: usize, n: usize) -> Self::Larger {
        assert!(axis <= N);

        let mut new = [0; N + 1];

        let (lhs, rhs) = self.split_at(axis);
        new[..axis].copy_from_slice(lhs);
        new[axis + 1..].copy_from_slice(rhs);
        new[axis] = n;

        new
    }
}

impl InsertDim for Vec<usize> {
    type Larger = Self;
    fn insert(&self, axis: usize, n: usize) -> Self::Larger {
        let mut new = self.clone();
        Vec::insert(&mut new, axis, n);
        new
    }
}
