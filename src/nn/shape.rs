use std::array;

use tracing::error;

use crate::nn::{
    DType,
    MAX_ITEMSIZE,
    TensorError,
};

pub const MAX_RANK: usize = 4;

const MAX_NUMEL: usize = 0x7FFF_FFFF;

const _: () = assert!(
    MAX_NUMEL.checked_mul(MAX_ITEMSIZE).is_some(),
    "the byte count of every shape fits in usize"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [usize; MAX_RANK],
    ndim: usize,
    numel: usize,
}

impl Shape {
    pub fn dims(&self) -> &[usize] { self.dims.get(..self.ndim).unwrap_or_default() }

    pub fn ndim(&self) -> usize { self.ndim }

    pub fn numel(&self) -> usize { self.numel }

    pub fn nbytes(
        &self,
        dtype: DType,
    ) -> usize {
        self.numel.saturating_mul(dtype.itemsize())
    }

    pub fn size(
        &self,
        dim: usize,
    ) -> Result<usize, TensorError> {
        self.dims().get(dim).copied().ok_or_else(|| {
            error!(
                message = "Dimension is out of range.",
                dim,
                ndim = self.ndim
            );
            TensorError::DimOutOfRange
        })
    }

    pub fn last_dim(&self) -> Result<usize, TensorError> {
        self.ndim.checked_sub(1).ok_or_else(|| {
            error!(message = "A scalar has no last dimension.");
            TensorError::DimOutOfRange
        })
    }

    pub fn broadcast(
        lhs: &Shape,
        rhs: &Shape,
    ) -> Result<Shape, TensorError> {
        let incompatible = || {
            error!(
                message = "Shapes cannot be broadcast together.",
                lhs = ?lhs.dims(),
                rhs = ?rhs.dims(),
            );
            TensorError::BroadcastIncompatible
        };
        let ndim = lhs.ndim.max(rhs.ndim);
        let mut dims = [1; MAX_RANK];
        dims.get_mut(..ndim)
            .unwrap_or_default()
            .iter_mut()
            .rev()
            .enumerate()
            .try_for_each(|(dim_from_end, slot)| {
                *slot = match (
                    size_from_end(lhs, dim_from_end),
                    size_from_end(rhs, dim_from_end),
                ) {
                    (lhs_size, rhs_size) if lhs_size == rhs_size => lhs_size,
                    (1, rhs_size) => rhs_size,
                    (lhs_size, 1) => lhs_size,
                    _ => return Err(incompatible()),
                };
                Ok(())
            })?;
        Shape::try_from(dims.get(..ndim).unwrap_or_default())
    }

    pub fn with_dim(
        &self,
        dim: usize,
        size: usize,
    ) -> Result<Shape, TensorError> {
        self.size(dim)?;
        let mut dims = self.dims;
        if let Some(slot) = dims.get_mut(dim) {
            *slot = size;
        }
        Shape::try_from(dims.get(..self.ndim).unwrap_or_default())
    }
}

impl TryFrom<&[usize]> for Shape {
    type Error = TensorError;

    fn try_from(dims: &[usize]) -> Result<Self, Self::Error> {
        let ndim = dims.len();
        if ndim > MAX_RANK {
            error!(
                message = "Rank exceeds the maximum rank.",
                ndim,
                max_rank = MAX_RANK
            );
            return Err(TensorError::RankTooLarge);
        }
        let numel_non_zero = dims
            .iter()
            .filter(|&&dim| dim != 0)
            .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
            .filter(|&count| count <= MAX_NUMEL)
            .ok_or_else(|| {
                error!(
                    message = "Shape has too many elements.",
                    ?dims,
                    max_numel = MAX_NUMEL,
                );
                TensorError::NumelOverflow
            })?;
        let numel = if dims.contains(&0) { 0 } else { numel_non_zero };
        Ok(Self {
            dims: array::from_fn(|dim| dims.get(dim).copied().unwrap_or(1)),
            ndim,
            numel,
        })
    }
}

fn size_from_end(
    shape: &Shape,
    dim_from_end: usize,
) -> usize {
    shape
        .dims()
        .iter()
        .rev()
        .nth(dim_from_end)
        .copied()
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use crate::nn::{
        Shape,
        TensorError,
    };

    #[test]
    fn broadcast_aligns_dims_from_the_right() {
        let result = Shape::broadcast(
            &Shape::try_from([2, 1, 4].as_slice()).expect("valid shape"),
            &Shape::try_from([3, 1].as_slice()).expect("valid shape"),
        )
        .expect("broadcastable");
        assert_eq!(result.dims(), &[2, 3, 4], "broadcast dims");
    }

    #[test]
    fn broadcast_rejects_mismatched_dims() {
        let result = Shape::broadcast(
            &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
            &Shape::try_from([4, 3].as_slice()).expect("valid shape"),
        );
        assert_eq!(
            result,
            Err(TensorError::BroadcastIncompatible),
            "dims 2 and 4 do not broadcast"
        );
    }

    #[test]
    fn broadcast_prepends_missing_dims_on_the_left() {
        let short = Shape::try_from([3].as_slice()).expect("valid shape");
        let long = Shape::try_from([2, 3].as_slice()).expect("valid shape");
        assert_eq!(
            Shape::broadcast(&short, &long),
            Ok(long),
            "the shorter shape on the left"
        );
        assert_eq!(
            Shape::broadcast(&long, &short),
            Ok(long),
            "the shorter shape on the right"
        );
    }

    #[test]
    fn broadcast_of_a_scalar_takes_the_other_shape() {
        let scalar = Shape::try_from([].as_slice()).expect("valid shape");
        let matrix = Shape::try_from([2, 3].as_slice()).expect("valid shape");
        assert_eq!(
            Shape::broadcast(&scalar, &matrix),
            Ok(matrix),
            "a scalar broadcasts to any shape"
        );
    }

    #[test]
    fn broadcast_beyond_the_element_limit_is_rejected() {
        assert_eq!(
            Shape::broadcast(
                &Shape::try_from([1 << 30, 1].as_slice()).expect("valid shape"),
                &Shape::try_from([1, 4].as_slice()).expect("valid shape"),
            ),
            Err(TensorError::NumelOverflow),
            "two valid shapes can broadcast to an invalid one"
        );
    }

    #[test]
    fn broadcast_of_two_scalars_is_a_scalar() {
        let scalar = Shape::try_from([].as_slice()).expect("valid shape");
        assert_eq!(
            Shape::broadcast(&scalar, &scalar),
            Ok(scalar),
            "two scalars"
        );
    }

    #[test]
    fn broadcast_of_rank_four_and_rank_one_aligns_the_last_dim() {
        assert_eq!(
            Shape::broadcast(
                &Shape::try_from([2, 3, 4, 1].as_slice()).expect("valid shape"),
                &Shape::try_from([5].as_slice()).expect("valid shape"),
            ),
            Shape::try_from([2, 3, 4, 5].as_slice()),
            "the rank one shape stretches the last dim"
        );
    }

    #[test]
    fn with_dim_replaces_one_size() {
        assert_eq!(
            Shape::try_from([2, 3, 4].as_slice())
                .expect("valid shape")
                .with_dim(1, 7),
            Shape::try_from([2, 7, 4].as_slice()),
            "dim 1 becomes 7"
        );
    }

    #[test]
    fn with_dim_to_zero_empties_the_shape() {
        assert_eq!(
            Shape::try_from([2, 3].as_slice())
                .expect("valid shape")
                .with_dim(0, 0)
                .map(|shape| shape.numel()),
            Ok(0),
            "a zero size empties the shape"
        );
    }

    #[test]
    fn with_dim_on_a_scalar_is_rejected() {
        assert_eq!(
            Shape::try_from([].as_slice())
                .expect("valid shape")
                .with_dim(0, 1),
            Err(TensorError::DimOutOfRange),
            "a scalar has no dim"
        );
    }

    #[test]
    fn with_dim_beyond_the_element_limit_is_rejected() {
        assert_eq!(
            Shape::try_from([2, 3].as_slice())
                .expect("valid shape")
                .with_dim(0, 1 << 30),
            Err(TensorError::NumelOverflow),
            "3 * 2^30 elements exceed the limit"
        );
    }

    #[test]
    fn with_dim_reports_the_dim_before_the_element_limit() {
        assert_eq!(
            Shape::try_from([2, 3].as_slice())
                .expect("valid shape")
                .with_dim(2, usize::MAX),
            Err(TensorError::DimOutOfRange),
            "the dim is checked first"
        );
    }

    #[test]
    fn rank_above_max_is_rejected() {
        assert_eq!(
            Shape::try_from([1, 2, 3, 4, 5].as_slice()),
            Err(TensorError::RankTooLarge),
            "rank 5 is rejected"
        );
    }

    #[test]
    fn rank_at_max_is_accepted() {
        assert_eq!(
            Shape::try_from([1, 2, 3, 4].as_slice()).map(|shape| shape.ndim()),
            Ok(4),
            "rank 4 is accepted"
        );
    }

    #[test]
    fn scalar_has_one_element() {
        assert_eq!(
            Shape::try_from([].as_slice()).map(|shape| shape.numel()),
            Ok(1),
            "a rank 0 shape holds one element"
        );
    }

    #[test]
    fn numel_overflow_is_rejected() {
        assert_eq!(
            Shape::try_from([usize::MAX, 2].as_slice()),
            Err(TensorError::NumelOverflow),
            "the product overflows usize"
        );
    }

    #[test]
    fn numel_above_i32_max_is_rejected() {
        assert_eq!(
            Shape::try_from([1 << 31].as_slice()),
            Err(TensorError::NumelOverflow),
            "one element above i32::MAX"
        );
    }

    #[test]
    fn numel_at_i32_max_is_accepted() {
        assert_eq!(
            Shape::try_from([(1 << 31) - 1].as_slice())
                .expect("valid shape")
                .numel(),
            (1 << 31) - 1,
            "element count"
        );
    }

    #[test]
    fn non_zero_product_above_i32_max_is_rejected_in_any_dim_order() {
        [
            [usize::MAX, 2, 0],
            [usize::MAX, 0, 2],
            [0, usize::MAX, 2],
            [1 << 30, 0, 2],
        ]
        .into_iter()
        .for_each(|dims| {
            assert_eq!(
                Shape::try_from(dims.as_slice()),
                Err(TensorError::NumelOverflow),
                "dims {dims:?}"
            );
        });
    }

    #[test]
    fn empty_shape_within_bound_is_accepted_in_any_dim_order() {
        [[(1 << 30) - 1, 0, 2], [0, (1 << 30) - 1, 2]]
            .into_iter()
            .for_each(|dims| {
                assert_eq!(
                    Shape::try_from(dims.as_slice())
                        .expect("valid shape")
                        .numel(),
                    0,
                    "dims {dims:?}"
                );
            });
    }
}
