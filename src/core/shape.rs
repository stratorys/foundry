use std::array;

use tracing::error;

use crate::core::{
    CoreError,
    DTYPE_SIZE_BYTES_MAX,
    DType,
};

pub const RANK_MAX: usize = 4;

const ELEMENT_COUNT_MAX: usize = 0x7FFF_FFFF;

const _: () = assert!(
    ELEMENT_COUNT_MAX
        .checked_mul(DTYPE_SIZE_BYTES_MAX)
        .is_some(),
    "the byte count of every shape fits in usize"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [usize; RANK_MAX],
    rank: usize,
    element_count: usize,
}

impl Shape {
    pub fn dims(&self) -> &[usize] { self.dims.get(..self.rank).unwrap_or_default() }

    pub fn rank(&self) -> usize { self.rank }

    pub fn element_count(&self) -> usize { self.element_count }

    pub fn byte_len(
        &self,
        dtype: DType,
    ) -> usize {
        self.element_count.saturating_mul(dtype.size_bytes())
    }

    pub fn dim(
        &self,
        axis: usize,
    ) -> Result<usize, CoreError> {
        self.dims().get(axis).copied().ok_or_else(|| {
            error!(message = "Axis is out of range.", axis, rank = self.rank);
            CoreError::AxisOutOfRange
        })
    }

    pub fn last_axis(&self) -> Result<usize, CoreError> {
        self.rank.checked_sub(1).ok_or_else(|| {
            error!(message = "A scalar has no last axis.");
            CoreError::AxisOutOfRange
        })
    }

    pub fn broadcast(
        lhs: &Shape,
        rhs: &Shape,
    ) -> Result<Shape, CoreError> {
        let incompatible = || {
            error!(
                message = "Shapes cannot be broadcast together.",
                lhs = ?lhs.dims(),
                rhs = ?rhs.dims(),
            );
            CoreError::BroadcastIncompatible
        };
        let dims_reversed = (0..lhs.rank.max(rhs.rank))
            .map(|axis_from_end| {
                match (
                    dim_from_end(lhs, axis_from_end),
                    dim_from_end(rhs, axis_from_end),
                ) {
                    (lhs_dim, rhs_dim) if lhs_dim == rhs_dim => Ok(lhs_dim),
                    (1, rhs_dim) => Ok(rhs_dim),
                    (lhs_dim, 1) => Ok(lhs_dim),
                    _ => Err(incompatible()),
                }
            })
            .collect::<Result<Vec<usize>, CoreError>>()?;
        let dims: Vec<usize> = dims_reversed.into_iter().rev().collect();
        Shape::try_from(dims.as_slice())
    }
}

impl TryFrom<&[usize]> for Shape {
    type Error = CoreError;

    fn try_from(dims: &[usize]) -> Result<Self, Self::Error> {
        let rank = dims.len();
        if rank > RANK_MAX {
            error!(
                message = "Rank exceeds the maximum rank.",
                rank,
                rank_max = RANK_MAX
            );
            return Err(CoreError::RankTooLarge);
        }
        let element_count_non_zero = dims
            .iter()
            .filter(|&&dim| dim != 0)
            .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
            .filter(|&count| count <= ELEMENT_COUNT_MAX)
            .ok_or_else(|| {
                error!(
                    message = "Shape has too many elements.",
                    ?dims,
                    element_count_max = ELEMENT_COUNT_MAX,
                );
                CoreError::ElementCountOverflow
            })?;
        let element_count = if dims.contains(&0) {
            0
        } else {
            element_count_non_zero
        };
        Ok(Self {
            dims: array::from_fn(|axis| dims.get(axis).copied().unwrap_or(1)),
            rank,
            element_count,
        })
    }
}

fn dim_from_end(
    shape: &Shape,
    axis_from_end: usize,
) -> usize {
    shape
        .dims()
        .iter()
        .rev()
        .nth(axis_from_end)
        .copied()
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use crate::core::{
        CoreError,
        Shape,
    };

    #[test]
    fn broadcast_aligns_axes_from_the_right() {
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
            Err(CoreError::BroadcastIncompatible),
            "dims 2 and 4 do not broadcast"
        );
    }

    #[test]
    fn broadcast_prepends_missing_axes_on_the_left() {
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
            Err(CoreError::ElementCountOverflow),
            "two valid shapes can broadcast to an invalid one"
        );
    }

    #[test]
    fn rank_above_max_is_rejected() {
        assert_eq!(
            Shape::try_from([1, 2, 3, 4, 5].as_slice()),
            Err(CoreError::RankTooLarge),
            "rank 5 is rejected"
        );
    }

    #[test]
    fn rank_at_max_is_accepted() {
        assert_eq!(
            Shape::try_from([1, 2, 3, 4].as_slice()).map(|shape| shape.rank()),
            Ok(4),
            "rank 4 is accepted"
        );
    }

    #[test]
    fn scalar_has_one_element() {
        assert_eq!(
            Shape::try_from([].as_slice()).map(|shape| shape.element_count()),
            Ok(1),
            "a rank 0 shape holds one element"
        );
    }

    #[test]
    fn element_count_overflow_is_rejected() {
        assert_eq!(
            Shape::try_from([usize::MAX, 2].as_slice()),
            Err(CoreError::ElementCountOverflow),
            "the product overflows usize"
        );
    }

    #[test]
    fn element_count_above_i32_max_is_rejected() {
        assert_eq!(
            Shape::try_from([1 << 31].as_slice()),
            Err(CoreError::ElementCountOverflow),
            "one element above i32::MAX"
        );
    }

    #[test]
    fn element_count_at_i32_max_is_accepted() {
        assert_eq!(
            Shape::try_from([(1 << 31) - 1].as_slice())
                .expect("valid shape")
                .element_count(),
            (1 << 31) - 1,
            "element count"
        );
    }

    #[test]
    fn non_zero_product_above_i32_max_is_rejected_in_any_axis_order() {
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
                Err(CoreError::ElementCountOverflow),
                "dims {dims:?}"
            );
        });
    }

    #[test]
    fn empty_shape_within_bound_is_accepted_in_any_axis_order() {
        [[(1 << 30) - 1, 0, 2], [0, (1 << 30) - 1, 2]]
            .into_iter()
            .for_each(|dims| {
                assert_eq!(
                    Shape::try_from(dims.as_slice())
                        .expect("valid shape")
                        .element_count(),
                    0,
                    "dims {dims:?}"
                );
            });
    }
}
