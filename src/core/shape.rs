use std::array;

use crate::core::CoreError;

pub const RANK_MAX: usize = 4;

const ELEMENT_COUNT_MAX: usize = 0x7FFF_FFFF;

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

    pub fn broadcast(
        lhs: &Shape,
        rhs: &Shape,
    ) -> Result<Shape, CoreError> {
        let incompatible = || CoreError::BroadcastIncompatible {
            lhs: lhs.dims().to_vec(),
            rhs: rhs.dims().to_vec(),
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
            return Err(CoreError::RankTooLarge {
                rank,
                rank_max: RANK_MAX,
            });
        }
        let element_count_non_zero = dims
            .iter()
            .filter(|&&dim| dim != 0)
            .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
            .filter(|&count| count <= ELEMENT_COUNT_MAX)
            .ok_or_else(|| CoreError::ElementCountOverflow {
                dims: dims.to_vec(),
                element_count_max: ELEMENT_COUNT_MAX,
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

    fn shape(dims: &[usize]) -> Shape { Shape::try_from(dims).expect("valid shape") }

    #[test]
    fn broadcast_aligns_axes_from_the_right() {
        let result = Shape::broadcast(&shape(&[2, 1, 4]), &shape(&[3, 1])).expect("broadcastable");
        assert_eq!(result.dims(), &[2, 3, 4], "broadcast dims");
    }

    #[test]
    fn broadcast_rejects_mismatched_dims() {
        let result = Shape::broadcast(&shape(&[2, 3]), &shape(&[4, 3]));
        assert!(
            matches!(result, Err(CoreError::BroadcastIncompatible { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn rank_above_max_is_rejected() {
        let result = Shape::try_from([1, 2, 3, 4, 5].as_slice());
        assert!(
            matches!(
                result,
                Err(CoreError::RankTooLarge {
                    rank: 5,
                    ..
                })
            ),
            "got {result:?}"
        );
    }

    #[test]
    fn element_count_overflow_is_rejected() {
        let result = Shape::try_from([usize::MAX, 2].as_slice());
        assert!(
            matches!(result, Err(CoreError::ElementCountOverflow { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn element_count_above_i32_max_is_rejected() {
        let result = Shape::try_from([1 << 31].as_slice());
        assert!(
            matches!(result, Err(CoreError::ElementCountOverflow { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn element_count_at_i32_max_is_accepted() {
        assert_eq!(
            shape(&[(1 << 31) - 1]).element_count(),
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
            let result = Shape::try_from(dims.as_slice());
            assert!(
                matches!(result, Err(CoreError::ElementCountOverflow { .. })),
                "dims {dims:?} got {result:?}"
            );
        });
    }

    #[test]
    fn empty_shape_within_bound_is_accepted_in_any_axis_order() {
        [[(1 << 30) - 1, 0, 2], [0, (1 << 30) - 1, 2]]
            .into_iter()
            .for_each(|dims| {
                assert_eq!(shape(&dims).element_count(), 0, "dims {dims:?}");
            });
    }
}
