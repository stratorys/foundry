use std::array;

use crate::core::CoreError;

pub const RANK_MAX: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [usize; RANK_MAX],
    rank: usize,
}

impl Shape {
    pub fn dims(&self) -> &[usize] { self.dims.get(..self.rank).unwrap_or_default() }

    pub fn rank(&self) -> usize { self.rank }

    pub fn element_count(&self) -> usize { self.dims().iter().product() }

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
        dims.iter()
            .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
            .ok_or_else(|| CoreError::ElementCountOverflow {
                dims: dims.to_vec(),
            })?;
        Ok(Self {
            dims: array::from_fn(|axis| dims.get(axis).copied().unwrap_or(1)),
            rank,
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
}
