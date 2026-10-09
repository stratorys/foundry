use std::array;

use crate::core::{
    CoreError,
    RANK_MAX,
    Shape,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Layout {
    shape: Shape,
    strides: [usize; RANK_MAX],
    offset: usize,
}

impl Layout {
    pub fn contiguous(shape: Shape) -> Result<Self, CoreError> {
        Ok(Self {
            shape,
            strides: contiguous_strides(&shape)?,
            offset: 0,
        })
    }

    pub fn shape(&self) -> &Shape { &self.shape }

    pub fn strides(&self) -> &[usize] { self.strides.get(..self.shape.rank()).unwrap_or_default() }

    pub fn offset(&self) -> usize { self.offset }

    pub fn is_contiguous(&self) -> bool {
        contiguous_strides(&self.shape).is_ok_and(|expected| {
            self.shape
                .dims()
                .iter()
                .zip(self.strides().iter().zip(expected.iter()))
                .all(|(&dim, (&stride, &stride_expected))| dim == 1 || stride == stride_expected)
        })
    }

    pub fn reshape(
        &self,
        shape: Shape,
    ) -> Result<Self, CoreError> {
        if !self.is_contiguous() {
            return Err(CoreError::ReshapeNonContiguous);
        }
        let (from, to) = (self.shape.element_count(), shape.element_count());
        if from != to {
            return Err(CoreError::ElementCountMismatch {
                from,
                to,
            });
        }
        Ok(Self {
            offset: self.offset,
            ..Self::contiguous(shape)?
        })
    }

    pub fn permute(
        &self,
        axes: &[usize],
    ) -> Result<Self, CoreError> {
        let rank = self.shape.rank();
        let invalid = || CoreError::InvalidPermutation {
            axes: axes.to_vec(),
            rank,
        };
        let is_permutation = axes.len() == rank && (0..rank).all(|axis| axes.contains(&axis));
        if !is_permutation {
            return Err(invalid());
        }
        let dims = permuted(self.shape.dims(), axes).ok_or_else(invalid)?;
        let strides = permuted(self.strides(), axes).ok_or_else(invalid)?;
        Ok(Self {
            shape: Shape::try_from(dims.as_slice())?,
            strides: padded(&strides),
            offset: self.offset,
        })
    }

    pub fn narrow(
        &self,
        axis: usize,
        start: usize,
        len: usize,
    ) -> Result<Self, CoreError> {
        let rank = self.shape.rank();
        let (dim, stride) = self
            .shape
            .dims()
            .get(axis)
            .copied()
            .zip(self.strides().get(axis).copied())
            .ok_or(CoreError::AxisOutOfRange {
                axis,
                rank,
            })?;
        let out_of_bounds = || CoreError::NarrowOutOfBounds {
            axis,
            start,
            len,
            dim,
        };
        let end = start.checked_add(len).ok_or_else(out_of_bounds)?;
        if end > dim {
            return Err(out_of_bounds());
        }
        let offset = start
            .checked_mul(stride)
            .and_then(|delta| self.offset.checked_add(delta))
            .ok_or(CoreError::OffsetOverflow {
                offset: self.offset,
                start,
                stride,
            })?;
        let dims: Vec<usize> = self
            .shape
            .dims()
            .iter()
            .enumerate()
            .map(|(index, &size)| if index == axis { len } else { size })
            .collect();
        Ok(Self {
            shape: Shape::try_from(dims.as_slice())?,
            strides: self.strides,
            offset,
        })
    }

    pub fn broadcast_as(
        &self,
        shape: Shape,
    ) -> Result<Self, CoreError> {
        let incompatible = || CoreError::BroadcastAsIncompatible {
            from: self.shape.dims().to_vec(),
            to: shape.dims().to_vec(),
        };
        let leading = shape
            .rank()
            .checked_sub(self.shape.rank())
            .ok_or_else(incompatible)?;
        let strides = shape
            .dims()
            .iter()
            .enumerate()
            .map(|(axis, &dim_target)| {
                let Some(axis_source) = axis.checked_sub(leading) else {
                    return Ok(0);
                };
                match (
                    self.shape.dims().get(axis_source).copied(),
                    self.strides().get(axis_source).copied(),
                ) {
                    (Some(dim_source), Some(stride)) if dim_source == dim_target => Ok(stride),
                    (Some(1), Some(_)) => Ok(0),
                    _ => Err(incompatible()),
                }
            })
            .collect::<Result<Vec<usize>, CoreError>>()?;
        Ok(Self {
            shape,
            strides: padded(&strides),
            offset: self.offset,
        })
    }
}

fn contiguous_strides(shape: &Shape) -> Result<[usize; RANK_MAX], CoreError> {
    let (strides_reversed, _) = shape
        .dims()
        .iter()
        .rev()
        .try_fold(
            (Vec::with_capacity(RANK_MAX), 1_usize),
            |(mut strides, stride_next), &dim| {
                strides.push(stride_next);
                stride_next.checked_mul(dim).map(|stride| (strides, stride))
            },
        )
        .ok_or_else(|| CoreError::StrideOverflow {
            dims: shape.dims().to_vec(),
        })?;
    let strides: Vec<usize> = strides_reversed.into_iter().rev().collect();
    Ok(padded(&strides))
}

fn permuted(
    values: &[usize],
    axes: &[usize],
) -> Option<Vec<usize>> {
    axes.iter().map(|&axis| values.get(axis).copied()).collect()
}

fn padded(values: &[usize]) -> [usize; RANK_MAX] {
    array::from_fn(|axis| values.get(axis).copied().unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use crate::core::{
        CoreError,
        Layout,
        Shape,
    };

    fn contiguous(dims: &[usize]) -> Layout {
        Layout::contiguous(Shape::try_from(dims).expect("valid shape")).expect("valid layout")
    }

    #[test]
    fn contiguous_strides_are_row_major() {
        let layout = contiguous(&[2, 3, 4]);
        assert_eq!(layout.strides(), &[12, 4, 1], "contiguous strides");
        assert!(
            layout.is_contiguous(),
            "contiguous layout reports contiguous"
        );
    }

    #[test]
    fn contiguous_rejects_stride_overflow() {
        let shape = Shape::try_from([0, usize::MAX, 2].as_slice()).expect("valid shape");
        let result = Layout::contiguous(shape);
        assert!(
            matches!(result, Err(CoreError::StrideOverflow { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn permute_reorders_dims_and_strides() {
        let layout = contiguous(&[2, 3, 4])
            .permute(&[2, 0, 1])
            .expect("valid permutation");
        assert_eq!(layout.shape().dims(), &[4, 2, 3], "permuted dims");
        assert_eq!(layout.strides(), &[1, 12, 4], "permuted strides");
        assert!(!layout.is_contiguous(), "permuted layout is not contiguous");
    }

    #[test]
    fn permute_rejects_invalid_permutation() {
        let layout = contiguous(&[2, 3, 4]);
        [[0, 0, 1].as_slice(), &[0, 1, 3], &[0, 1]]
            .into_iter()
            .for_each(|axes| {
                let result = layout.permute(axes);
                assert!(
                    matches!(result, Err(CoreError::InvalidPermutation { .. })),
                    "axes {axes:?} got {result:?}"
                );
            });
    }

    #[test]
    fn narrow_moves_offset_and_shrinks_axis() {
        let layout = contiguous(&[2, 3, 4]).narrow(1, 1, 2).expect("in bounds");
        assert_eq!(layout.shape().dims(), &[2, 2, 4], "narrowed dims");
        assert_eq!(layout.strides(), &[12, 4, 1], "strides unchanged");
        assert_eq!(layout.offset(), 4, "offset moved by one row");
    }

    #[test]
    fn narrow_rejects_out_of_bounds() {
        let layout = contiguous(&[2, 3, 4]);
        let result = layout.narrow(1, 2, 2);
        assert!(
            matches!(result, Err(CoreError::NarrowOutOfBounds { .. })),
            "got {result:?}"
        );
        let result = layout.narrow(3, 0, 1);
        assert!(
            matches!(
                result,
                Err(CoreError::AxisOutOfRange {
                    axis: 3,
                    rank: 3
                })
            ),
            "got {result:?}"
        );
    }

    #[test]
    fn reshape_keeps_offset_of_contiguous_layout() {
        let layout = contiguous(&[2, 3, 4]).narrow(0, 1, 1).expect("in bounds");
        let reshaped = layout
            .reshape(Shape::try_from([3, 4].as_slice()).expect("valid shape"))
            .expect("contiguous");
        assert_eq!(reshaped.strides(), &[4, 1], "reshaped strides");
        assert_eq!(reshaped.offset(), 12, "offset kept");
    }

    #[test]
    fn reshape_rejects_non_contiguous_layout() {
        let layout = contiguous(&[2, 3])
            .permute(&[1, 0])
            .expect("valid permutation");
        let result = layout.reshape(Shape::try_from([6].as_slice()).expect("valid shape"));
        assert!(
            matches!(result, Err(CoreError::ReshapeNonContiguous)),
            "got {result:?}"
        );
    }

    #[test]
    fn reshape_rejects_element_count_mismatch() {
        let result =
            contiguous(&[2, 3]).reshape(Shape::try_from([5].as_slice()).expect("valid shape"));
        assert!(
            matches!(
                result,
                Err(CoreError::ElementCountMismatch {
                    from: 6,
                    to: 5
                })
            ),
            "got {result:?}"
        );
    }

    #[test]
    fn broadcast_as_uses_zero_strides() {
        let layout = contiguous(&[3, 1])
            .broadcast_as(Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"))
            .expect("broadcastable");
        assert_eq!(layout.shape().dims(), &[2, 3, 4], "broadcast dims");
        assert_eq!(layout.strides(), &[0, 1, 0], "broadcast strides");
    }

    #[test]
    fn broadcast_as_rejects_incompatible_shape() {
        let result = contiguous(&[3, 2])
            .broadcast_as(Shape::try_from([3, 4].as_slice()).expect("valid shape"));
        assert!(
            matches!(result, Err(CoreError::BroadcastAsIncompatible { .. })),
            "got {result:?}"
        );
    }
}
