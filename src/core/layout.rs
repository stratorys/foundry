use std::array;

use tracing::error;

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
    pub fn contiguous(shape: Shape) -> Self {
        Self {
            shape,
            strides: contiguous_strides(&shape),
            offset: 0,
        }
    }

    pub fn shape(&self) -> &Shape { &self.shape }

    pub fn strides(&self) -> &[usize] { self.strides.get(..self.shape.rank()).unwrap_or_default() }

    pub fn offset(&self) -> usize { self.offset }

    pub fn stride(
        &self,
        axis: usize,
    ) -> Result<usize, CoreError> {
        self.strides().get(axis).copied().ok_or_else(|| {
            error!(
                message = "Axis is out of range.",
                axis,
                rank = self.shape.rank()
            );
            CoreError::AxisOutOfRange
        })
    }

    pub fn is_contiguous(&self) -> bool {
        let expected = contiguous_strides(&self.shape);
        self.shape
            .dims()
            .iter()
            .zip(self.strides().iter().zip(expected.iter()))
            .all(|(&dim, (&stride, &stride_expected))| dim == 1 || stride == stride_expected)
    }

    pub fn reshape(
        &self,
        shape: Shape,
    ) -> Result<Self, CoreError> {
        if !self.is_contiguous() {
            error!(message = "Reshape requires a contiguous layout.", layout = ?self);
            return Err(CoreError::ReshapeNonContiguous);
        }
        let (from, to) = (self.shape.element_count(), shape.element_count());
        if from != to {
            error!(message = "Element counts do not match.", from, to);
            return Err(CoreError::ElementCountMismatch);
        }
        Ok(Self {
            offset: self.offset,
            ..Self::contiguous(shape)
        })
    }

    pub fn permute(
        &self,
        axes: &[usize],
    ) -> Result<Self, CoreError> {
        let rank = self.shape.rank();
        let invalid = || {
            error!(
                message = "Axes are not a permutation of the rank.",
                ?axes,
                rank
            );
            CoreError::InvalidPermutation
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
        let dim = self.shape.dim(axis)?;
        let stride = self.stride(axis)?;
        let out_of_bounds = || {
            error!(
                message = "Narrow exceeds the dimension.",
                axis, start, len, dim
            );
            CoreError::NarrowOutOfBounds
        };
        let end = start.checked_add(len).ok_or_else(out_of_bounds)?;
        if end > dim {
            return Err(out_of_bounds());
        }
        let offset = start
            .checked_mul(stride)
            .and_then(|delta| self.offset.checked_add(delta))
            .ok_or_else(|| {
                error!(
                    message = "View offset overflows.",
                    offset = self.offset,
                    start,
                    stride,
                );
                CoreError::OffsetOverflow
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
        let incompatible = || {
            error!(
                message = "Shape cannot be broadcast to the target shape.",
                from = ?self.shape.dims(),
                to = ?shape.dims(),
            );
            CoreError::BroadcastAsIncompatible
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

fn contiguous_strides(shape: &Shape) -> [usize; RANK_MAX] {
    let (strides_reversed, _) = shape.dims().iter().rev().fold(
        (Vec::with_capacity(RANK_MAX), 1_usize),
        |(mut strides, stride_next), &dim| {
            strides.push(stride_next);
            (strides, stride_next.saturating_mul(dim))
        },
    );
    let strides: Vec<usize> = strides_reversed.into_iter().rev().collect();
    padded(&strides)
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

    #[test]
    fn contiguous_strides_are_row_major() {
        let layout =
            Layout::contiguous(Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"));
        assert_eq!(layout.strides(), &[12, 4, 1], "contiguous strides");
        assert!(
            layout.is_contiguous(),
            "contiguous layout reports contiguous"
        );
    }

    #[test]
    fn permute_reorders_dims_and_strides() {
        let layout =
            Layout::contiguous(Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"))
                .permute(&[2, 0, 1])
                .expect("valid permutation");
        assert_eq!(layout.shape().dims(), &[4, 2, 3], "permuted dims");
        assert_eq!(layout.strides(), &[1, 12, 4], "permuted strides");
        assert!(!layout.is_contiguous(), "permuted layout is not contiguous");
    }

    #[test]
    fn permute_rejects_invalid_permutation() {
        let layout =
            Layout::contiguous(Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"));
        [[0, 0, 1].as_slice(), &[0, 1, 3], &[0, 1]]
            .into_iter()
            .for_each(|axes| {
                assert_eq!(
                    layout.permute(axes),
                    Err(CoreError::InvalidPermutation),
                    "axes {axes:?}"
                );
            });
    }

    #[test]
    fn narrow_moves_offset_and_shrinks_axis() {
        let layout =
            Layout::contiguous(Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"))
                .narrow(1, 1, 2)
                .expect("in bounds");
        assert_eq!(layout.shape().dims(), &[2, 2, 4], "narrowed dims");
        assert_eq!(layout.strides(), &[12, 4, 1], "strides unchanged");
        assert_eq!(layout.offset(), 4, "offset moved by one row");
    }

    #[test]
    fn narrow_rejects_out_of_bounds() {
        let layout =
            Layout::contiguous(Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"));
        assert_eq!(
            layout.narrow(1, 2, 2),
            Err(CoreError::NarrowOutOfBounds),
            "the window ends after the axis"
        );
        assert_eq!(
            layout.narrow(3, 0, 1),
            Err(CoreError::AxisOutOfRange),
            "axis 3 does not exist"
        );
    }

    #[test]
    fn narrow_with_overflowing_end_is_rejected() {
        assert_eq!(
            Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape")).narrow(
                1,
                usize::MAX,
                1
            ),
            Err(CoreError::NarrowOutOfBounds),
            "start plus len overflows usize"
        );
    }

    #[test]
    fn reshape_keeps_offset_of_contiguous_layout() {
        let layout =
            Layout::contiguous(Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"))
                .narrow(0, 1, 1)
                .expect("in bounds");
        let reshaped = layout
            .reshape(Shape::try_from([3, 4].as_slice()).expect("valid shape"))
            .expect("contiguous");
        assert_eq!(reshaped.strides(), &[4, 1], "reshaped strides");
        assert_eq!(reshaped.offset(), 12, "offset kept");
    }

    #[test]
    fn reshape_rejects_non_contiguous_layout() {
        let layout = Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape"))
            .permute(&[1, 0])
            .expect("valid permutation");
        assert_eq!(
            layout.reshape(Shape::try_from([6].as_slice()).expect("valid shape")),
            Err(CoreError::ReshapeNonContiguous),
            "a transposed layout cannot be reshaped"
        );
    }

    #[test]
    fn reshape_rejects_element_count_mismatch() {
        assert_eq!(
            Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape"))
                .reshape(Shape::try_from([5].as_slice()).expect("valid shape")),
            Err(CoreError::ElementCountMismatch),
            "six elements cannot become five"
        );
    }

    #[test]
    fn broadcast_as_uses_zero_strides() {
        let layout = Layout::contiguous(Shape::try_from([3, 1].as_slice()).expect("valid shape"))
            .broadcast_as(Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"))
            .expect("broadcastable");
        assert_eq!(layout.shape().dims(), &[2, 3, 4], "broadcast dims");
        assert_eq!(layout.strides(), &[0, 1, 0], "broadcast strides");
    }

    const WIDENED_DIMS: [usize; 4] = [2, 6, 5, 7];

    fn addresses(layout: &Layout) -> Vec<usize> {
        let dims = layout.shape().dims();
        (0..layout.shape().element_count())
            .map(|flat| {
                dims.iter()
                    .zip(layout.strides())
                    .rev()
                    .fold(
                        (flat, layout.offset()),
                        |(rest, address), (&dim, &stride)| {
                            let index = rest
                                .checked_rem(dim)
                                .expect("a non-empty view has no zero dim");
                            let address = index
                                .checked_mul(stride)
                                .and_then(|delta| address.checked_add(delta))
                                .expect("the address fits in usize");
                            (rest.checked_div(dim).expect("dim is non-zero"), address)
                        },
                    )
                    .1
            })
            .collect()
    }

    fn with_dims(
        dims: &[usize],
        size: impl Fn(usize, usize) -> usize,
    ) -> Shape {
        let dims: Vec<usize> = dims
            .iter()
            .enumerate()
            .map(|(axis, &dim)| size(axis, dim))
            .collect();
        Shape::try_from(dims.as_slice()).expect("a widened shape is valid")
    }

    fn narrows(layout: &Layout) -> Vec<Layout> {
        layout
            .shape()
            .dims()
            .iter()
            .enumerate()
            .flat_map(|(axis, &dim)| {
                [
                    Some((0, dim)),
                    dim.checked_sub(1).map(|len| (1, len)),
                    Some((dim, 0)),
                ]
                .into_iter()
                .flatten()
                .map(move |(start, len)| {
                    layout
                        .narrow(axis, start, len)
                        .expect("the window is in bounds")
                })
            })
            .collect()
    }

    fn permutes(layout: &Layout) -> Vec<Layout> {
        let rank = layout.shape().rank();
        let reversed: Vec<usize> = (0..rank).rev().collect();
        let rotated: Vec<usize> = (1..rank).chain((rank > 0).then_some(0)).collect();
        [reversed, rotated]
            .iter()
            .map(|axes| layout.permute(axes).expect("the axes are a permutation"))
            .collect()
    }

    fn broadcasts(layout: &Layout) -> Vec<Layout> {
        let dims = layout.shape().dims();
        let leading: Vec<usize> = [2].into_iter().chain(dims.iter().copied()).collect();
        let widened_all = with_dims(dims, |axis, dim| {
            if dim == 1 {
                WIDENED_DIMS.get(axis).copied().unwrap_or(2)
            } else {
                dim
            }
        });
        let widened_each = dims
            .iter()
            .enumerate()
            .filter(|&(_, &dim)| dim == 1)
            .map(|(widened, _)| with_dims(dims, |axis, dim| if axis == widened { 2 } else { dim }));
        Shape::try_from(leading.as_slice())
            .ok()
            .into_iter()
            .chain([widened_all])
            .chain(widened_each)
            .map(|shape| {
                layout
                    .broadcast_as(shape)
                    .expect("the shape is broadcastable")
            })
            .collect()
    }

    fn reshapes(layout: &Layout) -> Vec<Layout> {
        Shape::try_from([layout.shape().element_count()].as_slice())
            .ok()
            .and_then(|shape| layout.reshape(shape).ok())
            .into_iter()
            .collect()
    }

    fn views(base: &Layout) -> Vec<Layout> {
        let narrowed = narrows(base);
        let permuted = permutes(base);
        let after_narrow = narrowed.iter().flat_map(|layout| {
            permutes(layout)
                .into_iter()
                .chain(reshapes(layout))
                .chain(broadcasts(layout))
        });
        let after_permute = permuted
            .iter()
            .flat_map(narrows)
            .flat_map(|layout| broadcasts(&layout));
        [*base]
            .into_iter()
            .chain(broadcasts(base))
            .chain(reshapes(base))
            .chain(after_narrow.collect::<Vec<Layout>>())
            .chain(after_permute.collect::<Vec<Layout>>())
            .chain(narrowed)
            .chain(permuted)
            .collect()
    }

    #[test]
    fn every_view_addresses_only_elements_of_its_base() {
        [[].as_slice(), &[2, 0, 3], &[5], &[2, 3, 4], &[1, 3, 1, 4]]
            .into_iter()
            .for_each(|dims| {
                let base = Layout::contiguous(Shape::try_from(dims).expect("valid shape"));
                let element_count = base.shape().element_count();
                views(&base).iter().for_each(|view| {
                    assert!(
                        addresses(view)
                            .iter()
                            .all(|&address| address < element_count),
                        "base {dims:?} view {:?} strides {:?} offset {} leaves {element_count} \
                         elements",
                        view.shape().dims(),
                        view.strides(),
                        view.offset()
                    );
                });
            });
    }

    #[test]
    fn empty_contiguous_strides_do_not_depend_on_axis_order() {
        assert_eq!(
            Layout::contiguous(
                Shape::try_from([(1 << 30) - 1, 0, 2].as_slice()).expect("valid shape")
            )
            .strides(),
            &[0, 2, 1],
            "zero dim in the middle"
        );
        assert_eq!(
            Layout::contiguous(
                Shape::try_from([0, (1 << 30) - 1, 2].as_slice()).expect("valid shape")
            )
            .strides(),
            &[(1 << 31) - 2, 2, 1],
            "zero dim first"
        );
    }

    #[test]
    fn broadcast_as_rejects_incompatible_shape() {
        assert_eq!(
            Layout::contiguous(Shape::try_from([3, 2].as_slice()).expect("valid shape"))
                .broadcast_as(Shape::try_from([3, 4].as_slice()).expect("valid shape")),
            Err(CoreError::BroadcastAsIncompatible),
            "dim 2 cannot become 4"
        );
    }

    #[test]
    fn broadcast_as_to_a_lower_rank_is_rejected() {
        assert_eq!(
            Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape"))
                .broadcast_as(Shape::try_from([3].as_slice()).expect("valid shape")),
            Err(CoreError::BroadcastAsIncompatible),
            "a broadcast cannot drop an axis"
        );
    }

    #[test]
    fn size_one_axis_with_any_stride_is_contiguous() {
        let layout =
            Layout::contiguous(Shape::try_from([2, 1, 3].as_slice()).expect("valid shape"))
                .permute(&[1, 0, 2])
                .expect("valid permutation");
        assert_eq!(
            layout.strides(),
            &[3, 3, 1],
            "the size one axis keeps stride 3"
        );
        assert!(
            layout.is_contiguous(),
            "moving a size one axis keeps the layout contiguous"
        );
    }

    #[test]
    fn broadcast_layout_is_not_contiguous() {
        let layout = Layout::contiguous(Shape::try_from([3].as_slice()).expect("valid shape"))
            .broadcast_as(Shape::try_from([2, 3].as_slice()).expect("valid shape"))
            .expect("broadcastable");
        assert!(
            !layout.is_contiguous(),
            "a zero stride on a dim of 2 is not contiguous"
        );
    }
}
