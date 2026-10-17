use std::array;

use tracing::error;

use crate::nn::{
    MAX_RANK,
    Shape,
    TensorError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Layout {
    shape: Shape,
    strides: [usize; MAX_RANK],
    storage_offset: usize,
}

impl Layout {
    pub fn contiguous(shape: Shape) -> Self {
        Self {
            shape,
            strides: contiguous_strides(&shape),
            storage_offset: 0,
        }
    }

    pub fn shape(&self) -> &Shape { &self.shape }

    pub fn strides(&self) -> &[usize] { self.strides.get(..self.shape.ndim()).unwrap_or_default() }

    pub fn storage_offset(&self) -> usize { self.storage_offset }

    pub fn stride(
        &self,
        dim: usize,
    ) -> Result<usize, TensorError> {
        self.strides().get(dim).copied().ok_or_else(|| {
            error!(
                message = "Dimension is out of range.",
                dim,
                ndim = self.shape.ndim()
            );
            TensorError::DimOutOfRange
        })
    }

    pub fn is_contiguous(&self) -> bool {
        let expected = contiguous_strides(&self.shape);
        self.shape
            .dims()
            .iter()
            .zip(self.strides().iter().zip(expected.iter()))
            .all(|(&size, (&stride, &stride_expected))| size == 1 || stride == stride_expected)
    }

    pub fn reshape(
        &self,
        shape: Shape,
    ) -> Result<Self, TensorError> {
        if !self.is_contiguous() {
            error!(message = "Reshape requires a contiguous layout.", layout = ?self);
            return Err(TensorError::ReshapeNonContiguous);
        }
        let from = self.shape.numel();
        let to = shape.numel();
        if from != to {
            error!(message = "Element counts do not match.", from, to);
            return Err(TensorError::NumelMismatch);
        }
        Ok(Self {
            storage_offset: self.storage_offset,
            ..Self::contiguous(shape)
        })
    }

    pub fn permute(
        &self,
        dims: &[usize],
    ) -> Result<Self, TensorError> {
        let ndim = self.shape.ndim();
        let is_permutation = dims.len() == ndim && (0..ndim).all(|dim| dims.contains(&dim));
        if !is_permutation {
            error!(
                message = "Dims are not a permutation of the dimensions.",
                ?dims,
                ndim
            );
            return Err(TensorError::InvalidPermutation);
        }
        let sizes: [usize; MAX_RANK] = array::from_fn(|index| {
            dims.get(index)
                .and_then(|&dim| self.shape.dims().get(dim))
                .copied()
                .unwrap_or(1)
        });
        Ok(Self {
            shape: Shape::try_from(sizes.get(..ndim).unwrap_or_default())?,
            strides: array::from_fn(|index| {
                dims.get(index)
                    .and_then(|&dim| self.strides().get(dim))
                    .copied()
                    .unwrap_or(0)
            }),
            storage_offset: self.storage_offset,
        })
    }

    pub fn narrow(
        &self,
        dim: usize,
        start: usize,
        length: usize,
    ) -> Result<Self, TensorError> {
        let size = self.shape.size(dim)?;
        let stride = self.stride(dim)?;
        let out_of_bounds = || {
            error!(
                message = "Narrow exceeds the dimension.",
                dim, start, length, size
            );
            TensorError::NarrowOutOfBounds
        };
        let end = start.checked_add(length).ok_or_else(out_of_bounds)?;
        if end > size {
            return Err(out_of_bounds());
        }
        let storage_offset = start
            .checked_mul(stride)
            .and_then(|delta| self.storage_offset.checked_add(delta))
            .ok_or_else(|| {
                error!(
                    message = "View offset overflows.",
                    storage_offset = self.storage_offset,
                    start,
                    stride,
                );
                TensorError::OffsetOverflow
            })?;
        Ok(Self {
            shape: self.shape.with_dim(dim, length)?,
            strides: self.strides,
            storage_offset,
        })
    }

    pub fn expand(
        &self,
        shape: Shape,
    ) -> Result<Self, TensorError> {
        let incompatible = || {
            error!(
                message = "Shape cannot be expanded to the target shape.",
                from = ?self.shape.dims(),
                to = ?shape.dims(),
            );
            TensorError::ExpandIncompatible
        };
        let leading = shape
            .ndim()
            .checked_sub(self.shape.ndim())
            .ok_or_else(incompatible)?;
        let mut strides = [0; MAX_RANK];
        strides
            .get_mut(..shape.ndim())
            .unwrap_or_default()
            .iter_mut()
            .zip(shape.dims())
            .enumerate()
            .try_for_each(|(dim, (slot, &size_target))| {
                let Some(dim_source) = dim.checked_sub(leading) else {
                    return Ok(());
                };
                *slot = match (
                    self.shape.dims().get(dim_source).copied(),
                    self.strides().get(dim_source).copied(),
                ) {
                    (Some(size_source), Some(stride)) if size_source == size_target => stride,
                    (Some(1), Some(_)) => 0,
                    _ => return Err(incompatible()),
                };
                Ok(())
            })?;
        Ok(Self {
            shape,
            strides,
            storage_offset: self.storage_offset,
        })
    }
}

fn contiguous_strides(shape: &Shape) -> [usize; MAX_RANK] {
    let mut strides = [0; MAX_RANK];
    let mut stride = 1_usize;
    for (slot, &size) in strides
        .get_mut(..shape.ndim())
        .unwrap_or_default()
        .iter_mut()
        .zip(shape.dims())
        .rev()
    {
        *slot = stride;
        stride = stride.saturating_mul(size);
    }
    strides
}

#[cfg(test)]
mod tests {
    use crate::nn::{
        Layout,
        Shape,
        TensorError,
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
            .for_each(|dims| {
                assert_eq!(
                    layout.permute(dims),
                    Err(TensorError::InvalidPermutation),
                    "dims {dims:?}"
                );
            });
    }

    #[test]
    fn narrow_moves_offset_and_shrinks_dim() {
        let layout =
            Layout::contiguous(Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"))
                .narrow(1, 1, 2)
                .expect("in bounds");
        assert_eq!(layout.shape().dims(), &[2, 2, 4], "narrowed dims");
        assert_eq!(layout.strides(), &[12, 4, 1], "strides unchanged");
        assert_eq!(layout.storage_offset(), 4, "offset moved by one row");
    }

    #[test]
    fn narrow_rejects_out_of_bounds() {
        let layout =
            Layout::contiguous(Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"));
        assert_eq!(
            layout.narrow(1, 2, 2),
            Err(TensorError::NarrowOutOfBounds),
            "the window ends after the dim"
        );
        assert_eq!(
            layout.narrow(3, 0, 1),
            Err(TensorError::DimOutOfRange),
            "dim 3 does not exist"
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
            Err(TensorError::NarrowOutOfBounds),
            "start plus length overflows usize"
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
        assert_eq!(reshaped.storage_offset(), 12, "offset kept");
    }

    #[test]
    fn reshape_rejects_non_contiguous_layout() {
        let layout = Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape"))
            .permute(&[1, 0])
            .expect("valid permutation");
        assert_eq!(
            layout.reshape(Shape::try_from([6].as_slice()).expect("valid shape")),
            Err(TensorError::ReshapeNonContiguous),
            "a transposed layout cannot be reshaped"
        );
    }

    #[test]
    fn reshape_rejects_numel_mismatch() {
        assert_eq!(
            Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape"))
                .reshape(Shape::try_from([5].as_slice()).expect("valid shape")),
            Err(TensorError::NumelMismatch),
            "six elements cannot become five"
        );
    }

    #[test]
    fn expand_uses_zero_strides() {
        let layout = Layout::contiguous(Shape::try_from([3, 1].as_slice()).expect("valid shape"))
            .expand(Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"))
            .expect("expandable");
        assert_eq!(layout.shape().dims(), &[2, 3, 4], "expanded dims");
        assert_eq!(layout.strides(), &[0, 1, 0], "expanded strides");
    }

    const WIDENED_DIMS: [usize; 4] = [2, 6, 5, 7];

    struct AddressFold {
        rest: usize,
        address: usize,
    }

    struct Window {
        start: usize,
        length: usize,
    }

    fn addresses(layout: &Layout) -> Vec<usize> {
        let dims = layout.shape().dims();
        (0..layout.shape().numel())
            .map(|flat| {
                dims.iter()
                    .zip(layout.strides())
                    .rev()
                    .fold(
                        AddressFold {
                            rest: flat,
                            address: layout.storage_offset(),
                        },
                        |fold, (&size, &stride)| {
                            let index = fold
                                .rest
                                .checked_rem(size)
                                .expect("a non-empty view has no zero size");
                            AddressFold {
                                rest: fold.rest.checked_div(size).expect("size is non-zero"),
                                address: index
                                    .checked_mul(stride)
                                    .and_then(|delta| fold.address.checked_add(delta))
                                    .expect("the address fits in usize"),
                            }
                        },
                    )
                    .address
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
            .map(|(dim, &current)| size(dim, current))
            .collect();
        Shape::try_from(dims.as_slice()).expect("a widened shape is valid")
    }

    fn narrows(layout: &Layout) -> Vec<Layout> {
        layout
            .shape()
            .dims()
            .iter()
            .enumerate()
            .flat_map(|(dim, &size)| {
                [
                    Some(Window {
                        start: 0,
                        length: size,
                    }),
                    size.checked_sub(1).map(|length| Window {
                        start: 1,
                        length,
                    }),
                    Some(Window {
                        start: size,
                        length: 0,
                    }),
                ]
                .into_iter()
                .flatten()
                .map(move |window| {
                    layout
                        .narrow(dim, window.start, window.length)
                        .expect("the window is in bounds")
                })
            })
            .collect()
    }

    fn permutes(layout: &Layout) -> Vec<Layout> {
        let ndim = layout.shape().ndim();
        let reversed: Vec<usize> = (0..ndim).rev().collect();
        let rotated: Vec<usize> = (1..ndim).chain((ndim > 0).then_some(0)).collect();
        [reversed, rotated]
            .iter()
            .map(|dims| layout.permute(dims).expect("the dims are a permutation"))
            .collect()
    }

    fn expands(layout: &Layout) -> Vec<Layout> {
        let dims = layout.shape().dims();
        let leading: Vec<usize> = [2].into_iter().chain(dims.iter().copied()).collect();
        let widened_all = with_dims(dims, |dim, size| {
            if size == 1 {
                WIDENED_DIMS.get(dim).copied().unwrap_or(2)
            } else {
                size
            }
        });
        let widened_each = dims
            .iter()
            .enumerate()
            .filter(|&(_, &size)| size == 1)
            .map(|(widened, _)| with_dims(dims, |dim, size| if dim == widened { 2 } else { size }));
        Shape::try_from(leading.as_slice())
            .ok()
            .into_iter()
            .chain([widened_all])
            .chain(widened_each)
            .map(|shape| layout.expand(shape).expect("the shape is expandable"))
            .collect()
    }

    fn reshapes(layout: &Layout) -> Vec<Layout> {
        Shape::try_from([layout.shape().numel()].as_slice())
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
                .chain(expands(layout))
        });
        let after_permute = permuted
            .iter()
            .flat_map(narrows)
            .flat_map(|layout| expands(&layout));
        [*base]
            .into_iter()
            .chain(expands(base))
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
                let numel = base.shape().numel();
                views(&base).iter().for_each(|view| {
                    assert!(
                        addresses(view).iter().all(|&address| address < numel),
                        "base {dims:?} view {:?} strides {:?} storage_offset {} leaves {numel} \
                         elements",
                        view.shape().dims(),
                        view.strides(),
                        view.storage_offset()
                    );
                });
            });
    }

    #[test]
    fn empty_contiguous_strides_do_not_depend_on_dim_order() {
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
    fn expand_rejects_incompatible_shape() {
        assert_eq!(
            Layout::contiguous(Shape::try_from([3, 2].as_slice()).expect("valid shape"))
                .expand(Shape::try_from([3, 4].as_slice()).expect("valid shape")),
            Err(TensorError::ExpandIncompatible),
            "dim 2 cannot become 4"
        );
    }

    #[test]
    fn expand_to_a_lower_ndim_is_rejected() {
        assert_eq!(
            Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape"))
                .expand(Shape::try_from([3].as_slice()).expect("valid shape")),
            Err(TensorError::ExpandIncompatible),
            "an expand cannot drop a dim"
        );
    }

    #[test]
    fn size_one_dim_with_any_stride_is_contiguous() {
        let layout =
            Layout::contiguous(Shape::try_from([2, 1, 3].as_slice()).expect("valid shape"))
                .permute(&[1, 0, 2])
                .expect("valid permutation");
        assert_eq!(
            layout.strides(),
            &[3, 3, 1],
            "the size one dim keeps stride 3"
        );
        assert!(
            layout.is_contiguous(),
            "moving a size one dim keeps the layout contiguous"
        );
    }

    #[test]
    fn scalar_layout_has_no_strides_and_is_contiguous() {
        let layout = Layout::contiguous(Shape::try_from([].as_slice()).expect("valid shape"));
        assert_eq!(layout.strides(), &[0_usize; 0], "no strides");
        assert!(layout.is_contiguous(), "a scalar is contiguous");
    }

    #[test]
    fn expand_of_a_size_one_dim_with_a_non_zero_stride_uses_a_zero_stride() {
        let layout =
            Layout::contiguous(Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"))
                .narrow(1, 1, 1)
                .expect("in bounds")
                .expand(Shape::try_from([2, 5, 4].as_slice()).expect("valid shape"))
                .expect("expandable");
        assert_eq!(layout.strides(), &[12, 0, 1], "the stride 4 becomes 0");
        assert_eq!(layout.storage_offset(), 4, "offset kept");
    }

    #[test]
    fn views_keep_the_storage_offset() {
        let layout =
            Layout::contiguous(Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"))
                .narrow(0, 1, 1)
                .expect("in bounds")
                .permute(&[2, 1, 0])
                .expect("valid permutation")
                .expand(Shape::try_from([5, 4, 3, 1].as_slice()).expect("valid shape"))
                .expect("expandable");
        assert_eq!(layout.shape().dims(), &[5, 4, 3, 1], "expanded dims");
        assert_eq!(layout.strides(), &[0, 1, 4, 12], "expanded strides");
        assert_eq!(layout.storage_offset(), 12, "offset of the second block");
    }

    #[test]
    fn identity_views_equal_the_base_layout() {
        let base = Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape"));
        assert_eq!(
            base.permute(&[1, 0])
                .and_then(|layout| layout.permute(&[1, 0])),
            Ok(base),
            "a transpose twice"
        );
        assert_eq!(base.narrow(1, 0, 3), Ok(base), "a full narrow");
    }

    #[test]
    fn narrow_reports_the_dim_before_the_bounds() {
        assert_eq!(
            Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape")).narrow(
                2,
                usize::MAX,
                1
            ),
            Err(TensorError::DimOutOfRange),
            "dim 2 does not exist"
        );
    }

    #[test]
    fn expanded_layout_is_not_contiguous() {
        let layout = Layout::contiguous(Shape::try_from([3].as_slice()).expect("valid shape"))
            .expand(Shape::try_from([2, 3].as_slice()).expect("valid shape"))
            .expect("expandable");
        assert!(
            !layout.is_contiguous(),
            "a zero stride on a dim of 2 is not contiguous"
        );
    }
}
