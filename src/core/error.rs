use crate::core::DType;
use crate::core::primitive::ReduceOp;

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("Rank {rank} exceeds the maximum rank {rank_max}.")]
    RankTooLarge { rank: usize, rank_max: usize },

    #[error("Shape {dims:?} has more than {element_count_max} non-zero elements.")]
    ElementCountOverflow {
        dims: Vec<usize>,
        element_count_max: usize,
    },

    #[error("Offset {offset} plus start {start} times stride {stride} overflows usize.")]
    OffsetOverflow {
        offset: usize,
        start: usize,
        stride: usize,
    },

    #[error("Shapes {lhs:?} and {rhs:?} cannot be broadcast together.")]
    BroadcastIncompatible { lhs: Vec<usize>, rhs: Vec<usize> },

    #[error("Axes {axes:?} are not a permutation of rank {rank}.")]
    InvalidPermutation { axes: Vec<usize>, rank: usize },

    #[error("Axis {axis} is out of range for rank {rank}.")]
    AxisOutOfRange { axis: usize, rank: usize },

    #[error("Narrow of axis {axis} from {start} with length {len} exceeds dimension {dim}.")]
    NarrowOutOfBounds {
        axis: usize,
        start: usize,
        len: usize,
        dim: usize,
    },

    #[error("Reshape requires a contiguous layout.")]
    ReshapeNonContiguous,

    #[error("Element count {from} does not match element count {to}.")]
    ElementCountMismatch { from: usize, to: usize },

    #[error("Shape {from:?} cannot be broadcast to shape {to:?}.")]
    BroadcastAsIncompatible { from: Vec<usize>, to: Vec<usize> },

    #[error("DType {dtype:?} is not a float dtype.")]
    DTypeNotFloat { dtype: DType },

    #[error("DType {lhs:?} does not match dtype {rhs:?}.")]
    DTypeMismatch { lhs: DType, rhs: DType },

    #[error("DType {dtype:?} was given where {dtype_expected:?} is expected.")]
    DTypeUnexpected { dtype: DType, dtype_expected: DType },

    #[error("Shapes {lhs:?} and {rhs:?} cannot be multiplied.")]
    MatmulIncompatible { lhs: Vec<usize>, rhs: Vec<usize> },

    #[error("Table shape {table:?} and indices shape {indices:?} cannot be gathered.")]
    GatherIncompatible {
        table: Vec<usize>,
        indices: Vec<usize>,
    },

    #[error("Shapes {lhs:?} and {rhs:?} cannot be concatenated on axis {axis}.")]
    ConcatIncompatible {
        axis: usize,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },

    #[error(
        "Update shape {update:?} at start {start} on axis {axis} does not fit target shape \
         {target:?}."
    )]
    SliceUpdateOutOfBounds {
        axis: usize,
        start: usize,
        target: Vec<usize>,
        update: Vec<usize>,
    },

    #[error("Reduction {op:?} over empty axis {axis} of shape {dims:?} has no result.")]
    EmptyReduction {
        op: ReduceOp,
        axis: usize,
        dims: Vec<usize>,
    },

    #[error("Slice update target shares its storage with another tensor.")]
    SharedStorage,

    #[error("Slice update requires a contiguous target layout.")]
    SliceUpdateNonContiguous,
}
