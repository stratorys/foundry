#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("Rank {rank} exceeds the maximum rank {rank_max}.")]
    RankTooLarge { rank: usize, rank_max: usize },

    #[error("Element count of shape {dims:?} overflows usize.")]
    ElementCountOverflow { dims: Vec<usize> },

    #[error("Contiguous strides of shape {dims:?} overflow usize.")]
    StrideOverflow { dims: Vec<usize> },

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
}
