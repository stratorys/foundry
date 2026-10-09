#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CoreError {
    #[error("Rank exceeds the maximum rank.")]
    RankTooLarge,

    #[error("Shape has too many elements.")]
    ElementCountOverflow,

    #[error("View offset overflows.")]
    OffsetOverflow,

    #[error("Shapes cannot be broadcast together.")]
    BroadcastIncompatible,

    #[error("Axes are not a permutation of the rank.")]
    InvalidPermutation,

    #[error("Axis is out of range.")]
    AxisOutOfRange,

    #[error("Narrow exceeds the dimension.")]
    NarrowOutOfBounds,

    #[error("Dimension is too large to convert exactly to f32.")]
    DimensionTooLargeForF32,

    #[error("Reshape requires a contiguous layout.")]
    ReshapeNonContiguous,

    #[error("Element counts do not match.")]
    ElementCountMismatch,

    #[error("Shape cannot be broadcast to the target shape.")]
    BroadcastAsIncompatible,

    #[error("DType is not a float dtype.")]
    DTypeNotFloat,

    #[error("DTypes do not match.")]
    DTypeMismatch,

    #[error("Gather indices are not u32.")]
    IndicesNotU32,

    #[error("Shapes cannot be multiplied.")]
    MatmulIncompatible,

    #[error("Table and indices cannot be gathered.")]
    GatherIncompatible,

    #[error("Shapes cannot be concatenated.")]
    ConcatIncompatible,

    #[error("Slice update does not fit the target.")]
    SliceUpdateOutOfBounds,

    #[error("Reduction over an empty axis has no result.")]
    EmptyReduction,

    #[error("Slice update target shares its storage with another tensor.")]
    SharedStorage,

    #[error("Slice update requires a contiguous target layout.")]
    SliceUpdateNonContiguous,

    #[error("Byte length does not match the shape and dtype.")]
    ByteLengthMismatch,

    #[error("Download requires a contiguous layout.")]
    DownloadNonContiguous,

    #[error("Query, keys and values are incompatible for attention.")]
    AttentionIncompatible,

    #[error("Attention head dim does not match.")]
    AttentionHeadDimMismatch,

    #[error("Keys and values do not have the same shape.")]
    KvIncompatible,

    #[error("Backend primitive failed.")]
    Backend,
}
