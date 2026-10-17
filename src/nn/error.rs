#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TensorError {
    #[error("Rank exceeds the maximum rank.")]
    RankTooLarge,

    #[error("Shape has too many elements.")]
    NumelOverflow,

    #[error("View offset overflows.")]
    OffsetOverflow,

    #[error("Shapes cannot be broadcast together.")]
    BroadcastIncompatible,

    #[error("Dims are not a permutation of the dimensions.")]
    InvalidPermutation,

    #[error("Dimension is out of range.")]
    DimOutOfRange,

    #[error("Narrow exceeds the dimension.")]
    NarrowOutOfBounds,

    #[error("Dimension is too large to convert exactly to f32.")]
    DimensionTooLargeForF32,

    #[error("Reshape requires a contiguous layout.")]
    ReshapeNonContiguous,

    #[error("Element counts do not match.")]
    NumelMismatch,

    #[error("Shape cannot be expanded to the target shape.")]
    ExpandIncompatible,

    #[error("DType is not a float dtype.")]
    DTypeNotFloat,

    #[error("DTypes do not match.")]
    DTypeMismatch,

    #[error("Index select indices are not u32.")]
    IndicesNotU32,

    #[error("Shapes cannot be multiplied.")]
    MatmulIncompatible,

    #[error("Indices cannot select rows of the table.")]
    IndexSelectIncompatible,

    #[error("Shapes cannot be concatenated.")]
    CatIncompatible,

    #[error("Slice scatter does not fit the target.")]
    SliceScatterOutOfBounds,

    #[error("Reduction over an empty dimension has no result.")]
    EmptyReduction,

    #[error("Slice scatter target shares its storage with another tensor.")]
    SharedStorage,

    #[error("Slice scatter requires a contiguous target layout.")]
    SliceScatterNonContiguous,

    #[error("Byte length does not match the shape and dtype.")]
    ByteLengthMismatch,

    #[error("Reading bytes requires a contiguous layout.")]
    ToBytesNonContiguous,

    #[error("Query, keys and values are incompatible for attention.")]
    AttentionIncompatible,

    #[error("Keys and values do not have the same shape.")]
    KvIncompatible,

    #[error("Layer index exceeds the cache layers.")]
    CacheLayerOutOfRange,

    #[error("Operand count does not match the operation.")]
    OperandCount,

    #[error("Backend operation failed.")]
    Backend,
}
