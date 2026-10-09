#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum WeightsError {
    #[error("Reading the safetensors index failed.")]
    IndexRead,

    #[error("Safetensors index exceeds the maximum size.")]
    IndexTooLarge,

    #[error("Safetensors index is not valid JSON.")]
    IndexJson,

    #[error("Opening a safetensors shard failed.")]
    ShardOpen,

    #[error("Shard name is not a plain file name.")]
    InvalidShardName,

    #[error("Safetensors header is truncated.")]
    TruncatedHeader,

    #[error("Safetensors header exceeds the maximum size.")]
    HeaderTooLarge,

    #[error("Safetensors header is not valid JSON.")]
    HeaderJson,

    #[error("Tensor header entry is malformed.")]
    TensorHeaderJson,

    #[error("Tensor has an unknown dtype.")]
    UnknownDType,

    #[error("Tensor has an invalid shape.")]
    InvalidShape,

    #[error("Tensor data offsets begin after they end.")]
    InvalidRange,

    #[error("Tensor ends outside the data region.")]
    OutOfDataRegion,

    #[error("Tensor byte count overflows usize.")]
    TensorByteCountOverflow,

    #[error("Tensor byte count does not match its shape and dtype.")]
    SizeMismatch,

    #[error("Tensors have overlapping byte ranges.")]
    OverlappingRanges,

    #[error("Tensor appears in more than one shard.")]
    DuplicateTensor,

    #[error("Index maps a tensor to a shard that does not contain it.")]
    IndexMismatch,
}
