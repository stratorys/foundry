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

    #[error("Safetensors header is invalid.")]
    InvalidHeader,

    #[error("Tensor has an unknown dtype.")]
    UnknownDType,

    #[error("Tensor has an invalid shape.")]
    InvalidShape,

    #[error("Tensor appears in more than one shard.")]
    DuplicateTensor,

    #[error("Index maps a tensor to a shard that does not contain it.")]
    IndexMismatch,
}
