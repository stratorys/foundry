use tracing::error;

use crate::core::CoreError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LlamaError {
    #[error("Reading the config file failed.")]
    ConfigRead,

    #[error("Config file exceeds the maximum size.")]
    ConfigTooLarge,

    #[error("Config is not valid JSON or misses a field.")]
    ConfigJson,

    #[error("Model type is not llama.")]
    ModelType,

    #[error("Hidden size does not equal attention heads times head dim.")]
    HiddenSizeMismatch,

    #[error("Attention heads are not a multiple of key-value heads.")]
    HeadRatio,

    #[error("RoPE type is not llama3.")]
    RopeType,

    #[error("Word embeddings are not tied.")]
    UntiedEmbeddings,

    #[error("Tensor is not in the weights.")]
    TensorNotFound,

    #[error("Tensor shape does not match the config.")]
    TensorShapeMismatch,

    #[error("Tensor dtype is not the model dtype.")]
    TensorDTypeMismatch,

    #[error("Key-value width overflows usize.")]
    KvDimOverflow,

    #[error("Uploaded byte count overflows usize.")]
    UploadedBytesOverflow,

    #[error("Maximum sequence length exceeds the allowed length.")]
    SeqLenMaxTooLarge,

    #[error("Head dim is not a positive even number.")]
    HeadDimInvalid,

    #[error("Forward input holds no token.")]
    EmptyInput,

    #[error("Tokens exceed the cache length.")]
    CacheOverflow,

    #[error("Layer cache count does not match the layer count.")]
    CacheLayerCount,

    #[error("Tensor operation failed.")]
    Tensor,
}

pub fn tensor_failed(error: CoreError) -> LlamaError {
    error!(message = "Tensor operation failed.", %error);
    LlamaError::Tensor
}
