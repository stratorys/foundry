use tracing::error;

use crate::nn::TensorError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LlamaConfigError {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LlamaError {
    #[error("Tensor is not in the weights.")]
    TensorNotFound,

    #[error("Tensor shape does not match the config.")]
    TensorShapeMismatch,

    #[error("Tensor dtype is not the model dtype.")]
    TensorDTypeMismatch,

    #[error("Key-value width overflows usize.")]
    KvDimOverflow,

    #[error("Maximum sequence length exceeds the allowed length.")]
    MaxCacheLenTooLarge,

    #[error("Head dim is not a positive even number.")]
    HeadDimInvalid,

    #[error("Dimension is too large to convert exactly to f32.")]
    DimensionTooLargeForF32,

    #[error("Forward input holds no token.")]
    EmptyInput,

    #[error("Tokens exceed the cache length.")]
    CacheOverflow,

    #[error("Layer cache count does not match the layer count.")]
    CacheLayerCount,

    #[error("Input IDs are not [batch, seq_len].")]
    InputIdsShape,

    #[error("Tensor operation failed.")]
    Tensor,
}

pub fn tensor_failed(error: TensorError) -> LlamaError {
    error!(message = "Tensor operation failed.", %error);
    LlamaError::Tensor
}

#[cfg(test)]
mod tests {
    use crate::models::llama::LlamaError;
    use crate::models::llama::error::tensor_failed;
    use crate::nn::TensorError;

    #[test]
    fn tensor_failures_become_the_tensor_variant() {
        assert_eq!(
            [
                tensor_failed(TensorError::Device),
                tensor_failed(TensorError::DimOutOfRange)
            ],
            [LlamaError::Tensor, LlamaError::Tensor],
            "device and validation failures are logged, then reported as a tensor failure"
        );
    }
}
