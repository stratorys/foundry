use std::convert::Infallible;
use std::io;
use std::path::PathBuf;

use crate::core::DType;
use crate::weights::WeightsError;

#[derive(Debug, thiserror::Error)]
pub enum LlamaError<E = Infallible> {
    #[error("I/O on {path:?} failed.")]
    Io {
        path: PathBuf,
        #[source]
        error: io::Error,
    },

    #[error("Config file {path:?} of {bytes} bytes exceeds the maximum of {bytes_max} bytes.")]
    ConfigTooLarge {
        path: PathBuf,
        bytes: u64,
        bytes_max: u64,
    },

    #[error("Config is not valid JSON or misses a field.")]
    Json(#[source] serde_json::Error),

    #[error("Model type {model_type:?} is not \"llama\".")]
    ModelType { model_type: String },

    #[error(
        "Hidden size {hidden_size} does not equal {num_attention_heads} attention heads times \
         head dim {head_dim}."
    )]
    HiddenSizeMismatch {
        hidden_size: usize,
        num_attention_heads: usize,
        head_dim: usize,
    },

    #[error(
        "{num_attention_heads} attention heads are not a multiple of {num_key_value_heads} \
         key-value heads."
    )]
    HeadRatio {
        num_attention_heads: usize,
        num_key_value_heads: usize,
    },

    #[error("RoPE type {rope_type:?} is not \"llama3\".")]
    RopeType { rope_type: String },

    #[error("Word embeddings are not tied.")]
    UntiedEmbeddings,

    #[error(transparent)]
    Weights(#[from] WeightsError),

    #[error(transparent)]
    Backend(E),

    #[error("Tensor {name} has shape {dims:?}; the config expects {dims_expected:?}.")]
    ShapeMismatch {
        name: String,
        dims: Vec<usize>,
        dims_expected: Vec<usize>,
    },

    #[error("Tensor {name} has dtype {dtype:?}; {dtype_expected:?} is expected.")]
    DTypeMismatch {
        name: String,
        dtype: DType,
        dtype_expected: DType,
    },

    #[error("{kv_heads} key-value heads times head dim {head_dim} overflows usize.")]
    DimensionOverflow { kv_heads: usize, head_dim: usize },

    #[error("Uploaded byte count overflows usize.")]
    ByteCountOverflow,

    #[error("Maximum sequence length {seq_len_max} exceeds the allowed {seq_len_max_allowed}.")]
    SeqLenMaxTooLarge {
        seq_len_max: usize,
        seq_len_max_allowed: usize,
    },

    #[error("Head dim {head_dim} is not a positive even number, as rotate-half RoPE requires.")]
    HeadDimInvalid { head_dim: usize },

    #[error("{name} {dim} exceeds {dim_max}, the largest value converted exactly to f32.")]
    DimensionTooLargeForF32 {
        name: &'static str,
        dim: usize,
        dim_max: usize,
    },

    #[error("Forward input holds no token.")]
    EmptyInput,

    #[error(
        "{seq_len} tokens at position {position} exceed the cache length of {seq_len_max} \
         positions."
    )]
    CacheOverflow {
        position: usize,
        seq_len: usize,
        seq_len_max: usize,
    },

    #[error("{caches} layer caches were given for {layers} layers.")]
    CacheLayerCount { caches: usize, layers: usize },
}
