use std::io;
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum LlamaConfigError {
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
}
