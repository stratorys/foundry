use crate::core::{
    CoreError,
    TensorError,
};

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

#[derive(Debug, thiserror::Error)]
pub enum LlamaError<E: std::error::Error + 'static> {
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
    Tensor(#[from] TensorError<E>),

    #[error(transparent)]
    Validation(#[from] CoreError),
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use crate::backend::cpu::CpuError;
    use crate::core::{
        CoreError,
        TensorError,
    };
    use crate::models::llama::LlamaError;

    #[test]
    fn model_propagation_preserves_tensor_and_backend_sources() {
        fn propagate() -> Result<(), LlamaError<CpuError>> {
            Err::<(), _>(TensorError::Backend {
                operation: "upload",
                source: CpuError::IndexOutOfStorage,
            })?;
            Ok(())
        }

        let error = propagate().expect_err("the backend failure propagates");
        assert!(
            matches!(
                &error,
                LlamaError::Tensor(TensorError::Backend {
                    operation: "upload",
                    source: CpuError::IndexOutOfStorage,
                })
            ),
            "propagation retains the concrete backend error"
        );
        let tensor = error
            .source()
            .expect("the model failure has a tensor source");
        assert!(
            tensor.is::<TensorError<CpuError>>(),
            "the intermediate error type is retained"
        );
        let backend = tensor
            .source()
            .expect("the tensor failure has a backend source");
        assert_eq!(
            backend.downcast_ref::<CpuError>(),
            Some(&CpuError::IndexOutOfStorage),
            "the original backend cause is available without tracing"
        );
    }

    #[test]
    fn model_propagation_preserves_direct_validation_failure() {
        fn propagate() -> Result<(), LlamaError<CpuError>> {
            Err::<(), _>(CoreError::AxisOutOfRange)?;
            Ok(())
        }

        let error = propagate().expect_err("the validation failure propagates");
        assert!(
            matches!(error, LlamaError::Validation(CoreError::AxisOutOfRange)),
            "pure validation failures do not become backend errors"
        );
    }
}
