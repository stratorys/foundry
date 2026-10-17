#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EvalError {
    #[error("Reading the activations file failed.")]
    ActivationsRead,

    #[error("Activations file exceeds the maximum size.")]
    ActivationsTooLarge,

    #[error("Activations file is not a valid safetensors file.")]
    ActivationsInvalid,

    #[error("Activation has no matching input or output.")]
    UnpairedActivation,

    #[error("Creating a submodule input tensor failed.")]
    Input,

    #[error("Submodule forward failed.")]
    Forward,

    #[error("Reading a submodule output failed.")]
    Output,

    #[error("Activation dtype cannot be compared.")]
    DTypeNotComparable,

    #[error("Activation bytes do not hold whole elements.")]
    PartialElement,

    #[error("Compared activations do not have the same element count.")]
    NumelMismatch,

    #[error("Element count does not fit in u32.")]
    NumelOverflow,
}
