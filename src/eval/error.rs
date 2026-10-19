#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EvalError {
    #[error("Reading a reference file failed.")]
    FileRead,

    #[error("Reference file exceeds the maximum size.")]
    FileTooLarge,

    #[error("Reference file is not a valid safetensors file.")]
    SafetensorsInvalid,

    #[error("Reference generation is not valid JSON or misses a field.")]
    GenerationJson,

    #[error("Reference generation is inconsistent.")]
    GenerationInvalid,

    #[error("Reference first-step logits are missing.")]
    LastLogitsMissing,

    #[error("References do not hold the same prompts.")]
    PromptMismatch,

    #[error("Generating with the model failed.")]
    Generate,

    #[error("Token position overflows usize.")]
    PositionOverflow,

    #[error("Vocabulary index does not fit in u32.")]
    VocabOverflow,

    #[error("Activation has no matching input or output.")]
    UnpairedActivation,

    #[error("Reference activations hold no tensor.")]
    ActivationsEmpty,

    #[error("Activation has no matching submodule.")]
    ModuleMissing,

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
