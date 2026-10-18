#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("Model id is not of the form owner/name.")]
pub struct ModelIdError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CliError {
    #[error("Model cannot be resolved from the Hugging Face cache.")]
    Resolve,

    #[error("Opening the model weights failed.")]
    Weights,

    #[error("Opening the model config failed.")]
    Config,

    #[error("Measuring the model weights failed.")]
    WeightsSize,

    #[error("Writing the report to stdout failed.")]
    Output,

    #[cfg(target_os = "macos")]
    #[error("Creating the Metal device failed.")]
    Device,

    #[cfg(not(target_os = "macos"))]
    #[error("No device is available on this platform.")]
    DeviceUnavailable,

    #[error("Loading the model failed.")]
    Load,

    #[error("Prompt and generated token counts overflow usize.")]
    SeqLenOverflow,

    #[error("Generation failed.")]
    Generate,

    #[error("Vocabulary size is zero or does not fit in u32.")]
    VocabSize,

    #[error("Token id is outside the vocabulary.")]
    TokenIdOutOfVocab,

    #[error("Benchmark failed.")]
    Bench,

    #[error("Writing the report failed.")]
    Report,

    #[error("Reading a reference failed.")]
    Reference,

    #[error("No reference holds layer activations.")]
    NoActivations,

    #[error("Evaluation failed.")]
    Eval,

    #[error("The model does not match a reference.")]
    EvalMismatch,

    #[error("System clock cannot be formatted as an RFC 3339 time.")]
    Timestamp,
}
