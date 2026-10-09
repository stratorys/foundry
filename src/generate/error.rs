#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GenerateError {
    #[error("Prompt holds no token.")]
    EmptyPrompt,

    #[error("Token position does not fit in u32.")]
    PositionOverflow,

    #[error("Creating the key-value cache failed.")]
    Cache,

    #[error("Uploading a generation input failed.")]
    Upload,

    #[error("Model forward failed.")]
    Forward,

    #[error("Selecting the next token failed.")]
    NextToken,

    #[error("Next token download does not hold one u32.")]
    TokenBytes,
}
