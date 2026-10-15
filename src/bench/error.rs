#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BenchError {
    #[error("Benchmark holds neither prompt nor generated tokens.")]
    NoToken,

    #[error("Drawing a random token ID failed.")]
    Random,

    #[error("Creating the decoder failed.")]
    Decoder,

    #[error("Benchmark step failed.")]
    Step,

    #[error("Token position overflows usize.")]
    PositionOverflow,

    #[error("Step duration does not fit in u64 nanoseconds.")]
    DurationOverflow,

    #[error("Run count overflows u32.")]
    RunCountOverflow,

    #[error("Statistics need at least one sample.")]
    NoSample,

    #[error("Sample count does not fit in u32.")]
    SampleCountOverflow,
}
