use crate::core::{
    CoreError,
    DType,
};

#[derive(Debug, thiserror::Error)]
pub enum CpuError {
    #[error("Byte count of shape {dims:?} overflows usize.")]
    ByteCountOverflow { dims: Vec<usize> },

    #[error("Byte length {bytes} does not match the expected byte length {bytes_expected}.")]
    ByteLengthMismatch { bytes: usize, bytes_expected: usize },

    #[error("Primitive {primitive} does not support dtype {dtype:?} on the CPU.")]
    UnsupportedDType {
        primitive: &'static str,
        dtype: DType,
    },

    #[error("Index of linear element {linear} overflows usize.")]
    IndexOverflow { linear: usize },

    #[error("Element {index} is outside a storage of {bytes_len} bytes.")]
    IndexOutOfStorage { index: usize, bytes_len: usize },

    #[error("Download requires a contiguous layout.")]
    DownloadNonContiguous,

    #[error(transparent)]
    Core(#[from] CoreError),
}
