use crate::core::{
    CoreError,
    DType,
};

#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("No default Metal device is available.")]
    DeviceUnavailable,

    #[error("Metal command queue creation failed.")]
    QueueCreation,

    #[error("Metal library compilation failed: {message}")]
    LibraryCompilation { message: String },

    #[error("Buffer of {bytes} bytes exceeds the maximum buffer length {bytes_max}.")]
    BufferTooLarge { bytes: usize, bytes_max: usize },

    #[error("Metal buffer allocation of {bytes} bytes failed.")]
    BufferAllocation { bytes: usize },

    #[error("Byte count of shape {dims:?} overflows usize.")]
    ByteCountOverflow { dims: Vec<usize> },

    #[error("Byte length {bytes} does not match the expected byte length {bytes_expected}.")]
    ByteLengthMismatch { bytes: usize, bytes_expected: usize },

    #[error("Metal command buffer creation failed.")]
    CommandBufferCreation,

    #[error("Metal blit command encoder creation failed.")]
    BlitEncoderCreation,

    #[error("Metal compute command encoder creation failed.")]
    ComputeEncoderCreation,

    #[error("Kernel {name} is not in the Metal library.")]
    KernelNotFound { name: &'static str },

    #[error("Metal pipeline creation for kernel {name} failed: {message}")]
    PipelineCreation { name: &'static str, message: String },

    #[error("Value {value} does not fit in a 32-bit kernel index.")]
    IndexTooLarge { value: usize },

    #[error("Primitive {primitive} does not support dtype {dtype:?} on Metal.")]
    UnsupportedDType {
        primitive: &'static str,
        dtype: DType,
    },

    #[error(
        "Kernel {name} allows {threads} threads per threadgroup; {threads_required} are required."
    )]
    ThreadgroupTooSmall {
        name: &'static str,
        threads: usize,
        threads_required: usize,
    },

    #[error("Metal command buffer failed: {message}")]
    CommandBufferFailed { message: String },

    #[error("Download requires a contiguous layout.")]
    DownloadNonContiguous,

    #[error(
        "Download of {bytes} bytes from byte offset {bytes_offset} exceeds buffer length \
         {bytes_len}."
    )]
    DownloadOutOfBounds {
        bytes: usize,
        bytes_offset: usize,
        bytes_len: usize,
    },

    #[error("Primitive {primitive} is not implemented on Metal.")]
    NotImplemented { primitive: &'static str },

    #[error(transparent)]
    Core(#[from] CoreError),
}
