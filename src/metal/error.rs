use tracing::error;

use crate::nn::DType;
use crate::nn::primitive::Op;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MetalError {
    #[error("No default Metal device is available.")]
    DeviceUnavailable,

    #[error("Metal command queue creation failed.")]
    QueueCreation,

    #[error("Metal library compilation failed.")]
    LibraryCompilation,

    #[error("Buffer exceeds the maximum Metal buffer length.")]
    BufferTooLarge,

    #[error("Metal buffer allocation failed.")]
    BufferAllocation,

    #[error("Metal allocated size does not fit in u64.")]
    AllocatedSizeOverflow,

    #[error("Metal command buffer creation failed.")]
    CommandBufferCreation,

    #[error("Metal compute command encoder creation failed.")]
    ComputeEncoderCreation,

    #[error("Kernel is not in the Metal library.")]
    KernelNotFound,

    #[error("Metal pipeline creation failed.")]
    PipelineCreation,

    #[error("Value does not fit in a 32-bit kernel index.")]
    KernelIndexTooLarge,

    #[error("Kernel allows too few threads per threadgroup.")]
    ThreadgroupTooSmall,

    #[error("Metal command buffer failed.")]
    CommandBufferFailed,

    #[error("Storage was produced by another Metal device instance.")]
    ForeignStorage,

    #[error("Reading bytes exceeds the buffer length.")]
    ToBytesOutOfBounds,

    #[error("Operation does not match its operands.")]
    InvalidOperation,

    #[error("Kernel needs a float dtype.")]
    DTypeNotFloat,
}

pub fn not_float(dtype: DType) -> MetalError {
    error!(message = "Kernel needs a float dtype.", ?dtype);
    MetalError::DTypeNotFloat
}

pub fn invalid_operation(op: Op) -> MetalError {
    error!(message = "Operation does not match its operands.", ?op);
    MetalError::InvalidOperation
}
