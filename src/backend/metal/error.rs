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

    #[error("Metal command buffer creation failed.")]
    CommandBufferCreation,

    #[error("Metal blit command encoder creation failed.")]
    BlitEncoderCreation,

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

    #[error("Storage was produced by another Metal backend instance.")]
    ForeignStorage,

    #[error("Download exceeds the buffer length.")]
    DownloadOutOfBounds,
}
