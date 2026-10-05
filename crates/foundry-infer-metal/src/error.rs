use std::error::Error;
use std::fmt;

use foundry_infer_core::{
    ByteSize,
    Id,
    OpId,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalError {
    DeviceUnavailable,
    UnsupportedDevice {
        name: String,
        requirement: &'static str,
    },
    EmptyAllocation,
    AllocationTooLarge {
        requested: ByteSize,
        limit: ByteSize,
    },
    AllocationFailed {
        bytes: ByteSize,
    },
    CopyTooLarge {
        source: ByteSize,
        destination: ByteSize,
    },
    PayloadTooLarge {
        bytes: ByteSize,
    },
    TooManyWeights {
        count: usize,
    },
    QueueCreation,
    CommandBufferCreation,
    EncoderCreation,
    EventCreation,
    ShaderCompilation(String),
    MissingKernel,
    PipelineCreation(String),
    Gpu(String),
    GpuWithoutError,
    NotCompleted,
    UnknownOp(OpId),
    #[cfg(test)]
    InjectedFault,
}

impl fmt::Display for MetalError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::DeviceUnavailable => write!(formatter, "no Metal device is available"),
            Self::UnsupportedDevice {
                name,
                requirement,
            } => write!(
                formatter,
                "Metal device {name} does not support {requirement}"
            ),
            Self::EmptyAllocation => write!(formatter, "Metal buffers cannot be empty"),
            Self::AllocationTooLarge {
                requested,
                limit,
            } => write!(
                formatter,
                "Metal allocation of {requested} exceeds the {limit} buffer length limit"
            ),
            Self::AllocationFailed {
                bytes,
            } => write!(formatter, "Metal failed to allocate {bytes}"),
            Self::CopyTooLarge {
                source,
                destination,
            } => write!(
                formatter,
                "copy source of {source} exceeds its {destination} destination"
            ),
            Self::PayloadTooLarge {
                bytes,
            } => write!(
                formatter,
                "synthetic compute payload of {bytes} exceeds 4 GiB"
            ),
            Self::TooManyWeights {
                count,
            } => write!(
                formatter,
                "synthetic compute over {count} weights overflows its checksum buffer"
            ),
            Self::QueueCreation => write!(formatter, "Metal command queue creation failed"),
            Self::CommandBufferCreation => {
                write!(formatter, "Metal command buffer creation failed")
            }
            Self::EncoderCreation => write!(formatter, "Metal command encoder creation failed"),
            Self::EventCreation => write!(formatter, "Metal event creation failed"),
            Self::ShaderCompilation(message) => {
                write!(formatter, "Metal shader compilation failed: {message}")
            }
            Self::MissingKernel => write!(formatter, "Metal kernel is missing from its library"),
            Self::PipelineCreation(message) => {
                write!(formatter, "Metal pipeline creation failed: {message}")
            }
            Self::Gpu(message) => write!(formatter, "Metal command buffer failed: {message}"),
            Self::GpuWithoutError => {
                write!(formatter, "Metal command buffer failed without an error")
            }
            Self::NotCompleted => write!(formatter, "Metal command buffer has not completed"),
            Self::UnknownOp(op) => write!(formatter, "op {} was never launched", op.index()),
            #[cfg(test)]
            Self::InjectedFault => write!(formatter, "injected fault"),
        }
    }
}

impl Error for MetalError {}
