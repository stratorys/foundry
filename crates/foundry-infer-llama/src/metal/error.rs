use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuError {
    DeviceUnavailable,
    UnsupportedDevice {
        name: String,
        requirement: &'static str,
    },
    QueueCreation,
    CommandBufferCreation,
    EncoderCreation,
    ShaderCompilation(String),
    MissingKernel(&'static str),
    PipelineCreation {
        kernel: &'static str,
        message: String,
    },
    ThreadgroupTooLarge {
        kernel: &'static str,
        required: usize,
        limit: usize,
    },
    AllocationTooLarge {
        bytes: usize,
        limit: usize,
    },
    AllocationFailed {
        bytes: usize,
    },
    Execution(String),
    ExecutionWithoutError,
    NotCompleted,
    Overflow {
        what: &'static str,
    },
    OutOfBounds {
        what: &'static str,
    },
}

impl fmt::Display for GpuError {
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
            Self::QueueCreation => write!(formatter, "Metal command queue creation failed"),
            Self::CommandBufferCreation => {
                write!(formatter, "Metal command buffer creation failed")
            }
            Self::EncoderCreation => write!(formatter, "Metal compute encoder creation failed"),
            Self::ShaderCompilation(message) => {
                write!(formatter, "Metal shader compilation failed: {message}")
            }
            Self::MissingKernel(kernel) => write!(formatter, "Metal kernel {kernel} is missing"),
            Self::PipelineCreation {
                kernel,
                message,
            } => write!(
                formatter,
                "Metal pipeline {kernel} creation failed: {message}"
            ),
            Self::ThreadgroupTooLarge {
                kernel,
                required,
                limit,
            } => write!(
                formatter,
                "Metal kernel {kernel} needs {required} threads per threadgroup, the limit is \
                 {limit}"
            ),
            Self::AllocationTooLarge {
                bytes,
                limit,
            } => write!(
                formatter,
                "Metal allocation of {bytes} bytes exceeds the {limit} byte buffer limit"
            ),
            Self::AllocationFailed {
                bytes,
            } => write!(formatter, "Metal allocation of {bytes} bytes failed"),
            Self::Execution(message) => write!(formatter, "Metal execution failed: {message}"),
            Self::ExecutionWithoutError => {
                write!(formatter, "Metal execution failed without an error")
            }
            Self::NotCompleted => write!(formatter, "Metal command buffer did not complete"),
            Self::Overflow {
                what,
            } => write!(formatter, "{what} overflows"),
            Self::OutOfBounds {
                what,
            } => write!(formatter, "{what} is out of bounds"),
        }
    }
}

impl Error for GpuError {}
