use std::error::Error;
use std::path::PathBuf;
use std::{
    fmt,
    io,
};

#[derive(Debug)]
pub enum ConfigError {
    Parse(serde_json::Error),
    Unsupported {
        field: &'static str,
        expected: String,
        actual: String,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Parse(error) => write!(formatter, "invalid config.json: {error}"),
            Self::Unsupported {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "unsupported configuration: {field} is {actual}, expected {expected}"
            ),
        }
    }
}

impl Error for ConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Parse(error) => Some(error),
            Self::Unsupported {
                ..
            } => None,
        }
    }
}

#[derive(Debug)]
pub enum SafetensorsError {
    TooShort {
        file_bytes: u64,
    },
    HeaderLength {
        header_bytes: u64,
        limit_bytes: u64,
        file_bytes: u64,
    },
    Header(serde_json::Error),
    Metadata,
    Dtype {
        name: String,
        dtype: String,
    },
    ShapeOverflow {
        name: String,
    },
    InvalidOffsets {
        name: String,
        start: u64,
        end: u64,
    },
    SizeMismatch {
        name: String,
        expected_bytes: u64,
        actual_bytes: u64,
    },
    Gap {
        name: String,
        expected_start: u64,
        actual_start: u64,
    },
    Coverage {
        covered_bytes: u64,
        data_bytes: u64,
    },
}

impl fmt::Display for SafetensorsError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::TooShort {
                file_bytes,
            } => write!(
                formatter,
                "safetensors file of {file_bytes} bytes has no header length"
            ),
            Self::HeaderLength {
                header_bytes,
                limit_bytes,
                file_bytes,
            } => write!(
                formatter,
                "safetensors header of {header_bytes} bytes exceeds the {limit_bytes} byte limit \
                 or the {file_bytes} byte file"
            ),
            Self::Header(error) => write!(formatter, "invalid safetensors header: {error}"),
            Self::Metadata => write!(
                formatter,
                "safetensors __metadata__ must map strings to strings"
            ),
            Self::Dtype {
                name,
                dtype,
            } => write!(formatter, "tensor {name} has unsupported dtype {dtype}"),
            Self::ShapeOverflow {
                name,
            } => write!(formatter, "tensor {name} shape overflows"),
            Self::InvalidOffsets {
                name,
                start,
                end,
            } => write!(
                formatter,
                "tensor {name} has invalid offsets [{start}, {end})"
            ),
            Self::SizeMismatch {
                name,
                expected_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "tensor {name} spans {actual_bytes} bytes but its shape requires {expected_bytes}"
            ),
            Self::Gap {
                name,
                expected_start,
                actual_start,
            } => write!(
                formatter,
                "tensor {name} starts at {actual_start}, expected {expected_start} for contiguous \
                 data"
            ),
            Self::Coverage {
                covered_bytes,
                data_bytes,
            } => write!(
                formatter,
                "tensors cover {covered_bytes} bytes of a {data_bytes} byte data region"
            ),
        }
    }
}

impl Error for SafetensorsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Header(error) => Some(error),
            Self::TooShort {
                ..
            }
            | Self::HeaderLength {
                ..
            }
            | Self::Metadata
            | Self::Dtype {
                ..
            }
            | Self::ShapeOverflow {
                ..
            }
            | Self::InvalidOffsets {
                ..
            }
            | Self::SizeMismatch {
                ..
            }
            | Self::Gap {
                ..
            }
            | Self::Coverage {
                ..
            } => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum CheckpointError {
    Missing {
        name: String,
    },
    Unexpected {
        name: String,
    },
    Dtype {
        name: String,
        expected: &'static str,
        actual: &'static str,
    },
    Shape {
        name: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    Overflow,
}

impl fmt::Display for CheckpointError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Missing {
                name,
            } => write!(formatter, "checkpoint tensor {name} is missing"),
            Self::Unexpected {
                name,
            } => write!(
                formatter,
                "checkpoint tensor {name} is not part of the model"
            ),
            Self::Dtype {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "tensor {name} has dtype {actual}, expected {expected}"
            ),
            Self::Shape {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "tensor {name} has shape {actual:?}, expected {expected:?}"
            ),
            Self::Overflow => write!(formatter, "checkpoint dimensions overflow"),
        }
    }
}

impl Error for CheckpointError {}

#[derive(Debug)]
pub enum ManifestError {
    Parse(serde_json::Error),
    Identity {
        field: &'static str,
        expected: String,
        actual: String,
    },
    TensorTable {
        name: String,
    },
    FileSize {
        path: String,
        expected_bytes: u64,
        actual_bytes: u64,
    },
}

impl fmt::Display for ManifestError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Parse(error) => write!(formatter, "invalid manifest: {error}"),
            Self::Identity {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "manifest {field} is {actual}, expected {expected}"
            ),
            Self::TensorTable {
                name,
            } => write!(
                formatter,
                "safetensors header disagrees with the manifest at tensor {name}"
            ),
            Self::FileSize {
                path,
                expected_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "{path} has {actual_bytes} bytes, the manifest records {expected_bytes}"
            ),
        }
    }
}

impl Error for ManifestError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Parse(error) => Some(error),
            Self::Identity {
                ..
            }
            | Self::TensorTable {
                ..
            }
            | Self::FileSize {
                ..
            } => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum RequestError {
    EmptyPrompt,
    PromptTooLong {
        tokens: usize,
        limit: usize,
    },
    CapacityExceeded {
        position: usize,
        requested: usize,
        capacity: usize,
    },
    NoTokensRequested,
    TokenOutOfRange {
        index: usize,
        token: u32,
        vocab: u32,
    },
    SessionNotFresh {
        position: usize,
    },
    NotPrepared,
}

impl fmt::Display for RequestError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::EmptyPrompt => write!(formatter, "the prompt is empty"),
            Self::PromptTooLong {
                tokens,
                limit,
            } => write!(
                formatter,
                "the prompt has {tokens} tokens, the limit is {limit}"
            ),
            Self::CapacityExceeded {
                position,
                requested,
                capacity,
            } => write!(
                formatter,
                "{requested} positions from position {position} exceed the {capacity} position KV \
                 cache"
            ),
            Self::NoTokensRequested => write!(formatter, "at least one token must be generated"),
            Self::TokenOutOfRange {
                index,
                token,
                vocab,
            } => write!(
                formatter,
                "token {token} at index {index} is outside the {vocab} token vocabulary"
            ),
            Self::SessionNotFresh {
                position,
            } => write!(
                formatter,
                "generation requires a fresh session, this one is at position {position}"
            ),
            Self::NotPrepared => write!(formatter, "no prompt was prepared for generation"),
        }
    }
}

impl Error for RequestError {}

#[derive(Debug)]
pub enum LlamaError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    Config(ConfigError),
    Safetensors(SafetensorsError),
    Checkpoint(CheckpointError),
    Manifest(ManifestError),
    Request(RequestError),
    #[cfg(target_os = "macos")]
    Gpu(crate::metal::GpuError),
}

impl LlamaError {
    pub(crate) fn io(
        path: impl Into<PathBuf>,
        source: io::Error,
    ) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

impl fmt::Display for LlamaError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Io {
                path,
                source,
            } => write!(formatter, "cannot read {}: {source}", path.display()),
            Self::Config(error) => error.fmt(formatter),
            Self::Safetensors(error) => error.fmt(formatter),
            Self::Checkpoint(error) => error.fmt(formatter),
            Self::Manifest(error) => error.fmt(formatter),
            Self::Request(error) => error.fmt(formatter),
            #[cfg(target_os = "macos")]
            Self::Gpu(error) => error.fmt(formatter),
        }
    }
}

impl Error for LlamaError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io {
                source, ..
            } => Some(source),
            Self::Config(error) => Some(error),
            Self::Safetensors(error) => Some(error),
            Self::Checkpoint(error) => Some(error),
            Self::Manifest(error) => Some(error),
            Self::Request(error) => Some(error),
            #[cfg(target_os = "macos")]
            Self::Gpu(error) => Some(error),
        }
    }
}

impl From<ConfigError> for LlamaError {
    fn from(error: ConfigError) -> Self { Self::Config(error) }
}

impl From<SafetensorsError> for LlamaError {
    fn from(error: SafetensorsError) -> Self { Self::Safetensors(error) }
}

impl From<CheckpointError> for LlamaError {
    fn from(error: CheckpointError) -> Self { Self::Checkpoint(error) }
}

impl From<ManifestError> for LlamaError {
    fn from(error: ManifestError) -> Self { Self::Manifest(error) }
}

impl From<RequestError> for LlamaError {
    fn from(error: RequestError) -> Self { Self::Request(error) }
}

#[cfg(target_os = "macos")]
impl From<crate::metal::GpuError> for LlamaError {
    fn from(error: crate::metal::GpuError) -> Self { Self::Gpu(error) }
}
