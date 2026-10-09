use std::io;
use std::path::PathBuf;

use crate::core::CoreError;

#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct IoError(#[from] io::Error);

impl PartialEq for IoError {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.0.kind() == other.0.kind()
    }
}

impl Eq for IoError {}

#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct JsonError(#[from] serde_json::Error);

impl PartialEq for JsonError {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        (self.0.classify(), self.0.line(), self.0.column())
            == (other.0.classify(), other.0.line(), other.0.column())
    }
}

impl Eq for JsonError {}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum WeightsError {
    #[error("I/O on {path:?} failed.")]
    Io {
        path: PathBuf,
        #[source]
        error: IoError,
    },

    #[error("Index file {path:?} of {bytes} bytes exceeds the maximum of {bytes_max} bytes.")]
    IndexTooLarge {
        path: PathBuf,
        bytes: u64,
        bytes_max: u64,
    },

    #[error("Index file {path:?} is not valid JSON.")]
    IndexJson {
        path: PathBuf,
        #[source]
        error: JsonError,
    },

    #[error("Shard name {name:?} is not a plain file name.")]
    InvalidShardName { name: String },

    #[error("Header of {path:?} is truncated; the file has {bytes} bytes.")]
    TruncatedHeader { path: PathBuf, bytes: usize },

    #[error("Header of {path:?} declares {bytes} bytes; the maximum is {bytes_max}.")]
    HeaderTooLarge {
        path: PathBuf,
        bytes: u64,
        bytes_max: usize,
    },

    #[error("Header of {path:?} is not valid JSON.")]
    HeaderJson {
        path: PathBuf,
        #[source]
        error: JsonError,
    },

    #[error("Header entry of tensor {name} in {path:?} is malformed.")]
    TensorHeaderJson {
        path: PathBuf,
        name: String,
        #[source]
        error: JsonError,
    },

    #[error("Tensor {name} has unknown dtype {dtype:?}.")]
    UnknownDType { name: String, dtype: String },

    #[error("Tensor {name} has an invalid shape.")]
    InvalidShape {
        name: String,
        #[source]
        error: CoreError,
    },

    #[error("Tensor {name} has data offsets [{begin}, {end}] with begin after end.")]
    InvalidRange {
        name: String,
        begin: usize,
        end: usize,
    },

    #[error("Tensor {name} ends at byte {end}, outside the data region of {bytes_data} bytes.")]
    OutOfDataRegion {
        name: String,
        end: usize,
        bytes_data: usize,
    },

    #[error("Byte count of tensor {name} overflows usize.")]
    ByteCountOverflow { name: String },

    #[error("Tensor {name} has {bytes} bytes; its shape and dtype require {bytes_expected}.")]
    SizeMismatch {
        name: String,
        bytes: usize,
        bytes_expected: usize,
    },

    #[error("Tensors {first} and {second} have overlapping byte ranges.")]
    OverlappingRanges { first: String, second: String },

    #[error("Tensor {name} appears in more than one shard.")]
    DuplicateTensor { name: String },

    #[error("Index maps tensor {name} to shard {shard}, which does not contain it.")]
    IndexMismatch { name: String, shard: String },

    #[error("Tensor {name} is not in the weights.")]
    TensorNotFound { name: String },
}
