#![cfg(target_os = "macos")]

mod backend;
mod device;
mod error;
mod pipeline;
mod submission;

pub use backend::{
    MetalBackend,
    MetalBuffer,
    MetalEvent,
    MetalStaging,
    MetalStream,
};
pub use error::MetalError;
pub use pipeline::reference_checksum;
