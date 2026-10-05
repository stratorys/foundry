#![cfg(target_os = "macos")]

mod backend;
mod clock;
mod device;
mod diagnostics;
mod error;
mod pipeline;
mod submission;
mod tracked;

pub use backend::{
    MetalBackend,
    MetalBuffer,
    MetalEvent,
    MetalStaging,
    MetalStream,
};
pub use clock::{
    HostClock,
    HostInstant,
};
pub use diagnostics::{
    AllocationCategory,
    AllocationChange,
    AllocationEvent,
    Boundary,
    CHECKSUM_KERNEL,
    CommandLabel,
    DiagnosticLimits,
    GpuInterval,
    GpuTiming,
    GpuWork,
    MemorySnapshot,
    MetalDiagnostics,
    TimestampFault,
    Usage,
};
pub use error::MetalError;
pub use pipeline::reference_checksum;
