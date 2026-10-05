mod backend;
mod error;
mod execute;
#[cfg(test)]
mod fake;
mod observe;

pub use backend::Backend;
pub use error::{
    BackendOperation,
    ExecutionFailure,
    RuntimeError,
};
pub use execute::{
    execute,
    execute_observed,
};
pub use observe::{
    CommandContext,
    CpuActivity,
    CpuRecord,
    ExecutionObserver,
    ExecutionOutcome,
};
