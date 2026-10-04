mod backend;
mod error;
mod execute;
#[cfg(test)]
mod fake;

pub use backend::Backend;
pub use error::{
    BackendOperation,
    ExecutionFailure,
    RuntimeError,
};
pub use execute::execute;
