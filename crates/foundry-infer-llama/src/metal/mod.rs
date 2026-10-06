mod context;
mod error;
mod kernels;
mod model;
mod session;
mod weights;

pub use error::GpuError;
pub use model::{
    LlamaModel,
    ModelMemory,
};
pub use session::{
    CONTEXT_CAPACITY,
    PREFILL_CAPACITY,
    Session,
    SessionMemory,
    Stage,
};

#[cfg(test)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "test shapes are small constants"
)]
mod tests;

#[cfg(test)]
#[expect(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::indexing_slicing,
    reason = "the CPU reference model indexes small test tensors"
)]
mod model_tests;
