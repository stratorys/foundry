mod backend;
mod dtype;
mod error;
mod layout;
mod memory;
pub mod primitive;
mod rng;
mod shape;
mod tensor;

pub use self::backend::Backend;
pub use self::dtype::{
    DType,
    FloatDType,
    MAX_ITEMSIZE,
    exact_f32,
};
pub use self::error::{
    CoreError,
    TensorError,
};
pub use self::layout::Layout;
pub use self::memory::{
    AllocatedMemory,
    AllocatedMemorySource,
};
pub use self::rng::SplitMix64;
pub use self::shape::{
    MAX_RANK,
    Shape,
};
pub use self::tensor::{
    Operand,
    OperandMut,
    Tensor,
};
