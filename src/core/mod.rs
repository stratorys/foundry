mod backend;
mod dtype;
mod error;
mod layout;
mod memory;
pub mod primitive;
mod shape;
mod tensor;

pub use self::backend::Backend;
pub use self::dtype::{
    DTYPE_SIZE_BYTES_MAX,
    DType,
    FloatDType,
    exact_f32,
};
pub use self::error::{
    CoreError,
    TensorError,
};
pub use self::layout::Layout;
pub use self::memory::{
    PeakMemory,
    PeakMemorySource,
};
pub use self::shape::{
    RANK_MAX,
    Shape,
};
pub use self::tensor::{
    Operand,
    OperandMut,
    Tensor,
};
