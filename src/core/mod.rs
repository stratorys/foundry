mod backend;
mod dtype;
mod error;
mod layout;
pub mod primitive;
mod shape;
mod tensor;

pub use self::backend::Backend;
pub use self::dtype::DType;
pub use self::error::CoreError;
pub use self::layout::Layout;
pub use self::shape::{
    RANK_MAX,
    Shape,
};
pub use self::tensor::{
    Operand,
    OperandMut,
    Tensor,
};
