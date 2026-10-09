mod dtype;
mod error;
mod layout;
pub mod primitive;
mod shape;

pub use self::dtype::DType;
pub use self::error::CoreError;
pub use self::layout::Layout;
pub use self::shape::{
    RANK_MAX,
    Shape,
};
