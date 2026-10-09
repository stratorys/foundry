use crate::core::primitive::{
    BinaryOp,
    ReduceOp,
    UnaryOp,
};
use crate::core::{
    CoreError,
    DType,
    Layout,
    Shape,
};

pub struct Operand<'storage, S> {
    pub storage: &'storage S,
    pub layout: &'storage Layout,
    pub dtype: DType,
}

pub trait Backend {
    type Storage;
    type Error: From<CoreError>;

    fn upload(
        &mut self,
        bytes: &[u8],
        dtype: DType,
        shape: &Shape,
    ) -> Result<Self::Storage, Self::Error>;

    fn zeros(
        &mut self,
        dtype: DType,
        shape: &Shape,
    ) -> Result<Self::Storage, Self::Error>;

    fn download(
        &mut self,
        input: Operand<'_, Self::Storage>,
    ) -> Result<Vec<u8>, Self::Error>;

    fn unary(
        &mut self,
        op: UnaryOp,
        input: Operand<'_, Self::Storage>,
    ) -> Result<Self::Storage, Self::Error>;

    fn binary(
        &mut self,
        op: BinaryOp,
        lhs: Operand<'_, Self::Storage>,
        rhs: Operand<'_, Self::Storage>,
    ) -> Result<Self::Storage, Self::Error>;

    fn reduce(
        &mut self,
        op: ReduceOp,
        input: Operand<'_, Self::Storage>,
        axis: usize,
    ) -> Result<Self::Storage, Self::Error>;

    fn matmul(
        &mut self,
        lhs: Operand<'_, Self::Storage>,
        rhs: Operand<'_, Self::Storage>,
    ) -> Result<Self::Storage, Self::Error>;

    fn copy(
        &mut self,
        input: Operand<'_, Self::Storage>,
    ) -> Result<Self::Storage, Self::Error>;

    fn cast(
        &mut self,
        input: Operand<'_, Self::Storage>,
        dtype: DType,
    ) -> Result<Self::Storage, Self::Error>;

    fn gather(
        &mut self,
        table: Operand<'_, Self::Storage>,
        indices: Operand<'_, Self::Storage>,
    ) -> Result<Self::Storage, Self::Error>;

    fn concat(
        &mut self,
        lhs: Operand<'_, Self::Storage>,
        rhs: Operand<'_, Self::Storage>,
        axis: usize,
    ) -> Result<Self::Storage, Self::Error>;

    fn slice_update(
        &mut self,
        target: Operand<'_, Self::Storage>,
        update: Operand<'_, Self::Storage>,
        axis: usize,
        start: usize,
    ) -> Result<(), Self::Error>;
}
