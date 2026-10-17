use std::error::Error;

use crate::nn::primitive::{
    BinaryOp,
    CatSpec,
    IndexSelectSpec,
    MatmulSpec,
    ReduceSpec,
    SliceScatterSpec,
    UnaryOp,
};
use crate::nn::{
    AllocatedMemory,
    FloatDType,
    Operand,
    OperandMut,
};

pub trait Backend {
    type Storage;
    type Error: Error + 'static;

    fn device_name(&self) -> String;

    fn current_allocated_memory(&self) -> Option<AllocatedMemory>;

    fn storage_from_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<Self::Storage, Self::Error>;

    fn zeros(
        &mut self,
        nbytes: usize,
    ) -> Result<Self::Storage, Self::Error>;

    fn storage_to_bytes(
        &mut self,
        input: Operand<'_, Self::Storage>,
    ) -> Result<Vec<u8>, Self::Error>;

    fn unary(
        &mut self,
        op: UnaryOp,
        input: Operand<'_, Self::Storage, FloatDType>,
    ) -> Result<Self::Storage, Self::Error>;

    fn binary(
        &mut self,
        op: BinaryOp,
        lhs: Operand<'_, Self::Storage, FloatDType>,
        rhs: Operand<'_, Self::Storage, FloatDType>,
    ) -> Result<Self::Storage, Self::Error>;

    fn reduce(
        &mut self,
        input: Operand<'_, Self::Storage, FloatDType>,
        spec: &ReduceSpec,
    ) -> Result<Self::Storage, Self::Error>;

    fn matmul(
        &mut self,
        lhs: Operand<'_, Self::Storage, FloatDType>,
        rhs: Operand<'_, Self::Storage, FloatDType>,
        spec: &MatmulSpec,
    ) -> Result<Self::Storage, Self::Error>;

    fn copy(
        &mut self,
        input: Operand<'_, Self::Storage>,
    ) -> Result<Self::Storage, Self::Error>;

    fn to_dtype(
        &mut self,
        input: Operand<'_, Self::Storage, FloatDType>,
        dtype: FloatDType,
    ) -> Result<Self::Storage, Self::Error>;

    fn index_select(
        &mut self,
        table: Operand<'_, Self::Storage, FloatDType>,
        indices: Operand<'_, Self::Storage>,
        spec: &IndexSelectSpec,
    ) -> Result<Self::Storage, Self::Error>;

    fn cat(
        &mut self,
        lhs: Operand<'_, Self::Storage, FloatDType>,
        rhs: Operand<'_, Self::Storage, FloatDType>,
        spec: &CatSpec,
    ) -> Result<Self::Storage, Self::Error>;

    fn slice_scatter(
        &mut self,
        target: OperandMut<'_, Self::Storage, FloatDType>,
        src: Operand<'_, Self::Storage, FloatDType>,
        spec: &SliceScatterSpec,
    ) -> Result<(), Self::Error>;
}
