use std::fmt::Display;

use crate::core::primitive::{
    BinaryOp,
    ConcatSpec,
    GatherSpec,
    MatmulSpec,
    ReduceSpec,
    SliceUpdateSpec,
    UnaryOp,
};
use crate::core::{
    FloatDType,
    Operand,
    OperandMut,
};

pub trait Backend {
    type Storage;
    type Error: Display;

    fn upload(
        &mut self,
        bytes: &[u8],
    ) -> Result<Self::Storage, Self::Error>;

    fn zeros(
        &mut self,
        byte_len: usize,
    ) -> Result<Self::Storage, Self::Error>;

    fn download(
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

    fn cast(
        &mut self,
        input: Operand<'_, Self::Storage, FloatDType>,
        dtype: FloatDType,
    ) -> Result<Self::Storage, Self::Error>;

    fn gather(
        &mut self,
        table: Operand<'_, Self::Storage, FloatDType>,
        indices: Operand<'_, Self::Storage>,
        spec: &GatherSpec,
    ) -> Result<Self::Storage, Self::Error>;

    fn concat(
        &mut self,
        lhs: Operand<'_, Self::Storage, FloatDType>,
        rhs: Operand<'_, Self::Storage, FloatDType>,
        spec: &ConcatSpec,
    ) -> Result<Self::Storage, Self::Error>;

    fn slice_update(
        &mut self,
        target: OperandMut<'_, Self::Storage, FloatDType>,
        update: Operand<'_, Self::Storage, FloatDType>,
        spec: &SliceUpdateSpec,
    ) -> Result<(), Self::Error>;
}
