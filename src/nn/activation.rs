use crate::core::{
    Backend,
    CoreError,
    DType,
    Shape,
    Tensor,
};

pub struct Silu<B: Backend> {
    one: Tensor<B>,
}

impl<B: Backend> Silu<B> {
    pub fn new(backend: &mut B) -> Result<Self, B::Error> {
        Ok(Self {
            one: Tensor::upload(
                backend,
                &1.0_f32.to_le_bytes(),
                DType::F32,
                Shape::try_from([1].as_slice())?,
            )?,
        })
    }

    pub fn forward(
        &self,
        backend: &mut B,
        x: &Tensor<B>,
    ) -> Result<Tensor<B>, B::Error> {
        let x_f32 = x.cast(backend, DType::F32)?;
        let denominator = x_f32.neg(backend)?.exp(backend)?.add(backend, &self.one)?;
        x_f32.div(backend, &denominator)?.cast(backend, x.dtype())
    }
}

pub fn softmax_last_axis<B: Backend>(
    backend: &mut B,
    x: &Tensor<B>,
) -> Result<Tensor<B>, B::Error> {
    let axis = x
        .shape()
        .rank()
        .checked_sub(1)
        .ok_or(CoreError::AxisOutOfRange {
            axis: 0,
            rank: 0,
        })?;
    let x_f32 = x.cast(backend, DType::F32)?;
    let row_max = x_f32.max(backend, axis)?;
    let exps = x_f32.sub(backend, &row_max)?.exp(backend)?;
    let row_sum = exps.sum(backend, axis)?;
    exps.div(backend, &row_sum)?.cast(backend, x.dtype())
}
