use crate::nn::{
    DType,
    Device,
    Shape,
    Tensor,
    TensorError,
};

pub struct Silu<D: Device> {
    one: Tensor<D>,
}

impl<D: Device> Silu<D> {
    pub fn new(device: &mut D) -> Result<Self, TensorError> {
        Ok(Self {
            one: Tensor::from_bytes(
                device,
                &1.0_f32.to_le_bytes(),
                DType::F32,
                Shape::try_from([1].as_slice())?,
            )?,
        })
    }

    pub fn forward(
        &self,
        device: &mut D,
        x: &Tensor<D>,
    ) -> Result<Tensor<D>, TensorError> {
        let x_f32 = x.to_dtype(device, DType::F32)?;
        let denominator = x_f32.neg(device)?.exp(device)?.add(device, &self.one)?;
        x_f32.div(device, &denominator)?.to_dtype(device, x.dtype())
    }
}

pub fn softmax<D: Device>(
    device: &mut D,
    x: &Tensor<D>,
    dim: usize,
) -> Result<Tensor<D>, TensorError> {
    let x_f32 = x.to_dtype(device, DType::F32)?;
    let row_max = x_f32.amax_keepdim(device, dim)?;
    let exps = x_f32.sub(device, &row_max)?.exp(device)?;
    let row_sum = exps.sum_keepdim(device, dim)?;
    exps.div(device, &row_sum)?.to_dtype(device, x.dtype())
}
