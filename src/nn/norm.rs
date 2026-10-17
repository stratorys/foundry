use crate::nn::{
    DType,
    Device,
    Module,
    Shape,
    Tensor,
    TensorError,
    exact_f32,
};

pub struct RmsNorm<D: Device> {
    weight: Tensor<D>,
    eps: Tensor<D>,
    dim_inv: Tensor<D>,
}

impl<D: Device> RmsNorm<D> {
    pub fn new(
        device: &mut D,
        weight: Tensor<D>,
        eps: f32,
    ) -> Result<Self, TensorError> {
        let dim_f32 = exact_f32(weight.shape().numel())?;
        let scalar_shape = Shape::try_from([1].as_slice())?;
        Ok(Self {
            eps: Tensor::from_bytes(device, &eps.to_le_bytes(), DType::F32, scalar_shape)?,
            dim_inv: Tensor::from_bytes(
                device,
                &dim_f32.recip().to_le_bytes(),
                DType::F32,
                scalar_shape,
            )?,
            weight,
        })
    }
}

impl<D: Device> Module<D> for RmsNorm<D> {
    fn forward(
        &self,
        device: &mut D,
        x: &Tensor<D>,
    ) -> Result<Tensor<D>, TensorError> {
        let dim = x.shape().last_dim()?;
        let x_f32 = x.to_dtype(device, DType::F32)?;
        let scale = x_f32
            .mul(device, &x_f32)?
            .sum_keepdim(device, dim)?
            .mul(device, &self.dim_inv)?
            .add(device, &self.eps)?
            .sqrt(device)?
            .reciprocal(device)?;
        x_f32
            .mul(device, &scale)?
            .to_dtype(device, x.dtype())?
            .mul(device, &self.weight)
    }
}
