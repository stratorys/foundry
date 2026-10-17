use crate::core::{
    Backend,
    DType,
    Shape,
    Tensor,
    TensorError,
    exact_f32,
};

pub struct RmsNorm<B: Backend> {
    weight: Tensor<B>,
    eps: Tensor<B>,
    dim_inv: Tensor<B>,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(
        backend: &mut B,
        weight: Tensor<B>,
        eps: f32,
    ) -> Result<Self, TensorError<B::Error>> {
        let dim_f32 = exact_f32(weight.shape().numel())?;
        let scalar_shape = Shape::try_from([1].as_slice())?;
        Ok(Self {
            eps: Tensor::from_bytes(backend, &eps.to_le_bytes(), DType::F32, scalar_shape)?,
            dim_inv: Tensor::from_bytes(
                backend,
                &dim_f32.recip().to_le_bytes(),
                DType::F32,
                scalar_shape,
            )?,
            weight,
        })
    }

    pub fn forward(
        &self,
        backend: &mut B,
        x: &Tensor<B>,
    ) -> Result<Tensor<B>, TensorError<B::Error>> {
        let dim = x.shape().last_dim()?;
        let x_f32 = x.to_dtype(backend, DType::F32)?;
        let scale = x_f32
            .mul(backend, &x_f32)?
            .sum_keepdim(backend, dim)?
            .mul(backend, &self.dim_inv)?
            .add(backend, &self.eps)?
            .sqrt(backend)?
            .reciprocal(backend)?;
        x_f32
            .mul(backend, &scale)?
            .to_dtype(backend, x.dtype())?
            .mul(backend, &self.weight)
    }
}
