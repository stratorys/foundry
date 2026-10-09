use crate::core::{
    Backend,
    CoreError,
    DType,
    Shape,
    Tensor,
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
    ) -> Result<Self, CoreError> {
        let dim_f32 = exact_f32(weight.shape().element_count())?;
        let scalar_shape = Shape::try_from([1].as_slice())?;
        Ok(Self {
            eps: Tensor::upload(backend, &eps.to_le_bytes(), DType::F32, scalar_shape)?,
            dim_inv: Tensor::upload(
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
    ) -> Result<Tensor<B>, CoreError> {
        let axis = x.shape().last_axis()?;
        let x_f32 = x.cast(backend, DType::F32)?;
        let scale = x_f32
            .mul(backend, &x_f32)?
            .sum(backend, axis)?
            .mul(backend, &self.dim_inv)?
            .add(backend, &self.eps)?
            .sqrt(backend)?
            .recip(backend)?;
        x_f32
            .mul(backend, &scale)?
            .cast(backend, x.dtype())?
            .mul(backend, &self.weight)
    }
}
