use crate::nn::{
    Backend,
    Shape,
    Tensor,
    TensorError,
};

pub struct Linear<B: Backend> {
    weight: Tensor<B>,
}

impl<B: Backend> Linear<B> {
    pub fn new(weight: Tensor<B>) -> Self {
        Self {
            weight,
        }
    }

    pub fn forward(
        &self,
        backend: &mut B,
        x: &Tensor<B>,
    ) -> Result<Tensor<B>, TensorError> {
        let weight_t = self.weight.permute(&[1, 0])?;
        let dims = x.shape().dims();
        let in_features = x.shape().size(x.shape().last_dim()?)?;
        let rows = x.shape().numel().checked_div(in_features).unwrap_or(0);
        let out_features = weight_t.shape().size(1)?;
        let output_dims = dims
            .iter()
            .take(dims.len().saturating_sub(1))
            .copied()
            .chain([out_features])
            .collect::<Vec<usize>>();
        x.reshape(Shape::try_from([rows, in_features].as_slice())?)?
            .matmul(backend, &weight_t)?
            .reshape(Shape::try_from(output_dims.as_slice())?)
    }
}

pub struct Embedding<B: Backend> {
    weight: Tensor<B>,
}

impl<B: Backend> Embedding<B> {
    pub fn new(weight: Tensor<B>) -> Self {
        Self {
            weight,
        }
    }

    pub fn forward(
        &self,
        backend: &mut B,
        input_ids: &Tensor<B>,
    ) -> Result<Tensor<B>, TensorError> {
        self.weight.index_select(backend, 0, input_ids)
    }

    pub fn weight(&self) -> &Tensor<B> { &self.weight }
}
