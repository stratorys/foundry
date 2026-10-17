use crate::core::{
    Backend,
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
    ) -> Result<Tensor<B>, TensorError<B::Error>> {
        x.matmul(backend, &self.weight.permute(&[1, 0])?)
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
    ) -> Result<Tensor<B>, TensorError<B::Error>> {
        self.weight.index_select(backend, 0, input_ids)
    }
}
