use crate::core::{
    Backend,
    Tensor,
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
    ) -> Result<Tensor<B>, B::Error> {
        x.matmul(backend, &self.weight.permute(&[1, 0])?)
    }
}

pub struct Embedding<B: Backend> {
    table: Tensor<B>,
}

impl<B: Backend> Embedding<B> {
    pub fn new(table: Tensor<B>) -> Self {
        Self {
            table,
        }
    }

    pub fn forward(
        &self,
        backend: &mut B,
        token_ids: &Tensor<B>,
    ) -> Result<Tensor<B>, B::Error> {
        self.table.gather(backend, token_ids)
    }
}
