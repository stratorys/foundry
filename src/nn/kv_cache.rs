use tracing::error;

use crate::core::{
    Backend,
    CoreError,
    DType,
    Shape,
    Tensor,
    TensorError,
};

const SEQ_DIM: usize = 2;

pub struct KvCache<B: Backend> {
    keys: Tensor<B>,
    values: Tensor<B>,
}

impl<B: Backend> KvCache<B> {
    pub fn new(
        backend: &mut B,
        dtype: DType,
        batch_size: usize,
        kv_heads: usize,
        max_cache_len: usize,
        head_dim: usize,
    ) -> Result<Self, TensorError<B::Error>> {
        let shape = Shape::try_from([batch_size, kv_heads, max_cache_len, head_dim].as_slice())?;
        Ok(Self {
            keys: Tensor::zeros(backend, dtype, shape)?,
            values: Tensor::zeros(backend, dtype, shape)?,
        })
    }

    pub fn update(
        &mut self,
        backend: &mut B,
        keys: &Tensor<B>,
        values: &Tensor<B>,
        past_seen_tokens: usize,
    ) -> Result<(), TensorError<B::Error>> {
        if keys.shape() != values.shape() {
            error!(
                message = "Keys and values do not have the same shape.",
                keys = ?keys.shape().dims(),
                values = ?values.shape().dims(),
            );
            return Err(CoreError::KvIncompatible.into());
        }
        if keys.dtype() != values.dtype() {
            error!(
                message = "DTypes do not match.",
                keys = ?keys.dtype(),
                values = ?values.dtype(),
            );
            return Err(CoreError::DTypeMismatch.into());
        }
        self.keys
            .slice_scatter(backend, keys, SEQ_DIM, past_seen_tokens)?;
        self.values
            .slice_scatter(backend, values, SEQ_DIM, past_seen_tokens)
    }

    pub fn keys(&self) -> &Tensor<B> { &self.keys }

    pub fn values(&self) -> &Tensor<B> { &self.values }
}
