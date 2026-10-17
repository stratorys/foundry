use tracing::error;

use crate::nn::{
    Backend,
    CoreError,
    DType,
    Shape,
    Tensor,
    TensorError,
};

const SEQ_DIM: usize = 2;

pub struct StaticCache<B: Backend> {
    layers: Vec<(Tensor<B>, Tensor<B>)>,
}

impl<B: Backend> StaticCache<B> {
    pub fn new(
        backend: &mut B,
        dtype: DType,
        num_layers: usize,
        batch_size: usize,
        num_key_value_heads: usize,
        max_cache_len: usize,
        head_dim: usize,
    ) -> Result<Self, TensorError<B::Error>> {
        let shape =
            Shape::try_from([batch_size, num_key_value_heads, max_cache_len, head_dim].as_slice())?;
        let layers = (0..num_layers)
            .map(|_| {
                Ok((
                    Tensor::zeros(backend, dtype, shape)?,
                    Tensor::zeros(backend, dtype, shape)?,
                ))
            })
            .collect::<Result<Vec<(Tensor<B>, Tensor<B>)>, TensorError<B::Error>>>()?;
        Ok(Self {
            layers,
        })
    }

    pub fn update(
        &mut self,
        backend: &mut B,
        key_states: &Tensor<B>,
        value_states: &Tensor<B>,
        layer_idx: usize,
        past_seen_tokens: usize,
    ) -> Result<[Tensor<B>; 2], TensorError<B::Error>> {
        if key_states.shape() != value_states.shape() {
            error!(
                message = "Keys and values do not have the same shape.",
                keys = ?key_states.shape().dims(),
                values = ?value_states.shape().dims(),
            );
            return Err(CoreError::KvIncompatible.into());
        }
        if key_states.dtype() != value_states.dtype() {
            error!(
                message = "DTypes do not match.",
                keys = ?key_states.dtype(),
                values = ?value_states.dtype(),
            );
            return Err(CoreError::DTypeMismatch.into());
        }
        let num_layers = self.layers.len();
        let Some((keys, values)) = self.layers.get_mut(layer_idx) else {
            error!(
                message = "Layer index exceeds the cache layers.",
                layer_idx, num_layers,
            );
            return Err(CoreError::CacheLayerOutOfRange.into());
        };
        keys.slice_scatter(backend, key_states, SEQ_DIM, past_seen_tokens)?;
        values.slice_scatter(backend, value_states, SEQ_DIM, past_seen_tokens)?;
        let kv_length = past_seen_tokens
            .checked_add(key_states.shape().size(SEQ_DIM)?)
            .ok_or_else(|| {
                error!(message = "Narrow exceeds the dimension.", past_seen_tokens);
                CoreError::NarrowOutOfBounds
            })?;
        Ok([
            keys.narrow(SEQ_DIM, 0, kv_length)?,
            values.narrow(SEQ_DIM, 0, kv_length)?,
        ])
    }

    pub fn num_layers(&self) -> usize { self.layers.len() }
}
