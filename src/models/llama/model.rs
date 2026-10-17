use std::num::NonZeroUsize;

use tracing::error;

use crate::core::{
    Backend,
    DType,
    Shape,
    Tensor,
    TensorError,
};
use crate::generate::CausalLm;
use crate::models::llama::rope::{
    LlamaRope,
    RopeAngles,
};
use crate::models::llama::{
    LlamaConfig,
    LlamaError,
    LlamaLayerWeights,
    LlamaWeights,
};
use crate::nn::{
    Attention,
    KvCache,
};

const CACHE_DTYPE: DType = DType::BF16;
const SEQ_DIM: usize = 1;
const HEAD_MAJOR: [usize; 4] = [0, 2, 1, 3];

pub struct Llama<B: Backend> {
    weights: LlamaWeights<B>,
    rope: LlamaRope<B>,
    attention: Attention<B>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    max_cache_len: usize,
}

struct ForwardStep<'input, B: Backend> {
    angles: RopeAngles<B>,
    attention_mask: &'input Tensor<B>,
    past_seen_tokens: usize,
    batch_size: usize,
    seq_len: usize,
    kv_length: usize,
}

impl<B: Backend> Llama<B> {
    pub fn new(
        backend: &mut B,
        config: &LlamaConfig,
        weights: LlamaWeights<B>,
        max_cache_len: usize,
    ) -> Result<Self, LlamaError<B::Error>> {
        let head_dim = config.head_dim();
        let rope = LlamaRope::new(backend, config, max_cache_len)?;
        let head_dim_non_zero = NonZeroUsize::new(head_dim).ok_or_else(|| {
            error!(
                message = "Head dim is not a positive even number.",
                head_dim
            );
            LlamaError::HeadDimInvalid
        })?;
        Ok(Self {
            weights,
            rope,
            attention: Attention::new(backend, head_dim_non_zero)?,
            heads: config.num_attention_heads(),
            kv_heads: config.num_key_value_heads(),
            head_dim,
            max_cache_len,
        })
    }
}

impl<B: Backend> CausalLm<B> for Llama<B> {
    type Cache = Vec<KvCache<B>>;
    type Error = LlamaError<B::Error>;

    fn cache(
        &self,
        backend: &mut B,
        batch_size: NonZeroUsize,
    ) -> Result<Self::Cache, Self::Error> {
        self.weights
            .layers()
            .iter()
            .map(|_| {
                KvCache::new(
                    backend,
                    CACHE_DTYPE,
                    batch_size.get(),
                    self.kv_heads,
                    self.max_cache_len,
                    self.head_dim,
                )
            })
            .collect::<Result<Vec<KvCache<B>>, TensorError<B::Error>>>()
            .map_err(LlamaError::from)
    }

    fn forward(
        &self,
        backend: &mut B,
        input_ids: &Tensor<B>,
        position_ids: &Tensor<B>,
        attention_mask: &Tensor<B>,
        past_key_values: &mut Self::Cache,
        past_seen_tokens: usize,
    ) -> Result<Tensor<B>, Self::Error> {
        let (batch_size, kv_length) = forward_dims(
            input_ids.shape().numel(),
            position_ids.shape().numel(),
            past_seen_tokens,
            self.max_cache_len,
            past_key_values.len(),
            self.weights.layers().len(),
        )?;
        let step = ForwardStep {
            angles: self.rope.at(backend, position_ids)?,
            attention_mask,
            past_seen_tokens,
            batch_size,
            seq_len: position_ids.shape().numel(),
            kv_length,
        };
        self.logits(backend, input_ids, past_key_values, &step)
            .map_err(LlamaError::from)
    }
}

fn forward_dims<E: std::error::Error + 'static>(
    token_count: usize,
    seq_len: usize,
    past_seen_tokens: usize,
    max_cache_len: usize,
    cache_count: usize,
    layer_count: usize,
) -> Result<(usize, usize), LlamaError<E>> {
    if token_count == 0 {
        error!(message = "Forward input holds no token.");
        return Err(LlamaError::EmptyInput);
    }
    let Some(batch_size) = token_count
        .checked_div(seq_len)
        .filter(|batch_size| batch_size.checked_mul(seq_len) == Some(token_count))
    else {
        error!(
            message = "Token count is not a multiple of the position count.",
            token_count, seq_len,
        );
        return Err(LlamaError::BatchMismatch);
    };
    let kv_length = past_seen_tokens
        .checked_add(seq_len)
        .filter(|&kv_length| kv_length <= max_cache_len)
        .ok_or_else(|| {
            error!(
                message = "Tokens exceed the cache length.",
                past_seen_tokens, seq_len, max_cache_len,
            );
            LlamaError::CacheOverflow
        })?;
    if cache_count != layer_count {
        error!(
            message = "Layer cache count does not match the layer count.",
            cache_count, layer_count,
        );
        return Err(LlamaError::CacheLayerCount);
    }
    Ok((batch_size, kv_length))
}

impl<B: Backend> Llama<B> {
    fn logits(
        &self,
        backend: &mut B,
        input_ids: &Tensor<B>,
        past_key_values: &mut [KvCache<B>],
        step: &ForwardStep<'_, B>,
    ) -> Result<Tensor<B>, TensorError<B::Error>> {
        let embedded = self.weights.embedding().forward(backend, input_ids)?;
        let hidden = self.weights.layers().iter().zip(past_key_values).try_fold(
            embedded,
            |hidden, (layer, layer_cache)| {
                self.decoder_layer(backend, layer, layer_cache, &hidden, step)
            },
        )?;
        let hidden_size = hidden.shape().size(hidden.shape().last_dim()?)?;
        let last = hidden
            .reshape(Shape::try_from(
                [step.batch_size, step.seq_len, hidden_size].as_slice(),
            )?)?
            .narrow(SEQ_DIM, step.seq_len.saturating_sub(1), 1)?
            .contiguous(backend)?
            .reshape(Shape::try_from([step.batch_size, hidden_size].as_slice())?)?;
        let normed = self.weights.norm().forward(backend, &last)?;
        self.weights.lm_head().forward(backend, &normed)
    }

    fn decoder_layer(
        &self,
        backend: &mut B,
        layer: &LlamaLayerWeights<B>,
        cache: &mut KvCache<B>,
        hidden: &Tensor<B>,
        step: &ForwardStep<'_, B>,
    ) -> Result<Tensor<B>, TensorError<B::Error>> {
        let normed = layer.input_norm().forward(backend, hidden)?;
        let query_shape =
            Shape::try_from([step.batch_size, step.seq_len, self.heads, self.head_dim].as_slice())?;
        let kv_shape = Shape::try_from(
            [step.batch_size, step.seq_len, self.kv_heads, self.head_dim].as_slice(),
        )?;
        let query_projected = layer
            .q_proj()
            .forward(backend, &normed)?
            .reshape(query_shape)?;
        let query = step
            .angles
            .apply(backend, &query_projected)?
            .permute(&HEAD_MAJOR)?;
        let key_projected = layer
            .k_proj()
            .forward(backend, &normed)?
            .reshape(kv_shape)?;
        let key = step
            .angles
            .apply(backend, &key_projected)?
            .permute(&HEAD_MAJOR)?;
        let value = layer
            .v_proj()
            .forward(backend, &normed)?
            .reshape(kv_shape)?
            .permute(&HEAD_MAJOR)?;
        cache.update(backend, &key, &value, step.past_seen_tokens)?;
        let attended =
            self.attention
                .forward(backend, &query, cache, step.kv_length, step.attention_mask)?;
        let attention_output = layer.o_proj().forward(backend, &attended)?;
        let hidden = hidden.add(backend, &attention_output)?;
        let mlp_input = layer.post_attention_norm().forward(backend, &hidden)?;
        let mlp_output = layer.mlp().forward(backend, &mlp_input)?;
        hidden.add(backend, &mlp_output)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::CoreError;
    use crate::models::llama::LlamaError;
    use crate::models::llama::model::forward_dims;

    #[test]
    fn forward_dims_reads_the_batch_size_and_the_cache_length() {
        assert_eq!(
            forward_dims::<CoreError>(6, 3, 4, 8, 2, 2),
            Ok((2, 7)),
            "six tokens over three positions from position 4, in a cache of 8"
        );
    }

    #[test]
    fn forward_dims_rejects_invalid_steps() {
        assert_eq!(
            [
                forward_dims::<CoreError>(0, 0, 0, 8, 2, 2),
                forward_dims::<CoreError>(5, 3, 0, 8, 2, 2),
                forward_dims::<CoreError>(6, 3, 6, 8, 2, 2),
                forward_dims::<CoreError>(6, 3, usize::MAX, usize::MAX, 2, 2),
                forward_dims::<CoreError>(6, 3, 0, 8, 1, 2),
            ],
            [
                Err(LlamaError::EmptyInput),
                Err(LlamaError::BatchMismatch),
                Err(LlamaError::CacheOverflow),
                Err(LlamaError::CacheOverflow),
                Err(LlamaError::CacheLayerCount),
            ],
            "no token, ragged step, past the cache end, position overflow, missing layer cache"
        );
    }
}
