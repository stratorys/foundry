use std::num::NonZeroUsize;

use tracing::error;

use crate::core::{
    Backend,
    DType,
    Shape,
    Tensor,
    TensorError,
};
use crate::models::llama::rope::{
    Llama3Rope,
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
const TOKEN_AXIS: usize = 0;
const HEAD_MAJOR: [usize; 3] = [1, 0, 2];

pub struct Llama<B: Backend> {
    weights: LlamaWeights<B>,
    rope: Llama3Rope<B>,
    attention: Attention<B>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    seq_len_max: usize,
}

struct ForwardStep<'input, B: Backend> {
    angles: RopeAngles<B>,
    mask: &'input Tensor<B>,
    position: usize,
    seq_len: usize,
    seq_len_total: usize,
}

impl<B: Backend> Llama<B> {
    pub fn new(
        backend: &mut B,
        config: &LlamaConfig,
        weights: LlamaWeights<B>,
        seq_len_max: usize,
    ) -> Result<Self, LlamaError<B::Error>> {
        let head_dim = config.head_dim();
        let rope = Llama3Rope::new(backend, config, seq_len_max)?;
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
            seq_len_max,
        })
    }

    pub fn cache(
        &self,
        backend: &mut B,
    ) -> Result<Vec<KvCache<B>>, LlamaError<B::Error>> {
        self.weights
            .layers()
            .iter()
            .map(|_| {
                KvCache::new(
                    backend,
                    CACHE_DTYPE,
                    self.kv_heads,
                    self.seq_len_max,
                    self.head_dim,
                )
            })
            .collect::<Result<Vec<KvCache<B>>, TensorError<B::Error>>>()
            .map_err(LlamaError::from)
    }

    pub fn forward(
        &self,
        backend: &mut B,
        token_ids: &Tensor<B>,
        positions: &Tensor<B>,
        mask: &Tensor<B>,
        cache: &mut [KvCache<B>],
        position: usize,
    ) -> Result<Tensor<B>, LlamaError<B::Error>> {
        let seq_len = token_ids.shape().element_count();
        if seq_len == 0 {
            error!(message = "Forward input holds no token.");
            return Err(LlamaError::EmptyInput);
        }
        let seq_len_total = position
            .checked_add(seq_len)
            .filter(|&seq_len_total| seq_len_total <= self.seq_len_max)
            .ok_or_else(|| {
                error!(
                    message = "Tokens exceed the cache length.",
                    position,
                    seq_len,
                    seq_len_max = self.seq_len_max,
                );
                LlamaError::CacheOverflow
            })?;
        let layers = self.weights.layers().len();
        if cache.len() != layers {
            error!(
                message = "Layer cache count does not match the layer count.",
                caches = cache.len(),
                layers,
            );
            return Err(LlamaError::CacheLayerCount);
        }
        let step = ForwardStep {
            angles: self.rope.at(backend, positions)?,
            mask,
            position,
            seq_len,
            seq_len_total,
        };
        self.logits(backend, token_ids, cache, &step)
            .map_err(LlamaError::from)
    }

    fn logits(
        &self,
        backend: &mut B,
        token_ids: &Tensor<B>,
        cache: &mut [KvCache<B>],
        step: &ForwardStep<'_, B>,
    ) -> Result<Tensor<B>, TensorError<B::Error>> {
        let embedded = self.weights.embedding().forward(backend, token_ids)?;
        let hidden = self.weights.layers().iter().zip(cache).try_fold(
            embedded,
            |hidden, (layer, layer_cache)| {
                self.decoder_layer(backend, layer, layer_cache, &hidden, step)
            },
        )?;
        let last = hidden.narrow(TOKEN_AXIS, step.seq_len.saturating_sub(1), 1)?;
        let normed = self.weights.norm().forward(backend, &last)?;
        let logits = self.weights.lm_head().forward(backend, &normed)?;
        let vocab_size = logits.shape().element_count();
        Ok(logits.reshape(Shape::try_from([vocab_size].as_slice())?)?)
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
        let query_shape = Shape::try_from([step.seq_len, self.heads, self.head_dim].as_slice())?;
        let kv_shape = Shape::try_from([step.seq_len, self.kv_heads, self.head_dim].as_slice())?;
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
        cache.update(backend, &key, &value, step.position)?;
        let attended =
            self.attention
                .forward(backend, &query, cache, step.seq_len_total, step.mask)?;
        let attention_output = layer.o_proj().forward(backend, &attended)?;
        let hidden = hidden.add(backend, &attention_output)?;
        let mlp_input = layer.post_attention_norm().forward(backend, &hidden)?;
        let mlp_output = layer.mlp().forward(backend, &mlp_input)?;
        hidden.add(backend, &mlp_output)
    }
}
