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
const SEQ_AXIS: usize = 1;
const HEAD_MAJOR: [usize; 4] = [0, 2, 1, 3];

pub struct Llama<B: Backend> {
    weights: LlamaWeights<B>,
    rope: LlamaRope<B>,
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
    batch_size: usize,
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
        let rope = LlamaRope::new(backend, config, seq_len_max)?;
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
                    self.seq_len_max,
                    self.head_dim,
                )
            })
            .collect::<Result<Vec<KvCache<B>>, TensorError<B::Error>>>()
            .map_err(LlamaError::from)
    }

    fn forward(
        &self,
        backend: &mut B,
        token_ids: &Tensor<B>,
        positions: &Tensor<B>,
        mask: &Tensor<B>,
        cache: &mut Self::Cache,
        position: usize,
    ) -> Result<Tensor<B>, Self::Error> {
        let token_count = token_ids.shape().element_count();
        if token_count == 0 {
            error!(message = "Forward input holds no token.");
            return Err(LlamaError::EmptyInput);
        }
        let seq_len = positions.shape().element_count();
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
            batch_size,
            seq_len,
            seq_len_total,
        };
        self.logits(backend, token_ids, cache, &step)
            .map_err(LlamaError::from)
    }
}

impl<B: Backend> Llama<B> {
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
        let hidden_size = hidden.shape().dim(hidden.shape().last_axis()?)?;
        let last = hidden
            .reshape(Shape::try_from(
                [step.batch_size, step.seq_len, hidden_size].as_slice(),
            )?)?
            .narrow(SEQ_AXIS, step.seq_len.saturating_sub(1), 1)?
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
