use std::num::NonZeroUsize;

use tracing::error;

use crate::generate::CausalLm;
use crate::models::llama::attention::LlamaAttention;
use crate::models::llama::error::tensor_failed;
use crate::models::llama::mlp::LlamaMlp;
use crate::models::llama::rotary_emb::LlamaRotaryEmbedding;
use crate::models::llama::weights::load_tensor;
use crate::models::llama::{
    LlamaConfig,
    LlamaError,
};
use crate::nn::{
    Backend,
    DType,
    Embedding,
    Linear,
    RmsNorm,
    Shape,
    StaticCache,
    Tensor,
    TensorError,
};
use crate::weights::Weights;

const CACHE_DTYPE: DType = DType::BF16;
const SEQ_DIM: usize = 1;

pub struct LlamaForCausalLm<B: Backend> {
    model: LlamaModel<B>,
    lm_head: Linear<B>,
    config: LlamaConfig,
}

pub struct LlamaModel<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<LlamaDecoderLayer<B>>,
    norm: RmsNorm<B>,
    rotary_emb: LlamaRotaryEmbedding<B>,
    max_cache_len: usize,
}

pub struct LlamaDecoderLayer<B: Backend> {
    self_attn: LlamaAttention<B>,
    mlp: LlamaMlp<B>,
    input_layernorm: RmsNorm<B>,
    post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> LlamaForCausalLm<B> {
    pub fn load<S: AsRef<[u8]>>(
        backend: &mut B,
        config: &LlamaConfig,
        weights: &Weights<S>,
        max_cache_len: usize,
    ) -> Result<Self, LlamaError> {
        let model = LlamaModel::load(backend, config, weights, "model", max_cache_len)?;
        let lm_head = Linear::new(model.embed_tokens.weight().clone());
        Ok(Self {
            model,
            lm_head,
            config: config.clone(),
        })
    }
}

impl<B: Backend> CausalLm<B> for LlamaForCausalLm<B> {
    type Cache = StaticCache<B>;
    type Error = LlamaError;

    fn cache(
        &self,
        backend: &mut B,
        batch_size: NonZeroUsize,
    ) -> Result<Self::Cache, Self::Error> {
        StaticCache::new(
            backend,
            CACHE_DTYPE,
            self.config.num_hidden_layers(),
            batch_size.get(),
            self.config.num_key_value_heads(),
            self.model.max_cache_len,
            self.config.head_dim(),
        )
        .map_err(tensor_failed)
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
        check_forward(
            input_ids.shape().numel(),
            position_ids.shape().numel(),
            past_seen_tokens,
            self.model.max_cache_len,
            past_key_values.num_layers(),
            self.model.layers.len(),
        )?;
        let hidden_states = self
            .model
            .forward(
                backend,
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                past_seen_tokens,
            )
            .map_err(tensor_failed)?;
        self.last_token_logits(backend, &hidden_states)
            .map_err(tensor_failed)
    }
}

impl<B: Backend> LlamaForCausalLm<B> {
    fn last_token_logits(
        &self,
        backend: &mut B,
        hidden_states: &Tensor<B>,
    ) -> Result<Tensor<B>, TensorError> {
        let &[batch_size, seq_len, hidden_size] = hidden_states.shape().dims() else {
            error!(
                message = "Model output is not [batch, seq, hidden].",
                dims = ?hidden_states.shape().dims(),
            );
            return Err(TensorError::DimOutOfRange);
        };
        let last_hidden_state = hidden_states
            .narrow(SEQ_DIM, seq_len.saturating_sub(1), 1)?
            .contiguous(backend)?
            .reshape(Shape::try_from([batch_size, hidden_size].as_slice())?)?;
        self.lm_head.forward(backend, &last_hidden_state)
    }
}

impl<B: Backend> LlamaModel<B> {
    fn load<S: AsRef<[u8]>>(
        backend: &mut B,
        config: &LlamaConfig,
        weights: &Weights<S>,
        prefix: &str,
        max_cache_len: usize,
    ) -> Result<Self, LlamaError> {
        let rotary_emb = LlamaRotaryEmbedding::new(backend, config, max_cache_len)?;
        let embed_tokens = Embedding::new(load_tensor(
            backend,
            weights,
            &format!("{prefix}.embed_tokens.weight"),
            &[config.vocab_size(), config.hidden_size()],
        )?);
        let layers = (0..config.num_hidden_layers())
            .map(|layer_idx| {
                LlamaDecoderLayer::load(
                    backend,
                    config,
                    weights,
                    &format!("{prefix}.layers.{layer_idx}"),
                    layer_idx,
                )
            })
            .collect::<Result<Vec<LlamaDecoderLayer<B>>, LlamaError>>()?;
        let norm = load_rms_norm(backend, config, weights, &format!("{prefix}.norm.weight"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            max_cache_len,
        })
    }

    fn forward(
        &self,
        backend: &mut B,
        input_ids: &Tensor<B>,
        position_ids: &Tensor<B>,
        attention_mask: &Tensor<B>,
        past_key_values: &mut StaticCache<B>,
        past_seen_tokens: usize,
    ) -> Result<Tensor<B>, TensorError> {
        let seq_len = position_ids.shape().numel();
        let batch_size = input_ids
            .shape()
            .numel()
            .checked_div(seq_len)
            .ok_or_else(|| {
                error!(message = "Forward input holds no position.");
                TensorError::NumelMismatch
            })?;
        let inputs_embeds = self.embed_tokens.forward(backend, input_ids)?;
        let hidden_size = inputs_embeds
            .shape()
            .size(inputs_embeds.shape().last_dim()?)?;
        let inputs_embeds = inputs_embeds.reshape(Shape::try_from(
            [batch_size, seq_len, hidden_size].as_slice(),
        )?)?;
        let [cos, sin] = self.rotary_emb.forward(backend, position_ids)?;
        let hidden_states =
            self.layers
                .iter()
                .try_fold(inputs_embeds, |hidden_states, layer| {
                    layer.forward(
                        backend,
                        &hidden_states,
                        [&cos, &sin],
                        attention_mask,
                        past_key_values,
                        past_seen_tokens,
                    )
                })?;
        self.norm.forward(backend, &hidden_states)
    }
}

impl<B: Backend> LlamaDecoderLayer<B> {
    fn load<S: AsRef<[u8]>>(
        backend: &mut B,
        config: &LlamaConfig,
        weights: &Weights<S>,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, LlamaError> {
        let input_layernorm = load_rms_norm(
            backend,
            config,
            weights,
            &format!("{prefix}.input_layernorm.weight"),
        )?;
        let self_attn = LlamaAttention::load(
            backend,
            config,
            weights,
            &format!("{prefix}.self_attn"),
            layer_idx,
        )?;
        let post_attention_layernorm = load_rms_norm(
            backend,
            config,
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        )?;
        let mlp = LlamaMlp::load(backend, config, weights, &format!("{prefix}.mlp"))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        backend: &mut B,
        hidden_states: &Tensor<B>,
        position_embeddings: [&Tensor<B>; 2],
        attention_mask: &Tensor<B>,
        past_key_values: &mut StaticCache<B>,
        past_seen_tokens: usize,
    ) -> Result<Tensor<B>, TensorError> {
        let normed = self.input_layernorm.forward(backend, hidden_states)?;
        let attn_output = self.self_attn.forward(
            backend,
            &normed,
            position_embeddings,
            attention_mask,
            past_key_values,
            past_seen_tokens,
        )?;
        let hidden_states = hidden_states.add(backend, &attn_output)?;
        let normed = self
            .post_attention_layernorm
            .forward(backend, &hidden_states)?;
        let mlp_output = self.mlp.forward(backend, &normed)?;
        hidden_states.add(backend, &mlp_output)
    }
}

fn load_rms_norm<B: Backend, S: AsRef<[u8]>>(
    backend: &mut B,
    config: &LlamaConfig,
    weights: &Weights<S>,
    name: &str,
) -> Result<RmsNorm<B>, LlamaError> {
    let weight = load_tensor(backend, weights, name, &[config.hidden_size()])?;
    RmsNorm::new(backend, weight, config.rms_norm_eps()).map_err(tensor_failed)
}

fn check_forward(
    token_count: usize,
    seq_len: usize,
    past_seen_tokens: usize,
    max_cache_len: usize,
    cache_count: usize,
    layer_count: usize,
) -> Result<(), LlamaError> {
    if token_count == 0 {
        error!(message = "Forward input holds no token.");
        return Err(LlamaError::EmptyInput);
    }
    let is_whole_batch = token_count
        .checked_div(seq_len)
        .and_then(|batch_size| batch_size.checked_mul(seq_len))
        == Some(token_count);
    if !is_whole_batch {
        error!(
            message = "Token count is not a multiple of the position count.",
            token_count, seq_len,
        );
        return Err(LlamaError::BatchMismatch);
    }
    past_seen_tokens
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::models::llama::LlamaError;
    use crate::models::llama::model::check_forward;

    #[test]
    fn check_forward_accepts_whole_batches_that_fit_the_cache() {
        assert_eq!(
            [
                check_forward(6, 3, 4, 8, 2, 2),
                check_forward(6, 3, 5, 8, 2, 2)
            ],
            [Ok(()), Ok(())],
            "six tokens over three positions from position 4, then up to the cache end"
        );
    }

    #[test]
    fn check_forward_rejects_invalid_steps() {
        assert_eq!(
            [
                check_forward(0, 0, 0, 8, 2, 2),
                check_forward(5, 3, 0, 8, 2, 2),
                check_forward(6, 3, 6, 8, 2, 2),
                check_forward(6, 3, usize::MAX, usize::MAX, 2, 2),
                check_forward(6, 3, 0, 8, 1, 2),
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
