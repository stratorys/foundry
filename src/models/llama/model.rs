use std::num::NonZeroUsize;

use tracing::error;

use crate::eval::Submodules;
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
    DType,
    Device,
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

pub struct LlamaForCausalLm<D: Device> {
    model: LlamaModel<D>,
    lm_head: Linear<D>,
    config: LlamaConfig,
}

pub struct LlamaModel<D: Device> {
    embed_tokens: Embedding<D>,
    layers: Vec<LlamaDecoderLayer<D>>,
    norm: RmsNorm<D>,
    rotary_emb: LlamaRotaryEmbedding<D>,
    max_cache_len: usize,
}

pub struct LlamaDecoderLayer<D: Device> {
    self_attn: LlamaAttention<D>,
    mlp: LlamaMlp<D>,
    input_layernorm: RmsNorm<D>,
    post_attention_layernorm: RmsNorm<D>,
}

impl<D: Device> LlamaForCausalLm<D> {
    pub fn load<S: AsRef<[u8]>>(
        device: &mut D,
        config: &LlamaConfig,
        weights: &Weights<S>,
        max_cache_len: usize,
    ) -> Result<Self, LlamaError> {
        let model = LlamaModel::load(device, config, weights, "model", max_cache_len)?;
        let lm_head = Linear::new(model.embed_tokens.weight().clone());
        Ok(Self {
            model,
            lm_head,
            config: config.clone(),
        })
    }
}

impl<D: Device> CausalLm<D> for LlamaForCausalLm<D> {
    type Cache = StaticCache<D>;
    type Error = LlamaError;

    fn cache(
        &self,
        device: &mut D,
        batch_size: NonZeroUsize,
    ) -> Result<Self::Cache, Self::Error> {
        StaticCache::new(
            device,
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
        device: &mut D,
        input_ids: &Tensor<D>,
        position_ids: &Tensor<D>,
        attention_mask: &Tensor<D>,
        past_key_values: &mut Self::Cache,
        past_seen_tokens: usize,
    ) -> Result<Tensor<D>, Self::Error> {
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
                device,
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                past_seen_tokens,
            )
            .map_err(tensor_failed)?;
        self.last_token_logits(device, &hidden_states)
            .map_err(tensor_failed)
    }
}

impl<D: Device> Submodules<D> for LlamaForCausalLm<D> {
    type Error = LlamaError;

    fn forward_submodule(
        &self,
        device: &mut D,
        target: &str,
        input: &Tensor<D>,
    ) -> Result<Option<Tensor<D>>, Self::Error> {
        let Some(target) = target.strip_prefix("model.") else {
            return Ok(None);
        };
        self.model
            .forward_submodule(device, target, input)
            .map_err(tensor_failed)
    }
}

impl<D: Device> LlamaForCausalLm<D> {
    fn last_token_logits(
        &self,
        device: &mut D,
        hidden_states: &Tensor<D>,
    ) -> Result<Tensor<D>, TensorError> {
        let &[batch_size, seq_len, hidden_size] = hidden_states.shape().dims() else {
            error!(
                message = "Model output is not [batch, seq, hidden].",
                dims = ?hidden_states.shape().dims(),
            );
            return Err(TensorError::DimOutOfRange);
        };
        let last_hidden_state = hidden_states
            .narrow(SEQ_DIM, seq_len.saturating_sub(1), 1)?
            .contiguous(device)?
            .reshape(Shape::try_from([batch_size, hidden_size].as_slice())?)?;
        self.lm_head.forward(device, &last_hidden_state)
    }
}

impl<D: Device> LlamaModel<D> {
    fn load<S: AsRef<[u8]>>(
        device: &mut D,
        config: &LlamaConfig,
        weights: &Weights<S>,
        prefix: &str,
        max_cache_len: usize,
    ) -> Result<Self, LlamaError> {
        let rotary_emb = LlamaRotaryEmbedding::new(device, config, max_cache_len)?;
        let embed_tokens = Embedding::new(load_tensor(
            device,
            weights,
            &format!("{prefix}.embed_tokens.weight"),
            &[config.vocab_size(), config.hidden_size()],
        )?);
        let layers = (0..config.num_hidden_layers())
            .map(|layer_idx| {
                LlamaDecoderLayer::load(
                    device,
                    config,
                    weights,
                    &format!("{prefix}.layers.{layer_idx}"),
                    layer_idx,
                )
            })
            .collect::<Result<Vec<LlamaDecoderLayer<D>>, LlamaError>>()?;
        let norm = load_rms_norm(device, config, weights, &format!("{prefix}.norm.weight"))?;
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
        device: &mut D,
        input_ids: &Tensor<D>,
        position_ids: &Tensor<D>,
        attention_mask: &Tensor<D>,
        past_key_values: &mut StaticCache<D>,
        past_seen_tokens: usize,
    ) -> Result<Tensor<D>, TensorError> {
        let seq_len = position_ids.shape().numel();
        let batch_size = input_ids
            .shape()
            .numel()
            .checked_div(seq_len)
            .ok_or_else(|| {
                error!(message = "Forward input holds no position.");
                TensorError::NumelMismatch
            })?;
        let inputs_embeds = self.embed_tokens.forward(device, input_ids)?;
        let hidden_size = inputs_embeds
            .shape()
            .size(inputs_embeds.shape().last_dim()?)?;
        let inputs_embeds = inputs_embeds.reshape(Shape::try_from(
            [batch_size, seq_len, hidden_size].as_slice(),
        )?)?;
        let [cos, sin] = self.rotary_emb.forward(device, position_ids)?;
        let hidden_states =
            self.layers
                .iter()
                .try_fold(inputs_embeds, |hidden_states, layer| {
                    layer.forward(
                        device,
                        &hidden_states,
                        [&cos, &sin],
                        attention_mask,
                        past_key_values,
                        past_seen_tokens,
                    )
                })?;
        self.norm.forward(device, &hidden_states)
    }
}

impl<D: Device> LlamaModel<D> {
    fn forward_submodule(
        &self,
        device: &mut D,
        target: &str,
        input: &Tensor<D>,
    ) -> Result<Option<Tensor<D>>, TensorError> {
        if target == "embed_tokens" {
            return self.embed(device, input).map(Some);
        }
        let Some((layer_idx, target)) = layer_target(target) else {
            return Ok(None);
        };
        match self.layers.get(layer_idx) {
            Some(layer) => layer.forward_submodule(device, target, input),
            None => Ok(None),
        }
    }

    fn embed(
        &self,
        device: &mut D,
        input_ids: &Tensor<D>,
    ) -> Result<Tensor<D>, TensorError> {
        let flat_ids =
            input_ids.reshape(Shape::try_from([input_ids.shape().numel()].as_slice())?)?;
        let inputs_embeds = self.embed_tokens.forward(device, &flat_ids)?;
        let hidden_size = inputs_embeds
            .shape()
            .size(inputs_embeds.shape().last_dim()?)?;
        let dims: Vec<usize> = input_ids
            .shape()
            .dims()
            .iter()
            .copied()
            .chain([hidden_size])
            .collect();
        inputs_embeds.reshape(Shape::try_from(dims.as_slice())?)
    }
}

impl<D: Device> LlamaDecoderLayer<D> {
    fn load<S: AsRef<[u8]>>(
        device: &mut D,
        config: &LlamaConfig,
        weights: &Weights<S>,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, LlamaError> {
        let input_layernorm = load_rms_norm(
            device,
            config,
            weights,
            &format!("{prefix}.input_layernorm.weight"),
        )?;
        let self_attn = LlamaAttention::load(
            device,
            config,
            weights,
            &format!("{prefix}.self_attn"),
            layer_idx,
        )?;
        let post_attention_layernorm = load_rms_norm(
            device,
            config,
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        )?;
        let mlp = LlamaMlp::load(device, config, weights, &format!("{prefix}.mlp"))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        device: &mut D,
        hidden_states: &Tensor<D>,
        position_embeddings: [&Tensor<D>; 2],
        attention_mask: &Tensor<D>,
        past_key_values: &mut StaticCache<D>,
        past_seen_tokens: usize,
    ) -> Result<Tensor<D>, TensorError> {
        let normed = self.input_layernorm.forward(device, hidden_states)?;
        let attn_output = self.self_attn.forward(
            device,
            &normed,
            position_embeddings,
            attention_mask,
            past_key_values,
            past_seen_tokens,
        )?;
        let hidden_states = hidden_states.add(device, &attn_output)?;
        let normed = self
            .post_attention_layernorm
            .forward(device, &hidden_states)?;
        let mlp_output = self.mlp.forward(device, &normed)?;
        hidden_states.add(device, &mlp_output)
    }
}

impl<D: Device> LlamaDecoderLayer<D> {
    fn forward_submodule(
        &self,
        device: &mut D,
        target: &str,
        input: &Tensor<D>,
    ) -> Result<Option<Tensor<D>>, TensorError> {
        match target {
            "input_layernorm" => self.input_layernorm.forward(device, input).map(Some),
            "post_attention_layernorm" => self
                .post_attention_layernorm
                .forward(device, input)
                .map(Some),
            "mlp" => self.mlp.forward(device, input).map(Some),
            _ => match target.strip_prefix("self_attn.") {
                Some(target) => self.self_attn.forward_submodule(device, target, input),
                None => Ok(None),
            },
        }
    }
}

fn layer_target(target: &str) -> Option<(usize, &str)> {
    let (layer_idx, target) = target.strip_prefix("layers.")?.split_once('.')?;
    Some((layer_idx.parse().ok()?, target))
}

fn load_rms_norm<D: Device, S: AsRef<[u8]>>(
    device: &mut D,
    config: &LlamaConfig,
    weights: &Weights<S>,
    name: &str,
) -> Result<RmsNorm<D>, LlamaError> {
    let weight = load_tensor(device, weights, name, &[config.hidden_size()])?;
    RmsNorm::new(device, weight, config.rms_norm_eps()).map_err(tensor_failed)
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
    use crate::models::llama::model::{
        check_forward,
        layer_target,
    };

    #[test]
    fn layer_target_splits_the_layer_index_from_the_submodule() {
        assert_eq!(
            [
                layer_target("layers.0.mlp"),
                layer_target("layers.27.self_attn.q_proj"),
                layer_target("layers.x.mlp"),
                layer_target("layers.0"),
                layer_target("norm"),
            ],
            [
                Some((0, "mlp")),
                Some((27, "self_attn.q_proj")),
                None,
                None,
                None,
            ],
            "a layer submodule, a nested one, a non-numeric index, no submodule, no layer"
        );
    }

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
