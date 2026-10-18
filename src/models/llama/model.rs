use std::num::NonZeroUsize;

use tracing::error;

use crate::models::CausalLm;
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
    Module,
    RmsNorm,
    StaticCache,
    Tensor,
    TensorError,
    create_causal_mask,
    position_ids,
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
        past_key_values: &mut Self::Cache,
        past_seen_tokens: usize,
        logits_to_keep: usize,
    ) -> Result<Tensor<D>, Self::Error> {
        let seq_len = check_forward(
            input_ids.shape().dims(),
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
                past_key_values,
                past_seen_tokens,
                seq_len,
            )
            .map_err(tensor_failed)?;
        let (start, kept) = logits_range(seq_len, logits_to_keep);
        hidden_states
            .narrow(SEQ_DIM, start, kept)
            .and_then(|hidden_states| hidden_states.contiguous(device))
            .and_then(|hidden_states| self.lm_head.forward(device, &hidden_states))
            .map_err(tensor_failed)
    }
}

impl<D: Device> LlamaForCausalLm<D> {
    pub fn named_modules(&self) -> Vec<(String, &dyn Module<D>)> {
        self.model.named_modules("model")
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
        past_key_values: &mut StaticCache<D>,
        past_seen_tokens: usize,
        seq_len: usize,
    ) -> Result<Tensor<D>, TensorError> {
        let position_ids = position_ids(device, past_seen_tokens, seq_len)?;
        let attention_mask =
            create_causal_mask(device, past_seen_tokens, seq_len)?.to_dtype(device, DType::F32)?;
        let inputs_embeds = self.embed_tokens.forward(device, input_ids)?;
        let [cos, sin] = self.rotary_emb.forward(device, &position_ids)?;
        let hidden_states =
            self.layers
                .iter()
                .try_fold(inputs_embeds, |hidden_states, layer| {
                    layer.forward(
                        device,
                        &hidden_states,
                        [&cos, &sin],
                        &attention_mask,
                        past_key_values,
                        past_seen_tokens,
                    )
                })?;
        self.norm.forward(device, &hidden_states)
    }
}

impl<D: Device> LlamaModel<D> {
    fn named_modules(
        &self,
        prefix: &str,
    ) -> Vec<(String, &dyn Module<D>)> {
        let embed_tokens: (String, &dyn Module<D>) =
            (format!("{prefix}.embed_tokens"), &self.embed_tokens);
        [embed_tokens]
            .into_iter()
            .chain(
                self.layers
                    .iter()
                    .enumerate()
                    .flat_map(|(layer_idx, layer)| {
                        layer.named_modules(&format!("{prefix}.layers.{layer_idx}"))
                    }),
            )
            .collect()
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
    fn named_modules(
        &self,
        prefix: &str,
    ) -> Vec<(String, &dyn Module<D>)> {
        let input_layernorm: [(String, &dyn Module<D>); 1] =
            [(format!("{prefix}.input_layernorm"), &self.input_layernorm)];
        let post_attention: [(String, &dyn Module<D>); 2] = [
            (
                format!("{prefix}.post_attention_layernorm"),
                &self.post_attention_layernorm,
            ),
            (format!("{prefix}.mlp"), &self.mlp),
        ];
        input_layernorm
            .into_iter()
            .chain(self.self_attn.named_modules(&format!("{prefix}.self_attn")))
            .chain(post_attention)
            .collect()
    }
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
    dims: &[usize],
    past_seen_tokens: usize,
    max_cache_len: usize,
    cache_count: usize,
    layer_count: usize,
) -> Result<usize, LlamaError> {
    let &[batch_size, seq_len] = dims else {
        error!(message = "Input IDs are not [batch, seq_len].", ?dims);
        return Err(LlamaError::InputIdsShape);
    };
    if batch_size == 0 || seq_len == 0 {
        error!(
            message = "Forward input holds no token.",
            batch_size, seq_len
        );
        return Err(LlamaError::EmptyInput);
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
    Ok(seq_len)
}

fn logits_range(
    seq_len: usize,
    logits_to_keep: usize,
) -> (usize, usize) {
    let kept = match logits_to_keep {
        0 => seq_len,
        logits_to_keep => logits_to_keep.min(seq_len),
    };
    (seq_len.saturating_sub(kept), kept)
}

pub fn module_atol(name: &str) -> f32 {
    match name.rsplit('.').next() {
        Some("embed_tokens") => 1e-3,
        Some("q_proj" | "k_proj" | "o_proj") => 2e-2,
        _ => 1e-2,
    }
}

#[cfg(test)]
mod tests {
    use crate::models::llama::LlamaError;
    use crate::models::llama::model::{
        check_forward,
        logits_range,
        module_atol,
    };

    #[test]
    fn module_atol_depends_on_the_last_module_name() {
        assert_eq!(
            [
                "model.embed_tokens",
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.o_proj",
                "model.layers.0.self_attn.v_proj",
                "model.layers.0.mlp",
                "model.layers.0.input_layernorm",
            ]
            .map(module_atol),
            [1e-3, 2e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2],
            "embedding, three projections, then the other modules"
        );
    }

    #[test]
    fn check_forward_accepts_whole_batches_that_fit_the_cache() {
        assert_eq!(
            [
                check_forward(&[2, 3], 4, 8, 2, 2),
                check_forward(&[2, 3], 5, 8, 2, 2)
            ],
            [Ok(3), Ok(3)],
            "two sequences of three tokens from position 4, then up to the cache end"
        );
    }

    #[test]
    fn check_forward_rejects_invalid_steps() {
        assert_eq!(
            [
                check_forward(&[6], 0, 8, 2, 2),
                check_forward(&[2, 0], 0, 8, 2, 2),
                check_forward(&[0, 3], 0, 8, 2, 2),
                check_forward(&[2, 3], 6, 8, 2, 2),
                check_forward(&[2, 3], usize::MAX, usize::MAX, 2, 2),
                check_forward(&[2, 3], 0, 8, 1, 2),
            ],
            [
                Err(LlamaError::InputIdsShape),
                Err(LlamaError::EmptyInput),
                Err(LlamaError::EmptyInput),
                Err(LlamaError::CacheOverflow),
                Err(LlamaError::CacheOverflow),
                Err(LlamaError::CacheLayerCount),
            ],
            "flat ids, no position, no sequence, past the cache end, position overflow, missing \
             layer cache"
        );
    }

    #[test]
    fn logits_range_keeps_the_last_positions() {
        assert_eq!(
            [(5, 0), (5, 1), (5, 3), (5, 5), (5, 9)]
                .map(|(seq_len, logits_to_keep)| logits_range(seq_len, logits_to_keep)),
            [(0, 5), (4, 1), (2, 3), (0, 5), (0, 5)],
            "every position, the last one, the last three, exactly all, more than the sequence"
        );
    }
}
