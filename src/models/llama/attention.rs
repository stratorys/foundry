use tracing::error;

use crate::models::llama::error::tensor_failed;
use crate::models::llama::rotary_emb::apply_rotary_pos_emb;
use crate::models::llama::weights::load_tensor;
use crate::models::llama::{
    LlamaConfig,
    LlamaError,
};
use crate::nn::{
    Device,
    Linear,
    Module,
    Shape,
    StaticCache,
    Tensor,
    TensorError,
    exact_f32,
    scaled_dot_product_attention,
};
use crate::weights::Weights;

const HEAD_MAJOR: [usize; 4] = [0, 2, 1, 3];

pub struct LlamaAttention<D: Device> {
    q_proj: Linear<D>,
    k_proj: Linear<D>,
    v_proj: Linear<D>,
    o_proj: Linear<D>,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scaling: Tensor<D>,
    layer_idx: usize,
}

impl<D: Device> LlamaAttention<D> {
    pub fn load<S: AsRef<[u8]>>(
        device: &mut D,
        config: &LlamaConfig,
        weights: &Weights<S>,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, LlamaError> {
        let hidden_size = config.hidden_size();
        let num_key_value_heads = config.num_key_value_heads();
        let head_dim = config.head_dim();
        let kv_dim = num_key_value_heads.saturating_mul(head_dim);
        let mut linear = |name: &str, dims: [usize; 2]| {
            load_tensor(device, weights, &format!("{prefix}.{name}.weight"), &dims).map(Linear::new)
        };
        let q_proj = linear("q_proj", [hidden_size, hidden_size])?;
        let k_proj = linear("k_proj", [kv_dim, hidden_size])?;
        let v_proj = linear("v_proj", [kv_dim, hidden_size])?;
        let o_proj = linear("o_proj", [hidden_size, hidden_size])?;
        let scaling = exact_f32(head_dim)
            .and_then(|dim| Tensor::scalar_f32(device, dim.sqrt().recip()))
            .map_err(tensor_failed)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads(),
            num_key_value_heads,
            head_dim,
            scaling,
            layer_idx,
        })
    }

    pub fn forward(
        &self,
        device: &mut D,
        hidden_states: &Tensor<D>,
        [cos, sin]: [&Tensor<D>; 2],
        attention_mask: &Tensor<D>,
        past_key_values: &mut StaticCache<D>,
        past_seen_tokens: usize,
    ) -> Result<Tensor<D>, TensorError> {
        let &[batch_size, seq_len, _] = hidden_states.shape().dims() else {
            error!(
                message = "Attention input is not [batch, seq, hidden].",
                dims = ?hidden_states.shape().dims(),
            );
            return Err(TensorError::AttentionIncompatible);
        };
        let query_shape =
            Shape::try_from([batch_size, seq_len, self.num_heads, self.head_dim].as_slice())?;
        let kv_shape = Shape::try_from(
            [batch_size, seq_len, self.num_key_value_heads, self.head_dim].as_slice(),
        )?;
        let query_states = self
            .q_proj
            .forward(device, hidden_states)?
            .reshape(query_shape)?;
        let key_states = self
            .k_proj
            .forward(device, hidden_states)?
            .reshape(kv_shape)?;
        let value_states = self
            .v_proj
            .forward(device, hidden_states)?
            .reshape(kv_shape)?
            .permute(&HEAD_MAJOR)?;
        let [query_states, key_states] =
            apply_rotary_pos_emb(device, &query_states, &key_states, cos, sin)?;
        let [keys, values] = past_key_values.update(
            device,
            &key_states.permute(&HEAD_MAJOR)?,
            &value_states,
            self.layer_idx,
            past_seen_tokens,
        )?;
        let attn_output = scaled_dot_product_attention(
            device,
            &query_states.permute(&HEAD_MAJOR)?,
            &keys,
            &values,
            attention_mask,
            &self.scaling,
        )?
        .permute(&HEAD_MAJOR)?
        .contiguous(device)?
        .reshape(Shape::try_from(
            [
                batch_size,
                seq_len,
                self.num_heads.saturating_mul(self.head_dim),
            ]
            .as_slice(),
        )?)?;
        self.o_proj.forward(device, &attn_output)
    }

    pub fn named_modules(
        &self,
        prefix: &str,
    ) -> [(String, &dyn Module<D>); 4] {
        [
            (format!("{prefix}.q_proj"), &self.q_proj),
            (format!("{prefix}.k_proj"), &self.k_proj),
            (format!("{prefix}.v_proj"), &self.v_proj),
            (format!("{prefix}.o_proj"), &self.o_proj),
        ]
    }
}
