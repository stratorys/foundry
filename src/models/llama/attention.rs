use tracing::error;

use crate::models::llama::rotary_emb::apply_rotary_pos_emb;
use crate::models::llama::weights::load_tensor;
use crate::models::llama::{
    LlamaConfig,
    LlamaError,
};
use crate::nn::{
    Backend,
    CoreError,
    DType,
    Linear,
    Shape,
    StaticCache,
    Tensor,
    TensorError,
    exact_f32,
    scaled_dot_product_attention,
};
use crate::weights::Weights;

const HEAD_MAJOR: [usize; 4] = [0, 2, 1, 3];

pub struct LlamaAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scaling: Tensor<B>,
    layer_idx: usize,
}

impl<B: Backend> LlamaAttention<B> {
    pub fn load<S: AsRef<[u8]>>(
        backend: &mut B,
        config: &LlamaConfig,
        weights: &Weights<S>,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, LlamaError<B::Error>> {
        let hidden_size = config.hidden_size();
        let num_key_value_heads = config.num_key_value_heads();
        let head_dim = config.head_dim();
        let kv_dim = num_key_value_heads.checked_mul(head_dim).ok_or_else(|| {
            error!(
                message = "Key-value width overflows usize.",
                num_key_value_heads, head_dim,
            );
            LlamaError::KvDimOverflow
        })?;
        let mut linear = |name: &str, dims: [usize; 2]| {
            load_tensor(backend, weights, &format!("{prefix}.{name}.weight"), &dims)
                .map(Linear::new)
        };
        let q_proj = linear("q_proj", [hidden_size, hidden_size])?;
        let k_proj = linear("k_proj", [kv_dim, hidden_size])?;
        let v_proj = linear("v_proj", [kv_dim, hidden_size])?;
        let o_proj = linear("o_proj", [hidden_size, hidden_size])?;
        let head_dim_f32 = exact_f32(head_dim)?;
        let scaling = Tensor::from_bytes(
            backend,
            &head_dim_f32.sqrt().recip().to_le_bytes(),
            DType::F32,
            Shape::try_from([1].as_slice())?,
        )?;
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
        backend: &mut B,
        hidden_states: &Tensor<B>,
        [cos, sin]: [&Tensor<B>; 2],
        attention_mask: &Tensor<B>,
        past_key_values: &mut StaticCache<B>,
        past_seen_tokens: usize,
    ) -> Result<Tensor<B>, TensorError<B::Error>> {
        let &[batch_size, seq_len, _] = hidden_states.shape().dims() else {
            error!(
                message = "Attention input is not [batch, seq, hidden].",
                dims = ?hidden_states.shape().dims(),
            );
            return Err(CoreError::AttentionIncompatible.into());
        };
        let query_shape =
            Shape::try_from([batch_size, seq_len, self.num_heads, self.head_dim].as_slice())?;
        let kv_shape = Shape::try_from(
            [batch_size, seq_len, self.num_key_value_heads, self.head_dim].as_slice(),
        )?;
        let query_states = self
            .q_proj
            .forward(backend, hidden_states)?
            .reshape(query_shape)?;
        let key_states = self
            .k_proj
            .forward(backend, hidden_states)?
            .reshape(kv_shape)?;
        let value_states = self
            .v_proj
            .forward(backend, hidden_states)?
            .reshape(kv_shape)?
            .permute(&HEAD_MAJOR)?;
        let [query_states, key_states] =
            apply_rotary_pos_emb(backend, &query_states, &key_states, cos, sin)?;
        let [keys, values] = past_key_values.update(
            backend,
            &key_states.permute(&HEAD_MAJOR)?,
            &value_states,
            self.layer_idx,
            past_seen_tokens,
        )?;
        let attn_output = scaled_dot_product_attention(
            backend,
            &query_states.permute(&HEAD_MAJOR)?,
            &keys,
            &values,
            attention_mask,
            &self.scaling,
        )?
        .permute(&HEAD_MAJOR)?
        .contiguous(backend)?
        .reshape(Shape::try_from(
            [
                batch_size,
                seq_len,
                self.num_heads.saturating_mul(self.head_dim),
            ]
            .as_slice(),
        )?)?;
        self.o_proj.forward(backend, &attn_output)
    }
}
