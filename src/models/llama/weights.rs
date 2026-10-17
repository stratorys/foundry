use tracing::error;

use crate::core::{
    Backend,
    DType,
    Tensor,
};
use crate::models::llama::{
    LlamaConfig,
    LlamaError,
};
use crate::nn::{
    Embedding,
    Linear,
    RmsNorm,
    SwigluMlp,
};
use crate::weights::{
    WeightView,
    Weights,
};

const DTYPE: DType = DType::BF16;

pub struct LlamaWeights<B: Backend> {
    embedding: Embedding<B>,
    layers: Vec<LlamaLayerWeights<B>>,
    norm: RmsNorm<B>,
    lm_head: Linear<B>,
    bytes_uploaded: usize,
}

pub struct LlamaLayerWeights<B: Backend> {
    input_norm: RmsNorm<B>,
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    post_attention_norm: RmsNorm<B>,
    mlp: SwigluMlp<B>,
}

struct Loader<'load, B: Backend, S> {
    backend: &'load mut B,
    weights: &'load Weights<S>,
    bytes_uploaded: usize,
}

impl<B: Backend> LlamaWeights<B> {
    pub fn load<S: AsRef<[u8]>>(
        backend: &mut B,
        config: &LlamaConfig,
        weights: &Weights<S>,
    ) -> Result<Self, LlamaError<B::Error>> {
        let kv_heads = config.num_key_value_heads();
        let head_dim = config.head_dim();
        let kv_dim = kv_heads.checked_mul(head_dim).ok_or_else(|| {
            error!(
                message = "Key-value width overflows usize.",
                kv_heads, head_dim,
            );
            LlamaError::KvDimOverflow
        })?;
        let mut loader = Loader {
            backend,
            weights,
            bytes_uploaded: 0,
        };
        let embed_tokens = loader.tensor(
            "model.embed_tokens.weight",
            &[config.vocab_size(), config.hidden_size()],
        )?;
        let layers = (0..config.num_hidden_layers())
            .map(|index| loader.layer(config, index, kv_dim))
            .collect::<Result<Vec<LlamaLayerWeights<B>>, LlamaError<B::Error>>>()?;
        let norm = loader.rms_norm("model.norm.weight", config)?;
        Ok(Self {
            embedding: Embedding::new(embed_tokens.clone()),
            layers,
            norm,
            lm_head: Linear::new(embed_tokens),
            bytes_uploaded: loader.bytes_uploaded,
        })
    }

    pub fn embedding(&self) -> &Embedding<B> { &self.embedding }

    pub fn layers(&self) -> &[LlamaLayerWeights<B>] { &self.layers }

    pub fn norm(&self) -> &RmsNorm<B> { &self.norm }

    pub fn lm_head(&self) -> &Linear<B> { &self.lm_head }

    pub fn bytes_uploaded(&self) -> usize { self.bytes_uploaded }
}

impl<B: Backend> LlamaLayerWeights<B> {
    pub fn input_norm(&self) -> &RmsNorm<B> { &self.input_norm }

    pub fn q_proj(&self) -> &Linear<B> { &self.q_proj }

    pub fn k_proj(&self) -> &Linear<B> { &self.k_proj }

    pub fn v_proj(&self) -> &Linear<B> { &self.v_proj }

    pub fn o_proj(&self) -> &Linear<B> { &self.o_proj }

    pub fn post_attention_norm(&self) -> &RmsNorm<B> { &self.post_attention_norm }

    pub fn mlp(&self) -> &SwigluMlp<B> { &self.mlp }
}

impl<B: Backend, S: AsRef<[u8]>> Loader<'_, B, S> {
    fn layer(
        &mut self,
        config: &LlamaConfig,
        index: usize,
        kv_dim: usize,
    ) -> Result<LlamaLayerWeights<B>, LlamaError<B::Error>> {
        let hidden = config.hidden_size();
        let intermediate = config.intermediate_size();
        let prefix = format!("model.layers.{index}");
        let input_norm = self.rms_norm(&format!("{prefix}.input_layernorm.weight"), config)?;
        let q_proj = self.linear(
            &format!("{prefix}.self_attn.q_proj.weight"),
            [hidden, hidden],
        )?;
        let k_proj = self.linear(
            &format!("{prefix}.self_attn.k_proj.weight"),
            [kv_dim, hidden],
        )?;
        let v_proj = self.linear(
            &format!("{prefix}.self_attn.v_proj.weight"),
            [kv_dim, hidden],
        )?;
        let o_proj = self.linear(
            &format!("{prefix}.self_attn.o_proj.weight"),
            [hidden, hidden],
        )?;
        let post_attention_norm =
            self.rms_norm(&format!("{prefix}.post_attention_layernorm.weight"), config)?;
        let gate = self.linear(
            &format!("{prefix}.mlp.gate_proj.weight"),
            [intermediate, hidden],
        )?;
        let up = self.linear(
            &format!("{prefix}.mlp.up_proj.weight"),
            [intermediate, hidden],
        )?;
        let down = self.linear(
            &format!("{prefix}.mlp.down_proj.weight"),
            [hidden, intermediate],
        )?;
        let mlp = SwigluMlp::new(self.backend, gate, up, down)?;
        Ok(LlamaLayerWeights {
            input_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            post_attention_norm,
            mlp,
        })
    }

    fn linear(
        &mut self,
        name: &str,
        dims_expected: [usize; 2],
    ) -> Result<Linear<B>, LlamaError<B::Error>> {
        Ok(Linear::new(self.tensor(name, &dims_expected)?))
    }

    fn rms_norm(
        &mut self,
        name: &str,
        config: &LlamaConfig,
    ) -> Result<RmsNorm<B>, LlamaError<B::Error>> {
        let weight = self.tensor(name, &[config.hidden_size()])?;
        RmsNorm::new(self.backend, weight, config.rms_norm_eps()).map_err(LlamaError::from)
    }

    fn tensor(
        &mut self,
        name: &str,
        dims_expected: &[usize],
    ) -> Result<Tensor<B>, LlamaError<B::Error>> {
        let view = checked_view(self.weights.tensor(name), name, dims_expected)?;
        let tensor = Tensor::from_bytes(self.backend, view.bytes, view.dtype, view.shape)?;
        self.bytes_uploaded = self
            .bytes_uploaded
            .checked_add(view.bytes.len())
            .ok_or_else(|| {
                error!(message = "Uploaded byte count overflows usize.");
                LlamaError::UploadedBytesOverflow
            })?;
        Ok(tensor)
    }
}

fn checked_view<'weights, E: std::error::Error + 'static>(
    view: Option<WeightView<'weights>>,
    name: &str,
    dims_expected: &[usize],
) -> Result<WeightView<'weights>, LlamaError<E>> {
    let view = view.ok_or_else(|| {
        error!(message = "Tensor is not in the weights.", name);
        LlamaError::TensorNotFound
    })?;
    if view.dtype != DTYPE {
        error!(
            message = "Tensor dtype is not the model dtype.",
            name,
            dtype = ?view.dtype,
            dtype_expected = ?DTYPE,
        );
        return Err(LlamaError::TensorDTypeMismatch);
    }
    if view.shape.dims() != dims_expected {
        error!(
            message = "Tensor shape does not match the config.",
            name,
            dims = ?view.shape.dims(),
            ?dims_expected,
        );
        return Err(LlamaError::TensorShapeMismatch);
    }
    Ok(view)
}

#[cfg(test)]
mod tests {
    use crate::core::{
        CoreError,
        DType,
        Shape,
    };
    use crate::models::llama::LlamaError;
    use crate::models::llama::weights::checked_view;
    use crate::weights::WeightView;

    #[test]
    fn checked_view_accepts_a_bf16_tensor_of_the_expected_shape() {
        let view = WeightView {
            name: "model.norm.weight",
            dtype: DType::BF16,
            shape: Shape::try_from([4].as_slice()).expect("the shape is valid"),
            bytes: &[0; 8],
        };
        assert_eq!(
            checked_view::<CoreError>(Some(view), "model.norm.weight", &[4]),
            Ok(view),
            "a bf16 tensor of 4 elements"
        );
    }

    #[test]
    fn checked_view_rejects_a_missing_tensor_a_wrong_dtype_or_a_wrong_shape() {
        let view = |dtype| WeightView {
            name: "model.norm.weight",
            dtype,
            shape: Shape::try_from([4].as_slice()).expect("the shape is valid"),
            bytes: &[0; 16],
        };
        assert_eq!(
            [
                checked_view::<CoreError>(None, "model.norm.weight", &[4]),
                checked_view::<CoreError>(Some(view(DType::F32)), "model.norm.weight", &[4]),
                checked_view::<CoreError>(Some(view(DType::BF16)), "model.norm.weight", &[8]),
            ],
            [
                Err(LlamaError::TensorNotFound),
                Err(LlamaError::TensorDTypeMismatch),
                Err(LlamaError::TensorShapeMismatch),
            ],
            "absent, then f32, then 4 elements where 8 are expected"
        );
    }
}
