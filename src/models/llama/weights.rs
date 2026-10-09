use crate::core::{
    Backend,
    DType,
    Tensor,
};
use crate::models::llama::{
    LlamaConfig,
    LlamaWeightsError,
};
use crate::nn::{
    Embedding,
    Linear,
    RmsNorm,
    SwigluMlp,
};
use crate::weights::Weights;

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

struct Loader<'load, B: Backend> {
    backend: &'load mut B,
    weights: &'load Weights,
    bytes_uploaded: usize,
}

impl<B: Backend> LlamaWeights<B> {
    pub fn load(
        backend: &mut B,
        config: &LlamaConfig,
        weights: &Weights,
    ) -> Result<Self, LlamaWeightsError<B::Error>> {
        let kv_heads = config.num_key_value_heads();
        let head_dim = config.head_dim();
        let kv_dim =
            kv_heads
                .checked_mul(head_dim)
                .ok_or(LlamaWeightsError::DimensionOverflow {
                    kv_heads,
                    head_dim,
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
            .collect::<Result<Vec<LlamaLayerWeights<B>>, LlamaWeightsError<B::Error>>>()?;
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

impl<B: Backend> Loader<'_, B> {
    fn layer(
        &mut self,
        config: &LlamaConfig,
        index: usize,
        kv_dim: usize,
    ) -> Result<LlamaLayerWeights<B>, LlamaWeightsError<B::Error>> {
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
        let mlp =
            SwigluMlp::new(self.backend, gate, up, down).map_err(LlamaWeightsError::Backend)?;
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
    ) -> Result<Linear<B>, LlamaWeightsError<B::Error>> {
        Ok(Linear::new(self.tensor(name, &dims_expected)?))
    }

    fn rms_norm(
        &mut self,
        name: &str,
        config: &LlamaConfig,
    ) -> Result<RmsNorm<B>, LlamaWeightsError<B::Error>> {
        let weight = self.tensor(name, &[config.hidden_size()])?;
        RmsNorm::new(self.backend, weight, config.rms_norm_eps())
            .map_err(LlamaWeightsError::Backend)
    }

    fn tensor(
        &mut self,
        name: &str,
        dims_expected: &[usize],
    ) -> Result<Tensor<B>, LlamaWeightsError<B::Error>> {
        let view = self.weights.get(name)?;
        if view.dtype != DTYPE {
            return Err(LlamaWeightsError::DTypeMismatch {
                name: name.to_owned(),
                dtype: view.dtype,
                dtype_expected: DTYPE,
            });
        }
        if view.shape.dims() != dims_expected {
            return Err(LlamaWeightsError::ShapeMismatch {
                name: name.to_owned(),
                dims: view.shape.dims().to_vec(),
                dims_expected: dims_expected.to_vec(),
            });
        }
        let tensor = Tensor::upload(self.backend, view.bytes, view.dtype, view.shape)
            .map_err(LlamaWeightsError::Backend)?;
        self.bytes_uploaded = self
            .bytes_uploaded
            .checked_add(view.bytes.len())
            .ok_or(LlamaWeightsError::ByteCountOverflow)?;
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;

    use crate::backend::cpu::CpuBackend;
    use crate::core::DType;
    use crate::models::llama::{
        LlamaConfig,
        LlamaWeights,
        LlamaWeightsError,
    };
    use crate::weights::{
        Weights,
        WeightsError,
    };

    #[test]
    fn llama_weights_load_every_tensor_once() {
        let directory =
            std::env::temp_dir().join(format!("foundry-llama-weights-load-{}", std::process::id()));
        fs::create_dir_all(&directory).expect("the test directory is created");
        let config = json!({
            "model_type": "llama",
            "hidden_size": 4,
            "intermediate_size": 6,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500_000.0,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192
            },
            "max_position_embeddings": 16,
            "vocab_size": 8,
            "tie_word_embeddings": true,
            "eos_token_id": [7]
        });
        fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&config).expect("the config serializes"),
        )
        .expect("the config is written");
        let tensors: Vec<(String, &str, Vec<usize>)> = [
            ("model.embed_tokens.weight".to_owned(), "BF16", vec![8, 4]),
            ("model.norm.weight".to_owned(), "BF16", vec![4]),
        ]
        .into_iter()
        .chain((0..2).flat_map(|index| {
            [
                (
                    format!("model.layers.{index}.input_layernorm.weight"),
                    "BF16",
                    vec![4],
                ),
                (
                    format!("model.layers.{index}.self_attn.q_proj.weight"),
                    "BF16",
                    vec![4, 4],
                ),
                (
                    format!("model.layers.{index}.self_attn.k_proj.weight"),
                    "BF16",
                    vec![2, 4],
                ),
                (
                    format!("model.layers.{index}.self_attn.v_proj.weight"),
                    "BF16",
                    vec![2, 4],
                ),
                (
                    format!("model.layers.{index}.self_attn.o_proj.weight"),
                    "BF16",
                    vec![4, 4],
                ),
                (
                    format!("model.layers.{index}.post_attention_layernorm.weight"),
                    "BF16",
                    vec![4],
                ),
                (
                    format!("model.layers.{index}.mlp.gate_proj.weight"),
                    "BF16",
                    vec![6, 4],
                ),
                (
                    format!("model.layers.{index}.mlp.up_proj.weight"),
                    "BF16",
                    vec![6, 4],
                ),
                (
                    format!("model.layers.{index}.mlp.down_proj.weight"),
                    "BF16",
                    vec![4, 6],
                ),
            ]
        }))
        .collect();
        let (header, bytes_data) = tensors.iter().fold(
            (serde_json::Map::new(), 0_usize),
            |(mut header, begin), (name, dtype, dims)| {
                let element_bytes: usize = if *dtype == "F32" { 4 } else { 2 };
                let end = dims
                    .iter()
                    .try_fold(element_bytes, |bytes, &dim| bytes.checked_mul(dim))
                    .and_then(|bytes| begin.checked_add(bytes))
                    .expect("the tensor end fits in usize");
                header.insert(
                    name.clone(),
                    json!({ "dtype": dtype, "shape": dims, "data_offsets": [begin, end] }),
                );
                (header, end)
            },
        );
        let header = serde_json::to_vec(&header).expect("the header serializes");
        let header_len = u64::try_from(header.len()).expect("the header length fits in u64");
        fs::write(
            directory.join("model.safetensors"),
            header_len
                .to_le_bytes()
                .into_iter()
                .chain(header)
                .chain(vec![0_u8; bytes_data])
                .collect::<Vec<u8>>(),
        )
        .expect("the safetensors file is written");
        let weights = Weights::open(&directory).expect("the safetensors file is valid");
        let config = LlamaConfig::open(&directory).expect("the config is valid");

        let llama = LlamaWeights::load(&mut CpuBackend::new(), &config, &weights)
            .expect("the weights load");
        assert_eq!(llama.layers().len(), 2, "one entry per layer");
        assert_eq!(
            llama.bytes_uploaded(),
            bytes_data,
            "every tensor, the tied embedding included, is uploaded once"
        );
    }

    #[test]
    fn llama_weights_reject_a_tensor_with_the_wrong_shape() {
        let directory = std::env::temp_dir().join(format!(
            "foundry-llama-weights-shape-{}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).expect("the test directory is created");
        let config = json!({
            "model_type": "llama",
            "hidden_size": 4,
            "intermediate_size": 6,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500_000.0,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192
            },
            "max_position_embeddings": 16,
            "vocab_size": 8,
            "tie_word_embeddings": true,
            "eos_token_id": [7]
        });
        fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&config).expect("the config serializes"),
        )
        .expect("the config is written");
        let tensors: Vec<(String, &str, Vec<usize>)> = [
            ("model.embed_tokens.weight".to_owned(), "BF16", vec![8, 4]),
            ("model.norm.weight".to_owned(), "BF16", vec![4]),
        ]
        .into_iter()
        .chain((0..2).flat_map(|index| {
            [
                (
                    format!("model.layers.{index}.input_layernorm.weight"),
                    "BF16",
                    vec![4],
                ),
                (
                    format!("model.layers.{index}.self_attn.q_proj.weight"),
                    "BF16",
                    vec![4, 4],
                ),
                (
                    format!("model.layers.{index}.self_attn.k_proj.weight"),
                    "BF16",
                    vec![2, 4],
                ),
                (
                    format!("model.layers.{index}.self_attn.v_proj.weight"),
                    "BF16",
                    vec![2, 4],
                ),
                (
                    format!("model.layers.{index}.self_attn.o_proj.weight"),
                    "BF16",
                    vec![4, 4],
                ),
                (
                    format!("model.layers.{index}.post_attention_layernorm.weight"),
                    "BF16",
                    vec![4],
                ),
                (
                    format!("model.layers.{index}.mlp.gate_proj.weight"),
                    "BF16",
                    vec![6, 4],
                ),
                (
                    format!("model.layers.{index}.mlp.up_proj.weight"),
                    "BF16",
                    vec![6, 4],
                ),
                (
                    format!("model.layers.{index}.mlp.down_proj.weight"),
                    "BF16",
                    vec![4, 6],
                ),
            ]
        }))
        .map(|(name, dtype, dims)| {
            if name == "model.layers.1.self_attn.k_proj.weight" {
                (name, dtype, vec![4, 4])
            } else {
                (name, dtype, dims)
            }
        })
        .collect();
        let (header, bytes_data) = tensors.iter().fold(
            (serde_json::Map::new(), 0_usize),
            |(mut header, begin), (name, dtype, dims)| {
                let element_bytes: usize = if *dtype == "F32" { 4 } else { 2 };
                let end = dims
                    .iter()
                    .try_fold(element_bytes, |bytes, &dim| bytes.checked_mul(dim))
                    .and_then(|bytes| begin.checked_add(bytes))
                    .expect("the tensor end fits in usize");
                header.insert(
                    name.clone(),
                    json!({ "dtype": dtype, "shape": dims, "data_offsets": [begin, end] }),
                );
                (header, end)
            },
        );
        let header = serde_json::to_vec(&header).expect("the header serializes");
        let header_len = u64::try_from(header.len()).expect("the header length fits in u64");
        fs::write(
            directory.join("model.safetensors"),
            header_len
                .to_le_bytes()
                .into_iter()
                .chain(header)
                .chain(vec![0_u8; bytes_data])
                .collect::<Vec<u8>>(),
        )
        .expect("the safetensors file is written");
        let weights = Weights::open(&directory).expect("the safetensors file is valid");
        let config = LlamaConfig::open(&directory).expect("the config is valid");

        let result = LlamaWeights::load(&mut CpuBackend::new(), &config, &weights);
        assert!(
            matches!(
                &result,
                Err(LlamaWeightsError::ShapeMismatch { name, dims, dims_expected })
                    if name == "model.layers.1.self_attn.k_proj.weight"
                        && dims == &[4, 4]
                        && dims_expected == &[2, 4]
            ),
            "got {:?}",
            result.err()
        );
    }

    #[test]
    fn llama_weights_reject_a_non_bf16_tensor() {
        let directory = std::env::temp_dir().join(format!(
            "foundry-llama-weights-dtype-{}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).expect("the test directory is created");
        let config = json!({
            "model_type": "llama",
            "hidden_size": 4,
            "intermediate_size": 6,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500_000.0,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192
            },
            "max_position_embeddings": 16,
            "vocab_size": 8,
            "tie_word_embeddings": true,
            "eos_token_id": [7]
        });
        fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&config).expect("the config serializes"),
        )
        .expect("the config is written");
        let tensors: Vec<(String, &str, Vec<usize>)> = [
            ("model.embed_tokens.weight".to_owned(), "BF16", vec![8, 4]),
            ("model.norm.weight".to_owned(), "BF16", vec![4]),
        ]
        .into_iter()
        .chain((0..2).flat_map(|index| {
            [
                (
                    format!("model.layers.{index}.input_layernorm.weight"),
                    "BF16",
                    vec![4],
                ),
                (
                    format!("model.layers.{index}.self_attn.q_proj.weight"),
                    "BF16",
                    vec![4, 4],
                ),
                (
                    format!("model.layers.{index}.self_attn.k_proj.weight"),
                    "BF16",
                    vec![2, 4],
                ),
                (
                    format!("model.layers.{index}.self_attn.v_proj.weight"),
                    "BF16",
                    vec![2, 4],
                ),
                (
                    format!("model.layers.{index}.self_attn.o_proj.weight"),
                    "BF16",
                    vec![4, 4],
                ),
                (
                    format!("model.layers.{index}.post_attention_layernorm.weight"),
                    "BF16",
                    vec![4],
                ),
                (
                    format!("model.layers.{index}.mlp.gate_proj.weight"),
                    "BF16",
                    vec![6, 4],
                ),
                (
                    format!("model.layers.{index}.mlp.up_proj.weight"),
                    "BF16",
                    vec![6, 4],
                ),
                (
                    format!("model.layers.{index}.mlp.down_proj.weight"),
                    "BF16",
                    vec![4, 6],
                ),
            ]
        }))
        .map(|(name, dtype, dims)| {
            if name == "model.norm.weight" {
                (name, "F32", dims)
            } else {
                (name, dtype, dims)
            }
        })
        .collect();
        let (header, bytes_data) = tensors.iter().fold(
            (serde_json::Map::new(), 0_usize),
            |(mut header, begin), (name, dtype, dims)| {
                let element_bytes: usize = if *dtype == "F32" { 4 } else { 2 };
                let end = dims
                    .iter()
                    .try_fold(element_bytes, |bytes, &dim| bytes.checked_mul(dim))
                    .and_then(|bytes| begin.checked_add(bytes))
                    .expect("the tensor end fits in usize");
                header.insert(
                    name.clone(),
                    json!({ "dtype": dtype, "shape": dims, "data_offsets": [begin, end] }),
                );
                (header, end)
            },
        );
        let header = serde_json::to_vec(&header).expect("the header serializes");
        let header_len = u64::try_from(header.len()).expect("the header length fits in u64");
        fs::write(
            directory.join("model.safetensors"),
            header_len
                .to_le_bytes()
                .into_iter()
                .chain(header)
                .chain(vec![0_u8; bytes_data])
                .collect::<Vec<u8>>(),
        )
        .expect("the safetensors file is written");
        let weights = Weights::open(&directory).expect("the safetensors file is valid");
        let config = LlamaConfig::open(&directory).expect("the config is valid");

        let result = LlamaWeights::load(&mut CpuBackend::new(), &config, &weights);
        assert!(
            matches!(
                &result,
                Err(LlamaWeightsError::DTypeMismatch {
                    name,
                    dtype: DType::F32,
                    dtype_expected: DType::BF16,
                }) if name == "model.norm.weight"
            ),
            "got {:?}",
            result.err()
        );
    }

    #[test]
    fn llama_weights_reject_a_missing_tensor() {
        let directory = std::env::temp_dir().join(format!(
            "foundry-llama-weights-missing-{}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).expect("the test directory is created");
        let config = json!({
            "model_type": "llama",
            "hidden_size": 4,
            "intermediate_size": 6,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500_000.0,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192
            },
            "max_position_embeddings": 16,
            "vocab_size": 8,
            "tie_word_embeddings": true,
            "eos_token_id": [7]
        });
        fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&config).expect("the config serializes"),
        )
        .expect("the config is written");
        let tensors: Vec<(String, &str, Vec<usize>)> = [
            ("model.embed_tokens.weight".to_owned(), "BF16", vec![8, 4]),
            ("model.norm.weight".to_owned(), "BF16", vec![4]),
        ]
        .into_iter()
        .chain((0..2).flat_map(|index| {
            [
                (
                    format!("model.layers.{index}.input_layernorm.weight"),
                    "BF16",
                    vec![4],
                ),
                (
                    format!("model.layers.{index}.self_attn.q_proj.weight"),
                    "BF16",
                    vec![4, 4],
                ),
                (
                    format!("model.layers.{index}.self_attn.k_proj.weight"),
                    "BF16",
                    vec![2, 4],
                ),
                (
                    format!("model.layers.{index}.self_attn.v_proj.weight"),
                    "BF16",
                    vec![2, 4],
                ),
                (
                    format!("model.layers.{index}.self_attn.o_proj.weight"),
                    "BF16",
                    vec![4, 4],
                ),
                (
                    format!("model.layers.{index}.post_attention_layernorm.weight"),
                    "BF16",
                    vec![4],
                ),
                (
                    format!("model.layers.{index}.mlp.gate_proj.weight"),
                    "BF16",
                    vec![6, 4],
                ),
                (
                    format!("model.layers.{index}.mlp.up_proj.weight"),
                    "BF16",
                    vec![6, 4],
                ),
                (
                    format!("model.layers.{index}.mlp.down_proj.weight"),
                    "BF16",
                    vec![4, 6],
                ),
            ]
        }))
        .filter(|(name, _, _)| name != "model.norm.weight")
        .collect();
        let (header, bytes_data) = tensors.iter().fold(
            (serde_json::Map::new(), 0_usize),
            |(mut header, begin), (name, dtype, dims)| {
                let element_bytes: usize = if *dtype == "F32" { 4 } else { 2 };
                let end = dims
                    .iter()
                    .try_fold(element_bytes, |bytes, &dim| bytes.checked_mul(dim))
                    .and_then(|bytes| begin.checked_add(bytes))
                    .expect("the tensor end fits in usize");
                header.insert(
                    name.clone(),
                    json!({ "dtype": dtype, "shape": dims, "data_offsets": [begin, end] }),
                );
                (header, end)
            },
        );
        let header = serde_json::to_vec(&header).expect("the header serializes");
        let header_len = u64::try_from(header.len()).expect("the header length fits in u64");
        fs::write(
            directory.join("model.safetensors"),
            header_len
                .to_le_bytes()
                .into_iter()
                .chain(header)
                .chain(vec![0_u8; bytes_data])
                .collect::<Vec<u8>>(),
        )
        .expect("the safetensors file is written");
        let weights = Weights::open(&directory).expect("the safetensors file is valid");
        let config = LlamaConfig::open(&directory).expect("the config is valid");

        let result = LlamaWeights::load(&mut CpuBackend::new(), &config, &weights);
        assert!(
            matches!(
                &result,
                Err(LlamaWeightsError::Weights(WeightsError::TensorNotFound { name }))
                    if name == "model.norm.weight"
            ),
            "got {:?}",
            result.err()
        );
    }
}
