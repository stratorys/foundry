use std::fs;
use std::path::Path;

use serde::Deserialize;
use tracing::error;

use crate::models::llama::LlamaError;

const CONFIG_FILE: &str = "config.json";
const CONFIG_BYTES_MAX: u64 = 1024 * 1024;
const MODEL_TYPE: &str = "llama";
const ROPE_TYPE: &str = "llama3";

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_scaling: Llama3RopeScaling,
    max_position_embeddings: usize,
    vocab_size: usize,
    eos_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Llama3RopeScaling {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
}

#[derive(Deserialize)]
struct ModelTypeFile {
    model_type: String,
}

#[derive(Deserialize)]
struct ConfigFile {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_scaling: RopeScalingFile,
    max_position_embeddings: usize,
    vocab_size: usize,
    tie_word_embeddings: bool,
    eos_token_id: EosTokenIds,
}

#[derive(Deserialize)]
struct RopeScalingFile {
    rope_type: String,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position_embeddings: usize,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EosTokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl From<EosTokenIds> for Vec<u32> {
    fn from(eos_token_ids: EosTokenIds) -> Self {
        match eos_token_ids {
            EosTokenIds::One(eos_token_id) => vec![eos_token_id],
            EosTokenIds::Many(eos_token_ids) => eos_token_ids,
        }
    }
}

impl LlamaConfig {
    pub fn open(directory: &Path) -> Result<Self, LlamaError> {
        let path = directory.join(CONFIG_FILE);
        let read_error = |error| {
            error!(
                message = "Reading the config file failed.",
                path = %path.display(),
                %error,
            );
            LlamaError::ConfigRead
        };
        let bytes = fs::metadata(&path).map_err(read_error)?.len();
        if bytes > CONFIG_BYTES_MAX {
            error!(
                message = "Config file exceeds the maximum size.",
                path = %path.display(),
                bytes,
                bytes_max = CONFIG_BYTES_MAX,
            );
            return Err(LlamaError::ConfigTooLarge);
        }
        let json = fs::read(&path).map_err(read_error)?;
        Self::parse(&json)
    }

    pub fn parse(json: &[u8]) -> Result<Self, LlamaError> {
        let ModelTypeFile {
            model_type,
        } = serde_json::from_slice(json).map_err(config_json_error)?;
        check_model_type(&model_type)?;
        let file: ConfigFile = serde_json::from_slice(json).map_err(config_json_error)?;
        Self::try_from(file)
    }

    pub const fn hidden_size(&self) -> usize { self.hidden_size }

    pub const fn intermediate_size(&self) -> usize { self.intermediate_size }

    pub const fn num_hidden_layers(&self) -> usize { self.num_hidden_layers }

    pub const fn num_attention_heads(&self) -> usize { self.num_attention_heads }

    pub const fn num_key_value_heads(&self) -> usize { self.num_key_value_heads }

    pub const fn head_dim(&self) -> usize { self.head_dim }

    pub const fn rms_norm_eps(&self) -> f32 { self.rms_norm_eps }

    pub const fn rope_theta(&self) -> f32 { self.rope_theta }

    pub const fn rope_scaling(&self) -> &Llama3RopeScaling { &self.rope_scaling }

    pub const fn max_position_embeddings(&self) -> usize { self.max_position_embeddings }

    pub const fn vocab_size(&self) -> usize { self.vocab_size }

    pub fn eos_token_ids(&self) -> &[u32] { &self.eos_token_ids }
}

impl TryFrom<ConfigFile> for LlamaConfig {
    type Error = LlamaError;

    fn try_from(file: ConfigFile) -> Result<Self, Self::Error> {
        check_model_type(&file.model_type)?;
        if file.num_attention_heads.checked_mul(file.head_dim) != Some(file.hidden_size) {
            error!(
                message = "Hidden size does not equal attention heads times head dim.",
                hidden_size = file.hidden_size,
                num_attention_heads = file.num_attention_heads,
                head_dim = file.head_dim,
            );
            return Err(LlamaError::HiddenSizeMismatch);
        }
        if file
            .num_attention_heads
            .checked_rem(file.num_key_value_heads)
            != Some(0)
        {
            error!(
                message = "Attention heads are not a multiple of key-value heads.",
                num_attention_heads = file.num_attention_heads,
                num_key_value_heads = file.num_key_value_heads,
            );
            return Err(LlamaError::HeadRatio);
        }
        if file.rope_scaling.rope_type != ROPE_TYPE {
            error!(
                message = "RoPE type is not llama3.",
                rope_type = file.rope_scaling.rope_type,
            );
            return Err(LlamaError::RopeType);
        }
        if !file.tie_word_embeddings {
            error!(message = "Word embeddings are not tied.");
            return Err(LlamaError::UntiedEmbeddings);
        }
        Ok(Self {
            hidden_size: file.hidden_size,
            intermediate_size: file.intermediate_size,
            num_hidden_layers: file.num_hidden_layers,
            num_attention_heads: file.num_attention_heads,
            num_key_value_heads: file.num_key_value_heads,
            head_dim: file.head_dim,
            rms_norm_eps: file.rms_norm_eps,
            rope_theta: file.rope_theta,
            rope_scaling: Llama3RopeScaling {
                factor: file.rope_scaling.factor,
                low_freq_factor: file.rope_scaling.low_freq_factor,
                high_freq_factor: file.rope_scaling.high_freq_factor,
                original_max_position_embeddings: file
                    .rope_scaling
                    .original_max_position_embeddings,
            },
            max_position_embeddings: file.max_position_embeddings,
            vocab_size: file.vocab_size,
            eos_token_ids: file.eos_token_id.into(),
        })
    }
}

fn config_json_error(error: serde_json::Error) -> LlamaError {
    error!(message = "Config is not valid JSON or misses a field.", %error);
    LlamaError::ConfigJson
}

fn check_model_type(model_type: &str) -> Result<(), LlamaError> {
    if model_type == MODEL_TYPE {
        Ok(())
    } else {
        error!(message = "Model type is not llama.", model_type);
        Err(LlamaError::ModelType)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{
        Value,
        json,
    };

    use crate::models::llama::{
        Llama3RopeScaling,
        LlamaConfig,
        LlamaError,
    };

    fn instruct_config() -> Value {
        json!({
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 128000,
            "eos_token_id": [128001, 128008, 128009],
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 3072,
            "initializer_range": 0.02,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "mlp_bias": false,
            "model_type": "llama",
            "num_attention_heads": 24,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            },
            "rope_theta": 500000.0,
            "tie_word_embeddings": true,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.45.0.dev0",
            "use_cache": true,
            "vocab_size": 128256
        })
    }

    fn with(
        config: Value,
        field: &str,
        value: Value,
    ) -> Value {
        let mut config = config;
        config
            .as_object_mut()
            .expect("the config is an object")
            .insert(field.to_owned(), value);
        config
    }

    fn parse(config: &Value) -> Result<LlamaConfig, LlamaError> {
        LlamaConfig::parse(&serde_json::to_vec(config).expect("the config serializes"))
    }

    #[test]
    fn instruct_values_are_accepted() {
        let config = parse(&instruct_config()).expect("the config is valid");
        assert_eq!(config.hidden_size(), 3072, "hidden size");
        assert_eq!(config.intermediate_size(), 8192, "intermediate size");
        assert_eq!(config.num_hidden_layers(), 28, "layers");
        assert_eq!(config.num_attention_heads(), 24, "attention heads");
        assert_eq!(config.num_key_value_heads(), 8, "key-value heads");
        assert_eq!(config.head_dim(), 128, "head dim");
        assert_eq!(config.rms_norm_eps(), 1e-5, "rms norm eps");
        assert_eq!(config.rope_theta(), 500_000.0, "rope theta");
        assert_eq!(
            config.rope_scaling(),
            &Llama3RopeScaling {
                factor: 32.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8192,
            },
            "rope scaling"
        );
        assert_eq!(config.max_position_embeddings(), 131_072, "max positions");
        assert_eq!(config.vocab_size(), 128_256, "vocab size");
        assert_eq!(
            config.eos_token_ids(),
            &[128_001, 128_008, 128_009],
            "eos token ids"
        );
    }

    #[test]
    fn single_eos_token_id_becomes_a_list() {
        let config = parse(&with(instruct_config(), "eos_token_id", json!(128_001)))
            .expect("the config is valid");
        assert_eq!(config.eos_token_ids(), &[128_001], "eos token ids");
    }

    #[test]
    fn other_model_type_is_rejected_before_missing_fields() {
        let result = parse(&json!({ "model_type": "qwen3", "hidden_size": 1024 }));
        assert_eq!(result, Err(LlamaError::ModelType), "the config is rejected");
    }

    #[test]
    fn hidden_size_not_matching_heads_is_rejected() {
        let result = parse(&with(instruct_config(), "hidden_size", json!(3000)));
        assert_eq!(
            result,
            Err(LlamaError::HiddenSizeMismatch),
            "the config is rejected"
        );
    }

    #[test]
    fn heads_not_a_multiple_of_key_value_heads_are_rejected() {
        let result = parse(&with(instruct_config(), "num_key_value_heads", json!(7)));
        assert_eq!(result, Err(LlamaError::HeadRatio), "the config is rejected");
    }

    #[test]
    fn zero_key_value_heads_are_rejected() {
        let result = parse(&with(instruct_config(), "num_key_value_heads", json!(0)));
        assert_eq!(result, Err(LlamaError::HeadRatio), "the config is rejected");
    }

    #[test]
    fn other_rope_type_is_rejected() {
        let rope_scaling = json!({
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "default"
        });
        let result = parse(&with(instruct_config(), "rope_scaling", rope_scaling));
        assert_eq!(result, Err(LlamaError::RopeType), "the config is rejected");
    }

    #[test]
    fn untied_embeddings_are_rejected() {
        let result = parse(&with(
            instruct_config(),
            "tie_word_embeddings",
            json!(false),
        ));
        assert_eq!(
            result,
            Err(LlamaError::UntiedEmbeddings),
            "the config is rejected"
        );
    }

    #[test]
    fn missing_rope_scaling_is_rejected() {
        let mut config = instruct_config();
        config
            .as_object_mut()
            .expect("the config is an object")
            .remove("rope_scaling");
        let result = parse(&config);
        assert_eq!(
            result,
            Err(LlamaError::ConfigJson),
            "the config is rejected"
        );
    }
}
