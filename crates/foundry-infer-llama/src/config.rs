use serde::Deserialize;

use crate::error::ConfigError;

pub const GROUP_SIZE: u32 = 64;
pub const BITS: u32 = 4;
pub const HEAD_DIM: u32 = 128;
const TILE: u32 = 64;

#[derive(Deserialize)]
struct QuantizationDto {
    group_size: u32,
    bits: u32,
    mode: Option<String>,
}

#[derive(Deserialize)]
struct RopeScalingDto {
    rope_type: String,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position_embeddings: u32,
}

#[derive(Deserialize)]
struct ConfigDto {
    model_type: String,
    hidden_act: String,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: Option<u32>,
    vocab_size: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_scaling: RopeScalingDto,
    rope_traditional: Option<bool>,
    tie_word_embeddings: bool,
    attention_bias: bool,
    mlp_bias: bool,
    quantization: QuantizationDto,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeConfig {
    pub theta: f32,
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_context: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LlamaConfig {
    pub layers: u32,
    pub hidden: u32,
    pub intermediate: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub vocab: u32,
    pub rms_eps: f32,
    pub rope: RopeConfig,
}

impl LlamaConfig {
    pub fn parse(text: &str) -> Result<Self, ConfigError> {
        let dto: ConfigDto = serde_json::from_str(text).map_err(ConfigError::Parse)?;
        Self::try_from(dto)
    }

    pub fn query_dim(&self) -> u32 { self.heads.saturating_mul(self.head_dim) }

    pub fn kv_dim(&self) -> u32 { self.kv_heads.saturating_mul(self.head_dim) }

    pub fn gqa_factor(&self) -> u32 { self.heads.checked_div(self.kv_heads).unwrap_or(0) }
}

fn require<T: PartialEq + std::fmt::Debug>(
    field: &'static str,
    actual: T,
    expected: T,
) -> Result<(), ConfigError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ConfigError::Unsupported {
            field,
            expected: format!("{expected:?}"),
            actual: format!("{actual:?}"),
        })
    }
}

fn require_multiple(
    field: &'static str,
    value: u32,
    divisor: u32,
) -> Result<(), ConfigError> {
    if value > 0 && value.checked_rem(divisor) == Some(0) {
        Ok(())
    } else {
        Err(ConfigError::Unsupported {
            field,
            expected: format!("a positive multiple of {divisor}"),
            actual: value.to_string(),
        })
    }
}

fn require_positive(
    field: &'static str,
    value: f32,
) -> Result<(), ConfigError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ConfigError::Unsupported {
            field,
            expected: "a positive finite number".to_owned(),
            actual: value.to_string(),
        })
    }
}

impl TryFrom<ConfigDto> for LlamaConfig {
    type Error = ConfigError;

    fn try_from(dto: ConfigDto) -> Result<Self, Self::Error> {
        require("model_type", dto.model_type.as_str(), "llama")?;
        require("hidden_act", dto.hidden_act.as_str(), "silu")?;
        require("tie_word_embeddings", dto.tie_word_embeddings, true)?;
        require("attention_bias", dto.attention_bias, false)?;
        require("mlp_bias", dto.mlp_bias, false)?;
        require(
            "rope_traditional",
            dto.rope_traditional.unwrap_or(false),
            false,
        )?;
        require("quantization.bits", dto.quantization.bits, BITS)?;
        require(
            "quantization.group_size",
            dto.quantization.group_size,
            GROUP_SIZE,
        )?;
        require(
            "quantization.mode",
            dto.quantization.mode.as_deref().unwrap_or("affine"),
            "affine",
        )?;
        require(
            "rope_scaling.rope_type",
            dto.rope_scaling.rope_type.as_str(),
            "llama3",
        )?;
        let head_dim = dto.head_dim.unwrap_or(HEAD_DIM);
        require("head_dim", head_dim, HEAD_DIM)?;
        require_multiple("num_hidden_layers", dto.num_hidden_layers, 1)?;
        require_multiple("num_key_value_heads", dto.num_key_value_heads, 1)?;
        require_multiple(
            "num_attention_heads",
            dto.num_attention_heads,
            dto.num_key_value_heads,
        )?;
        require_multiple("hidden_size", dto.hidden_size, TILE)?;
        require_multiple("intermediate_size", dto.intermediate_size, TILE)?;
        require_multiple("vocab_size", dto.vocab_size, 8)?;
        require_positive("rms_norm_eps", dto.rms_norm_eps)?;
        require_positive("rope_theta", dto.rope_theta)?;
        require_positive("rope_scaling.factor", dto.rope_scaling.factor)?;
        require_positive(
            "rope_scaling.low_freq_factor",
            dto.rope_scaling.low_freq_factor,
        )?;
        require_positive(
            "rope_scaling.high_freq_factor",
            dto.rope_scaling.high_freq_factor,
        )?;
        if dto.rope_scaling.high_freq_factor <= dto.rope_scaling.low_freq_factor {
            return Err(ConfigError::Unsupported {
                field: "rope_scaling.high_freq_factor",
                expected: "a value above low_freq_factor".to_owned(),
                actual: dto.rope_scaling.high_freq_factor.to_string(),
            });
        }
        require_multiple(
            "rope_scaling.original_max_position_embeddings",
            dto.rope_scaling.original_max_position_embeddings,
            1,
        )?;
        let config = Self {
            layers: dto.num_hidden_layers,
            hidden: dto.hidden_size,
            intermediate: dto.intermediate_size,
            heads: dto.num_attention_heads,
            kv_heads: dto.num_key_value_heads,
            head_dim,
            vocab: dto.vocab_size,
            rms_eps: dto.rms_norm_eps,
            rope: RopeConfig {
                theta: dto.rope_theta,
                factor: dto.rope_scaling.factor,
                low_freq_factor: dto.rope_scaling.low_freq_factor,
                high_freq_factor: dto.rope_scaling.high_freq_factor,
                original_context: dto.rope_scaling.original_max_position_embeddings,
            },
        };
        let query_dim = dto
            .num_attention_heads
            .checked_mul(head_dim)
            .ok_or_else(|| ConfigError::Unsupported {
                field: "num_attention_heads",
                expected: "a query dimension that fits in u32".to_owned(),
                actual: dto.num_attention_heads.to_string(),
            })?;
        require_multiple("num_attention_heads * head_dim", query_dim, TILE)?;
        Ok(config)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use serde_json::{
        Value,
        json,
    };

    use super::LlamaConfig;
    use crate::error::ConfigError;
    use crate::test_support::set;

    pub(crate) fn pinned() -> Value {
        json!({
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": false,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "mlp_bias": false,
            "model_type": "llama",
            "num_attention_heads": 24,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "quantization": {"group_size": 64, "bits": 4},
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
            "vocab_size": 128256
        })
    }

    pub(crate) fn tiny() -> Value {
        let mut config = pinned();
        set(&mut config, "/hidden_size", json!(128));
        set(&mut config, "/intermediate_size", json!(192));
        set(&mut config, "/num_attention_heads", json!(2));
        set(&mut config, "/num_key_value_heads", json!(1));
        set(&mut config, "/num_hidden_layers", json!(2));
        set(&mut config, "/vocab_size", json!(64));
        config
    }

    fn parse(value: &Value) -> Result<LlamaConfig, ConfigError> {
        LlamaConfig::parse(&value.to_string())
    }

    #[test]
    fn the_pinned_configuration_is_supported() -> Result<(), ConfigError> {
        let config = parse(&pinned())?;
        assert_eq!(config.layers, 28, "layers");
        assert_eq!(config.query_dim(), 3072, "query dimension");
        assert_eq!(config.kv_dim(), 1024, "kv dimension");
        assert_eq!(config.gqa_factor(), 3, "gqa factor");
        assert_eq!(config.rope.original_context, 8192, "original context");
        Ok(())
    }

    #[test]
    fn unsupported_configurations_are_rejected() {
        let cases: [(&str, Value); 9] = [
            ("model_type", json!("mistral")),
            ("tie_word_embeddings", json!(false)),
            ("attention_bias", json!(true)),
            ("head_dim", json!(64)),
            ("num_key_value_heads", json!(5)),
            ("hidden_size", json!(3000)),
            ("rms_norm_eps", json!(0.0)),
            ("quantization", json!({"group_size": 32, "bits": 4})),
            (
                "quantization",
                json!({"group_size": 64, "bits": 4, "mode": "mxfp4"}),
            ),
        ];
        for (field, value) in cases {
            let mut config = pinned();
            set(&mut config, &format!("/{field}"), value.clone());
            assert!(
                matches!(parse(&config), Err(ConfigError::Unsupported { .. })),
                "{field} = {value} must be rejected"
            );
        }
        let mut config = pinned();
        set(&mut config, "/rope_scaling/rope_type", json!("linear"));
        assert!(parse(&config).is_err(), "non-llama3 RoPE must be rejected");
        let mut config = pinned();
        if let Some(object) = config.as_object_mut() {
            object.remove("vocab_size");
        }
        assert!(
            matches!(parse(&config), Err(ConfigError::Parse(_))),
            "missing fields must be rejected"
        );
    }
}
