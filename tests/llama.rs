use foundry::backend::cpu::CpuBackend;
use foundry::models::llama::{
    LlamaConfig,
    LlamaError,
    LlamaWeights,
};
use foundry::weights::{
    Shard,
    Weights,
};
use hf_hub::HFClientSync;
use serde_json::json;

#[test]
#[ignore = "needs meta-llama/Llama-3.2-3B-Instruct in the Hugging Face cache"]
fn llama_weights_load_the_official_snapshot() {
    let directory = HFClientSync::new()
        .expect("the Hugging Face client is created")
        .model("meta-llama", "Llama-3.2-3B-Instruct")
        .snapshot_download()
        .local_files_only(true)
        .send()
        .expect("meta-llama/Llama-3.2-3B-Instruct is in the Hugging Face cache");
    let config = LlamaConfig::open(&directory).expect("the config is valid");
    let weights = Weights::open(&directory).expect("the safetensors files are valid");

    let llama =
        LlamaWeights::load(&mut CpuBackend::new(), &config, &weights).expect("the weights load");
    assert_eq!(llama.layers().len(), 28, "one entry per layer");
    assert_eq!(
        u64::try_from(llama.bytes_uploaded()).ok(),
        weights.bytes_declared(),
        "the uploaded bytes equal the total size declared by the index"
    );
}

#[test]
fn llama_weights_load_every_tensor_once() {
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
    struct TensorEntry {
        name: String,
        dtype: &'static str,
        dims: Vec<usize>,
    }
    struct HeaderFold {
        entries: serde_json::Map<String, serde_json::Value>,
        end: usize,
    }
    let tensors: Vec<TensorEntry> = [
        TensorEntry {
            name: "model.embed_tokens.weight".to_owned(),
            dtype: "BF16",
            dims: vec![8, 4],
        },
        TensorEntry {
            name: "model.norm.weight".to_owned(),
            dtype: "BF16",
            dims: vec![4],
        },
    ]
    .into_iter()
    .chain((0..2).flat_map(|index| {
        [
            TensorEntry {
                name: format!("model.layers.{index}.input_layernorm.weight"),
                dtype: "BF16",
                dims: vec![4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.self_attn.q_proj.weight"),
                dtype: "BF16",
                dims: vec![4, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.self_attn.k_proj.weight"),
                dtype: "BF16",
                dims: vec![2, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.self_attn.v_proj.weight"),
                dtype: "BF16",
                dims: vec![2, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.self_attn.o_proj.weight"),
                dtype: "BF16",
                dims: vec![4, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.post_attention_layernorm.weight"),
                dtype: "BF16",
                dims: vec![4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.mlp.gate_proj.weight"),
                dtype: "BF16",
                dims: vec![6, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.mlp.up_proj.weight"),
                dtype: "BF16",
                dims: vec![6, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.mlp.down_proj.weight"),
                dtype: "BF16",
                dims: vec![4, 6],
            },
        ]
    }))
    .collect();
    let header = tensors.iter().fold(
        HeaderFold {
            entries: serde_json::Map::new(),
            end: 0,
        },
        |mut fold, entry| {
            let element_bytes: usize = if entry.dtype == "F32" { 4 } else { 2 };
            let end = entry
                .dims
                .iter()
                .try_fold(element_bytes, |bytes, &dim| bytes.checked_mul(dim))
                .and_then(|bytes| fold.end.checked_add(bytes))
                .expect("the tensor end fits in usize");
            fold.entries.insert(
                entry.name.clone(),
                json!({ "dtype": entry.dtype, "shape": entry.dims, "data_offsets": [fold.end, end] }),
            );
            HeaderFold {
                entries: fold.entries,
                end,
            }
        },
    );
    let bytes_data = header.end;
    let header = serde_json::to_vec(&header.entries).expect("the header serializes");
    let header_len = u64::try_from(header.len()).expect("the header length fits in u64");
    let shard: Vec<u8> = header_len
        .to_le_bytes()
        .into_iter()
        .chain(header)
        .chain(vec![0_u8; bytes_data])
        .collect::<Vec<u8>>();
    let weights = Weights::from_shards(
        None,
        vec![Shard {
            name: "model.safetensors".to_owned(),
            bytes: shard,
        }],
    )
    .expect("the safetensors bytes are valid");
    let config = LlamaConfig::parse(&serde_json::to_vec(&config).expect("the config serializes"))
        .expect("the config is valid");

    let llama =
        LlamaWeights::load(&mut CpuBackend::new(), &config, &weights).expect("the weights load");
    assert_eq!(llama.layers().len(), 2, "one entry per layer");
    assert_eq!(
        llama.bytes_uploaded(),
        bytes_data,
        "every tensor, the tied embedding included, is uploaded once"
    );
}

#[test]
fn llama_weights_reject_a_tensor_with_the_wrong_shape() {
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
    struct TensorEntry {
        name: String,
        dtype: &'static str,
        dims: Vec<usize>,
    }
    struct HeaderFold {
        entries: serde_json::Map<String, serde_json::Value>,
        end: usize,
    }
    let tensors: Vec<TensorEntry> = [
        TensorEntry {
            name: "model.embed_tokens.weight".to_owned(),
            dtype: "BF16",
            dims: vec![8, 4],
        },
        TensorEntry {
            name: "model.norm.weight".to_owned(),
            dtype: "BF16",
            dims: vec![4],
        },
    ]
    .into_iter()
    .chain((0..2).flat_map(|index| {
        [
            TensorEntry {
                name: format!("model.layers.{index}.input_layernorm.weight"),
                dtype: "BF16",
                dims: vec![4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.self_attn.q_proj.weight"),
                dtype: "BF16",
                dims: vec![4, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.self_attn.k_proj.weight"),
                dtype: "BF16",
                dims: vec![2, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.self_attn.v_proj.weight"),
                dtype: "BF16",
                dims: vec![2, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.self_attn.o_proj.weight"),
                dtype: "BF16",
                dims: vec![4, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.post_attention_layernorm.weight"),
                dtype: "BF16",
                dims: vec![4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.mlp.gate_proj.weight"),
                dtype: "BF16",
                dims: vec![6, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.mlp.up_proj.weight"),
                dtype: "BF16",
                dims: vec![6, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.mlp.down_proj.weight"),
                dtype: "BF16",
                dims: vec![4, 6],
            },
        ]
    }))
    .map(|entry| {
        if entry.name == "model.layers.1.self_attn.k_proj.weight" {
            TensorEntry {
                dims: vec![4, 4],
                ..entry
            }
        } else {
            entry
        }
    })
    .collect();
    let header = tensors.iter().fold(
        HeaderFold {
            entries: serde_json::Map::new(),
            end: 0,
        },
        |mut fold, entry| {
            let element_bytes: usize = if entry.dtype == "F32" { 4 } else { 2 };
            let end = entry
                .dims
                .iter()
                .try_fold(element_bytes, |bytes, &dim| bytes.checked_mul(dim))
                .and_then(|bytes| fold.end.checked_add(bytes))
                .expect("the tensor end fits in usize");
            fold.entries.insert(
                entry.name.clone(),
                json!({ "dtype": entry.dtype, "shape": entry.dims, "data_offsets": [fold.end, end] }),
            );
            HeaderFold {
                entries: fold.entries,
                end,
            }
        },
    );
    let bytes_data = header.end;
    let header = serde_json::to_vec(&header.entries).expect("the header serializes");
    let header_len = u64::try_from(header.len()).expect("the header length fits in u64");
    let shard: Vec<u8> = header_len
        .to_le_bytes()
        .into_iter()
        .chain(header)
        .chain(vec![0_u8; bytes_data])
        .collect::<Vec<u8>>();
    let weights = Weights::from_shards(
        None,
        vec![Shard {
            name: "model.safetensors".to_owned(),
            bytes: shard,
        }],
    )
    .expect("the safetensors bytes are valid");
    let config = LlamaConfig::parse(&serde_json::to_vec(&config).expect("the config serializes"))
        .expect("the config is valid");

    let result = LlamaWeights::load(&mut CpuBackend::new(), &config, &weights);
    assert_eq!(
        result.err(),
        Some(LlamaError::TensorShapeMismatch),
        "k_proj of shape [4, 4] where the config expects [2, 4] is rejected"
    );
}

#[test]
fn llama_weights_reject_a_non_bf16_tensor() {
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
    struct TensorEntry {
        name: String,
        dtype: &'static str,
        dims: Vec<usize>,
    }
    struct HeaderFold {
        entries: serde_json::Map<String, serde_json::Value>,
        end: usize,
    }
    let tensors: Vec<TensorEntry> = [
        TensorEntry {
            name: "model.embed_tokens.weight".to_owned(),
            dtype: "BF16",
            dims: vec![8, 4],
        },
        TensorEntry {
            name: "model.norm.weight".to_owned(),
            dtype: "BF16",
            dims: vec![4],
        },
    ]
    .into_iter()
    .chain((0..2).flat_map(|index| {
        [
            TensorEntry {
                name: format!("model.layers.{index}.input_layernorm.weight"),
                dtype: "BF16",
                dims: vec![4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.self_attn.q_proj.weight"),
                dtype: "BF16",
                dims: vec![4, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.self_attn.k_proj.weight"),
                dtype: "BF16",
                dims: vec![2, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.self_attn.v_proj.weight"),
                dtype: "BF16",
                dims: vec![2, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.self_attn.o_proj.weight"),
                dtype: "BF16",
                dims: vec![4, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.post_attention_layernorm.weight"),
                dtype: "BF16",
                dims: vec![4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.mlp.gate_proj.weight"),
                dtype: "BF16",
                dims: vec![6, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.mlp.up_proj.weight"),
                dtype: "BF16",
                dims: vec![6, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.mlp.down_proj.weight"),
                dtype: "BF16",
                dims: vec![4, 6],
            },
        ]
    }))
    .map(|entry| {
        if entry.name == "model.norm.weight" {
            TensorEntry {
                dtype: "F32",
                ..entry
            }
        } else {
            entry
        }
    })
    .collect();
    let header = tensors.iter().fold(
        HeaderFold {
            entries: serde_json::Map::new(),
            end: 0,
        },
        |mut fold, entry| {
            let element_bytes: usize = if entry.dtype == "F32" { 4 } else { 2 };
            let end = entry
                .dims
                .iter()
                .try_fold(element_bytes, |bytes, &dim| bytes.checked_mul(dim))
                .and_then(|bytes| fold.end.checked_add(bytes))
                .expect("the tensor end fits in usize");
            fold.entries.insert(
                entry.name.clone(),
                json!({ "dtype": entry.dtype, "shape": entry.dims, "data_offsets": [fold.end, end] }),
            );
            HeaderFold {
                entries: fold.entries,
                end,
            }
        },
    );
    let bytes_data = header.end;
    let header = serde_json::to_vec(&header.entries).expect("the header serializes");
    let header_len = u64::try_from(header.len()).expect("the header length fits in u64");
    let shard: Vec<u8> = header_len
        .to_le_bytes()
        .into_iter()
        .chain(header)
        .chain(vec![0_u8; bytes_data])
        .collect::<Vec<u8>>();
    let weights = Weights::from_shards(
        None,
        vec![Shard {
            name: "model.safetensors".to_owned(),
            bytes: shard,
        }],
    )
    .expect("the safetensors bytes are valid");
    let config = LlamaConfig::parse(&serde_json::to_vec(&config).expect("the config serializes"))
        .expect("the config is valid");

    let result = LlamaWeights::load(&mut CpuBackend::new(), &config, &weights);
    assert_eq!(
        result.err(),
        Some(LlamaError::TensorDTypeMismatch),
        "an f32 model.norm.weight is rejected"
    );
}

#[test]
fn llama_weights_reject_a_missing_tensor() {
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
    struct TensorEntry {
        name: String,
        dtype: &'static str,
        dims: Vec<usize>,
    }
    struct HeaderFold {
        entries: serde_json::Map<String, serde_json::Value>,
        end: usize,
    }
    let tensors: Vec<TensorEntry> = [
        TensorEntry {
            name: "model.embed_tokens.weight".to_owned(),
            dtype: "BF16",
            dims: vec![8, 4],
        },
        TensorEntry {
            name: "model.norm.weight".to_owned(),
            dtype: "BF16",
            dims: vec![4],
        },
    ]
    .into_iter()
    .chain((0..2).flat_map(|index| {
        [
            TensorEntry {
                name: format!("model.layers.{index}.input_layernorm.weight"),
                dtype: "BF16",
                dims: vec![4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.self_attn.q_proj.weight"),
                dtype: "BF16",
                dims: vec![4, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.self_attn.k_proj.weight"),
                dtype: "BF16",
                dims: vec![2, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.self_attn.v_proj.weight"),
                dtype: "BF16",
                dims: vec![2, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.self_attn.o_proj.weight"),
                dtype: "BF16",
                dims: vec![4, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.post_attention_layernorm.weight"),
                dtype: "BF16",
                dims: vec![4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.mlp.gate_proj.weight"),
                dtype: "BF16",
                dims: vec![6, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.mlp.up_proj.weight"),
                dtype: "BF16",
                dims: vec![6, 4],
            },
            TensorEntry {
                name: format!("model.layers.{index}.mlp.down_proj.weight"),
                dtype: "BF16",
                dims: vec![4, 6],
            },
        ]
    }))
    .filter(|entry| entry.name != "model.norm.weight")
    .collect();
    let header = tensors.iter().fold(
        HeaderFold {
            entries: serde_json::Map::new(),
            end: 0,
        },
        |mut fold, entry| {
            let element_bytes: usize = if entry.dtype == "F32" { 4 } else { 2 };
            let end = entry
                .dims
                .iter()
                .try_fold(element_bytes, |bytes, &dim| bytes.checked_mul(dim))
                .and_then(|bytes| fold.end.checked_add(bytes))
                .expect("the tensor end fits in usize");
            fold.entries.insert(
                entry.name.clone(),
                json!({ "dtype": entry.dtype, "shape": entry.dims, "data_offsets": [fold.end, end] }),
            );
            HeaderFold {
                entries: fold.entries,
                end,
            }
        },
    );
    let bytes_data = header.end;
    let header = serde_json::to_vec(&header.entries).expect("the header serializes");
    let header_len = u64::try_from(header.len()).expect("the header length fits in u64");
    let shard: Vec<u8> = header_len
        .to_le_bytes()
        .into_iter()
        .chain(header)
        .chain(vec![0_u8; bytes_data])
        .collect::<Vec<u8>>();
    let weights = Weights::from_shards(
        None,
        vec![Shard {
            name: "model.safetensors".to_owned(),
            bytes: shard,
        }],
    )
    .expect("the safetensors bytes are valid");
    let config = LlamaConfig::parse(&serde_json::to_vec(&config).expect("the config serializes"))
        .expect("the config is valid");

    let result = LlamaWeights::load(&mut CpuBackend::new(), &config, &weights);
    assert_eq!(
        result.err(),
        Some(LlamaError::TensorNotFound),
        "a missing model.norm.weight is rejected"
    );
}
