use foundry::backend::cpu::CpuBackend;
use foundry::core::{
    DType,
    Shape,
    Tensor,
};
use foundry::generate::CausalLm;
use foundry::models::llama::{
    Llama,
    LlamaConfig,
    LlamaError,
    LlamaWeights,
};
use foundry::weights::{
    Shard,
    Weights,
};
use half::bf16;
use hf_hub::HFClientSync;
use serde_json::json;

#[test]
#[ignore = "needs HuggingFaceTB/SmolLM2-135M-Instruct in the Hugging Face cache"]
fn llama_weights_load_the_smollm2_snapshot() {
    let directory = HFClientSync::new()
        .expect("the Hugging Face client is created")
        .model("HuggingFaceTB", "SmolLM2-135M-Instruct")
        .snapshot_download()
        .local_files_only(true)
        .send()
        .expect("HuggingFaceTB/SmolLM2-135M-Instruct is in the Hugging Face cache");
    let config = LlamaConfig::open(&directory).expect("the config is valid");
    let weights = Weights::open(&directory).expect("the safetensors files are valid");

    let llama =
        LlamaWeights::load(&mut CpuBackend::new(), &config, &weights).expect("the weights load");
    assert_eq!(llama.layers().len(), 30, "one entry per layer");
    assert_eq!(
        llama.bytes_uploaded(),
        weights
            .tensors()
            .map(|tensor| tensor.bytes.len())
            .sum::<usize>(),
        "every tensor of the snapshot is uploaded once"
    );
}

#[test]
#[ignore = "needs HuggingFaceTB/SmolLM2-135M-Instruct in the Hugging Face cache"]
fn llama_logprobs_match_mlx_on_smollm2() {
    const PROMPT_IDS: [u32; 5] = [504, 3575, 282, 4649, 314];
    const ABSOLUTE_TOLERANCE: f32 = 0.125;
    const RELATIVE_TOLERANCE: f32 = 1e-2;

    #[derive(Debug, PartialEq)]
    struct MlxLogprob {
        id: usize,
        logprob: f32,
    }

    #[derive(Debug, PartialEq)]
    struct LogprobMismatch {
        id: usize,
        mlx: f32,
        foundry: Option<f32>,
    }

    let mlx_top_logprobs = [
        MlxLogprob {
            id: 7042,
            logprob: -0.7742,
        },
        MlxLogprob {
            id: 260,
            logprob: -1.3367,
        },
        MlxLogprob {
            id: 3807,
            logprob: -2.4617,
        },
        MlxLogprob {
            id: 1217,
            logprob: -3.9617,
        },
        MlxLogprob {
            id: 2250,
            logprob: -4.2117,
        },
        MlxLogprob {
            id: 253,
            logprob: -4.7117,
        },
        MlxLogprob {
            id: 1343,
            logprob: -4.9617,
        },
        MlxLogprob {
            id: 5145,
            logprob: -5.1492,
        },
        MlxLogprob {
            id: 13010,
            logprob: -5.1492,
        },
        MlxLogprob {
            id: 3692,
            logprob: -5.2117,
        },
    ];

    let directory = HFClientSync::new()
        .expect("the Hugging Face client is created")
        .model("HuggingFaceTB", "SmolLM2-135M-Instruct")
        .snapshot_download()
        .local_files_only(true)
        .send()
        .expect("HuggingFaceTB/SmolLM2-135M-Instruct is in the Hugging Face cache");
    let config = LlamaConfig::open(&directory).expect("the config is valid");
    let weights = Weights::open(&directory).expect("the safetensors files are valid");
    let backend = &mut CpuBackend::new();
    let seq_len = PROMPT_IDS.len();
    let llama_weights = LlamaWeights::load(backend, &config, &weights).expect("the weights load");
    let model = Llama::new(backend, &config, llama_weights, seq_len).expect("the model is built");
    let mut cache = model.cache(backend).expect("the cache is allocated");
    let shape = |dims: &[usize]| Shape::try_from(dims).expect("the shape is valid");

    let token_ids = Tensor::upload(
        backend,
        &PROMPT_IDS
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect::<Vec<u8>>(),
        DType::U32,
        shape(&[seq_len]),
    )
    .expect("the token ids upload");
    let positions = Tensor::upload(
        backend,
        &(0..seq_len)
            .map(|position| u32::try_from(position).expect("the position fits in u32"))
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<u8>>(),
        DType::U32,
        shape(&[seq_len]),
    )
    .expect("the positions upload");
    let mask = Tensor::upload(
        backend,
        &(0..seq_len)
            .flat_map(|row| {
                (0..seq_len).map(move |col| {
                    if col <= row {
                        bf16::ZERO
                    } else {
                        bf16::NEG_INFINITY
                    }
                })
            })
            .flat_map(bf16::to_le_bytes)
            .collect::<Vec<u8>>(),
        DType::BF16,
        shape(&[seq_len, seq_len]),
    )
    .expect("the causal mask uploads");

    let logits: Vec<f32> = model
        .forward(backend, &token_ids, &positions, &mask, &mut cache, 0)
        .expect("the forward pass runs")
        .download(backend)
        .expect("the logits download")
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| bf16::from_le_bytes(*bytes).to_f32())
        .collect();
    let logit_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = logit_max
        + logits
            .iter()
            .map(|logit| (logit - logit_max).exp())
            .sum::<f32>()
            .ln();
    let mismatches: Vec<LogprobMismatch> = mlx_top_logprobs
        .iter()
        .map(|mlx| LogprobMismatch {
            id: mlx.id,
            mlx: mlx.logprob,
            foundry: logits.get(mlx.id).map(|logit| logit - log_sum_exp),
        })
        .filter(|mismatch| {
            mismatch.foundry.is_none_or(|foundry| {
                (foundry - mismatch.mlx).abs()
                    > ABSOLUTE_TOLERANCE
                        + RELATIVE_TOLERANCE * foundry.abs().max(mismatch.mlx.abs())
            })
        })
        .collect();

    assert_eq!(mismatches, Vec::new());
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
    assert!(
        matches!(result.err(), Some(LlamaError::TensorShapeMismatch)),
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
    assert!(
        matches!(result.err(), Some(LlamaError::TensorDTypeMismatch)),
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
    assert!(
        matches!(result.err(), Some(LlamaError::TensorNotFound)),
        "a missing model.norm.weight is rejected"
    );
}
