use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use half::f16;

use crate::checkpoint::tests::{
    header_for,
    tensor_specs,
};
use crate::config::LlamaConfig;
use crate::config::tests::tiny;
use crate::error::{
    LlamaError,
    RequestError,
};
use crate::metal::model::LlamaModel;
use crate::metal::session::{
    CONTEXT_CAPACITY,
    PREFILL_CAPACITY,
};
use crate::metal::tests::gpu;
use crate::rope::llama3_frequencies;
use crate::test_support::Rng;
use crate::test_support::reference::{
    self,
    AttentionShape,
};

type TestResult = Result<(), Box<dyn Error>>;

struct TinyModel {
    directory: PathBuf,
    config: LlamaConfig,
    tensors: BTreeMap<String, Vec<u8>>,
}

impl TinyModel {
    fn write(name: &str) -> Result<Self, Box<dyn Error>> {
        let config_json = tiny();
        let config = LlamaConfig::parse(&config_json.to_string())?;
        let specs = tensor_specs(&config);
        let mut rng = Rng::new(23);
        let tensors: BTreeMap<String, Vec<u8>> = specs
            .iter()
            .map(|(tensor, dtype, shape)| {
                let count = usize::try_from(shape.iter().product::<u64>()).unwrap_or(0);
                let bytes: Vec<u8> = if *dtype == "U32" {
                    rng.words(count)
                        .iter()
                        .flat_map(|word| word.to_le_bytes())
                        .collect()
                } else {
                    let (low, high) = if tensor.ends_with("norm.weight") {
                        (0.8, 1.2)
                    } else if tensor.ends_with(".scales") {
                        (-0.04, 0.04)
                    } else {
                        (-0.03, 0.03)
                    };
                    rng.halves(count, low, high)
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect()
                };
                (tensor.clone(), bytes)
            })
            .collect();
        let (header, _) = header_for(&specs);
        let header = header.to_string().into_bytes();
        let directory =
            std::env::temp_dir().join(format!("foundry-llama-{}-{name}", std::process::id()));
        fs::create_dir_all(&directory)?;
        let mut file = u64::try_from(header.len())?.to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        for (tensor, ..) in &specs {
            file.extend_from_slice(tensors.get(tensor).map_or(&[][..], Vec::as_slice));
        }
        fs::write(directory.join("model.safetensors"), file)?;
        fs::write(directory.join("config.json"), config_json.to_string())?;
        Ok(Self {
            directory,
            config,
            tensors,
        })
    }

    fn load(&self) -> Result<LlamaModel, LlamaError> {
        LlamaModel::load_unpinned(
            &self.directory.join("config.json"),
            &self.directory.join("model.safetensors"),
        )
    }

    fn halves(
        &self,
        name: &str,
    ) -> Vec<f32> {
        self.tensors
            .get(name)
            .map(|bytes| {
                bytes
                    .chunks_exact(2)
                    .map(|pair| f16::from_le_bytes([pair[0], pair[1]]).to_f32())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn dense(
        &self,
        name: &str,
        rows: u32,
        cols: u32,
    ) -> Vec<f32> {
        let words: Vec<u32> = self
            .tensors
            .get(&format!("{name}.weight"))
            .map(|bytes| {
                bytes
                    .chunks_exact(4)
                    .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                    .collect()
            })
            .unwrap_or_default();
        reference::dequantize(
            &words,
            &self.halves(&format!("{name}.scales")),
            &self.halves(&format!("{name}.biases")),
            rows as usize,
            cols as usize,
        )
    }

    fn reference_logits(
        &self,
        tokens: &[u32],
    ) -> Vec<Vec<f32>> {
        let c = &self.config;
        let (hidden, query, kv, inner, vocab) = (
            c.hidden as usize,
            c.query_dim() as usize,
            c.kv_dim() as usize,
            c.intermediate as usize,
            c.vocab as usize,
        );
        let rows = tokens.len();
        let embed = self.dense("model.embed_tokens", c.vocab, c.hidden);
        let mut x: Vec<f32> = tokens
            .iter()
            .flat_map(|&token| {
                let start = token as usize * hidden;
                embed[start..start + hidden].to_vec()
            })
            .collect();
        let frequencies = llama3_frequencies(&c.rope, c.head_dim);
        let rotate = |values: &mut Vec<f32>| {
            let width = values.len() / rows;
            for (position, row) in values.chunks_mut(width).enumerate() {
                for head in row.chunks_mut(128) {
                    reference::rope(head, position, &frequencies);
                }
            }
        };
        for layer in 0..c.layers {
            let prefix = format!("model.layers.{layer}");
            let normed = reference::rms_norm(
                &x,
                &self.halves(&format!("{prefix}.input_layernorm.weight")),
                c.rms_eps,
            );
            let project = |input: &[f32], name: &str, out: usize, inner: usize| {
                let weights = self.dense(&format!("{prefix}.{name}"), out as u32, inner as u32);
                reference::matmul(input, &weights, rows, inner, out)
            };
            let mut q = project(&normed, "self_attn.q_proj", query, hidden);
            let mut k = project(&normed, "self_attn.k_proj", kv, hidden);
            let v = project(&normed, "self_attn.v_proj", kv, hidden);
            rotate(&mut q);
            rotate(&mut k);
            let attention = reference::attention(
                &q,
                &k,
                &v,
                0,
                &AttentionShape {
                    heads: c.heads as usize,
                    kv_heads: c.kv_heads as usize,
                    head_dim: 128,
                },
            );
            let output = project(&attention, "self_attn.o_proj", hidden, query);
            x.iter_mut()
                .zip(output)
                .for_each(|(value, delta)| *value += delta);
            let normed = reference::rms_norm(
                &x,
                &self.halves(&format!("{prefix}.post_attention_layernorm.weight")),
                c.rms_eps,
            );
            let gate = project(&normed, "mlp.gate_proj", inner, hidden);
            let up = project(&normed, "mlp.up_proj", inner, hidden);
            let activated: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| reference::silu(*g) * u)
                .collect();
            let output = project(&activated, "mlp.down_proj", hidden, inner);
            x.iter_mut()
                .zip(output)
                .for_each(|(value, delta)| *value += delta);
        }
        let normed = reference::rms_norm(&x, &self.halves("model.norm.weight"), c.rms_eps);
        reference::matmul(&normed, &embed, rows, hidden, vocab)
            .chunks(vocab)
            .map(<[f32]>::to_vec)
            .collect()
    }
}

impl Drop for TinyModel {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.directory); }
}

#[test]
fn generation_is_repeatable_and_requests_are_validated() -> TestResult {
    if gpu()?.is_none() {
        return Ok(());
    }
    let tiny = TinyModel::write("generate")?;
    let model = tiny.load()?;
    let prompt: Vec<u32> = (0..40).map(|index| (index * 7 + 3) % 64).collect();
    let run = || -> Result<(Vec<u32>, Vec<usize>), LlamaError> {
        let mut session = model.session()?;
        session.prepare(&prompt)?;
        let mut seen = Vec::new();
        let tokens = session.generate(12, |index, _| seen.push(index))?;
        Ok((tokens, seen))
    };
    let (first, order) = run()?;
    let (second, _) = run()?;
    let logits = tiny.reference_logits(&prompt).pop().unwrap_or_default();
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    if let [(best, top), (_, runner_up), ..] = ranked.as_slice()
        && top - runner_up > 0.05
    {
        assert_eq!(
            first.first().copied(),
            u32::try_from(*best).ok(),
            "first token is the reference argmax"
        );
    }
    assert_eq!(first.len(), 12, "token count");
    assert_eq!(first, second, "fresh sessions repeat greedy generation");
    assert_eq!(order, (0..12).collect::<Vec<_>>(), "tokens arrive in order");
    assert!(
        first.iter().all(|&token| token < 64),
        "tokens are in the vocabulary"
    );
    let request = |error: LlamaError| {
        if let LlamaError::Request(request) = error {
            Some(request)
        } else {
            None
        }
    };
    let mut session = model.session()?;
    assert_eq!(
        session.prepare(&[]).err().and_then(request),
        Some(RequestError::EmptyPrompt),
        "empty prompt"
    );
    assert_eq!(
        session
            .prepare(&vec![1; PREFILL_CAPACITY + 1])
            .err()
            .and_then(request),
        Some(RequestError::PromptTooLong {
            tokens: PREFILL_CAPACITY + 1,
            limit: PREFILL_CAPACITY
        }),
        "prompt longer than the prefill buffers"
    );
    assert_eq!(
        session.prepare(&[1, 64]).err().and_then(request),
        Some(RequestError::TokenOutOfRange {
            index: 1,
            token: 64,
            vocab: 64
        }),
        "token outside the vocabulary"
    );
    assert_eq!(
        session.generate(1, |_, _| {}).err().and_then(request),
        Some(RequestError::NotPrepared),
        "generation without a prompt"
    );
    session.prepare(&vec![1; PREFILL_CAPACITY])?;
    assert_eq!(
        session
            .generate(CONTEXT_CAPACITY - PREFILL_CAPACITY + 1, |_, _| {})
            .err()
            .and_then(request),
        Some(RequestError::CapacityExceeded {
            position: 0,
            requested: CONTEXT_CAPACITY + 1,
            capacity: CONTEXT_CAPACITY
        }),
        "generation beyond the KV cache"
    );
    session.prepare(&[1, 2])?;
    session.generate(2, |_, _| {})?;
    assert_eq!(
        session.prepare(&[1]).err().and_then(request),
        Some(RequestError::SessionNotFresh {
            position: 3
        }),
        "a used session cannot generate again"
    );
    Ok(())
}

#[cfg(feature = "probe")]
#[test]
fn tiny_model_logits_match_the_cpu_reference() -> TestResult {
    use crate::metal::session::Stage;
    use crate::test_support::relative_l2;

    if gpu()?.is_none() {
        return Ok(());
    }
    let tiny = TinyModel::write("probe")?;
    let model = tiny.load()?;
    let tokens: Vec<u32> = (0..36).map(|index| (index * 11 + 5) % 64).collect();
    let expected = tiny.reference_logits(&tokens);
    let mut session = model.session()?;
    let prefill = 32;
    let mut compare = |position: usize, chunk: &[u32]| -> TestResult {
        let mut logits = Vec::new();
        session.forward_probed(chunk, &mut |stage, values| {
            if stage == Stage::Logits {
                logits = values.iter().map(|value| value.to_f32()).collect();
            }
        })?;
        let reference = expected.get(position).cloned().unwrap_or_default();
        let error = relative_l2(&logits, &reference);
        assert!(
            error < 1e-2,
            "position {position}: relative L2 error {error}"
        );
        Ok(())
    };
    compare(prefill - 1, &tokens[..prefill])?;
    for position in prefill..tokens.len() {
        compare(position, &tokens[position..=position])?;
    }
    Ok(())
}
