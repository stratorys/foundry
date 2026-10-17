mod compare;
mod error;
mod fixtures;
mod generation;

use std::num::NonZeroUsize;
use std::path::{
    Path,
    PathBuf,
};

use serde::Serialize;
use tracing::error;

pub use self::compare::{
    CloseReport,
    Tolerance,
    compare_close,
    to_f32,
};
pub use self::error::EvalError;
pub use self::fixtures::{
    Activation,
    activations,
    read_file,
    read_safetensors,
};
pub use self::generation::{
    FirstStepReport,
    GenerationReport,
    ReferenceGeneration,
    ReferencePrompt,
    Steps,
    TokenLogprob,
    compare_generation,
    first_step,
    log_softmax,
    parse_generation,
    top_k,
};
use crate::generate::{
    Decoder,
    GenerateError,
};
use crate::models::CausalLm;
use crate::nn::{
    Device,
    Module,
    Tensor,
};
use crate::weights::Weights;

const RTOL: f32 = 1e-2;
const GENERATION_FILE: &str = "generation.json";
const LAST_LOGITS_FILE: &str = "last_logits.safetensors";
const ACTIVATIONS_FILE: &str = "activations.safetensors";
const MAX_GENERATION_BYTES: u64 = 64 * 1024 * 1024;

pub struct Reference {
    pub dir: PathBuf,
    pub generation: ReferenceGeneration,
    pub last_logits: Weights<Vec<u8>>,
    pub activations: Option<Weights<Vec<u8>>>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReferenceReport {
    pub engine: String,
    pub dir: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<LayerReport>>,
    pub generation: Vec<PromptReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PromptReport {
    pub prompt_index: usize,
    pub prompt_tokens: usize,
    #[serde(flatten)]
    pub generation: GenerationReport,
    pub first_step: FirstStepReport,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BaselineReport {
    pub engines: [String; 2],
    pub prompts: Vec<GenerationReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayerReport {
    pub name: String,
    #[serde(flatten)]
    pub tolerance: Tolerance,
    #[serde(flatten)]
    pub outcome: LayerOutcome,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum LayerOutcome {
    Passed(CloseReport),
    Failed(CloseReport),
    Mismatch {
        expected_dtype: String,
        actual_dtype: String,
        expected_dims: Vec<usize>,
        actual_dims: Vec<usize>,
    },
    Skipped,
}

impl LayerOutcome {
    pub fn is_failure(&self) -> bool { matches!(self, Self::Failed(_) | Self::Mismatch { .. }) }
}

impl ReferenceReport {
    pub fn is_failure(&self) -> bool {
        let layer_failed = self
            .layers
            .iter()
            .flatten()
            .any(|layer| layer.outcome.is_failure());
        layer_failed
            || self
                .generation
                .iter()
                .any(|prompt| !prompt.generation.close)
    }
}

pub fn read_reference(dir: &Path) -> Result<Reference, EvalError> {
    let generation = parse_generation(&read_file(
        &dir.join(GENERATION_FILE),
        MAX_GENERATION_BYTES,
    )?)?;
    let last_logits = read_safetensors(&dir.join(LAST_LOGITS_FILE))?;
    let activations_path = dir.join(ACTIVATIONS_FILE);
    let activations = activations_path
        .is_file()
        .then(|| read_safetensors(&activations_path))
        .transpose()?;
    Ok(Reference {
        dir: dir.to_path_buf(),
        generation,
        last_logits,
        activations,
    })
}

pub fn max_cache_len(generation: &ReferenceGeneration) -> Option<usize> {
    generation
        .prompts
        .iter()
        .map(|prompt| prompt.prompt_ids.len().checked_add(generation.max_tokens))
        .try_fold(0, |max_cache_len, cache_len| {
            Some(max_cache_len.max(cache_len?))
        })
}

pub fn baselines(references: &[Reference]) -> Result<Vec<BaselineReport>, EvalError> {
    references
        .iter()
        .enumerate()
        .flat_map(|(index, lhs)| {
            references
                .iter()
                .skip(index.saturating_add(1))
                .map(move |rhs| baseline(lhs, rhs))
        })
        .collect()
}

pub fn eval_reference<D: Device, M: CausalLm<D>>(
    device: &mut D,
    model: &M,
    modules: &[(String, &dyn Module<D>)],
    eos_token_id: &[u32],
    reference: &Reference,
) -> Result<ReferenceReport, EvalError> {
    let layers = reference
        .activations
        .as_ref()
        .map(|activations| eval_layers(device, modules, activations))
        .transpose()?;
    let generation = reference
        .generation
        .prompts
        .iter()
        .enumerate()
        .map(|(prompt_index, prompt)| {
            eval_prompt(
                device,
                model,
                eos_token_id,
                &reference.generation,
                &reference.last_logits,
                prompt_index,
                prompt,
            )
        })
        .collect::<Result<Vec<PromptReport>, EvalError>>()?;
    Ok(ReferenceReport {
        engine: reference.generation.engine.clone(),
        dir: reference.dir.display().to_string(),
        layers,
        generation,
    })
}

pub fn eval_layers<D: Device, S: AsRef<[u8]>>(
    device: &mut D,
    modules: &[(String, &dyn Module<D>)],
    weights: &Weights<S>,
) -> Result<Vec<LayerReport>, EvalError> {
    activations(weights)?
        .iter()
        .map(|activation| eval_layer(device, modules, activation))
        .collect()
}

fn eval_layer<D: Device>(
    device: &mut D,
    modules: &[(String, &dyn Module<D>)],
    activation: &Activation<'_>,
) -> Result<LayerReport, EvalError> {
    let Activation {
        name,
        input,
        output: expected,
    } = *activation;
    let report = |outcome| LayerReport {
        name: name.to_owned(),
        tolerance: tolerance(name),
        outcome,
    };
    let Some((_, module)) = modules.iter().find(|(module_name, _)| module_name == name) else {
        return Ok(report(LayerOutcome::Skipped));
    };
    let input =
        Tensor::from_bytes(device, input.bytes, input.dtype, input.shape).map_err(|error| {
            error!(message = "Creating a submodule input tensor failed.", name, %error);
            EvalError::Input
        })?;
    let actual = module.forward(device, &input).map_err(|error| {
        error!(message = "Submodule forward failed.", name, %error);
        EvalError::Forward
    })?;
    if actual.dtype() != expected.dtype || *actual.shape() != expected.shape {
        return Ok(report(LayerOutcome::Mismatch {
            expected_dtype: format!("{:?}", expected.dtype),
            actual_dtype: format!("{:?}", actual.dtype()),
            expected_dims: expected.shape.dims().to_vec(),
            actual_dims: actual.shape().dims().to_vec(),
        }));
    }
    let actual_bytes = actual
        .contiguous(device)
        .and_then(|actual| actual.to_bytes(device))
        .map_err(|error| {
            error!(message = "Reading a submodule output failed.", name, %error);
            EvalError::Output
        })?;
    let close = compare_close(
        &to_f32(actual.dtype(), &actual_bytes)?,
        &to_f32(expected.dtype, expected.bytes)?,
        tolerance(name),
    )?;
    Ok(report(if close.passed() {
        LayerOutcome::Passed(close)
    } else {
        LayerOutcome::Failed(close)
    }))
}

fn baseline(
    lhs: &Reference,
    rhs: &Reference,
) -> Result<BaselineReport, EvalError> {
    let prompts_match = lhs.generation.prompts.len() == rhs.generation.prompts.len()
        && lhs
            .generation
            .prompts
            .iter()
            .zip(&rhs.generation.prompts)
            .all(|(lhs, rhs)| lhs.prompt_ids == rhs.prompt_ids);
    if !prompts_match {
        error!(
            message = "References do not hold the same prompts.",
            lhs = lhs.generation.engine,
            rhs = rhs.generation.engine,
        );
        return Err(EvalError::PromptMismatch);
    }
    Ok(BaselineReport {
        engines: [lhs.generation.engine.clone(), rhs.generation.engine.clone()],
        prompts: lhs
            .generation
            .prompts
            .iter()
            .zip(&rhs.generation.prompts)
            .map(|(lhs, rhs)| compare_generation(lhs.steps(), rhs.steps()))
            .collect(),
    })
}

fn eval_prompt<D: Device, M: CausalLm<D>>(
    device: &mut D,
    model: &M,
    eos_token_id: &[u32],
    generation: &ReferenceGeneration,
    last_logits: &Weights<Vec<u8>>,
    prompt_index: usize,
    prompt: &ReferencePrompt,
) -> Result<PromptReport, EvalError> {
    let generate_failed = |error: GenerateError| {
        error!(message = "Generating with the model failed.", prompt_index, %error);
        EvalError::Generate
    };
    let mut decoder = Decoder::new(device, model, NonZeroUsize::MIN).map_err(generate_failed)?;
    let mut token_ids = Vec::with_capacity(generation.max_tokens);
    let mut top_logprobs = Vec::with_capacity(generation.max_tokens);
    let mut first_logits = Vec::new();
    let mut input_ids = prompt.prompt_ids.clone();
    let mut past_seen_tokens = 0_usize;
    for step in 0..generation.max_tokens {
        let (tokens, logits) = decoder
            .step_with_logits(&input_ids, past_seen_tokens)
            .map_err(generate_failed)?;
        let logits = to_f32(logits.dtype, &logits.bytes)?;
        let &[token] = tokens.as_slice() else {
            error!(
                message = "Batch-one step did not return one token.",
                prompt_index, step
            );
            return Err(EvalError::Generate);
        };
        top_logprobs.push(top_k(&log_softmax(&logits), generation.top_k)?);
        token_ids.push(token);
        if step == 0 {
            first_logits = logits;
        }
        if eos_token_id.contains(&token) {
            break;
        }
        past_seen_tokens = past_seen_tokens
            .checked_add(input_ids.len())
            .ok_or_else(|| {
                error!(
                    message = "Token position overflows usize.",
                    prompt_index, step
                );
                EvalError::PositionOverflow
            })?;
        input_ids = vec![token];
    }
    let expected_logits_name = format!("prompt.{prompt_index}");
    let Some(expected_logits) = last_logits.tensor(&expected_logits_name) else {
        error!(
            message = "Reference first-step logits are missing.",
            name = expected_logits_name,
        );
        return Err(EvalError::LastLogitsMissing);
    };
    Ok(PromptReport {
        prompt_index,
        prompt_tokens: prompt.prompt_ids.len(),
        generation: compare_generation(
            Steps {
                token_ids: &token_ids,
                top_logprobs: &top_logprobs,
            },
            prompt.steps(),
        ),
        first_step: first_step(
            &first_logits,
            &to_f32(expected_logits.dtype, expected_logits.bytes)?,
        )?,
    })
}

fn tolerance(name: &str) -> Tolerance {
    let atol = match name.rsplit('.').next() {
        Some("embed_tokens") => 1e-3,
        Some("q_proj" | "k_proj" | "o_proj") => 2e-2,
        _ => 1e-2,
    };
    Tolerance {
        atol,
        rtol: RTOL,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::eval::{
        CloseReport,
        LayerOutcome,
        LayerReport,
        Tolerance,
        tolerance,
    };

    #[test]
    fn tolerance_depends_on_the_last_module_name() {
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
            .map(|name| tolerance(name).atol),
            [1e-3, 2e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2],
            "embedding, three projections, then the other modules"
        );
    }

    #[test]
    fn is_failure_holds_for_failed_and_mismatched_layers() {
        let close = CloseReport {
            numel: 1,
            close_count: 1,
            close_fraction: 1.0,
            non_finite_count: 0,
            max_abs_error: 0.0,
        };
        assert_eq!(
            [
                LayerOutcome::Passed(close),
                LayerOutcome::Failed(close),
                LayerOutcome::Mismatch {
                    expected_dtype: "BF16".to_owned(),
                    actual_dtype: "F32".to_owned(),
                    expected_dims: vec![1],
                    actual_dims: vec![1],
                },
                LayerOutcome::Skipped,
            ]
            .map(|outcome| outcome.is_failure()),
            [false, true, true, false],
            "passed, failed, mismatched, skipped"
        );
    }

    #[test]
    fn layer_report_serializes_flat_with_a_status() {
        let report = |outcome| LayerReport {
            name: "model.norm".to_owned(),
            tolerance: Tolerance {
                atol: 0.5,
                rtol: 0.25,
            },
            outcome,
        };
        assert_eq!(
            [
                serde_json::to_value(report(LayerOutcome::Passed(CloseReport {
                    numel: 2,
                    close_count: 2,
                    close_fraction: 1.0,
                    non_finite_count: 0,
                    max_abs_error: 0.5,
                })))
                .ok(),
                serde_json::to_value(report(LayerOutcome::Skipped)).ok(),
            ],
            [
                Some(json!({
                    "name": "model.norm",
                    "atol": 0.5,
                    "rtol": 0.25,
                    "status": "passed",
                    "numel": 2,
                    "close_count": 2,
                    "close_fraction": 1.0,
                    "non_finite_count": 0,
                    "max_abs_error": 0.5,
                })),
                Some(json!({
                    "name": "model.norm",
                    "atol": 0.5,
                    "rtol": 0.25,
                    "status": "skipped",
                })),
            ],
            "a compared layer and a skipped layer"
        );
    }
}
