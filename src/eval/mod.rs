mod compare;
mod error;
mod fixtures;

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
    read_activations,
};
use crate::nn::{
    Backend,
    Tensor,
};
use crate::weights::Weights;

const RTOL: f32 = 1e-2;

pub trait Submodules<B: Backend> {
    type Error: std::error::Error;

    fn forward_submodule(
        &self,
        backend: &mut B,
        target: &str,
        input: &Tensor<B>,
    ) -> Result<Option<Tensor<B>>, Self::Error>;
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

pub fn eval_layers<B: Backend, M: Submodules<B>, S: AsRef<[u8]>>(
    backend: &mut B,
    model: &M,
    weights: &Weights<S>,
) -> Result<Vec<LayerReport>, EvalError> {
    activations(weights)?
        .iter()
        .map(|activation| eval_layer(backend, model, activation))
        .collect()
}

fn eval_layer<B: Backend, M: Submodules<B>>(
    backend: &mut B,
    model: &M,
    activation: &Activation<'_>,
) -> Result<LayerReport, EvalError> {
    let Activation {
        name,
        input,
        output: expected,
    } = *activation;
    let input =
        Tensor::from_bytes(backend, input.bytes, input.dtype, input.shape).map_err(|error| {
            error!(message = "Creating a submodule input tensor failed.", name, %error);
            EvalError::Input
        })?;
    let actual = model
        .forward_submodule(backend, name, &input)
        .map_err(|error| {
            error!(message = "Submodule forward failed.", name, %error);
            EvalError::Forward
        })?;
    let report = |outcome| LayerReport {
        name: name.to_owned(),
        tolerance: tolerance(name),
        outcome,
    };
    let Some(actual) = actual else {
        return Ok(report(LayerOutcome::Skipped));
    };
    if actual.dtype() != expected.dtype || *actual.shape() != expected.shape {
        return Ok(report(LayerOutcome::Mismatch {
            expected_dtype: format!("{:?}", expected.dtype),
            actual_dtype: format!("{:?}", actual.dtype()),
            expected_dims: expected.shape.dims().to_vec(),
            actual_dims: actual.shape().dims().to_vec(),
        }));
    }
    let actual_bytes = actual
        .contiguous(backend)
        .and_then(|actual| actual.to_bytes(backend))
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
