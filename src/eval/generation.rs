use serde::{
    Deserialize,
    Serialize,
};
use tracing::error;

use crate::eval::EvalError;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ReferenceGeneration {
    pub engine: String,
    pub max_tokens: usize,
    pub top_k: usize,
    pub prompts: Vec<ReferencePrompt>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ReferencePrompt {
    pub prompt_ids: Vec<u32>,
    pub output_ids: Vec<u32>,
    pub top_logprobs: Vec<Vec<TokenLogprob>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub id: u32,
    pub logprob: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Steps<'steps> {
    pub token_ids: &'steps [u32],
    pub top_logprobs: &'steps [Vec<TokenLogprob>],
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct GenerationReport {
    pub compared_tokens: usize,
    pub matched_tokens: usize,
    pub first_divergence: Option<usize>,
    pub close: bool,
    pub max_logprob_abs_diff: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct FirstStepReport {
    pub kl: f64,
    pub max_abs_logit_diff: f64,
}

impl ReferencePrompt {
    pub fn steps(&self) -> Steps<'_> {
        Steps {
            token_ids: &self.output_ids,
            top_logprobs: &self.top_logprobs,
        }
    }
}

pub fn parse_generation(bytes: &[u8]) -> Result<ReferenceGeneration, EvalError> {
    let generation: ReferenceGeneration = serde_json::from_slice(bytes).map_err(|error| {
        error!(message = "Reference generation is not valid JSON or misses a field.", %error);
        EvalError::GenerationJson
    })?;
    let invalid_prompt = generation.prompts.iter().position(|prompt| {
        prompt.prompt_ids.is_empty()
            || prompt.output_ids.len() > generation.max_tokens
            || prompt.top_logprobs.len() != prompt.output_ids.len()
            || prompt
                .top_logprobs
                .iter()
                .any(|top| top.len() != generation.top_k)
    });
    if generation.prompts.is_empty() || generation.top_k == 0 || invalid_prompt.is_some() {
        error!(
            message = "Reference generation is inconsistent.",
            engine = generation.engine,
            prompt_count = generation.prompts.len(),
            top_k = generation.top_k,
            ?invalid_prompt,
        );
        return Err(EvalError::GenerationInvalid);
    }
    Ok(generation)
}

pub fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = logits
        .iter()
        .map(|&logit| (f64::from(logit) - max).exp())
        .sum();
    let logsumexp = max + sum.ln();
    logits
        .iter()
        .map(|&logit| f64::from(logit) - logsumexp)
        .collect()
}

pub fn top_k(
    logprobs: &[f64],
    k: usize,
) -> Result<Vec<TokenLogprob>, EvalError> {
    let mut ranked: Vec<(usize, f64)> = logprobs.iter().copied().enumerate().collect();
    ranked.sort_by(|(lhs_id, lhs), (rhs_id, rhs)| rhs.total_cmp(lhs).then(lhs_id.cmp(rhs_id)));
    ranked
        .into_iter()
        .take(k)
        .map(|(id, logprob)| {
            let id = u32::try_from(id).map_err(|_| {
                error!(message = "Vocabulary index does not fit in u32.", id);
                EvalError::VocabOverflow
            })?;
            Ok(TokenLogprob {
                id,
                logprob,
            })
        })
        .collect()
}

pub fn compare_generation(
    actual: Steps<'_>,
    expected: Steps<'_>,
) -> GenerationReport {
    let compared_tokens = actual.token_ids.len().min(expected.token_ids.len());
    let first_divergence = actual
        .token_ids
        .iter()
        .zip(expected.token_ids)
        .position(|(actual, expected)| actual != expected);
    let in_top_k = |steps: Steps<'_>, step: usize, token_id: Option<&u32>| {
        steps
            .top_logprobs
            .get(step)
            .zip(token_id)
            .is_some_and(|(top, &token_id)| top.iter().any(|entry| entry.id == token_id))
    };
    let close = first_divergence.is_none_or(|step| {
        in_top_k(expected, step, actual.token_ids.get(step))
            && in_top_k(actual, step, expected.token_ids.get(step))
    });
    let matched_tokens = first_divergence.unwrap_or(compared_tokens);
    let max_logprob_abs_diff = actual
        .top_logprobs
        .iter()
        .zip(expected.top_logprobs)
        .take(matched_tokens)
        .flat_map(|(actual_top, expected_top)| {
            expected_top.iter().filter_map(|expected| {
                actual_top
                    .iter()
                    .find(|actual| actual.id == expected.id)
                    .map(|actual| (actual.logprob - expected.logprob).abs())
            })
        })
        .reduce(f64::max);
    GenerationReport {
        compared_tokens,
        matched_tokens,
        first_divergence,
        close,
        max_logprob_abs_diff,
    }
}

pub fn first_step(
    actual_logits: &[f32],
    expected_logits: &[f32],
) -> Result<FirstStepReport, EvalError> {
    if actual_logits.len() != expected_logits.len() {
        error!(
            message = "Compared activations do not have the same element count.",
            actual = actual_logits.len(),
            expected = expected_logits.len(),
        );
        return Err(EvalError::NumelMismatch);
    }
    let kl = log_softmax(expected_logits)
        .into_iter()
        .zip(log_softmax(actual_logits))
        .map(|(expected, actual)| expected.exp() * (expected - actual))
        .sum();
    let max_abs_logit_diff = actual_logits
        .iter()
        .zip(expected_logits)
        .map(|(&actual, &expected)| (f64::from(actual) - f64::from(expected)).abs())
        .fold(0.0, f64::max);
    Ok(FirstStepReport {
        kl,
        max_abs_logit_diff,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::eval::{
        EvalError,
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

    #[test]
    fn log_softmax_of_equal_logits_is_minus_log_count() {
        assert_eq!(
            [
                log_softmax(&[0.0, 0.0]),
                log_softmax(&[0.0, 0.0, 0.0]),
                log_softmax(&[0.0, 0.0, 0.0, 0.0])
            ],
            [
                vec![-(2.0_f64).ln(); 2],
                vec![-(3.0_f64).ln(); 3],
                vec![-(4.0_f64).ln(); 4]
            ],
            "uniform distributions over two, three and four tokens"
        );
    }

    #[test]
    fn top_k_ranks_by_logprob_then_by_lowest_id() {
        let top = |entries: &[(u32, f64)]| -> Vec<TokenLogprob> {
            entries
                .iter()
                .map(|&(id, logprob)| TokenLogprob {
                    id,
                    logprob,
                })
                .collect()
        };
        assert_eq!(
            top_k(&[-1.0, -0.5, -0.5, -3.0], 3),
            Ok(top(&[(1, -0.5), (2, -0.5), (0, -1.0)])),
            "the tie between ids 1 and 2 keeps the lower id first, as the argmax kernel"
        );
    }

    #[test]
    fn compare_generation_without_divergence_is_close() {
        let top = |entries: &[(u32, f64)]| -> Vec<TokenLogprob> {
            entries
                .iter()
                .map(|&(id, logprob)| TokenLogprob {
                    id,
                    logprob,
                })
                .collect()
        };
        let tops = [top(&[(7, -0.1)]), top(&[(8, -0.2)])];
        assert_eq!(
            compare_generation(
                Steps {
                    token_ids: &[7, 8],
                    top_logprobs: &tops,
                },
                Steps {
                    token_ids: &[7, 8],
                    top_logprobs: &[top(&[(7, -0.25)]), top(&[(8, -0.2)])],
                },
            ),
            GenerationReport {
                compared_tokens: 2,
                matched_tokens: 2,
                first_divergence: None,
                close: true,
                max_logprob_abs_diff: Some(0.15),
            },
            "two equal tokens, the logprob difference is reported"
        );
    }

    #[test]
    fn compare_generation_accepts_a_near_tie_divergence() {
        let top = |entries: &[(u32, f64)]| -> Vec<TokenLogprob> {
            entries
                .iter()
                .map(|&(id, logprob)| TokenLogprob {
                    id,
                    logprob,
                })
                .collect()
        };
        let transformers_first_step = top(&[(220, -1.5610992908477783), (16543, -1.686)]);
        assert_eq!(
            compare_generation(
                Steps {
                    token_ids: &[16543, 1],
                    top_logprobs: &[top(&[(16543, -1.568), (220, -1.630)]), top(&[(1, -0.1)])],
                },
                Steps {
                    token_ids: &[220, 83017],
                    top_logprobs: &[transformers_first_step, top(&[(83017, -0.2)])],
                },
            ),
            GenerationReport {
                compared_tokens: 2,
                matched_tokens: 0,
                first_divergence: Some(0),
                close: true,
                max_logprob_abs_diff: None,
            },
            "transformers' prompt 3 first step: 220 and 16543 are in each other's top-k"
        );
    }

    #[test]
    fn compare_generation_rejects_a_token_outside_the_other_top_k() {
        let top = |entries: &[(u32, f64)]| -> Vec<TokenLogprob> {
            entries
                .iter()
                .map(|&(id, logprob)| TokenLogprob {
                    id,
                    logprob,
                })
                .collect()
        };
        let reference = Steps {
            token_ids: &[5],
            top_logprobs: &[top(&[(5, -0.1), (6, -2.0)])],
        };
        assert_eq!(
            [
                compare_generation(
                    Steps {
                        token_ids: &[9],
                        top_logprobs: &[top(&[(9, -0.1), (5, -2.0)])],
                    },
                    reference,
                )
                .close,
                compare_generation(
                    Steps {
                        token_ids: &[6],
                        top_logprobs: &[top(&[(6, -0.1), (7, -2.0)])],
                    },
                    reference,
                )
                .close,
            ],
            [false, false],
            "Foundry's token outside the reference top-k, then the reference token outside \
             Foundry's top-k"
        );
    }

    #[test]
    fn compare_generation_stops_at_the_shorter_sequence() {
        let top = |entries: &[(u32, f64)]| -> Vec<TokenLogprob> {
            entries
                .iter()
                .map(|&(id, logprob)| TokenLogprob {
                    id,
                    logprob,
                })
                .collect()
        };
        let eos_step = [top(&[(2, -0.1)])];
        assert_eq!(
            compare_generation(
                Steps {
                    token_ids: &[2],
                    top_logprobs: &eos_step,
                },
                Steps {
                    token_ids: &[2, 4, 4],
                    top_logprobs: &[top(&[(2, -0.1)]), top(&[(4, -0.1)]), top(&[(4, -0.1)])],
                },
            ),
            GenerationReport {
                compared_tokens: 1,
                matched_tokens: 1,
                first_divergence: None,
                close: true,
                max_logprob_abs_diff: Some(0.0),
            },
            "an engine stopped at its end-of-sequence token after one step"
        );
    }

    #[test]
    fn first_step_measures_kl_and_logit_difference() {
        let kl_e6 = first_step(&[0.0, 3.0_f32.ln()], &[0.0, 0.0])
            .map(|report| ((report.kl * 1e6).round(), report.max_abs_logit_diff));
        assert_eq!(
            [
                first_step(&[1.0, 2.0], &[1.0, 2.0])
                    .map(|report| (report.kl, report.max_abs_logit_diff)),
                kl_e6.map(|(kl_e6, diff)| (kl_e6, (diff * 1e6).round())),
                first_step(&[1.0], &[1.0, 2.0])
                    .map(|report| (report.kl, report.max_abs_logit_diff)),
            ],
            [
                Ok((0.0, 0.0)),
                Ok((143_841.0, 1_098_612.0)),
                Err(EvalError::NumelMismatch)
            ],
            "identical logits, KL(1/2, 1/2 || 1/4, 3/4) = ln(4/3) / 2, different vocabularies"
        );
    }

    #[test]
    fn parse_generation_reads_the_reference_format() {
        let top = |entries: &[(u32, f64)]| -> Vec<TokenLogprob> {
            entries
                .iter()
                .map(|&(id, logprob)| TokenLogprob {
                    id,
                    logprob,
                })
                .collect()
        };
        let bytes = serde_json::to_vec(&json!({
            "model": "a/b",
            "engine": "transformers",
            "max_tokens": 2,
            "top_k": 1,
            "eos_token_ids": [2],
            "prompts": [{
                "text": "hi",
                "prompt_ids": [1, 3],
                "output_ids": [4, 2],
                "top_logprobs": [[{ "id": 4, "logprob": -0.5 }], [{ "id": 2, "logprob": -0.25 }]],
            }],
        }))
        .expect("the generation serializes");
        assert_eq!(
            parse_generation(&bytes),
            Ok(ReferenceGeneration {
                engine: "transformers".to_owned(),
                max_tokens: 2,
                top_k: 1,
                prompts: vec![ReferencePrompt {
                    prompt_ids: vec![1, 3],
                    output_ids: vec![4, 2],
                    top_logprobs: vec![top(&[(4, -0.5)]), top(&[(2, -0.25)])],
                }],
            }),
            "unused fields are ignored"
        );
    }

    #[test]
    fn parse_generation_rejects_missing_fields_and_inconsistent_steps() {
        let generation = |top_k, top_logprobs| {
            serde_json::to_vec(&json!({
                "engine": "mlx",
                "max_tokens": 2,
                "top_k": top_k,
                "prompts": [{ "prompt_ids": [1], "output_ids": [4], "top_logprobs": top_logprobs }],
            }))
            .expect("the generation serializes")
        };
        assert_eq!(
            [
                parse_generation(br#"{ "engine": "mlx" }"#),
                parse_generation(&generation(1, json!([]))),
                parse_generation(&generation(2, json!([[{ "id": 4, "logprob": -0.5 }]]))),
                parse_generation(&generation(0, json!([[]]))),
            ],
            [
                Err(EvalError::GenerationJson),
                Err(EvalError::GenerationInvalid),
                Err(EvalError::GenerationInvalid),
                Err(EvalError::GenerationInvalid),
            ],
            "missing fields, no step for a token, a top-k shorter than declared, top_k of zero"
        );
    }
}
