mod error;

use std::time::{
    Duration,
    Instant,
};

use half::bf16;
use tracing::error;

pub use self::error::GenerateError;
use crate::core::{
    Backend,
    DType,
    Shape,
    Tensor,
};
use crate::models::llama::Llama;
use crate::nn::KvCache;

const VOCAB_AXIS: usize = 0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    pub token_ids: Vec<u32>,
    pub prefill: Duration,
    pub decode_steps: Vec<Duration>,
}

struct Decoder<'run, B: Backend> {
    backend: &'run mut B,
    model: &'run Llama<B>,
    cache: Vec<KvCache<B>>,
}

pub fn generate<B: Backend>(
    backend: &mut B,
    model: &Llama<B>,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    stop_ids: &[u32],
) -> Result<Generation, GenerateError> {
    if prompt_ids.is_empty() {
        error!(message = "Prompt holds no token.");
        return Err(GenerateError::EmptyPrompt);
    }
    if max_new_tokens == 0 {
        return Ok(Generation {
            token_ids: Vec::new(),
            prefill: Duration::ZERO,
            decode_steps: Vec::new(),
        });
    }
    let cache = model.cache(backend).map_err(|error| {
        error!(message = "Creating the key-value cache failed.", %error);
        GenerateError::Cache
    })?;
    let mut decoder = Decoder {
        backend,
        model,
        cache,
    };
    let prefill_start = Instant::now();
    let first_token = decoder.step(prompt_ids, 0)?;
    let mut generation = Generation {
        token_ids: vec![first_token],
        prefill: prefill_start.elapsed(),
        decode_steps: Vec::new(),
    };
    let mut token = first_token;
    let mut position = prompt_ids.len();
    for _ in 1..max_new_tokens {
        if stop_ids.contains(&token) {
            break;
        }
        let step_start = Instant::now();
        token = decoder.step(&[token], position)?;
        generation.decode_steps.push(step_start.elapsed());
        generation.token_ids.push(token);
        position = position.checked_add(1).ok_or_else(|| {
            error!(message = "Token position overflows usize.", position);
            GenerateError::PositionOverflow
        })?;
    }
    Ok(generation)
}

impl<B: Backend> Decoder<'_, B> {
    fn step(
        &mut self,
        token_ids: &[u32],
        position: usize,
    ) -> Result<u32, GenerateError> {
        let seq_len = token_ids.len();
        let seq_len_total = position.checked_add(seq_len).ok_or_else(|| {
            error!(
                message = "Token position overflows usize.",
                position, seq_len
            );
            GenerateError::PositionOverflow
        })?;
        let token_bytes = token_ids
            .iter()
            .flat_map(|token_id| token_id.to_le_bytes())
            .collect::<Vec<u8>>();
        let tokens = self.upload(&token_bytes, DType::U32, &[seq_len])?;
        let positions =
            self.upload(&positions(position, seq_len_total)?, DType::U32, &[seq_len])?;
        let mask = self.upload(
            &causal_mask(position, seq_len_total),
            DType::BF16,
            &[seq_len, seq_len_total],
        )?;
        let logits = self
            .model
            .forward(
                self.backend,
                &tokens,
                &positions,
                &mask,
                &mut self.cache,
                position,
            )
            .map_err(|error| {
                error!(message = "Model forward failed.", position, seq_len, %error);
                GenerateError::Forward
            })?;
        self.next_token(&logits)
    }

    fn upload(
        &mut self,
        bytes: &[u8],
        dtype: DType,
        dims: &[usize],
    ) -> Result<Tensor<B>, GenerateError> {
        let upload_error = |error: &dyn std::error::Error| {
            error!(message = "Uploading a generation input failed.", ?dtype, ?dims, %error);
            GenerateError::Upload
        };
        let shape = Shape::try_from(dims).map_err(|error| upload_error(&error))?;
        Tensor::upload(self.backend, bytes, dtype, shape).map_err(|error| upload_error(&error))
    }

    fn next_token(
        &mut self,
        logits: &Tensor<B>,
    ) -> Result<u32, GenerateError> {
        let bytes = logits
            .argmax(self.backend, VOCAB_AXIS)
            .and_then(|token| token.download(self.backend))
            .map_err(|error| {
                error!(message = "Selecting the next token failed.", %error);
                GenerateError::NextToken
            })?;
        let token_bytes = <[u8; 4]>::try_from(bytes.as_slice()).map_err(|_| {
            error!(
                message = "Next token download does not hold one u32.",
                bytes = bytes.len()
            );
            GenerateError::TokenBytes
        })?;
        Ok(u32::from_le_bytes(token_bytes))
    }
}

fn positions(
    position: usize,
    seq_len_total: usize,
) -> Result<Vec<u8>, GenerateError> {
    (position..seq_len_total)
        .map(|token_position| {
            u32::try_from(token_position).map_err(|_| {
                error!(
                    message = "Token position does not fit in u32.",
                    token_position
                );
                GenerateError::PositionOverflow
            })
        })
        .try_fold(Vec::new(), |mut bytes, token_position| {
            bytes.extend(token_position?.to_le_bytes());
            Ok(bytes)
        })
}

fn causal_mask(
    position: usize,
    seq_len_total: usize,
) -> Vec<u8> {
    (position..seq_len_total)
        .flat_map(|query_position| {
            (0..seq_len_total).map(move |key_position| {
                if key_position <= query_position {
                    bf16::ZERO
                } else {
                    bf16::NEG_INFINITY
                }
            })
        })
        .flat_map(bf16::to_le_bytes)
        .collect()
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use crate::generate::{
        GenerateError,
        causal_mask,
        positions,
    };

    #[test]
    fn causal_mask_of_a_prefill_hides_later_keys() {
        let zero = bf16::ZERO;
        let hidden = bf16::NEG_INFINITY;
        assert_eq!(
            causal_mask(0, 3),
            [zero, hidden, hidden, zero, zero, hidden, zero, zero, zero]
                .into_iter()
                .flat_map(bf16::to_le_bytes)
                .collect::<Vec<u8>>(),
            "three queries over three keys, lower triangle visible"
        );
    }

    #[test]
    fn causal_mask_of_a_decode_step_shows_every_key() {
        assert_eq!(
            causal_mask(4, 5),
            [bf16::ZERO; 5]
                .into_iter()
                .flat_map(bf16::to_le_bytes)
                .collect::<Vec<u8>>(),
            "one query at position 4 sees keys 0 to 4"
        );
    }

    #[test]
    fn positions_count_from_the_cache_offset() {
        assert_eq!(
            positions(2, 5),
            Ok([2_u32, 3, 4]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<u8>>()),
            "positions 2, 3 and 4"
        );
    }

    #[test]
    fn positions_beyond_u32_are_rejected() {
        let position = usize::try_from(u32::MAX).expect("u32 fits in usize");
        assert_eq!(
            positions(position, position.saturating_add(2)),
            Err(GenerateError::PositionOverflow),
            "position u32::MAX + 1 does not fit"
        );
    }
}
