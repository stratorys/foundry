mod error;

use std::num::NonZeroUsize;
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

const VOCAB_AXIS: usize = 1;
const TOKEN_ID_BYTES: usize = 4;

pub trait CausalLm<B: Backend> {
    type Cache;
    type Error: std::error::Error;

    fn cache(
        &self,
        backend: &mut B,
        batch_size: NonZeroUsize,
    ) -> Result<Self::Cache, Self::Error>;

    fn forward(
        &self,
        backend: &mut B,
        token_ids: &Tensor<B>,
        positions: &Tensor<B>,
        mask: &Tensor<B>,
        cache: &mut Self::Cache,
        position: usize,
    ) -> Result<Tensor<B>, Self::Error>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    pub token_ids: Vec<Vec<u32>>,
    pub prefill: Duration,
    pub decode_steps: Vec<Duration>,
}

pub struct Decoder<'run, B: Backend, M: CausalLm<B>> {
    backend: &'run mut B,
    model: &'run M,
    cache: M::Cache,
    batch_size: NonZeroUsize,
}

pub fn generate<B: Backend, M: CausalLm<B>>(
    backend: &mut B,
    model: &M,
    prompts: &[&[u32]],
    max_new_tokens: usize,
    stop_ids: &[u32],
) -> Result<Generation, GenerateError> {
    let (batch_size, prompt_len) = batch_dims(prompts)?;
    if max_new_tokens == 0 {
        return Ok(Generation {
            token_ids: vec![Vec::new(); batch_size.get()],
            prefill: Duration::ZERO,
            decode_steps: Vec::new(),
        });
    }
    let mut decoder = Decoder::new(backend, model, batch_size)?;
    let prefill_start = Instant::now();
    let mut tokens = decoder.step(&prompts.concat(), 0)?;
    let mut generation = Generation {
        token_ids: tokens.iter().map(|&token| vec![token]).collect(),
        prefill: prefill_start.elapsed(),
        decode_steps: Vec::new(),
    };
    let mut position = prompt_len;
    for _ in 1..max_new_tokens {
        if generation
            .token_ids
            .iter()
            .all(|row| is_stopped(row, stop_ids))
        {
            break;
        }
        let step_start = Instant::now();
        tokens = decoder.step(&tokens, position)?;
        generation.decode_steps.push(step_start.elapsed());
        generation.token_ids = extend_rows(generation.token_ids, &tokens, stop_ids);
        position = position.checked_add(1).ok_or_else(|| {
            error!(message = "Token position overflows usize.", position);
            GenerateError::PositionOverflow
        })?;
    }
    Ok(generation)
}

impl<'run, B: Backend, M: CausalLm<B>> Decoder<'run, B, M> {
    pub fn new(
        backend: &'run mut B,
        model: &'run M,
        batch_size: NonZeroUsize,
    ) -> Result<Self, GenerateError> {
        let cache = model.cache(backend, batch_size).map_err(|error| {
            error!(message = "Creating the key-value cache failed.", %error);
            GenerateError::Cache
        })?;
        Ok(Self {
            backend,
            model,
            cache,
            batch_size,
        })
    }

    pub fn step(
        &mut self,
        token_ids: &[u32],
        position: usize,
    ) -> Result<Vec<u32>, GenerateError> {
        let token_count = token_ids.len();
        let (seq_len, seq_len_total) = step_lens(token_count, self.batch_size, position)?;
        let token_bytes = token_ids
            .iter()
            .flat_map(|token_id| token_id.to_le_bytes())
            .collect::<Vec<u8>>();
        let tokens = self.upload(&token_bytes, DType::U32, &[token_count])?;
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
        self.next_tokens(&logits)
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

    fn next_tokens(
        &mut self,
        logits: &Tensor<B>,
    ) -> Result<Vec<u32>, GenerateError> {
        let bytes = logits
            .argmax(self.backend, VOCAB_AXIS)
            .and_then(|token| token.download(self.backend))
            .map_err(|error| {
                error!(message = "Selecting the next token failed.", %error);
                GenerateError::NextToken
            })?;
        decode_token_ids(&bytes, self.batch_size)
    }
}

fn batch_dims(prompts: &[&[u32]]) -> Result<(NonZeroUsize, usize), GenerateError> {
    let Some(batch_size) = NonZeroUsize::new(prompts.len()) else {
        error!(message = "Batch holds no prompt.");
        return Err(GenerateError::EmptyBatch);
    };
    let prompt_len = prompts.first().map_or(0, |prompt| prompt.len());
    if prompt_len == 0 {
        error!(message = "Prompt holds no token.");
        return Err(GenerateError::EmptyPrompt);
    }
    if prompts.iter().any(|prompt| prompt.len() != prompt_len) {
        error!(
            message = "Prompts do not have the same length.",
            prompt_lens = ?prompts.iter().map(|prompt| prompt.len()).collect::<Vec<usize>>(),
        );
        return Err(GenerateError::RaggedBatch);
    }
    Ok((batch_size, prompt_len))
}

fn step_lens(
    token_count: usize,
    batch_size: NonZeroUsize,
    position: usize,
) -> Result<(usize, usize), GenerateError> {
    let batch_size = batch_size.get();
    let Some(seq_len) = token_count
        .checked_div(batch_size)
        .filter(|seq_len| seq_len.checked_mul(batch_size) == Some(token_count))
    else {
        error!(
            message = "Token count is not a multiple of the batch size.",
            token_count, batch_size
        );
        return Err(GenerateError::BatchMismatch);
    };
    let seq_len_total = position.checked_add(seq_len).ok_or_else(|| {
        error!(
            message = "Token position overflows usize.",
            position, seq_len
        );
        GenerateError::PositionOverflow
    })?;
    Ok((seq_len, seq_len_total))
}

fn decode_token_ids(
    bytes: &[u8],
    batch_size: NonZeroUsize,
) -> Result<Vec<u32>, GenerateError> {
    let (tokens, remainder) = bytes.as_chunks::<TOKEN_ID_BYTES>();
    if tokens.len() != batch_size.get() || !remainder.is_empty() {
        error!(
            message = "Next token download does not hold one u32 per sequence.",
            bytes = bytes.len(),
            batch_size = batch_size.get(),
        );
        return Err(GenerateError::TokenBytes);
    }
    Ok(tokens.iter().copied().map(u32::from_le_bytes).collect())
}

fn is_stopped(
    row: &[u32],
    stop_ids: &[u32],
) -> bool {
    row.last().is_some_and(|token| stop_ids.contains(token))
}

fn extend_rows(
    rows: Vec<Vec<u32>>,
    tokens: &[u32],
    stop_ids: &[u32],
) -> Vec<Vec<u32>> {
    rows.into_iter()
        .zip(tokens)
        .map(|(mut row, &token)| {
            if !is_stopped(&row, stop_ids) {
                row.push(token);
            }
            row
        })
        .collect()
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
    use std::num::NonZeroUsize;

    use half::bf16;

    use crate::generate::{
        GenerateError,
        batch_dims,
        causal_mask,
        decode_token_ids,
        extend_rows,
        positions,
        step_lens,
    };

    #[test]
    fn batch_dims_reads_the_batch_size_and_the_prompt_length() {
        assert_eq!(
            batch_dims(&[&[1, 2, 3], &[4, 5, 6]]),
            Ok((NonZeroUsize::new(2).expect("2 is not zero"), 3)),
            "two prompts of three tokens"
        );
    }

    #[test]
    fn batch_dims_rejects_an_empty_batch_an_empty_prompt_or_a_ragged_batch() {
        let empty: [&[u32]; 0] = [];
        assert_eq!(
            [&empty[..], &[&[], &[]], &[&[1, 2], &[3]]].map(batch_dims),
            [
                Err(GenerateError::EmptyBatch),
                Err(GenerateError::EmptyPrompt),
                Err(GenerateError::RaggedBatch),
            ],
            "no prompt, then empty prompts, then prompts of two and one tokens"
        );
    }

    #[test]
    fn step_lens_splits_the_tokens_over_the_batch() {
        assert_eq!(
            step_lens(6, NonZeroUsize::new(2).expect("2 is not zero"), 4),
            Ok((3, 7)),
            "six tokens over two sequences at position 4"
        );
    }

    #[test]
    fn step_lens_rejects_a_ragged_step_or_a_position_overflow() {
        let batch_size = NonZeroUsize::new(2).expect("2 is not zero");
        assert_eq!(
            [
                step_lens(3, batch_size, 0),
                step_lens(2, batch_size, usize::MAX)
            ],
            [
                Err(GenerateError::BatchMismatch),
                Err(GenerateError::PositionOverflow)
            ],
            "three tokens over two sequences, then one token past usize::MAX"
        );
    }

    #[test]
    fn decode_token_ids_reads_one_u32_per_sequence() {
        assert_eq!(
            decode_token_ids(
                &[7_u32, 9]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<u8>>(),
                NonZeroUsize::new(2).expect("2 is not zero"),
            ),
            Ok(vec![7, 9]),
            "tokens 7 and 9"
        );
    }

    #[test]
    fn decode_token_ids_rejects_a_wrong_byte_count() {
        let batch_size = NonZeroUsize::new(2).expect("2 is not zero");
        assert_eq!(
            [
                decode_token_ids(&[0; 4], batch_size),
                decode_token_ids(&[0; 9], batch_size)
            ],
            [
                Err(GenerateError::TokenBytes),
                Err(GenerateError::TokenBytes)
            ],
            "one token for two sequences, then a trailing byte"
        );
    }

    #[test]
    fn extend_rows_leaves_stopped_rows_unchanged() {
        assert_eq!(
            extend_rows(vec![vec![2], vec![3]], &[4, 5], &[3]),
            vec![vec![2, 4], vec![3]],
            "row 1 ended on stop id 3"
        );
    }

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
