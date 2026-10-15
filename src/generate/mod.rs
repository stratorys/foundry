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
    if max_new_tokens == 0 {
        return Ok(Generation {
            token_ids: vec![Vec::new(); batch_size.get()],
            prefill: Duration::ZERO,
            decode_steps: Vec::new(),
        });
    }
    let is_stopped = |row: &Vec<u32>| row.last().is_some_and(|token| stop_ids.contains(token));
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
        if generation.token_ids.iter().all(is_stopped) {
            break;
        }
        let step_start = Instant::now();
        tokens = decoder.step(&tokens, position)?;
        generation.decode_steps.push(step_start.elapsed());
        generation
            .token_ids
            .iter_mut()
            .zip(&tokens)
            .filter(|(row, _)| !is_stopped(row))
            .for_each(|(row, &token)| row.push(token));
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
        let batch_size = self.batch_size.get();
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
        let (tokens, remainder) = bytes.as_chunks::<TOKEN_ID_BYTES>();
        if tokens.len() != self.batch_size.get() || !remainder.is_empty() {
            error!(
                message = "Next token download does not hold one u32 per sequence.",
                bytes = bytes.len(),
                batch_size = self.batch_size.get(),
            );
            return Err(GenerateError::TokenBytes);
        }
        Ok(tokens.iter().copied().map(u32::from_le_bytes).collect())
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
    use std::num::NonZeroUsize;

    use half::bf16;

    use crate::backend::cpu::{
        CpuBackend,
        CpuError,
    };
    use crate::core::{
        DType,
        Shape,
        Tensor,
        TensorError,
    };
    use crate::generate::{
        CausalLm,
        Decoder,
        GenerateError,
        causal_mask,
        generate,
        positions,
    };

    const VOCAB_SIZE: usize = 16;

    struct PositionModel;

    impl CausalLm<CpuBackend> for PositionModel {
        type Cache = ();
        type Error = TensorError<CpuError>;

        fn cache(
            &self,
            _backend: &mut CpuBackend,
            _batch_size: NonZeroUsize,
        ) -> Result<Self::Cache, Self::Error> {
            Ok(())
        }

        fn forward(
            &self,
            backend: &mut CpuBackend,
            token_ids: &Tensor<CpuBackend>,
            positions: &Tensor<CpuBackend>,
            _mask: &Tensor<CpuBackend>,
            _cache: &mut Self::Cache,
            position: usize,
        ) -> Result<Tensor<CpuBackend>, Self::Error> {
            let seq_len = positions.shape().element_count();
            let batch_size = token_ids
                .shape()
                .element_count()
                .checked_div(seq_len)
                .expect("the step holds at least one position");
            let seq_len_total = position.saturating_add(seq_len);
            let logits = (0..batch_size)
                .flat_map(|sequence| {
                    (0..VOCAB_SIZE).map(move |token_id| {
                        f32::from(u8::from(token_id == seq_len_total.saturating_add(sequence)))
                    })
                })
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<u8>>();
            let shape = Shape::try_from([batch_size, VOCAB_SIZE].as_slice())?;
            Tensor::upload(backend, &logits, DType::F32, shape)
        }
    }

    #[test]
    fn generate_stops_each_sequence_at_its_own_stop_id() {
        let generation = generate(
            &mut CpuBackend::new(),
            &PositionModel,
            &[&[1, 2], &[3, 4]],
            4,
            &[3],
        )
        .expect("the generation runs");
        assert_eq!(
            (generation.token_ids, generation.decode_steps.len()),
            (vec![vec![2, 3], vec![3]], 1),
            "sequence 1 stops at the prefill, sequence 0 after one decode step"
        );
    }

    #[test]
    fn generate_rejects_an_empty_or_ragged_batch() {
        let empty: [&[u32]; 0] = [];
        assert_eq!(
            [&empty[..], &[&[1, 2], &[3]]].map(|prompts| {
                generate(&mut CpuBackend::new(), &PositionModel, prompts, 2, &[])
            }),
            [
                Err(GenerateError::EmptyBatch),
                Err(GenerateError::RaggedBatch)
            ],
            "no prompt, then prompts of two and one tokens"
        );
    }

    #[test]
    fn step_rejects_tokens_not_divisible_by_the_batch_size() {
        let mut backend = CpuBackend::new();
        let mut decoder = Decoder::new(
            &mut backend,
            &PositionModel,
            NonZeroUsize::new(2).expect("2 is not zero"),
        )
        .expect("the decoder is created");
        assert_eq!(
            decoder.step(&[1, 2, 3], 0),
            Err(GenerateError::BatchMismatch),
            "three tokens cannot be split over two sequences"
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
