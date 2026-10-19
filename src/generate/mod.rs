mod error;

use std::num::NonZeroUsize;
use std::time::{
    Duration,
    Instant,
};

use tracing::error;

pub use self::error::GenerateError;
use crate::models::CausalLm;
use crate::nn::{
    DType,
    Device,
    DeviceProfile,
    Shape,
    Tensor,
};

const VOCAB_DIM: usize = 2;
const LOGITS_TO_KEEP: usize = 1;
const TOKEN_ID_BYTES: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepLogits {
    pub dtype: DType,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    pub token_ids: Vec<Vec<u32>>,
    pub prefill: Duration,
    pub decode_steps: Vec<Duration>,
}

pub struct Decoder<'run, D: Device, M: CausalLm<D>> {
    device: &'run mut D,
    model: &'run M,
    cache: M::Cache,
    batch_size: NonZeroUsize,
}

pub fn generate<D: Device, M: CausalLm<D>>(
    device: &mut D,
    model: &M,
    prompts: &[&[u32]],
    max_new_tokens: usize,
    eos_token_id: &[u32],
) -> Result<Generation, GenerateError> {
    let (batch_size, prompt_len) = batch_dims(prompts)?;
    if max_new_tokens == 0 {
        return Ok(Generation {
            token_ids: vec![Vec::new(); batch_size.get()],
            prefill: Duration::ZERO,
            decode_steps: Vec::new(),
        });
    }
    let mut decoder = Decoder::new(device, model, batch_size)?;
    let prefill_start = Instant::now();
    let mut tokens = decoder.step(&prompts.concat(), 0)?;
    let mut generation = Generation {
        token_ids: tokens.iter().map(|&token| vec![token]).collect(),
        prefill: prefill_start.elapsed(),
        decode_steps: Vec::new(),
    };
    let mut past_seen_tokens = prompt_len;
    for _ in 1..max_new_tokens {
        if generation
            .token_ids
            .iter()
            .all(|row| is_stopped(row, eos_token_id))
        {
            break;
        }
        let step_start = Instant::now();
        tokens = decoder.step(&tokens, past_seen_tokens)?;
        generation.decode_steps.push(step_start.elapsed());
        generation.token_ids = extend_rows(generation.token_ids, &tokens, eos_token_id);
        past_seen_tokens = past_seen_tokens.checked_add(1).ok_or_else(|| {
            error!(
                message = "Token position overflows usize.",
                past_seen_tokens
            );
            GenerateError::PositionOverflow
        })?;
    }
    Ok(generation)
}

impl<'run, D: Device, M: CausalLm<D>> Decoder<'run, D, M> {
    pub fn new(
        device: &'run mut D,
        model: &'run M,
        batch_size: NonZeroUsize,
    ) -> Result<Self, GenerateError> {
        let cache = model.cache(device, batch_size).map_err(|error| {
            error!(message = "Creating the key-value cache failed.", %error);
            GenerateError::Cache
        })?;
        Ok(Self {
            device,
            model,
            cache,
            batch_size,
        })
    }

    pub fn step(
        &mut self,
        input_ids: &[u32],
        past_seen_tokens: usize,
    ) -> Result<Vec<u32>, GenerateError> {
        let logits = self.forward_logits(input_ids, past_seen_tokens)?;
        self.next_tokens(&logits)
    }

    pub fn step_with_logits(
        &mut self,
        input_ids: &[u32],
        past_seen_tokens: usize,
    ) -> Result<(Vec<u32>, StepLogits), GenerateError> {
        let logits = self.forward_logits(input_ids, past_seen_tokens)?;
        let token_ids = self.next_tokens(&logits)?;
        let bytes = logits
            .contiguous(self.device)
            .and_then(|logits| logits.to_bytes(self.device))
            .map_err(|error| {
                error!(message = "Reading the logits failed.", %error);
                GenerateError::Logits
            })?;
        Ok((
            token_ids,
            StepLogits {
                dtype: logits.dtype(),
                bytes,
            },
        ))
    }

    pub(crate) fn synchronize(&mut self) -> Result<(), GenerateError> {
        self.device.synchronize().map_err(|error| {
            error!(message = "Synchronizing device work failed.", %error);
            GenerateError::Synchronize
        })
    }

    pub fn take_profile(&mut self) -> Option<DeviceProfile> { self.device.take_profile() }

    pub(crate) fn forward_logits(
        &mut self,
        input_ids: &[u32],
        past_seen_tokens: usize,
    ) -> Result<Tensor<D>, GenerateError> {
        let seq_len = step_seq_len(input_ids.len(), self.batch_size)?;
        let token_bytes = input_ids
            .iter()
            .flat_map(|token_id| token_id.to_le_bytes())
            .collect::<Vec<u8>>();
        let input_ids =
            self.tensor_from_bytes(&token_bytes, DType::U32, &[self.batch_size.get(), seq_len])?;
        self.model
            .forward(
                self.device,
                &input_ids,
                &mut self.cache,
                past_seen_tokens,
                LOGITS_TO_KEEP,
            )
            .map_err(|error| {
                error!(message = "Model forward failed.", past_seen_tokens, seq_len, %error);
                GenerateError::Forward
            })
    }

    fn tensor_from_bytes(
        &mut self,
        bytes: &[u8],
        dtype: DType,
        dims: &[usize],
    ) -> Result<Tensor<D>, GenerateError> {
        let tensor_from_bytes_error = |error: &dyn std::error::Error| {
            error!(message = "Creating a generation input tensor failed.", ?dtype, ?dims, %error);
            GenerateError::FromBytes
        };
        let shape = Shape::try_from(dims).map_err(|error| tensor_from_bytes_error(&error))?;
        Tensor::from_bytes(self.device, bytes, dtype, shape)
            .map_err(|error| tensor_from_bytes_error(&error))
    }

    pub(crate) fn next_tokens(
        &mut self,
        logits: &Tensor<D>,
    ) -> Result<Vec<u32>, GenerateError> {
        let bytes = logits
            .argmax_keepdim(self.device, VOCAB_DIM)
            .and_then(|token| token.to_bytes(self.device))
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

fn step_seq_len(
    token_count: usize,
    batch_size: NonZeroUsize,
) -> Result<usize, GenerateError> {
    let batch_size = batch_size.get();
    token_count
        .checked_div(batch_size)
        .filter(|seq_len| seq_len.checked_mul(batch_size) == Some(token_count))
        .ok_or_else(|| {
            error!(
                message = "Token count is not a multiple of the batch size.",
                token_count, batch_size
            );
            GenerateError::BatchMismatch
        })
}

fn decode_token_ids(
    bytes: &[u8],
    batch_size: NonZeroUsize,
) -> Result<Vec<u32>, GenerateError> {
    let (tokens, remainder) = bytes.as_chunks::<TOKEN_ID_BYTES>();
    if tokens.len() != batch_size.get() || !remainder.is_empty() {
        error!(
            message = "Next token bytes do not hold one u32 per sequence.",
            bytes = bytes.len(),
            batch_size = batch_size.get(),
        );
        return Err(GenerateError::TokenBytes);
    }
    Ok(tokens.iter().copied().map(u32::from_le_bytes).collect())
}

fn is_stopped(
    row: &[u32],
    eos_token_id: &[u32],
) -> bool {
    row.last().is_some_and(|token| eos_token_id.contains(token))
}

fn extend_rows(
    rows: Vec<Vec<u32>>,
    tokens: &[u32],
    eos_token_id: &[u32],
) -> Vec<Vec<u32>> {
    rows.into_iter()
        .zip(tokens)
        .map(|(mut row, &token)| {
            if !is_stopped(&row, eos_token_id) {
                row.push(token);
            }
            row
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::generate::{
        GenerateError,
        batch_dims,
        decode_token_ids,
        extend_rows,
        step_seq_len,
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
    fn step_seq_len_splits_the_tokens_over_the_batch() {
        assert_eq!(
            step_seq_len(6, NonZeroUsize::new(2).expect("2 is not zero")),
            Ok(3),
            "six tokens over two sequences"
        );
    }

    #[test]
    fn step_seq_len_rejects_a_ragged_step() {
        assert_eq!(
            step_seq_len(3, NonZeroUsize::new(2).expect("2 is not zero")),
            Err(GenerateError::BatchMismatch),
            "three tokens over two sequences"
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
}
