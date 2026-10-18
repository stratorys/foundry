use half::bf16;
use tracing::error;

use crate::nn::{
    DType,
    Device,
    Shape,
    Tensor,
    TensorError,
};

pub fn position_ids<D: Device>(
    device: &mut D,
    past_seen_tokens: usize,
    seq_len: usize,
) -> Result<Tensor<D>, TensorError> {
    let kv_length = kv_length(past_seen_tokens, seq_len)?;
    Tensor::from_bytes(
        device,
        &position_id_bytes(past_seen_tokens, kv_length)?,
        DType::U32,
        Shape::try_from([seq_len])?,
    )
}

pub fn create_causal_mask<D: Device>(
    device: &mut D,
    past_seen_tokens: usize,
    seq_len: usize,
) -> Result<Tensor<D>, TensorError> {
    let (shape, bytes) = causal_mask(past_seen_tokens, seq_len)?;
    Tensor::from_bytes(device, &bytes, DType::BF16, shape)
}

fn causal_mask(
    past_seen_tokens: usize,
    seq_len: usize,
) -> Result<(Shape, Vec<u8>), TensorError> {
    let kv_length = kv_length(past_seen_tokens, seq_len)?;
    let shape = Shape::try_from([seq_len, kv_length])?;
    Ok((shape, causal_mask_bytes(past_seen_tokens, kv_length)))
}

fn kv_length(
    past_seen_tokens: usize,
    seq_len: usize,
) -> Result<usize, TensorError> {
    past_seen_tokens.checked_add(seq_len).ok_or_else(|| {
        error!(
            message = "Token position overflows usize.",
            past_seen_tokens, seq_len
        );
        TensorError::PositionOverflow
    })
}

fn position_id_bytes(
    past_seen_tokens: usize,
    kv_length: usize,
) -> Result<Vec<u8>, TensorError> {
    (past_seen_tokens..kv_length)
        .map(|token_position| {
            u32::try_from(token_position).map_err(|_| {
                error!(
                    message = "Token position does not fit in u32.",
                    token_position
                );
                TensorError::PositionOverflow
            })
        })
        .try_fold(Vec::new(), |mut bytes, token_position| {
            bytes.extend(token_position?.to_le_bytes());
            Ok(bytes)
        })
}

fn causal_mask_bytes(
    past_seen_tokens: usize,
    kv_length: usize,
) -> Vec<u8> {
    (past_seen_tokens..kv_length)
        .flat_map(|query_position| {
            (0..kv_length).map(move |key_position| {
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

    use crate::nn::TensorError;
    use crate::nn::masking::{
        causal_mask,
        causal_mask_bytes,
        kv_length,
        position_id_bytes,
    };

    #[test]
    fn causal_mask_bytes_of_a_prefill_hide_later_keys() {
        let zero = bf16::ZERO;
        let hidden = bf16::NEG_INFINITY;
        assert_eq!(
            causal_mask_bytes(0, 3),
            [zero, hidden, hidden, zero, zero, hidden, zero, zero, zero]
                .into_iter()
                .flat_map(bf16::to_le_bytes)
                .collect::<Vec<u8>>(),
            "three queries over three keys, lower triangle visible"
        );
    }

    #[test]
    fn causal_mask_bytes_of_a_decode_step_show_every_key() {
        assert_eq!(
            causal_mask_bytes(4, 5),
            [bf16::ZERO; 5]
                .into_iter()
                .flat_map(bf16::to_le_bytes)
                .collect::<Vec<u8>>(),
            "one query at position 4 sees keys 0 to 4"
        );
    }

    #[test]
    fn causal_mask_checks_the_shape_before_building_the_bytes() {
        let dims_and_bytes = |past_seen_tokens, seq_len| {
            causal_mask(past_seen_tokens, seq_len)
                .map(|(shape, bytes)| (shape.dims().to_vec(), bytes))
        };
        assert_eq!(
            [dims_and_bytes(4, 1), dims_and_bytes(0, 100_000)],
            [
                Ok((
                    vec![1, 5],
                    [bf16::ZERO; 5]
                        .into_iter()
                        .flat_map(bf16::to_le_bytes)
                        .collect::<Vec<u8>>()
                )),
                Err(TensorError::NumelOverflow),
            ],
            "one decode step over five keys, then a 100 000 token prefill that would build about \
             20 GB of mask bytes"
        );
    }

    #[test]
    fn position_id_bytes_count_from_the_cache_offset() {
        assert_eq!(
            position_id_bytes(2, 5),
            Ok([2_u32, 3, 4]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<u8>>()),
            "positions 2, 3 and 4"
        );
    }

    #[test]
    fn position_id_bytes_beyond_u32_are_rejected() {
        let past_seen_tokens = usize::try_from(u32::MAX).expect("u32 fits in usize");
        assert_eq!(
            position_id_bytes(past_seen_tokens, past_seen_tokens.saturating_add(2)),
            Err(TensorError::PositionOverflow),
            "position u32::MAX + 1 does not fit"
        );
    }

    #[test]
    fn kv_length_adds_the_new_tokens_to_the_cache_offset() {
        assert_eq!(
            [kv_length(4, 3), kv_length(usize::MAX, 1)],
            [Ok(7), Err(TensorError::PositionOverflow)],
            "three tokens after four, then one token past usize::MAX"
        );
    }
}
