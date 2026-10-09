use std::num::NonZeroUsize;

use crate::core::{
    Backend,
    CoreError,
    DType,
    Shape,
    Tensor,
};
use crate::nn::softmax_last_axis;

pub struct Attention<B: Backend> {
    head_dim: NonZeroUsize,
    scale: Tensor<B>,
}

#[derive(Debug, PartialEq, Eq)]
struct AttentionDims {
    heads: usize,
    kv_heads: usize,
    group: usize,
    seq_len: usize,
    seq_len_total: usize,
    head_dim: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        backend: &mut B,
        head_dim: NonZeroUsize,
    ) -> Result<Self, B::Error> {
        let head_dim_f32 = u16::try_from(head_dim.get()).map(f32::from).map_err(|_| {
            CoreError::DimensionTooLargeForF32 {
                dim: head_dim.get(),
                dim_max: u16::MAX.into(),
            }
        })?;
        Ok(Self {
            head_dim,
            scale: Tensor::upload(
                backend,
                &head_dim_f32.sqrt().recip().to_le_bytes(),
                DType::F32,
                Shape::try_from([1].as_slice())?,
            )?,
        })
    }

    pub fn forward(
        &self,
        backend: &mut B,
        query: &Tensor<B>,
        keys: &Tensor<B>,
        values: &Tensor<B>,
        mask: &Tensor<B>,
    ) -> Result<Tensor<B>, B::Error> {
        let AttentionDims {
            heads,
            kv_heads,
            group,
            seq_len,
            seq_len_total,
            head_dim,
        } = attention_dims(self.head_dim, query.shape(), keys.shape(), values.shape())?;
        let rows = group.saturating_mul(seq_len);
        let keys_t = keys.permute(&[0, 2, 1])?;
        let mask_f32 = mask.cast(backend, DType::F32)?;
        let scores = query
            .contiguous(backend)?
            .reshape(Shape::try_from([kv_heads, rows, head_dim].as_slice())?)?
            .matmul(backend, &keys_t)?
            .reshape(Shape::try_from(
                [kv_heads, group, seq_len, seq_len_total].as_slice(),
            )?)?
            .cast(backend, DType::F32)?
            .mul(backend, &self.scale)?
            .add(backend, &mask_f32)?;
        softmax_last_axis(backend, &scores)?
            .cast(backend, query.dtype())?
            .reshape(Shape::try_from([kv_heads, rows, seq_len_total].as_slice())?)?
            .matmul(backend, values)?
            .reshape(Shape::try_from([heads, seq_len, head_dim].as_slice())?)?
            .permute(&[1, 0, 2])?
            .contiguous(backend)?
            .reshape(Shape::try_from(
                [seq_len, heads.saturating_mul(head_dim)].as_slice(),
            )?)
            .map_err(Into::into)
    }
}

fn attention_dims(
    head_dim_expected: NonZeroUsize,
    query: &Shape,
    keys: &Shape,
    values: &Shape,
) -> Result<AttentionDims, CoreError> {
    let incompatible = || CoreError::AttentionIncompatible {
        query: query.dims().to_vec(),
        key: keys.dims().to_vec(),
        value: values.dims().to_vec(),
    };
    let (&[heads, seq_len, head_dim], &[kv_heads, seq_len_total, key_head_dim]) =
        (query.dims(), keys.dims())
    else {
        return Err(incompatible());
    };
    let is_grouped = heads.checked_rem(kv_heads) == Some(0);
    let is_compatible =
        is_grouped && seq_len <= seq_len_total && head_dim == key_head_dim && keys == values;
    if !is_compatible {
        return Err(incompatible());
    }
    if head_dim != head_dim_expected.get() {
        return Err(CoreError::AttentionHeadDimMismatch {
            head_dim,
            head_dim_expected: head_dim_expected.get(),
        });
    }
    Ok(AttentionDims {
        heads,
        kv_heads,
        group: heads.checked_div(kv_heads).ok_or_else(incompatible)?,
        seq_len,
        seq_len_total,
        head_dim,
    })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::backend::cpu::{
        CpuBackend,
        CpuError,
    };
    use crate::core::{
        CoreError,
        DType,
        Shape,
        Tensor,
    };
    use crate::nn::attention::{
        AttentionDims,
        attention_dims,
    };
    use crate::nn::{
        Attention,
        KvCache,
    };

    #[test]
    fn attention_dims_computes_the_group() {
        let [query, keys] = [[4, 1, 2], [2, 3, 2]]
            .map(|dims| Shape::try_from(dims.as_slice()).expect("the shape is valid"));
        assert_eq!(
            attention_dims(
                NonZeroUsize::new(2).expect("the head dim is non-zero"),
                &query,
                &keys,
                &keys,
            ),
            Ok(AttentionDims {
                heads: 4,
                kv_heads: 2,
                group: 2,
                seq_len: 1,
                seq_len_total: 3,
                head_dim: 2,
            }),
            "four query heads share two key-value heads"
        );
    }

    #[test]
    fn attention_dims_rejects_rank_other_than_three() {
        assert_eq!(
            attention_dims(
                NonZeroUsize::new(2).expect("the head dim is non-zero"),
                &Shape::try_from([4, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(CoreError::AttentionIncompatible {
                query: vec![4, 2],
                key: vec![2, 2, 2],
                value: vec![2, 2, 2],
            }),
            "a rank 2 query"
        );
        assert_eq!(
            attention_dims(
                NonZeroUsize::new(2).expect("the head dim is non-zero"),
                &Shape::try_from([4, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(CoreError::AttentionIncompatible {
                query: vec![4, 2, 2],
                key: vec![2, 2],
                value: vec![2, 2],
            }),
            "rank 2 keys"
        );
    }

    #[test]
    fn attention_dims_rejects_zero_kv_heads() {
        assert_eq!(
            attention_dims(
                NonZeroUsize::new(2).expect("the head dim is non-zero"),
                &Shape::try_from([4, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([0, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([0, 2, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(CoreError::AttentionIncompatible {
                query: vec![4, 2, 2],
                key: vec![0, 2, 2],
                value: vec![0, 2, 2],
            }),
            "zero key-value heads"
        );
    }

    #[test]
    fn attention_dims_rejects_heads_not_a_multiple_of_kv_heads() {
        assert_eq!(
            attention_dims(
                NonZeroUsize::new(2).expect("the head dim is non-zero"),
                &Shape::try_from([3, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(CoreError::AttentionIncompatible {
                query: vec![3, 2, 2],
                key: vec![2, 2, 2],
                value: vec![2, 2, 2],
            }),
            "three query heads over two key-value heads"
        );
    }

    #[test]
    fn attention_dims_rejects_query_longer_than_keys() {
        assert_eq!(
            attention_dims(
                NonZeroUsize::new(2).expect("the head dim is non-zero"),
                &Shape::try_from([4, 3, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(CoreError::AttentionIncompatible {
                query: vec![4, 3, 2],
                key: vec![2, 2, 2],
                value: vec![2, 2, 2],
            }),
            "three query positions over two key positions"
        );
    }

    #[test]
    fn attention_dims_rejects_key_head_dim_mismatch() {
        assert_eq!(
            attention_dims(
                NonZeroUsize::new(2).expect("the head dim is non-zero"),
                &Shape::try_from([4, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 4].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 4].as_slice()).expect("the shape is valid"),
            ),
            Err(CoreError::AttentionIncompatible {
                query: vec![4, 2, 2],
                key: vec![2, 2, 4],
                value: vec![2, 2, 4],
            }),
            "head dims 2 and 4 differ"
        );
    }

    #[test]
    fn attention_dims_rejects_values_shaped_unlike_keys() {
        assert_eq!(
            attention_dims(
                NonZeroUsize::new(2).expect("the head dim is non-zero"),
                &Shape::try_from([4, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 3, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(CoreError::AttentionIncompatible {
                query: vec![4, 2, 2],
                key: vec![2, 2, 2],
                value: vec![2, 3, 2],
            }),
            "values hold one more position than keys"
        );
    }

    #[test]
    fn attention_matches_hand_computed_values() {
        let mut backend = CpuBackend::new();
        let [query, key, value, mask] = [
            (
                vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.0],
                vec![2, 2, 2],
            ),
            (vec![1.0, 0.0, 0.0, 1.0], vec![1, 2, 2]),
            (vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]),
            (vec![0.0, f32::NEG_INFINITY, 0.0, 0.0], vec![2, 2]),
        ]
        .map(|(values, dims)| {
            Tensor::upload(
                &mut backend,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<u8>>(),
                DType::F32,
                Shape::try_from(dims.as_slice()).expect("the shape is valid"),
            )
            .expect("the upload succeeds")
            .cast(&mut backend, DType::BF16)
            .expect("the cast succeeds")
        });
        let mut cache =
            KvCache::new(&mut backend, DType::BF16, 1, 4, 2).expect("the cache is built");
        let attention = Attention::new(
            &mut backend,
            NonZeroUsize::new(2).expect("the head dim is non-zero"),
        )
        .expect("the attention is built");
        let (keys, values) = cache
            .update(&mut backend, &key, &value, 0)
            .expect("the cache update succeeds");
        assert_eq!(
            keys.shape().dims(),
            &[1, 2, 2],
            "keys view after the prefill"
        );
        let output = attention
            .forward(&mut backend, &query, &keys, &values, &mask)
            .expect("the attention succeeds");
        assert_eq!(output.dtype(), DType::BF16, "output keeps the input dtype");
        assert_eq!(output.shape().dims(), &[2, 4], "output dims");
        let actual: Vec<f32> = output
            .cast(&mut backend, DType::F32)
            .expect("the cast succeeds")
            .download(&mut backend)
            .expect("the download succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        let expected = [
            1.0_f32, 2.0, 1.0, 2.0, 2.339_52, 3.339_52, 1.391_15, 2.391_15,
        ];
        assert_eq!(actual.len(), expected.len(), "element count");
        actual.iter().zip(expected).for_each(|(&actual, expected)| {
            assert!(
                (actual - expected).abs() <= 1e-2 * expected.abs(),
                "attention: {actual} is not within 1e-2 of {expected}"
            );
        });
    }

    #[test]
    fn attention_decode_after_prefill_matches_full_prefill() {
        let mut backend = CpuBackend::new();
        let [query, key, value, causal_mask] = [
            (
                vec![
                    0.5_f32, -1.0, 1.0, 0.25, -0.5, 2.0, 1.0, 1.0, 0.0, -1.0, 1.5, 0.5,
                ],
                vec![2, 3, 2],
            ),
            (vec![1.0, 0.5, -1.0, 1.0, 0.25, -0.75], vec![1, 3, 2]),
            (vec![1.0, 2.0, -1.0, 0.5, 3.0, -2.0], vec![1, 3, 2]),
            (
                vec![
                    0.0,
                    f32::NEG_INFINITY,
                    f32::NEG_INFINITY,
                    0.0,
                    0.0,
                    f32::NEG_INFINITY,
                    0.0,
                    0.0,
                    0.0,
                ],
                vec![3, 3],
            ),
        ]
        .map(|(values, dims)| {
            Tensor::upload(
                &mut backend,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<u8>>(),
                DType::F32,
                Shape::try_from(dims.as_slice()).expect("the shape is valid"),
            )
            .expect("the upload succeeds")
            .cast(&mut backend, DType::BF16)
            .expect("the cast succeeds")
        });
        let attention = Attention::new(
            &mut backend,
            NonZeroUsize::new(2).expect("the head dim is non-zero"),
        )
        .expect("the attention is built");
        let mut prefill_cache =
            KvCache::new(&mut backend, DType::BF16, 1, 3, 2).expect("the cache is built");
        let (prefill_keys, prefill_values) = prefill_cache
            .update(&mut backend, &key, &value, 0)
            .expect("the cache update succeeds");
        let prefill: Vec<f32> = attention
            .forward(
                &mut backend,
                &query,
                &prefill_keys,
                &prefill_values,
                &causal_mask,
            )
            .expect("the prefill succeeds")
            .cast(&mut backend, DType::F32)
            .expect("the cast succeeds")
            .download(&mut backend)
            .expect("the download succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        let mut step_cache =
            KvCache::new(&mut backend, DType::BF16, 1, 3, 2).expect("the cache is built");
        let steps: Vec<f32> = (0..3)
            .flat_map(|position| {
                let mask = Tensor::zeros(
                    &mut backend,
                    DType::BF16,
                    Shape::try_from([1, position + 1].as_slice()).expect("the shape is valid"),
                )
                .expect("the zeros succeed");
                let [query, key, value] = [&query, &key, &value]
                    .map(|tensor| tensor.narrow(1, position, 1).expect("the narrow succeeds"));
                let (keys, values) = step_cache
                    .update(&mut backend, &key, &value, position)
                    .expect("the cache update succeeds");
                assert_eq!(
                    keys.shape().dims(),
                    &[1, position + 1, 2],
                    "keys view after the decode step"
                );
                attention
                    .forward(&mut backend, &query, &keys, &values, &mask)
                    .expect("the step succeeds")
                    .cast(&mut backend, DType::F32)
                    .expect("the cast succeeds")
                    .download(&mut backend)
                    .expect("the download succeeds")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&chunk| f32::from_le_bytes(chunk))
                    .collect::<Vec<f32>>()
            })
            .collect();
        assert_eq!(steps.len(), prefill.len(), "element count");
        steps.iter().zip(&prefill).for_each(|(&step, &full)| {
            assert!(
                (step - full).abs() <= 1e-2 * full.abs(),
                "decode: {step} is not within 1e-2 of prefill {full}"
            );
        });
    }

    #[test]
    fn attention_rejects_a_head_dim_other_than_its_own() {
        let mut backend = CpuBackend::new();
        let [query, keys, values, mask] =
            [&[2, 2, 2][..], &[1, 2, 2], &[1, 2, 2], &[2, 2]].map(|dims| {
                Tensor::zeros(
                    &mut backend,
                    DType::BF16,
                    Shape::try_from(dims).expect("the shape is valid"),
                )
                .expect("the zeros succeed")
            });
        let attention = Attention::new(
            &mut backend,
            NonZeroUsize::new(4).expect("the head dim is non-zero"),
        )
        .expect("the attention is built");
        assert!(
            matches!(
                attention.forward(&mut backend, &query, &keys, &values, &mask),
                Err(CpuError::Core(CoreError::AttentionHeadDimMismatch {
                    head_dim: 2,
                    head_dim_expected: 4,
                }))
            ),
            "a head dim of 2 is rejected by an attention built for 4"
        );
    }
}
