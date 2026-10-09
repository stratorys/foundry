use std::num::NonZeroUsize;

use foundry::backend::cpu::CpuBackend;
use foundry::core::{
    CoreError,
    DType,
    Shape,
    Tensor,
};
use foundry::nn::{
    Attention,
    Embedding,
    KvCache,
    Linear,
    RmsNorm,
    Silu,
    SwigluMlp,
    softmax_last_axis,
};

#[test]
fn softmax_rows_match_hand_computed_values() {
    let mut backend = CpuBackend::new();
    let x = Tensor::upload(
        &mut backend,
        &[1.0_f32, 2.0, 3.0, 0.0, 0.0, 0.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>(),
        DType::F32,
        Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
    )
    .expect("the upload succeeds")
    .cast(&mut backend, DType::BF16)
    .expect("the cast succeeds");
    let output = softmax_last_axis(&mut backend, &x).expect("the softmax succeeds");
    assert_eq!(output.dtype(), DType::BF16, "output keeps the input dtype");
    assert_eq!(output.shape().dims(), &[2, 3], "output dims");
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
    let third = 1.0_f32 / 3.0;
    let expected = [0.090_030_6_f32, 0.244_728_5, 0.665_241, third, third, third];
    assert_eq!(actual.len(), expected.len(), "element count");
    actual.iter().zip(expected).for_each(|(&actual, expected)| {
        assert!(
            (actual - expected).abs() <= 1e-2 * expected.abs(),
            "softmax: {actual} is not within 1e-2 of {expected}"
        );
    });
}

#[test]
fn silu_matches_hand_computed_values() {
    let mut backend = CpuBackend::new();
    let silu = Silu::new(&mut backend).expect("the silu is built");
    let x = Tensor::upload(
        &mut backend,
        &[0.0_f32, 1.0, -1.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>(),
        DType::F32,
        Shape::try_from([3].as_slice()).expect("the shape is valid"),
    )
    .expect("the upload succeeds")
    .cast(&mut backend, DType::BF16)
    .expect("the cast succeeds");
    let output = silu.forward(&mut backend, &x).expect("the silu succeeds");
    assert_eq!(output.dtype(), DType::BF16, "output keeps the input dtype");
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
    let expected = [0.0_f32, 0.731_058_6, -0.268_941_4];
    assert_eq!(actual.len(), expected.len(), "element count");
    actual.iter().zip(expected).for_each(|(&actual, expected)| {
        assert!(
            (actual - expected).abs() <= 1e-2 * expected.abs(),
            "silu: {actual} is not within 1e-2 of {expected}"
        );
    });
}

#[test]
fn rms_norm_matches_hand_computed_values() {
    let mut backend = CpuBackend::new();
    let weight = Tensor::upload(
        &mut backend,
        &[1.0_f32, 2.0, 0.5, -1.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>(),
        DType::F32,
        Shape::try_from([4].as_slice()).expect("the shape is valid"),
    )
    .expect("the upload succeeds")
    .cast(&mut backend, DType::BF16)
    .expect("the cast succeeds");
    let norm = RmsNorm::new(&mut backend, weight, 1e-5).expect("the norm is built");
    let x = Tensor::upload(
        &mut backend,
        &[1.0_f32, 2.0, 3.0, 4.0, -2.0, 0.0, 2.0, 4.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>(),
        DType::F32,
        Shape::try_from([2, 4].as_slice()).expect("the shape is valid"),
    )
    .expect("the upload succeeds")
    .cast(&mut backend, DType::BF16)
    .expect("the cast succeeds");
    let output = norm.forward(&mut backend, &x).expect("the norm succeeds");
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
        0.365_148_f32,
        1.460_593,
        0.547_722,
        -1.460_593,
        -0.816_496,
        0.0,
        0.408_248,
        -1.632_992,
    ];
    assert_eq!(actual.len(), expected.len(), "element count");
    actual.iter().zip(expected).for_each(|(&actual, expected)| {
        assert!(
            (actual - expected).abs() <= 1e-2 * expected.abs(),
            "rms_norm: {actual} is not within 1e-2 of {expected}"
        );
    });
}

#[test]
fn rms_norm_rejects_a_dimension_beyond_u16() {
    let mut backend = CpuBackend::new();
    let weight = Tensor::zeros(
        &mut backend,
        DType::BF16,
        Shape::try_from([65_536].as_slice()).expect("the shape is valid"),
    )
    .expect("the zeros succeed");
    assert_eq!(
        RmsNorm::new(&mut backend, weight, 1e-5).err(),
        Some(CoreError::DimensionTooLargeForF32),
        "a dimension above u16::MAX is rejected"
    );
}

#[test]
fn linear_multiplies_by_the_transposed_weight() {
    let mut backend = CpuBackend::new();
    let weight = Tensor::upload(
        &mut backend,
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>(),
        DType::F32,
        Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
    )
    .expect("the upload succeeds")
    .cast(&mut backend, DType::BF16)
    .expect("the cast succeeds");
    let linear = Linear::new(weight);
    let x = Tensor::upload(
        &mut backend,
        &[1.0_f32, 0.0, -1.0, 2.0, 1.0, 0.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>(),
        DType::F32,
        Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
    )
    .expect("the upload succeeds")
    .cast(&mut backend, DType::BF16)
    .expect("the cast succeeds");
    let output = linear
        .forward(&mut backend, &x)
        .expect("the linear succeeds");
    assert_eq!(output.shape().dims(), &[2, 2], "output dims");
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
    assert_eq!(
        actual,
        vec![-2.0, -2.0, 4.0, 13.0],
        "x times the transposed weight"
    );
}

#[test]
fn embedding_gathers_the_rows_of_the_token_ids() {
    let mut backend = CpuBackend::new();
    let table = Tensor::upload(
        &mut backend,
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>(),
        DType::F32,
        Shape::try_from([3, 2].as_slice()).expect("the shape is valid"),
    )
    .expect("the upload succeeds")
    .cast(&mut backend, DType::BF16)
    .expect("the cast succeeds");
    let embedding = Embedding::new(table);
    let token_ids = Tensor::upload(
        &mut backend,
        &[2_u32, 0]
            .iter()
            .flat_map(|token_id| token_id.to_le_bytes())
            .collect::<Vec<u8>>(),
        DType::U32,
        Shape::try_from([2].as_slice()).expect("the shape is valid"),
    )
    .expect("the upload succeeds");
    let output = embedding
        .forward(&mut backend, &token_ids)
        .expect("the embedding succeeds");
    assert_eq!(output.shape().dims(), &[2, 2], "output dims");
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
    assert_eq!(actual, vec![5.0, 6.0, 1.0, 2.0], "rows 2 then 0");
}

#[test]
fn swiglu_mlp_matches_hand_computed_values() {
    let mut backend = CpuBackend::new();
    let [gate, up, down] = [
        [1.0_f32, 0.0, 0.0, -1.0],
        [2.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, -1.0],
    ]
    .map(|values| {
        Linear::new(
            Tensor::upload(
                &mut backend,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<u8>>(),
                DType::F32,
                Shape::try_from([2, 2].as_slice()).expect("the shape is valid"),
            )
            .expect("the upload succeeds")
            .cast(&mut backend, DType::BF16)
            .expect("the cast succeeds"),
        )
    });
    let mlp = SwigluMlp::new(&mut backend, gate, up, down).expect("the mlp is built");
    let x = Tensor::upload(
        &mut backend,
        &[1.0_f32, 2.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>(),
        DType::F32,
        Shape::try_from([1, 2].as_slice()).expect("the shape is valid"),
    )
    .expect("the upload succeeds")
    .cast(&mut backend, DType::BF16)
    .expect("the cast succeeds");
    let output = mlp.forward(&mut backend, &x).expect("the mlp succeeds");
    assert_eq!(output.shape().dims(), &[1, 2], "output dims");
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
    let expected = [1.462_117_f32, 2.177_335];
    assert_eq!(actual.len(), expected.len(), "element count");
    actual.iter().zip(expected).for_each(|(&actual, expected)| {
        assert!(
            (actual - expected).abs() <= 1e-2 * expected.abs(),
            "swiglu mlp: {actual} is not within 1e-2 of {expected}"
        );
    });
}

#[test]
fn kv_cache_update_beyond_capacity_is_rejected() {
    let mut backend = CpuBackend::new();
    let [key, value] = [[1, 3, 2], [1, 3, 2]].map(|dims| {
        Tensor::zeros(
            &mut backend,
            DType::BF16,
            Shape::try_from(dims.as_slice()).expect("the shape is valid"),
        )
        .expect("the zeros succeed")
    });
    let mut cache = KvCache::new(&mut backend, DType::BF16, 1, 2, 2).expect("the cache is built");
    assert_eq!(
        cache.update(&mut backend, &key, &value, 0),
        Err(CoreError::SliceUpdateOutOfBounds),
        "writing three positions to a cache of two is rejected"
    );
}

#[test]
fn kv_cache_update_rejects_keys_and_values_of_different_shapes() {
    let mut backend = CpuBackend::new();
    let [key, value] = [[1, 2, 2], [1, 1, 2]].map(|dims| {
        Tensor::zeros(
            &mut backend,
            DType::BF16,
            Shape::try_from(dims.as_slice()).expect("the shape is valid"),
        )
        .expect("the zeros succeed")
    });
    let mut cache = KvCache::new(&mut backend, DType::BF16, 1, 4, 2).expect("the cache is built");
    assert_eq!(
        cache.update(&mut backend, &key, &value, 0),
        Err(CoreError::KvIncompatible),
        "keys of two positions with values of one position are rejected"
    );
}

#[test]
fn kv_cache_keeps_earlier_positions_across_updates() {
    struct PositionKeyValueCase<T0, T1, T2> {
        position: T0,
        key: T1,
        value: T2,
    }
    let mut backend = CpuBackend::new();
    let mut cache = KvCache::new(&mut backend, DType::F32, 1, 3, 2).expect("the cache is built");
    [
        PositionKeyValueCase {
            position: 0,
            key: [1.0_f32, 2.0],
            value: [5.0_f32, 6.0],
        },
        PositionKeyValueCase {
            position: 1,
            key: [3.0, 4.0],
            value: [7.0, 8.0],
        },
    ]
    .into_iter()
    .for_each(
        |PositionKeyValueCase {
             position,
             key,
             value,
         }| {
            let [key, value] = [key, value].map(|values| {
                Tensor::upload(
                    &mut backend,
                    &values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from([1, 1, 2].as_slice()).expect("the shape is valid"),
                )
                .expect("the upload succeeds")
            });
            cache
                .update(&mut backend, &key, &value, position)
                .expect("the cache update succeeds");
        },
    );
    let [keys, values] = [cache.keys(), cache.values()].map(|tensor| {
        tensor
            .download(&mut backend)
            .expect("the download succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect::<Vec<f32>>()
    });
    assert_eq!(
        keys,
        vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0],
        "keys of positions 0 and 1, position 2 untouched"
    );
    assert_eq!(
        values,
        vec![5.0, 6.0, 7.0, 8.0, 0.0, 0.0],
        "values of positions 0 and 1, position 2 untouched"
    );
}

#[test]
fn attention_matches_hand_computed_values() {
    struct ValuesDimsCase<T0, T1> {
        values: T0,
        dims: T1,
    }
    let mut backend = CpuBackend::new();
    let [query, key, value, mask] = [
        ValuesDimsCase {
            values: vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.0],
            dims: vec![2, 2, 2],
        },
        ValuesDimsCase {
            values: vec![1.0, 0.0, 0.0, 1.0],
            dims: vec![1, 2, 2],
        },
        ValuesDimsCase {
            values: vec![1.0, 2.0, 3.0, 4.0],
            dims: vec![1, 2, 2],
        },
        ValuesDimsCase {
            values: vec![0.0, f32::NEG_INFINITY, 0.0, 0.0],
            dims: vec![2, 2],
        },
    ]
    .map(
        |ValuesDimsCase {
             values,
             dims,
         }| {
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
        },
    );
    let mut cache = KvCache::new(&mut backend, DType::BF16, 1, 4, 2).expect("the cache is built");
    let attention = Attention::new(
        &mut backend,
        NonZeroUsize::new(2).expect("the head dim is non-zero"),
    )
    .expect("the attention is built");
    cache
        .update(&mut backend, &key, &value, 0)
        .expect("the cache update succeeds");
    let output = attention
        .forward(&mut backend, &query, &cache, 2, &mask)
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
    struct ValuesDimsCase<T0, T1> {
        values: T0,
        dims: T1,
    }
    let mut backend = CpuBackend::new();
    let [query, key, value, causal_mask] = [
        ValuesDimsCase {
            values: vec![
                0.5_f32, -1.0, 1.0, 0.25, -0.5, 2.0, 1.0, 1.0, 0.0, -1.0, 1.5, 0.5,
            ],
            dims: vec![2, 3, 2],
        },
        ValuesDimsCase {
            values: vec![1.0, 0.5, -1.0, 1.0, 0.25, -0.75],
            dims: vec![1, 3, 2],
        },
        ValuesDimsCase {
            values: vec![1.0, 2.0, -1.0, 0.5, 3.0, -2.0],
            dims: vec![1, 3, 2],
        },
        ValuesDimsCase {
            values: vec![
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
            dims: vec![3, 3],
        },
    ]
    .map(
        |ValuesDimsCase {
             values,
             dims,
         }| {
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
        },
    );
    let attention = Attention::new(
        &mut backend,
        NonZeroUsize::new(2).expect("the head dim is non-zero"),
    )
    .expect("the attention is built");
    let mut prefill_cache =
        KvCache::new(&mut backend, DType::BF16, 1, 3, 2).expect("the cache is built");
    prefill_cache
        .update(&mut backend, &key, &value, 0)
        .expect("the cache update succeeds");
    let prefill: Vec<f32> = attention
        .forward(&mut backend, &query, &prefill_cache, 3, &causal_mask)
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
            step_cache
                .update(&mut backend, &key, &value, position)
                .expect("the cache update succeeds");
            attention
                .forward(&mut backend, &query, &step_cache, position + 1, &mask)
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
    let [query, mask] = [&[2, 2, 2][..], &[2, 2]].map(|dims| {
        Tensor::zeros(
            &mut backend,
            DType::BF16,
            Shape::try_from(dims).expect("the shape is valid"),
        )
        .expect("the zeros succeed")
    });
    let cache = KvCache::new(&mut backend, DType::BF16, 1, 2, 2).expect("the cache is built");
    let attention = Attention::new(
        &mut backend,
        NonZeroUsize::new(4).expect("the head dim is non-zero"),
    )
    .expect("the attention is built");
    assert_eq!(
        attention
            .forward(&mut backend, &query, &cache, 2, &mask)
            .err(),
        Some(CoreError::AttentionHeadDimMismatch),
        "a head dim of 2 is rejected by an attention built for 4"
    );
}
