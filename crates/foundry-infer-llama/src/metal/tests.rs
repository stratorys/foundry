use std::env;
use std::error::Error;

use half::f16;
use objc2_metal::MTLCommandBuffer;

use crate::config::RopeConfig;
use crate::metal::context::{
    Buffer,
    Context,
    read,
    wait,
    write,
};
use crate::metal::error::GpuError;
use crate::metal::kernels::{
    AttentionParams,
    Encoder,
    KvCache,
    QuantView,
    RopeParams,
    View,
};
use crate::rope::llama3_frequencies;
use crate::test_support::reference::{
    self,
    AttentionShape,
};
use crate::test_support::{
    Rng,
    widen,
};

type TestResult = Result<(), Box<dyn Error>>;

const CAPACITY: usize = 640;
const HEAD_DIM: usize = 128;

#[expect(
    clippy::print_stderr,
    reason = "the test reports whether GPU behavior was verified or skipped"
)]
pub(crate) fn gpu() -> Result<Option<Context>, GpuError> {
    match Context::new() {
        Ok(context) => {
            eprintln!("metal: running on {}", context.device_name());
            Ok(Some(context))
        }
        Err(
            error @ (GpuError::DeviceUnavailable
            | GpuError::UnsupportedDevice {
                ..
            }),
        ) if env::var_os("FOUNDRY_REQUIRE_METAL").is_none() => {
            eprintln!("metal: SKIPPED, GPU behavior not verified: {error}");
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn run(
    context: &Context,
    encode: impl FnOnce(&Encoder<'_>) -> Result<(), GpuError>,
) -> Result<(), GpuError> {
    let command_buffer = context.command_buffer()?;
    let encoder = Encoder::new(&command_buffer, &context.kernels)?;
    encode(&encoder)?;
    encoder.end();
    command_buffer.commit();
    wait(&command_buffer)
}

fn halves(
    buffer: &Buffer,
    count: usize,
) -> Result<Vec<f32>, GpuError> {
    Ok(widen(&read::<f16>(buffer, 0, count)?))
}

fn assert_close(
    what: &str,
    actual: &[f32],
    expected: &[f32],
    magnitude: &[f32],
    relative: f32,
    absolute: f32,
) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    let worst = actual
        .iter()
        .zip(expected)
        .zip(magnitude)
        .enumerate()
        .map(|(index, ((a, e), m))| (index, (a - e).abs() - (relative * m.abs() + absolute)))
        .fold(
            (0, f32::MIN),
            |best, item| if item.1 > best.1 { item } else { best },
        );
    let (index, excess) = worst;
    assert!(
        excess <= 0.0,
        "{what}: element {index} is {:?}, expected {:?}",
        actual.get(index),
        expected.get(index)
    );
}

struct Quantized {
    words: Vec<u32>,
    scales: Vec<f16>,
    biases: Vec<f16>,
    rows: usize,
    cols: usize,
}

impl Quantized {
    fn random(
        rng: &mut Rng,
        rows: usize,
        cols: usize,
    ) -> Self {
        let groups = rows * cols / 64;
        Self {
            words: rng.words(rows * cols / 8),
            scales: rng.halves(groups, -0.05, 0.05),
            biases: rng.halves(groups, -0.05, 0.05),
            rows,
            cols,
        }
    }

    fn dense(&self) -> Vec<f32> {
        reference::dequantize(
            &self.words,
            &widen(&self.scales),
            &widen(&self.biases),
            self.rows,
            self.cols,
        )
    }
}

struct Resident {
    words: Buffer,
    scales: Buffer,
    biases: Buffer,
    rows: u32,
    cols: u32,
}

impl Resident {
    fn new(
        context: &Context,
        tensor: &Quantized,
    ) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            words: context.shared_from(&tensor.words)?,
            scales: context.shared_from(&tensor.scales)?,
            biases: context.shared_from(&tensor.biases)?,
            rows: u32::try_from(tensor.rows)?,
            cols: u32::try_from(tensor.cols)?,
        })
    }

    fn view(&self) -> QuantView<'_> {
        QuantView {
            weight: View::new(&self.words),
            scales: View::new(&self.scales),
            biases: View::new(&self.biases),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

#[test]
fn embedding_rows_decode_nibbles_with_negative_scales() -> TestResult {
    let Some(context) = gpu()? else {
        return Ok(());
    };
    let rows = 8;
    let cols = 128;
    let words: Vec<u32> = (0..rows * cols / 8)
        .map(|index| 0x7654_3210_u32.rotate_left(4 * u32::try_from(index % 8).unwrap_or(0)))
        .collect();
    let scales: Vec<f16> = (0..rows * cols / 64)
        .map(|index| f16::from_f32(if index % 3 == 0 { -0.5 } else { 0.25 }))
        .collect();
    let biases: Vec<f16> = (0..rows * cols / 64)
        .map(|index| f16::from_f32(0.125 * f32::from(u8::try_from(index % 5).unwrap_or(0))))
        .collect();
    let table = Quantized {
        words,
        scales,
        biases,
        rows,
        cols,
    };
    let resident = Resident::new(&context, &table)?;
    let ids = [3_u32, 0, 7, 3];
    let id_buffer = context.shared_from(&ids)?;
    let out = context.shared_elements::<f16>(ids.len() * cols)?;
    run(&context, |encoder| {
        encoder.embed(
            View::new(&id_buffer),
            &resident.view(),
            View::new(&out),
            u32::try_from(ids.len()).unwrap_or(0),
        )
    })?;
    let dense = table.dense();
    let expected: Vec<f32> = ids
        .iter()
        .flat_map(|&id| {
            let start = usize::try_from(id).unwrap_or(0) * cols;
            dense
                .iter()
                .skip(start)
                .take(cols)
                .map(|value| f16::from_f32(*value).to_f32())
        })
        .collect();
    let actual = halves(&out, ids.len() * cols)?;
    assert_eq!(
        actual, expected,
        "embedding rows match the f16-rounded dequantization"
    );
    assert_eq!(
        actual.get(..8),
        Some(
            [-0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5]
                .map(|value| value + 0.125)
                .as_slice()
        ),
        "nibble k of word 0 is input k, with a negative scale"
    );
    Ok(())
}

#[test]
fn quantized_matmuls_match_the_reference() -> TestResult {
    let Some(context) = gpu()? else {
        return Ok(());
    };
    let mut rng = Rng::new(7);
    for (rows, inner, outputs, residual) in [
        (1, 128, 64, false),
        (1, 256, 128, true),
        (3, 128, 128, false),
        (70, 192, 128, true),
    ] {
        let tensor = Quantized::random(&mut rng, outputs, inner);
        let resident = Resident::new(&context, &tensor)?;
        let x = rng.halves(rows * inner, -1.0, 1.0);
        let initial = rng.halves(rows * outputs, -1.0, 1.0);
        let x_buffer = context.shared_from(&x)?;
        let out = context.shared_from(&initial)?;
        run(&context, |encoder| {
            encoder.matmul(
                View::new(&x_buffer),
                &resident.view(),
                View::new(&out),
                u32::try_from(rows).unwrap_or(0),
                residual,
            )
        })?;
        let dense = tensor.dense();
        let x = widen(&x);
        let product = reference::matmul(&x, &dense, rows, inner, outputs);
        let magnitude = reference::matmul(
            &x.iter().map(|value| value.abs()).collect::<Vec<_>>(),
            &dense.iter().map(|value| value.abs()).collect::<Vec<_>>(),
            rows,
            inner,
            outputs,
        );
        let base = widen(&initial);
        let expected: Vec<f32> = if residual {
            product.iter().zip(&base).map(|(p, b)| p + b).collect()
        } else {
            product
        };
        let magnitude: Vec<f32> = if residual {
            magnitude
                .iter()
                .zip(&base)
                .map(|(m, b)| m + b.abs())
                .collect()
        } else {
            magnitude
        };
        let actual = halves(&out, rows * outputs)?;
        assert_close(
            &format!("matmul rows={rows} residual={residual}"),
            &actual,
            &expected,
            &magnitude,
            2e-3,
            1e-4,
        );
    }
    Ok(())
}

#[test]
fn rms_norm_and_swiglu_match_the_reference() -> TestResult {
    let Some(context) = gpu()? else {
        return Ok(());
    };
    let mut rng = Rng::new(11);
    let (rows, dim) = (3, 256);
    let x = rng.halves(rows * dim, -2.0, 2.0);
    let weight = rng.halves(dim, 0.5, 1.5);
    let x_buffer = context.shared_from(&x)?;
    let weight_buffer = context.shared_from(&weight)?;
    let out = context.shared_elements::<f16>(rows * dim)?;
    let gate = rng.halves(rows * dim, -6.0, 6.0);
    let up = rng.halves(rows * dim, -2.0, 2.0);
    let gate_buffer = context.shared_from(&gate)?;
    let up_buffer = context.shared_from(&up)?;
    run(&context, |encoder| {
        encoder.rms_norm(
            View::new(&x_buffer),
            View::new(&weight_buffer),
            View::new(&out),
            3,
            256,
            1e-5,
        )?;
        encoder.swiglu(View::new(&gate_buffer), View::new(&up_buffer), 768)
    })?;
    let expected = reference::rms_norm(&widen(&x), &widen(&weight), 1e-5);
    let actual = halves(&out, rows * dim)?;
    assert_close("rms_norm", &actual, &expected, &expected, 2e-3, 1e-4);
    let expected: Vec<f32> = widen(&gate)
        .iter()
        .zip(widen(&up))
        .map(|(g, u)| reference::silu(*g) * u)
        .collect();
    let actual = halves(&gate_buffer, rows * dim)?;
    assert_close("swiglu", &actual, &expected, &expected, 3e-3, 1e-4);
    Ok(())
}

fn pinned_rope() -> RopeConfig {
    RopeConfig {
        theta: 500_000.0,
        factor: 32.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_context: 8192,
    }
}

#[test]
fn rope_cache_writes_and_causal_gqa_attention_match_the_reference() -> TestResult {
    let Some(context) = gpu()? else {
        return Ok(());
    };
    let mut rng = Rng::new(13);
    let (heads, kv_heads, rows, offset) = (6, 2, 5, 600);
    let frequencies = llama3_frequencies(&pinned_rope(), 128);
    let q = rng.halves(rows * heads * HEAD_DIM, -1.0, 1.0);
    let k = rng.halves(rows * kv_heads * HEAD_DIM, -1.0, 1.0);
    let v = rng.halves(rows * kv_heads * HEAD_DIM, -1.0, 1.0);
    let cached_keys = rng.halves(kv_heads * CAPACITY * HEAD_DIM, -1.0, 1.0);
    let cached_values = rng.halves(kv_heads * CAPACITY * HEAD_DIM, -1.0, 1.0);
    let q_buffer = context.shared_from(&q)?;
    let k_buffer = context.shared_from(&k)?;
    let v_buffer = context.shared_from(&v)?;
    let key_cache = context.shared_from(&cached_keys)?;
    let value_cache = context.shared_from(&cached_values)?;
    let frequency_buffer = context.shared_from(&frequencies)?;
    let out = context.shared_elements::<f16>(rows * heads * HEAD_DIM)?;
    let cache = KvCache {
        keys: View::new(&key_cache),
        values: View::new(&value_cache),
    };
    let to_u32 = |value: usize| u32::try_from(value).unwrap_or(0);
    run(&context, |encoder| {
        encoder.rope_store(
            [
                View::new(&q_buffer),
                View::new(&k_buffer),
                View::new(&v_buffer),
            ],
            &cache,
            View::new(&frequency_buffer),
            &RopeParams {
                rows: to_u32(rows),
                offset: to_u32(offset),
                heads: to_u32(heads),
                kv_heads: to_u32(kv_heads),
                capacity: to_u32(CAPACITY),
            },
        )?;
        encoder.attention(
            View::new(&q_buffer),
            &cache,
            View::new(&out),
            &AttentionParams {
                rows: to_u32(rows),
                offset: to_u32(offset),
                heads: to_u32(heads),
                kv_heads: to_u32(kv_heads),
                capacity: to_u32(CAPACITY),
                scale: 1.0 / 128_f32.sqrt(),
            },
        )
    })?;
    let rotate = |values: &[f16], width: usize| -> Vec<f32> {
        widen(values)
            .chunks(HEAD_DIM)
            .enumerate()
            .flat_map(|(index, head)| {
                let mut head = head.to_vec();
                reference::rope(&mut head, offset + index / width, &frequencies);
                head
            })
            .collect()
    };
    let expected_q = rotate(&q, heads);
    let actual_q = halves(&q_buffer, q.len())?;
    assert_close(
        "rotated queries",
        &actual_q,
        &expected_q,
        &expected_q,
        2e-3,
        2e-3,
    );
    let expected_k = rotate(&k, kv_heads);
    let cache_layout = |values: &[f32]| -> Vec<f32> {
        (0..offset + rows)
            .flat_map(|position| {
                (0..kv_heads).flat_map(move |kv| {
                    let start = (kv * CAPACITY + position) * HEAD_DIM;
                    values
                        .get(start..start + HEAD_DIM)
                        .unwrap_or_default()
                        .to_vec()
                })
            })
            .collect()
    };
    let actual_keys = halves(&key_cache, cached_keys.len())?;
    let actual_values = halves(&value_cache, cached_values.len())?;
    let untouched = |actual: &[f32], original: &[f16]| {
        (0..kv_heads).all(|kv| {
            let range = kv * CAPACITY * HEAD_DIM..(kv * CAPACITY + offset) * HEAD_DIM;
            actual.get(range.clone())
                == Some(widen(original.get(range).unwrap_or_default()).as_slice())
        })
    };
    assert!(
        untouched(&actual_keys, &cached_keys),
        "earlier keys are preserved"
    );
    assert!(
        untouched(&actual_values, &cached_values),
        "earlier values are preserved"
    );
    let fresh = |actual: &[f32]| -> Vec<f32> {
        (0..rows)
            .flat_map(|row| {
                (0..kv_heads).flat_map(move |kv| {
                    let start = (kv * CAPACITY + offset + row) * HEAD_DIM;
                    actual
                        .get(start..start + HEAD_DIM)
                        .unwrap_or_default()
                        .to_vec()
                })
            })
            .collect()
    };
    assert_close(
        "new keys",
        &fresh(&actual_keys),
        &expected_k,
        &expected_k,
        2e-3,
        2e-3,
    );
    assert_eq!(
        fresh(&actual_values),
        widen(&v),
        "values are copied unrotated"
    );
    let keys = cache_layout(&actual_keys);
    let values = cache_layout(&actual_values);
    let shape = AttentionShape {
        heads,
        kv_heads,
        head_dim: HEAD_DIM,
    };
    let expected = reference::attention(&actual_q, &keys, &values, offset, &shape);
    let actual = halves(&out, rows * heads * HEAD_DIM)?;
    assert_close("attention", &actual, &expected, &expected, 3e-3, 1e-3);
    Ok(())
}

#[test]
fn attention_masks_future_positions_from_offset_zero() -> TestResult {
    let Some(context) = gpu()? else {
        return Ok(());
    };
    let mut rng = Rng::new(19);
    let (heads, kv_heads, rows) = (6_usize, 2_usize, 4_usize);
    let q = rng.halves(rows * heads * HEAD_DIM, -1.0, 1.0);
    let keys = rng.halves(kv_heads * CAPACITY * HEAD_DIM, -1.0, 1.0);
    let values = rng.halves(kv_heads * CAPACITY * HEAD_DIM, -1.0, 1.0);
    let q_buffer = context.shared_from(&q)?;
    let key_cache = context.shared_from(&keys)?;
    let value_cache = context.shared_from(&values)?;
    let out = context.shared_elements::<f16>(rows * heads * HEAD_DIM)?;
    run(&context, |encoder| {
        encoder.attention(
            View::new(&q_buffer),
            &KvCache {
                keys: View::new(&key_cache),
                values: View::new(&value_cache),
            },
            View::new(&out),
            &AttentionParams {
                rows: 4,
                offset: 0,
                heads: 6,
                kv_heads: 2,
                capacity: 640,
                scale: 1.0 / 128_f32.sqrt(),
            },
        )
    })?;
    let actual = halves(&out, rows * heads * HEAD_DIM)?;
    for head in 0..heads {
        let kv = head / 3;
        let start = head * HEAD_DIM;
        let expected = widen(
            values
                .get(kv * CAPACITY * HEAD_DIM..(kv * CAPACITY + 1) * HEAD_DIM)
                .unwrap_or_default(),
        );
        assert_eq!(
            actual.get(start..start + HEAD_DIM),
            Some(expected.as_slice()),
            "query head {head} at position 0 sees only value 0 of KV head {kv}"
        );
    }
    Ok(())
}

#[test]
fn argmax_prefers_the_lowest_index_on_ties() -> TestResult {
    let Some(context) = gpu()? else {
        return Ok(());
    };
    let mut rng = Rng::new(17);
    let mut logits = rng.halves(5000, -4.0, 4.0);
    for index in [4321, 77, 2048] {
        if let Some(slot) = logits.get_mut(index) {
            *slot = f16::from_f32(9.5);
        }
    }
    let logit_buffer = context.shared_from(&logits)?;
    let outputs = context.shared_from(&[0_u32; 4])?;
    let next = context.shared_from(&[0_u32])?;
    run(&context, |encoder| {
        encoder.argmax(
            View::new(&logit_buffer),
            View::new(&outputs),
            View::new(&next),
            5000,
            2,
        );
        Ok(())
    })?;
    assert_eq!(
        read::<u32>(&outputs, 0, 4)?,
        [0, 0, 77, 0],
        "slot 2 holds the token"
    );
    assert_eq!(
        read::<u32>(&next, 0, 1)?,
        [77],
        "the next input is the token"
    );
    write(&next, 0, &[0_u32])?;
    Ok(())
}
