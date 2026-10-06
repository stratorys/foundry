use half::f16;
use serde_json::Value;

pub(crate) fn set(
    value: &mut Value,
    pointer: &str,
    new: Value,
) {
    let (parent, key) = pointer.rsplit_once('/').unwrap_or(("", pointer));
    let object = value.pointer_mut(parent).and_then(Value::as_object_mut);
    assert!(object.is_some(), "{pointer} has no parent object");
    if let Some(object) = object {
        object.insert(key.to_owned(), new);
    }
}

pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self { Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1) }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub(crate) fn next_u32(&mut self) -> u32 { u32::try_from(self.next_u64() >> 32).unwrap_or(0) }

    pub(crate) fn uniform(
        &mut self,
        low: f32,
        high: f32,
    ) -> f32 {
        let unit = f32::from(u16::try_from(self.next_u64() >> 48).unwrap_or(0)) / 65_536.0;
        low + (high - low) * unit
    }

    pub(crate) fn halves(
        &mut self,
        count: usize,
        low: f32,
        high: f32,
    ) -> Vec<f16> {
        (0..count)
            .map(|_| f16::from_f32(self.uniform(low, high)))
            .collect()
    }

    pub(crate) fn words(
        &mut self,
        count: usize,
    ) -> Vec<u32> {
        (0..count).map(|_| self.next_u32()).collect()
    }
}

pub(crate) fn widen(values: &[f16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

#[cfg(feature = "probe")]
pub(crate) fn relative_l2(
    actual: &[f32],
    expected: &[f32],
) -> f32 {
    let (error, norm) =
        actual
            .iter()
            .zip(expected)
            .fold((0.0_f64, 0.0_f64), |(error, norm), (&a, &e)| {
                let difference = f64::from(a) - f64::from(e);
                (
                    error + difference * difference,
                    norm + f64::from(e) * f64::from(e),
                )
            });
    let ratio = (error / norm.max(f64::MIN_POSITIVE)).sqrt();
    ratio.to_string().parse().unwrap_or(f32::INFINITY)
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "CPU reference indices are bounded by small test shapes"
)]
pub(crate) mod reference {
    use crate::config::GROUP_SIZE;

    pub(crate) fn dequantize(
        words: &[u32],
        scales: &[f32],
        biases: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let group = usize::try_from(GROUP_SIZE).unwrap_or(64);
        let per_row = cols / 8;
        let groups = cols / group;
        (0..rows)
            .flat_map(|row| {
                (0..cols).map(move |col| {
                    let word = words.get(row * per_row + col / 8).copied().unwrap_or(0);
                    let nibble = (word >> (4 * (col % 8))) & 0xF;
                    let scale = scales
                        .get(row * groups + col / group)
                        .copied()
                        .unwrap_or(0.0);
                    let bias = biases
                        .get(row * groups + col / group)
                        .copied()
                        .unwrap_or(0.0);
                    let q = f32::from(u8::try_from(nibble).unwrap_or(0));
                    scale * q + bias
                })
            })
            .collect()
    }

    pub(crate) fn matmul(
        x: &[f32],
        w: &[f32],
        rows: usize,
        inner: usize,
        outputs: usize,
    ) -> Vec<f32> {
        (0..rows)
            .flat_map(|row| {
                (0..outputs).map(move |out| {
                    (0..inner)
                        .map(|k| {
                            f64::from(x.get(row * inner + k).copied().unwrap_or(0.0))
                                * f64::from(w.get(out * inner + k).copied().unwrap_or(0.0))
                        })
                        .sum::<f64>()
                })
            })
            .map(|value| value.to_string().parse().unwrap_or(f32::NAN))
            .collect()
    }

    pub(crate) fn rms_norm(
        x: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        x.chunks(weight.len())
            .flat_map(|row| {
                let mean = row.iter().map(|value| value * value).sum::<f32>()
                    / f32::from(u16::try_from(row.len()).unwrap_or(1));
                let inverse = 1.0 / (mean + eps).sqrt();
                row.iter()
                    .zip(weight)
                    .map(move |(value, w)| value * inverse * w)
            })
            .collect()
    }

    pub(crate) fn rope(
        x: &mut [f32],
        position: usize,
        frequencies: &[f32],
    ) {
        let half = frequencies.len();
        let position = f32::from(u16::try_from(position).unwrap_or(0));
        for (index, frequency) in frequencies.iter().enumerate() {
            let theta = position / frequency;
            let (sin, cos) = theta.sin_cos();
            let x1 = x.get(index).copied().unwrap_or(0.0);
            let x2 = x.get(index + half).copied().unwrap_or(0.0);
            if let Some(slot) = x.get_mut(index) {
                *slot = x1 * cos - x2 * sin;
            }
            if let Some(slot) = x.get_mut(index + half) {
                *slot = x1 * sin + x2 * cos;
            }
        }
    }

    pub(crate) struct AttentionShape {
        pub(crate) heads: usize,
        pub(crate) kv_heads: usize,
        pub(crate) head_dim: usize,
    }

    pub(crate) fn attention(
        q: &[f32],
        keys: &[f32],
        values: &[f32],
        offset: usize,
        shape: &AttentionShape,
    ) -> Vec<f32> {
        let AttentionShape {
            heads,
            kv_heads,
            head_dim,
        } = *shape;
        let rows = q.len() / (heads * head_dim);
        let scale = 1.0 / f32::from(u16::try_from(head_dim).unwrap_or(1)).sqrt();
        let element = |buffer: &[f32], index: usize| buffer.get(index).copied().unwrap_or(0.0);
        (0..rows)
            .flat_map(|row| {
                (0..heads).flat_map(move |head| {
                    let kv = head / (heads / kv_heads);
                    let query = (row * heads + head) * head_dim;
                    let last = offset + row;
                    let scores: Vec<f32> = (0..=last)
                        .map(|position| {
                            let key = (position * kv_heads + kv) * head_dim;
                            (0..head_dim)
                                .map(|d| element(q, query + d) * element(keys, key + d))
                                .sum::<f32>()
                                * scale
                        })
                        .collect();
                    let maximum = scores.iter().copied().fold(f32::MIN, f32::max);
                    let weights: Vec<f32> = scores.iter().map(|s| (s - maximum).exp()).collect();
                    let total: f32 = weights.iter().sum();
                    (0..head_dim).map(move |d| {
                        weights
                            .iter()
                            .enumerate()
                            .map(|(position, w)| {
                                w * element(values, (position * kv_heads + kv) * head_dim + d)
                            })
                            .sum::<f32>()
                            / total
                    })
                })
            })
            .collect()
    }

    pub(crate) fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
}
