use std::f32::consts::TAU;
use std::iter;

use tracing::error;

use crate::core::{
    Backend,
    DType,
    Shape,
    Tensor,
    TensorError,
    exact_f32,
};
use crate::models::llama::{
    Llama3RopeScaling,
    LlamaConfig,
    LlamaError,
};

const MAX_POSITION_COUNT: usize = 1 << f32::MANTISSA_DIGITS;
const LAST_DIM: usize = 3;

pub struct LlamaRope<B: Backend> {
    cos: Tensor<B>,
    sin: Tensor<B>,
    head_dim: usize,
}

pub struct RopeAngles<B: Backend> {
    cos: Tensor<B>,
    sin: Tensor<B>,
    half_dim: usize,
}

impl<B: Backend> LlamaRope<B> {
    pub fn new(
        backend: &mut B,
        config: &LlamaConfig,
        max_cache_len: usize,
    ) -> Result<Self, LlamaError<B::Error>> {
        let head_dim = config.head_dim();
        let angles = angles(
            &inv_frequencies(head_dim, config.rope_theta(), config.rope_scaling())?,
            max_cache_len,
            config.max_position_embeddings().min(MAX_POSITION_COUNT),
        )?;
        let functions: [fn(f32) -> f32; 2] = [f32::cos, f32::sin];
        let [cos, sin] = functions.map(|function| {
            Tensor::from_bytes(
                backend,
                &angles
                    .iter()
                    .flat_map(|&angle| function(angle).to_le_bytes())
                    .collect::<Vec<u8>>(),
                DType::F32,
                Shape::try_from([max_cache_len, head_dim].as_slice())?,
            )?
            .to_dtype(backend, DType::BF16)
        });
        Ok(Self {
            cos: cos?,
            sin: sin?,
            head_dim,
        })
    }

    pub fn at(
        &self,
        backend: &mut B,
        position_ids: &Tensor<B>,
    ) -> Result<RopeAngles<B>, TensorError<B::Error>> {
        let shape = Shape::try_from([position_ids.shape().numel(), 1, self.head_dim].as_slice())?;
        let [cos, sin] = [&self.cos, &self.sin].map(|table| {
            Ok::<_, TensorError<B::Error>>(
                table
                    .index_select(backend, 0, position_ids)?
                    .reshape(shape)?,
            )
        });
        Ok(RopeAngles {
            cos: cos?,
            sin: sin?,
            half_dim: self.head_dim / 2,
        })
    }
}

impl<B: Backend> RopeAngles<B> {
    pub fn apply(
        &self,
        backend: &mut B,
        x: &Tensor<B>,
    ) -> Result<Tensor<B>, TensorError<B::Error>> {
        let neg_x2 = x
            .narrow(LAST_DIM, self.half_dim, self.half_dim)?
            .neg(backend)?;
        let x1 = x.narrow(LAST_DIM, 0, self.half_dim)?;
        let rotated = Tensor::cat(backend, [&neg_x2, &x1], LAST_DIM)?.mul(backend, &self.sin)?;
        x.mul(backend, &self.cos)?.add(backend, &rotated)
    }
}

fn angles<E: std::error::Error + 'static>(
    inv_frequencies: &[f32],
    max_cache_len: usize,
    max_positions: usize,
) -> Result<Vec<f32>, LlamaError<E>> {
    if max_cache_len > max_positions {
        error!(
            message = "Maximum sequence length exceeds the allowed length.",
            max_cache_len, max_positions,
        );
        return Err(LlamaError::MaxCacheLenTooLarge);
    }
    Ok(
        iter::successors(Some(0.0_f32), |position| Some(position + 1.0))
            .take(max_cache_len)
            .flat_map(|position| {
                inv_frequencies
                    .iter()
                    .chain(inv_frequencies)
                    .map(move |inv_frequency| position * inv_frequency)
            })
            .collect(),
    )
}

fn inv_frequencies<E: std::error::Error + 'static>(
    head_dim: usize,
    theta: f32,
    scaling: Option<&Llama3RopeScaling>,
) -> Result<Vec<f32>, LlamaError<E>> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        error!(
            message = "Head dim is not a positive even number.",
            head_dim
        );
        return Err(LlamaError::HeadDimInvalid);
    }
    let head_dim_f32 = exact_f32(head_dim)?;
    (0..head_dim)
        .step_by(2)
        .map(|dim_index| {
            let inv_frequency = theta.powf(exact_f32(dim_index)? / head_dim_f32).recip();
            scaling.map_or(Ok(inv_frequency), |scaling| {
                llama3_scaled(inv_frequency, scaling)
            })
        })
        .collect()
}

fn llama3_scaled<E: std::error::Error + 'static>(
    inv_frequency: f32,
    scaling: &Llama3RopeScaling,
) -> Result<f32, LlamaError<E>> {
    let context_len = exact_f32(scaling.original_max_position_embeddings)?;
    let wavelen_low = context_len / scaling.low_freq_factor;
    let wavelen_high = context_len / scaling.high_freq_factor;
    let wavelen = TAU / inv_frequency;
    let inv_frequency_scaled = if wavelen > wavelen_low {
        inv_frequency / scaling.factor
    } else {
        inv_frequency
    };
    let smooth = (context_len / wavelen - scaling.low_freq_factor)
        / (scaling.high_freq_factor - scaling.low_freq_factor);
    let is_medium = wavelen >= wavelen_high && wavelen <= wavelen_low;
    Ok(if is_medium {
        (1.0 - smooth) * inv_frequency_scaled / scaling.factor + smooth * inv_frequency_scaled
    } else {
        inv_frequency_scaled
    })
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::f64::consts::TAU;

    use crate::models::llama::rope::{
        angles,
        inv_frequencies,
    };
    use crate::models::llama::{
        Llama3RopeScaling,
        LlamaError,
    };

    #[test]
    fn inv_frequencies_follow_the_llama3_formula_in_each_band() {
        let scaling = Llama3RopeScaling {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        };
        let inv_frequencies = inv_frequencies::<Infallible>(128, 500_000.0, Some(&scaling))
            .expect("the head dim is valid");
        assert_eq!(inv_frequencies.len(), 64, "one frequency per rotated pair");
        [0_u8, 32, 63].iter().for_each(|&dim_index| {
            let period = 500_000.0_f64.powf(f64::from(dim_index) * 2.0 / 128.0);
            let wavelen = TAU * period;
            let period_scaled = if wavelen > 8192.0 {
                32.0 * period
            } else {
                period
            };
            let smooth = (8192.0 / wavelen - 1.0) / (4.0 - 1.0);
            let period_expected = if 2048.0 < wavelen && wavelen < 8192.0 {
                period_scaled / ((1.0 - smooth) / 32.0 + smooth)
            } else {
                period_scaled
            };
            let expected = period_expected.recip();
            let actual = f64::from(
                *inv_frequencies
                    .get(usize::from(dim_index))
                    .expect("the index is in range"),
            );
            assert!(
                (actual - expected).abs() <= 1e-5 * expected.abs(),
                "j = {dim_index}: {actual} is not within 1e-5 of {expected}"
            );
        });
    }

    #[test]
    fn inv_frequencies_without_scaling_are_powers_of_theta() {
        let inv_frequencies =
            inv_frequencies::<Infallible>(64, 100_000.0, None).expect("the head dim is valid");
        assert_eq!(inv_frequencies.len(), 32, "one frequency per rotated pair");
        [0_u8, 16, 31].iter().for_each(|&dim_index| {
            let expected = 100_000.0_f64
                .powf(f64::from(dim_index) * 2.0 / 64.0)
                .recip();
            let actual = f64::from(
                *inv_frequencies
                    .get(usize::from(dim_index))
                    .expect("the index is in range"),
            );
            assert!(
                (actual - expected).abs() <= 1e-5 * expected.abs(),
                "j = {dim_index}: {actual} is not within 1e-5 of {expected}"
            );
        });
    }

    #[test]
    fn inv_frequencies_reject_an_odd_head_dim() {
        let scaling = Llama3RopeScaling {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        };
        assert_eq!(
            inv_frequencies::<Infallible>(3, 500_000.0, Some(&scaling)),
            Err(LlamaError::HeadDimInvalid),
            "an odd head dim is rejected"
        );
    }

    #[test]
    fn angles_repeat_the_frequencies_for_both_halves_at_each_position() {
        assert_eq!(
            angles::<Infallible>(&[1.0, 0.5], 3, 8),
            Ok(vec![
                0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.5, 2.0, 1.0, 2.0, 1.0
            ]),
            "positions 0 to 2 times frequencies 1 and 0.5, twice"
        );
    }

    #[test]
    fn angles_reject_a_cache_beyond_the_allowed_length() {
        assert_eq!(
            angles::<Infallible>(&[1.0], 9, 8),
            Err(LlamaError::MaxCacheLenTooLarge),
            "9 positions where 8 are allowed"
        );
    }
}
