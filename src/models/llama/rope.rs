use std::f32::consts::TAU;
use std::iter;

use crate::core::{
    Backend,
    DType,
    Shape,
    Tensor,
};
use crate::models::llama::{
    Llama3RopeScaling,
    LlamaConfig,
    LlamaError,
};

const POSITION_COUNT_MAX: usize = 1 << f32::MANTISSA_DIGITS;
const DIM_AXIS: usize = 2;

pub struct Llama3Rope<B: Backend> {
    cos: Tensor<B>,
    sin: Tensor<B>,
    head_dim: usize,
}

pub struct RopeAngles<B: Backend> {
    cos: Tensor<B>,
    sin: Tensor<B>,
    half_dim: usize,
}

impl<B: Backend> Llama3Rope<B> {
    pub fn new(
        backend: &mut B,
        config: &LlamaConfig,
        seq_len_max: usize,
    ) -> Result<Self, LlamaError<B::Error>> {
        let seq_len_max_allowed = config.max_position_embeddings().min(POSITION_COUNT_MAX);
        if seq_len_max > seq_len_max_allowed {
            return Err(LlamaError::SeqLenMaxTooLarge {
                seq_len_max,
                seq_len_max_allowed,
            });
        }
        let head_dim = config.head_dim();
        let inv_frequencies =
            inv_frequencies(head_dim, config.rope_theta(), config.rope_scaling())?;
        let angles: Vec<f32> = iter::successors(Some(0.0_f32), |position| Some(position + 1.0))
            .take(seq_len_max)
            .flat_map(|position| {
                inv_frequencies
                    .iter()
                    .chain(&inv_frequencies)
                    .map(move |inv_frequency| position * inv_frequency)
            })
            .collect();
        let functions: [fn(f32) -> f32; 2] = [f32::cos, f32::sin];
        let [cos, sin] = functions.map(|function| {
            Tensor::upload(
                backend,
                &angles
                    .iter()
                    .flat_map(|&angle| function(angle).to_le_bytes())
                    .collect::<Vec<u8>>(),
                DType::F32,
                Shape::try_from([seq_len_max, head_dim].as_slice())?,
            )?
            .cast(backend, DType::BF16)
        });
        Ok(Self {
            cos: cos.map_err(LlamaError::Backend)?,
            sin: sin.map_err(LlamaError::Backend)?,
            head_dim,
        })
    }

    pub fn at(
        &self,
        backend: &mut B,
        positions: &Tensor<B>,
    ) -> Result<RopeAngles<B>, B::Error> {
        let shape =
            Shape::try_from([positions.shape().element_count(), 1, self.head_dim].as_slice())?;
        let [cos, sin] = [&self.cos, &self.sin].map(|table| {
            Ok::<Tensor<B>, B::Error>(table.gather(backend, positions)?.reshape(shape)?)
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
    ) -> Result<Tensor<B>, B::Error> {
        let rotated = x
            .narrow(DIM_AXIS, self.half_dim, self.half_dim)?
            .neg(backend)?
            .concat(backend, &x.narrow(DIM_AXIS, 0, self.half_dim)?, DIM_AXIS)?
            .mul(backend, &self.sin)?;
        x.mul(backend, &self.cos)?.add(backend, &rotated)
    }
}

fn inv_frequencies<E>(
    head_dim: usize,
    theta: f32,
    scaling: &Llama3RopeScaling,
) -> Result<Vec<f32>, LlamaError<E>> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(LlamaError::HeadDimInvalid {
            head_dim,
        });
    }
    let head_dim_u16 =
        u16::try_from(head_dim).map_err(|_| LlamaError::DimensionTooLargeForF32 {
            name: "head dim",
            dim: head_dim,
            dim_max: u16::MAX.into(),
        })?;
    let context_len_u16 =
        u16::try_from(scaling.original_max_position_embeddings).map_err(|_| {
            LlamaError::DimensionTooLargeForF32 {
                name: "original max position embeddings",
                dim: scaling.original_max_position_embeddings,
                dim_max: u16::MAX.into(),
            }
        })?;
    let head_dim_f32 = f32::from(head_dim_u16);
    let context_len = f32::from(context_len_u16);
    let wavelen_low = context_len / scaling.low_freq_factor;
    let wavelen_high = context_len / scaling.high_freq_factor;
    Ok((0..head_dim_u16)
        .step_by(2)
        .map(|dim_index| {
            let inv_frequency = theta.powf(f32::from(dim_index) / head_dim_f32).recip();
            let wavelen = TAU / inv_frequency;
            let inv_frequency_scaled = if wavelen > wavelen_low {
                inv_frequency / scaling.factor
            } else {
                inv_frequency
            };
            let smooth = (context_len / wavelen - scaling.low_freq_factor)
                / (scaling.high_freq_factor - scaling.low_freq_factor);
            let is_medium = wavelen >= wavelen_high && wavelen <= wavelen_low;
            if is_medium {
                (1.0 - smooth) * inv_frequency_scaled / scaling.factor
                    + smooth * inv_frequency_scaled
            } else {
                inv_frequency_scaled
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::f64::consts::TAU;

    use crate::models::llama::rope::inv_frequencies;
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
        let inv_frequencies =
            inv_frequencies::<Infallible>(128, 500_000.0, &scaling).expect("the head dim is valid");
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
    fn inv_frequencies_reject_an_odd_head_dim() {
        let scaling = Llama3RopeScaling {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        };
        let result = inv_frequencies::<Infallible>(3, 500_000.0, &scaling);
        assert!(
            matches!(
                result,
                Err(LlamaError::HeadDimInvalid {
                    head_dim: 3
                })
            ),
            "got {result:?}"
        );
    }
}
