use serde::Serialize;
use tracing::error;

use crate::bench::error::BenchError;

const MAX_PERCENT: u32 = 100;
const U32_SPAN: f64 = 4_294_967_296.0;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Summary {
    pub count: u32,
    pub mean: f64,
    pub stdev: f64,
    pub variance: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p99: f64,
}

impl TryFrom<&[f64]> for Summary {
    type Error = BenchError;

    fn try_from(samples: &[f64]) -> Result<Self, Self::Error> {
        let count = u32::try_from(samples.len()).map_err(|_| {
            error!(
                message = "Sample count does not fit in u32.",
                samples = samples.len()
            );
            BenchError::SampleCountOverflow
        })?;
        let mut sorted = samples.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);
        let at = |percent| percentile(&sorted, percent).ok_or(BenchError::NoSample);
        let (Some(&min), Some(&max)) = (sorted.first(), sorted.last()) else {
            error!(message = "Statistics need at least one sample.");
            return Err(BenchError::NoSample);
        };
        let mean = mean(samples).ok_or(BenchError::NoSample)?;
        let variance = variance_sample(samples, mean);
        Ok(Self {
            count,
            mean,
            stdev: variance.sqrt(),
            variance,
            median: at(50)?,
            min,
            max,
            p10: at(10)?,
            p25: at(25)?,
            p50: at(50)?,
            p75: at(75)?,
            p90: at(90)?,
            p99: at(99)?,
        })
    }
}

pub fn mean(samples: &[f64]) -> Option<f64> {
    let count = u32::try_from(samples.len())
        .ok()
        .filter(|&count| count > 0)?;
    Some(samples.iter().sum::<f64>() / f64::from(count))
}

pub fn variance_sample(
    samples: &[f64],
    mean: f64,
) -> f64 {
    let Some(degrees) = u32::try_from(samples.len())
        .ok()
        .and_then(|count| count.checked_sub(1))
        .filter(|&degrees| degrees > 0)
    else {
        return 0.0;
    };
    samples
        .iter()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f64>()
        / f64::from(degrees)
}

pub fn median(samples: &[f64]) -> Option<f64> {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    percentile(&sorted, 50)
}

pub fn percentile(
    sorted: &[f64],
    percent: u32,
) -> Option<f64> {
    let last = u32::try_from(sorted.len()).ok()?.checked_sub(1)?;
    let rank_scaled = percent.min(MAX_PERCENT).checked_mul(last)?;
    let lower_index = usize::try_from(rank_scaled.checked_div(MAX_PERCENT)?).ok()?;
    let fraction_scaled = rank_scaled.checked_rem(MAX_PERCENT)?;
    let lower = *sorted.get(lower_index)?;
    let upper = sorted
        .get(lower_index.checked_add(1)?)
        .copied()
        .unwrap_or(lower);
    Some(lower + (upper - lower) * f64::from(fraction_scaled) / f64::from(MAX_PERCENT))
}

pub fn ns_to_f64(ns: u64) -> f64 {
    let [b0, b1, b2, b3, b4, b5, b6, b7] = ns.to_le_bytes();
    let low = u32::from_le_bytes([b0, b1, b2, b3]);
    let high = u32::from_le_bytes([b4, b5, b6, b7]);
    f64::from(high) * U32_SPAN + f64::from(low)
}

#[cfg(test)]
mod tests {
    use crate::bench::error::BenchError;
    use crate::bench::stats::{
        Summary,
        median,
        ns_to_f64,
        percentile,
    };

    #[test]
    fn summary_of_five_samples_matches_hand_values() {
        assert_eq!(
            Summary::try_from([30.0, 10.0, 50.0, 20.0, 40.0].as_slice()),
            Ok(Summary {
                count: 5,
                mean: 30.0,
                stdev: 250_f64.sqrt(),
                variance: 250.0,
                median: 30.0,
                min: 10.0,
                max: 50.0,
                p10: 14.0,
                p25: 20.0,
                p50: 30.0,
                p75: 40.0,
                p90: 46.0,
                p99: 49.6,
            }),
            "variance 1000 / 4, percentiles interpolated between ranks"
        );
    }

    #[test]
    fn summary_of_one_sample_has_zero_spread() {
        assert_eq!(
            Summary::try_from([7.0].as_slice()),
            Ok(Summary {
                count: 1,
                mean: 7.0,
                stdev: 0.0,
                variance: 0.0,
                median: 7.0,
                min: 7.0,
                max: 7.0,
                p10: 7.0,
                p25: 7.0,
                p50: 7.0,
                p75: 7.0,
                p90: 7.0,
                p99: 7.0,
            }),
            "one sample, as llama.cpp stdev returns 0 below two samples"
        );
    }

    #[test]
    fn summary_of_no_sample_is_rejected() {
        assert_eq!(
            Summary::try_from([].as_slice()),
            Err(BenchError::NoSample),
            "no sample"
        );
    }

    #[test]
    fn median_of_an_even_count_averages_the_middle_pair() {
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5), "between 2 and 3");
        assert_eq!(percentile(&[], 50), None, "no sample");
    }

    #[test]
    fn ns_to_f64_is_exact_across_the_u32_boundary() {
        assert_eq!(
            [0, 1_000, 4_294_967_296, 4_294_967_297].map(ns_to_f64),
            [0.0, 1_000.0, 4_294_967_296.0, 4_294_967_297.0],
            "integers below 2^53 convert exactly"
        );
    }
}
