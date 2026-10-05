use foundry_infer_core::ByteSize;

const GIB: f64 = 1_073_741_824.0;
const NANOS_PER_SECOND: f64 = 1e9;

#[expect(
    clippy::as_conversions,
    reason = "no lossless conversion from u128 to f64 exists; nanosecond sums of benchmark \
              samples stay far below 2^53"
)]
pub(crate) const fn lossy(value: u128) -> f64 { value as f64 }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Summary {
    pub(crate) median: u64,
    pub(crate) min: u64,
    pub(crate) max: u64,
}

impl Summary {
    pub(crate) fn new(samples: &[u64]) -> Option<Self> {
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        Some(Self {
            median: median_sorted(&sorted)?,
            min: *sorted.first()?,
            max: *sorted.last()?,
        })
    }
}

pub(crate) fn median_sorted(sorted: &[u64]) -> Option<u64> {
    let middle = sorted.len() / 2;
    let upper = *sorted.get(middle)?;
    if sorted.len() % 2 == 0 {
        let lower = *sorted.get(middle.checked_sub(1)?)?;
        lower.checked_add(upper.checked_sub(lower)? / 2)
    } else {
        Some(upper)
    }
}

pub(crate) fn mean(samples: &[u64]) -> Option<f64> {
    let total = samples.iter().try_fold(0_u128, |total, &sample| {
        total.checked_add(u128::from(sample))
    })?;
    let count = u128::try_from(samples.len()).ok()?;
    (count > 0)
        .then(|| lossy(total) / lossy(count))
        .filter(|mean| mean.is_finite())
}

pub(crate) fn stddev(samples: &[u64]) -> Option<f64> {
    let center = mean(samples)?;
    let degrees = lossy(u128::try_from(samples.len().checked_sub(1)?).ok()?);
    let squares = samples.iter().fold(0.0, |total, &sample| {
        let deviation = lossy(u128::from(sample)) - center;
        total + deviation * deviation
    });
    (degrees > 0.0)
        .then(|| (squares / degrees).sqrt())
        .filter(|deviation| deviation.is_finite())
}

pub(crate) fn percentile(
    sorted: &[f64],
    probability: f64,
) -> Option<f64> {
    let last = sorted.len().checked_sub(1)?;
    let position = lossy(u128::try_from(last).ok()?) * probability;
    let lower = position.floor();
    let fraction = position - lower;
    let index = floor_index(lower)?;
    let below = *sorted.get(index)?;
    let above = sorted
        .get(index.saturating_add(1))
        .copied()
        .unwrap_or(below);
    Some(below + fraction * (above - below)).filter(|value| value.is_finite())
}

#[expect(
    clippy::as_conversions,
    reason = "no conversion from f64 to usize exists; the value is checked to be a finite, \
              non-negative whole number first"
)]
fn floor_index(position: f64) -> Option<usize> {
    (position.is_finite() && position >= 0.0 && position < lossy(u128::from(u32::MAX)))
        .then_some(position as usize)
}

pub(crate) fn throughput_gib_s(
    bytes: ByteSize,
    elapsed_ns: u64,
) -> Option<f64> {
    (elapsed_ns > 0)
        .then(|| bytes.to_f64_lossy() / GIB / (lossy(u128::from(elapsed_ns)) / NANOS_PER_SECOND))
        .filter(|throughput| throughput.is_finite())
}

pub(crate) fn speedup(
    baseline_ns: u64,
    elapsed_ns: u64,
) -> Option<f64> {
    (baseline_ns > 0 && elapsed_ns > 0)
        .then(|| lossy(u128::from(baseline_ns)) / lossy(u128::from(elapsed_ns)))
        .filter(|speedup| speedup.is_finite())
}

#[cfg(test)]
mod tests {
    use foundry_infer_core::ByteSize;

    use super::{
        Summary,
        mean,
        percentile,
        speedup,
        stddev,
        throughput_gib_s,
    };

    const MS: u64 = 1_000_000;

    fn close(
        actual: Option<f64>,
        expected: f64,
    ) -> bool {
        actual.is_some_and(|actual| (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0))
    }

    #[test]
    fn an_odd_sample_count_uses_the_middle_value() {
        assert_eq!(
            Summary::new(&[30 * MS, 10 * MS, 20 * MS]),
            Some(Summary {
                median: 20 * MS,
                min: 10 * MS,
                max: 30 * MS,
            }),
            "three unsorted samples"
        );
    }

    #[test]
    fn an_even_sample_count_averages_the_two_middle_values() {
        assert_eq!(
            Summary::new(&[40 * MS, 10 * MS, 30 * MS, 15 * MS]),
            Some(Summary {
                median: 22_500_000,
                min: 10 * MS,
                max: 40 * MS,
            }),
            "four unsorted samples"
        );
        assert_eq!(
            Summary::new(&[1, 4]).map(|summary| summary.median),
            Some(2),
            "two samples average down to whole nanoseconds"
        );
    }

    #[test]
    fn a_single_sample_is_its_own_summary() {
        let sample = 7 * MS;
        assert_eq!(
            Summary::new(&[sample]),
            Some(Summary {
                median: sample,
                min: sample,
                max: sample,
            }),
            "one sample"
        );
        assert!(close(mean(&[sample]), 7e6), "the mean of one sample");
        assert_eq!(stddev(&[sample]), None, "one sample has no deviation");
    }

    #[test]
    fn no_samples_have_no_summary() {
        assert_eq!(Summary::new(&[]), None, "an empty sample set");
        assert_eq!(mean(&[]), None, "no mean");
        assert_eq!(stddev(&[]), None, "no deviation");
    }

    #[test]
    fn mean_and_sample_deviation_use_n_minus_one() {
        let samples = [2, 4, 4, 4, 5, 5, 7, 9];
        assert!(close(mean(&samples), 5.0), "the arithmetic mean");
        assert!(
            close(stddev(&samples), (32.0_f64 / 7.0).sqrt()),
            "the sample deviation divides by n - 1"
        );
        assert!(close(stddev(&[5, 5, 5]), 0.0), "equal samples do not vary");
        assert!(
            close(mean(&[u64::MAX, u64::MAX]), 1.844_674_407_370_955_2e19),
            "large samples do not overflow"
        );
    }

    #[test]
    fn percentiles_interpolate_linearly() {
        let sorted = [10.0, 20.0, 30.0, 40.0, 50.0];
        assert!(close(percentile(&sorted, 0.0), 10.0), "the minimum");
        assert!(close(percentile(&sorted, 1.0), 50.0), "the maximum");
        assert!(close(percentile(&sorted, 0.5), 30.0), "the middle");
        assert!(
            close(percentile(&sorted, 0.025), 11.0),
            "position (n - 1) x 0.025 = 0.1"
        );
        assert!(
            close(percentile(&sorted, 0.975), 49.0),
            "position (n - 1) x 0.975 = 3.9"
        );
        assert!(close(percentile(&[3.0], 0.975), 3.0), "one value");
        assert_eq!(percentile(&[], 0.5), None, "no values");
    }

    #[test]
    fn throughput_uses_binary_gibibytes() -> Result<(), Box<dyn std::error::Error>> {
        assert!(
            close(throughput_gib_s(ByteSize::from_gib(3)?, 1500 * MS), 2.0),
            "3 GiB in 1.5 s is 2 GiB/s"
        );
        assert!(
            close(throughput_gib_s(ByteSize::from_mib(512)?, 1000 * MS), 0.5),
            "512 MiB in 1 s is 0.5 GiB/s"
        );
        Ok(())
    }

    #[test]
    fn speedup_compares_against_the_baseline() {
        assert!(
            close(speedup(300 * MS, 200 * MS), 1.5),
            "300 ms against 200 ms"
        );
        assert!(
            close(speedup(300 * MS, 300 * MS), 1.0),
            "the baseline itself"
        );
    }

    #[test]
    fn zero_durations_have_no_throughput_or_speedup() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            throughput_gib_s(ByteSize::from_gib(1)?, 0),
            None,
            "a zero median has no throughput"
        );
        assert_eq!(speedup(5 * MS, 0), None, "a zero median has no speedup");
        assert_eq!(speedup(0, 5 * MS), None, "a zero baseline has no speedup");
        Ok(())
    }
}
