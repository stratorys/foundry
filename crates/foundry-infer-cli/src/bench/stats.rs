use std::time::Duration;

use foundry_infer_core::ByteSize;

const GIB: f64 = 1_073_741_824.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Summary {
    pub(crate) median: Duration,
    pub(crate) min: Duration,
    pub(crate) max: Duration,
}

impl Summary {
    pub(crate) fn new(samples: &[Duration]) -> Option<Self> {
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let min = *sorted.first()?;
        let max = *sorted.last()?;
        let middle = sorted.len() / 2;
        let upper = *sorted.get(middle)?;
        let median = if sorted.len() % 2 == 0 {
            let lower = *sorted.get(middle.checked_sub(1)?)?;
            lower.checked_add(upper.checked_sub(lower)?.checked_div(2)?)?
        } else {
            upper
        };
        Some(Self {
            median,
            min,
            max,
        })
    }
}

pub(crate) fn throughput_gib_s(
    bytes: ByteSize,
    elapsed: Duration,
) -> Option<f64> {
    (!elapsed.is_zero()).then(|| bytes.to_f64_lossy() / GIB / elapsed.as_secs_f64())
}

pub(crate) fn speedup(
    baseline: Duration,
    elapsed: Duration,
) -> Option<f64> {
    (!baseline.is_zero() && !elapsed.is_zero())
        .then(|| baseline.as_secs_f64() / elapsed.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use foundry_infer_core::ByteSize;

    use super::{
        Summary,
        speedup,
        throughput_gib_s,
    };

    fn millis(values: &[u64]) -> Vec<Duration> {
        values.iter().copied().map(Duration::from_millis).collect()
    }

    #[test]
    fn an_odd_sample_count_uses_the_middle_value() {
        assert_eq!(
            Summary::new(&millis(&[30, 10, 20])),
            Some(Summary {
                median: Duration::from_millis(20),
                min: Duration::from_millis(10),
                max: Duration::from_millis(30),
            }),
            "three unsorted samples"
        );
    }

    #[test]
    fn an_even_sample_count_averages_the_two_middle_values() {
        assert_eq!(
            Summary::new(&millis(&[40, 10, 30, 15])),
            Some(Summary {
                median: Duration::from_micros(22_500),
                min: Duration::from_millis(10),
                max: Duration::from_millis(40),
            }),
            "four unsorted samples"
        );
        assert_eq!(
            Summary::new(&[Duration::from_nanos(1), Duration::from_nanos(4)])
                .map(|summary| summary.median),
            Some(Duration::from_nanos(2)),
            "two samples average to whole nanoseconds"
        );
    }

    #[test]
    fn a_single_sample_is_its_own_summary() {
        let sample = Duration::from_millis(7);
        assert_eq!(
            Summary::new(&[sample]),
            Some(Summary {
                median: sample,
                min: sample,
                max: sample,
            }),
            "one sample"
        );
    }

    #[test]
    fn no_samples_have_no_summary() {
        assert_eq!(Summary::new(&[]), None, "an empty sample set");
    }

    #[test]
    fn throughput_uses_binary_gibibytes() -> Result<(), Box<dyn std::error::Error>> {
        let throughput = throughput_gib_s(ByteSize::from_gib(3)?, Duration::from_millis(1500))
            .ok_or("3 GiB in 1.5 s has a throughput")?;
        assert!(
            (throughput - 2.0).abs() < 1e-12,
            "3 GiB in 1.5 s is 2 GiB/s, got {throughput}"
        );
        let throughput = throughput_gib_s(ByteSize::from_mib(512)?, Duration::from_secs(1))
            .ok_or("512 MiB in 1 s has a throughput")?;
        assert!(
            (throughput - 0.5).abs() < 1e-12,
            "512 MiB in 1 s is 0.5 GiB/s, got {throughput}"
        );
        Ok(())
    }

    #[test]
    fn speedup_compares_against_the_baseline() -> Result<(), Box<dyn std::error::Error>> {
        let faster = speedup(Duration::from_millis(300), Duration::from_millis(200))
            .ok_or("300 ms against 200 ms has a speedup")?;
        assert!(
            (faster - 1.5).abs() < 1e-12,
            "300 ms against 200 ms is 1.5x, got {faster}"
        );
        let same = speedup(Duration::from_millis(300), Duration::from_millis(300))
            .ok_or("the baseline against itself has a speedup")?;
        assert!(
            (same - 1.0).abs() < 1e-12,
            "the baseline against itself is 1x, got {same}"
        );
        Ok(())
    }

    #[test]
    fn zero_durations_have_no_throughput_or_speedup() -> Result<(), Box<dyn std::error::Error>> {
        let sample = Duration::from_millis(5);
        assert_eq!(
            throughput_gib_s(ByteSize::from_gib(1)?, Duration::ZERO),
            None,
            "a zero median has no throughput"
        );
        assert_eq!(
            speedup(sample, Duration::ZERO),
            None,
            "a zero median has no speedup"
        );
        assert_eq!(
            speedup(Duration::ZERO, sample),
            None,
            "a zero baseline has no speedup"
        );
        Ok(())
    }
}
