use std::num::NonZeroU32;

use crate::bench::random::SplitMix64;
use crate::bench::stats::{
    lossy,
    mean,
    median_sorted,
    percentile,
    speedup,
};

pub(crate) const PROBABILITY_LOWER: f64 = 0.025;
pub(crate) const PROBABILITY_UPPER: f64 = 0.975;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Interval {
    pub(crate) lower: f64,
    pub(crate) upper: f64,
}

pub(crate) type Estimate = Result<Interval, &'static str>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Marginal {
    pub(crate) median: Estimate,
    pub(crate) mean: Estimate,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Bootstrap {
    pub(crate) marginals: Vec<(u32, Marginal)>,
    pub(crate) speedups: Vec<(u32, Estimate)>,
}

fn interval(mut replicates: Vec<f64>) -> Estimate {
    replicates.sort_unstable_by(f64::total_cmp);
    match (
        percentile(&replicates, PROBABILITY_LOWER),
        percentile(&replicates, PROBABILITY_UPPER),
    ) {
        (Some(lower), Some(upper)) => Ok(Interval {
            lower,
            upper,
        }),
        (None, _) | (_, None) => Err("the bootstrap produced no finite percentile"),
    }
}

fn marginal(
    samples: &[u64],
    resamples: NonZeroU32,
    generator: &mut SplitMix64,
) -> Marginal {
    if samples.len() < 2 {
        let reason = "confidence intervals need at least two measured samples";
        return Marginal {
            median: Err(reason),
            mean: Err(reason),
        };
    }
    let mut scratch = Vec::with_capacity(samples.len());
    let mut medians = Vec::with_capacity(usize::try_from(resamples.get()).unwrap_or_default());
    let mut means = Vec::with_capacity(medians.capacity());
    for _ in 0..resamples.get() {
        scratch.clear();
        scratch.extend(
            samples
                .iter()
                .filter_map(|_| generator.index_below(samples.len()))
                .filter_map(|index| samples.get(index).copied()),
        );
        means.push(mean(&scratch));
        scratch.sort_unstable();
        medians.push(median_sorted(&scratch).map(|median| lossy(u128::from(median))));
    }
    let collect = |values: Vec<Option<f64>>| {
        values
            .into_iter()
            .collect::<Option<Vec<f64>>>()
            .ok_or("a bootstrap replicate had no estimate")
            .and_then(interval)
    };
    Marginal {
        median: collect(medians),
        mean: collect(means),
    }
}

fn paired(
    baseline: &[u64],
    other: &[u64],
    resamples: NonZeroU32,
    generator: &mut SplitMix64,
) -> Estimate {
    if baseline.len() != other.len() {
        return Err("paired resampling needs complete measured rounds");
    }
    if baseline.len() < 2 {
        return Err("confidence intervals need at least two measured rounds");
    }
    let rounds = baseline.len();
    let mut first = Vec::with_capacity(rounds);
    let mut second = Vec::with_capacity(rounds);
    let replicates = (0..resamples.get())
        .map(|_| {
            first.clear();
            second.clear();
            for _ in 0..rounds {
                let round = generator.index_below(rounds)?;
                first.push(*baseline.get(round)?);
                second.push(*other.get(round)?);
            }
            first.sort_unstable();
            second.sort_unstable();
            speedup(median_sorted(&first)?, median_sorted(&second)?)
        })
        .collect::<Vec<Option<f64>>>();
    replicates
        .into_iter()
        .collect::<Option<Vec<f64>>>()
        .ok_or("a resampled median was zero, so its speedup is undefined")
        .and_then(interval)
}

pub(crate) fn bootstrap(
    configurations: &[(u32, Vec<u64>)],
    resamples: NonZeroU32,
    seed: u64,
) -> Bootstrap {
    let mut generator = SplitMix64::new(seed);
    let marginals = configurations
        .iter()
        .map(|(buffers, samples)| (*buffers, marginal(samples, resamples, &mut generator)))
        .collect();
    let baseline = configurations.first();
    let speedups = configurations
        .iter()
        .skip(1)
        .map(|(buffers, samples)| {
            let estimate = baseline.map_or(Err("the baseline configuration is missing"), |base| {
                paired(&base.1, samples, resamples, &mut generator)
            });
            (*buffers, estimate)
        })
        .collect();
    Bootstrap {
        marginals,
        speedups,
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::{
        Interval,
        bootstrap,
        paired,
    };
    use crate::bench::random::SplitMix64;

    fn resamples(count: u32) -> NonZeroU32 { NonZeroU32::new(count).unwrap_or(NonZeroU32::MIN) }

    fn configurations() -> Vec<(u32, Vec<u64>)> {
        vec![
            (1, vec![100, 104, 98, 101, 99, 103, 97, 102]),
            (2, vec![90, 95, 88, 92, 91, 89, 94, 93]),
            (3, vec![100, 0, 101, 99, 98, 102, 100, 97]),
        ]
    }

    fn finite(interval: Interval) -> bool {
        interval.lower.is_finite() && interval.upper.is_finite() && interval.lower <= interval.upper
    }

    #[test]
    fn the_same_seed_reproduces_every_interval() {
        let first = bootstrap(&configurations(), resamples(500), 42);
        assert_eq!(
            first,
            bootstrap(&configurations(), resamples(500), 42),
            "the same seed gives the same result"
        );
        assert_ne!(
            first,
            bootstrap(&configurations(), resamples(500), 43),
            "another seed resamples differently"
        );
        let buffers: Vec<u32> = first
            .marginals
            .iter()
            .map(|(buffers, _)| *buffers)
            .collect();
        assert_eq!(buffers, [1, 2, 3], "configurations keep their order");
        for (buffers, marginal) in &first.marginals {
            assert!(
                marginal.median.is_ok_and(finite) && marginal.mean.is_ok_and(finite),
                "{buffers} buffers have finite intervals"
            );
        }
    }

    #[test]
    fn intervals_bracket_the_resampled_statistics() {
        let result = bootstrap(&configurations(), resamples(2000), 0);
        let (_, baseline) = result.marginals.first().copied().unwrap_or((
            0,
            super::Marginal {
                median: Err("missing"),
                mean: Err("missing"),
            },
        ));
        let median = baseline.median.unwrap_or(Interval {
            lower: f64::NAN,
            upper: f64::NAN,
        });
        assert!(
            (97.0..=104.0).contains(&median.lower) && (97.0..=104.0).contains(&median.upper),
            "the median interval stays within the samples, got {median:?}"
        );
        assert!(
            median.lower < 100.5 && median.upper > 100.5,
            "it covers the median"
        );
    }

    #[test]
    fn single_samples_have_no_interval() {
        let result = bootstrap(&[(1, vec![5]), (2, vec![4])], resamples(100), 0);
        assert!(
            result
                .marginals
                .iter()
                .all(|(_, marginal)| marginal.median.is_err() && marginal.mean.is_err()),
            "one sample has no interval"
        );
        assert!(
            result.speedups.iter().all(|(_, speedup)| speedup.is_err()),
            "one round has no speedup interval"
        );
    }

    #[test]
    fn speedups_resample_whole_rounds() {
        let baseline = [100, 200, 300, 400];
        let doubled = [50, 100, 150, 200];
        let speedup = paired(&baseline, &doubled, resamples(500), &mut SplitMix64::new(9));
        assert_eq!(
            speedup,
            Ok(Interval {
                lower: 2.0,
                upper: 2.0,
            }),
            "pairs drawn together keep their exact ratio"
        );
        assert!(
            paired(
                &baseline,
                &doubled[..3],
                resamples(10),
                &mut SplitMix64::new(0)
            )
            .is_err(),
            "incomplete rounds cannot be paired"
        );
    }

    #[test]
    fn zero_medians_never_become_infinite() {
        let result = bootstrap(
            &[(1, vec![100, 100, 100]), (2, vec![0, 0, 0])],
            resamples(100),
            0,
        );
        assert!(
            result.speedups.iter().all(|(_, speedup)| speedup.is_err()),
            "a zero median has no speedup interval"
        );
        assert!(
            result
                .marginals
                .iter()
                .all(|(_, marginal)| marginal.median.is_ok_and(finite)),
            "zero durations still have finite marginal intervals"
        );
    }
}
