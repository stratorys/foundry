use std::num::NonZeroU32;

use foundry_infer_core::ByteSize;

use crate::bench::bootstrap::{
    Estimate,
    PROBABILITY_LOWER,
    PROBABILITY_UPPER,
    bootstrap,
};
use crate::bench::report::{
    Available,
    ChecksumStatus,
    ConfigurationStatistics,
    Execution,
    ExecutionOutcomeDto,
    IntervalDto,
    Method,
    PhaseDto,
    Protocol,
    Statistics,
    Units,
};
use crate::bench::stats::{
    Summary,
    mean,
    speedup,
    stddev,
    throughput_gib_s,
};
use crate::bench::workload::BUFFER_COUNTS;

pub(crate) const RECALCULATION_TOLERANCE: f64 = 1e-9;

pub(crate) fn protocol(
    warmup: u32,
    iterations: NonZeroU32,
) -> Protocol {
    Protocol {
        configurations: BUFFER_COUNTS.to_vec(),
        configuration_order: "each round runs every configuration once; round r runs [1, 2, 3] \
                              rotated left by r mod 3"
            .to_owned(),
        warmup_rounds: warmup,
        measured_rounds: iterations.get(),
        timer: "std::time::Instant, monotonic".to_owned(),
        timed_region: "from right before the runtime interpreter call until all submitted GPU \
                       work has drained: runtime validation, allocations, host staging, \
                       transfers, checksum kernels, synchronization, and runtime cleanup (slot \
                       release and a final drain)"
            .to_owned(),
        untimed: [
            "graph and plan construction",
            "payload generation and expected checksums",
            "backend setup",
            "diagnostic setup and extraction when tracing",
            "checksum verification",
            "release of checksum results with clear_checksums()",
            "statistics, console output and file export",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        units: Units {
            durations: "integer nanoseconds (suffix _ns)".to_owned(),
            sizes: "integer bytes (suffix _bytes)".to_owned(),
            throughput: "GiB/s with 1 GiB = 2^30 bytes".to_owned(),
            trace_timestamps: "microseconds since the host-time origin".to_owned(),
        },
    }
}

pub(crate) fn method(
    resamples: NonZeroU32,
    seed: u64,
) -> Method {
    Method {
        samples: "completed measured executions with verified checksums; warmup executions are \
                  recorded but excluded"
            .to_owned(),
        median: "middle sample, or lower + (upper - lower) / 2 in integer nanoseconds for an even \
                 count"
            .to_owned(),
        mean: "arithmetic mean: integer sum of nanoseconds divided by the count in f64".to_owned(),
        standard_deviation: "sample standard deviation, two-pass, divided by n - 1; unavailable \
                             for one sample"
            .to_owned(),
        confidence_level: 0.95,
        interval: "percentile bootstrap; each replicate draws the original number of observations \
                   with replacement"
            .to_owned(),
        quantile: "sort replicates ascending, then interpolate linearly at index (n - 1) x p"
            .to_owned(),
        probabilities: [PROBABILITY_LOWER, PROBABILITY_UPPER],
        resamples: resamples.get(),
        seed,
        generator: "SplitMix64: state += 0x9E3779B97F4A7C15; z = (state ^ (state >> 30)) * \
                    0xBF58476D1CE4E5B9; z = (z ^ (z >> 27)) * 0x94D049BB133111EB; z ^ (z >> 31), \
                    all wrapping in 64 bits"
            .to_owned(),
        bounded_draws: "index below n: threshold = (2^64 - n) mod n; draw until value >= \
                        threshold (at most 128 draws); index = value mod n"
            .to_owned(),
        traversal: "one generator seeded once; for each configuration in ascending buffer order \
                    and each replicate, draw n sample indices and compute the median and mean of \
                    that resample; then, for each non-baseline configuration in ascending order \
                    and each replicate, draw R measured round indices shared by the baseline and \
                    that configuration; configurations with fewer than two samples draw nothing"
            .to_owned(),
        speedup: "median of the one-buffer configuration divided by the median of the \
                  configuration; its interval resamples complete measured rounds in pairs"
            .to_owned(),
        recalculation_tolerance: format!(
            "offline recalculation matches within a relative difference of \
             {RECALCULATION_TOLERANCE:e}"
        ),
    }
}

pub(crate) fn measured(executions: &[Execution]) -> Vec<(u32, Vec<u64>)> {
    BUFFER_COUNTS
        .into_iter()
        .map(|buffers| {
            let samples = executions
                .iter()
                .filter(|execution| {
                    execution.buffers == buffers
                        && execution.phase == PhaseDto::Measured
                        && execution.outcome == ExecutionOutcomeDto::Completed
                        && execution.checksums == ChecksumStatus::Verified
                })
                .filter_map(|execution| execution.elapsed_ns)
                .collect();
            (buffers, samples)
        })
        .collect()
}

fn interval(estimate: Estimate) -> Available<IntervalDto> {
    Available::from_result(estimate.map(|interval| IntervalDto {
        lower: interval.lower,
        upper: interval.upper,
    }))
}

pub(crate) fn statistics(
    executions: &[Execution],
    payload: ByteSize,
    resamples: NonZeroU32,
    seed: u64,
) -> Result<Statistics, String> {
    let configurations = measured(executions);
    let summaries = configurations
        .iter()
        .map(|(buffers, samples)| {
            Summary::new(samples).ok_or_else(|| format!("{buffers} buffers have no samples"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let baseline = summaries
        .first()
        .map(|summary| summary.median)
        .ok_or("the one-buffer baseline is missing")?;
    let resampled = bootstrap(&configurations, resamples, seed);
    let rows = configurations
        .iter()
        .zip(&summaries)
        .zip(&resampled.marginals)
        .map(|(((buffers, samples), summary), (_, marginal))| {
            let speedup_ci = if *buffers == 1 {
                Available::unknown("the baseline is compared with itself")
            } else {
                resampled
                    .speedups
                    .iter()
                    .find(|(other, _)| other == buffers)
                    .map_or_else(
                        || Available::unknown("no paired speedup was resampled"),
                        |(_, estimate)| interval(*estimate),
                    )
            };
            ConfigurationStatistics {
                buffers: *buffers,
                samples: u32::try_from(samples.len()).unwrap_or(u32::MAX),
                median_ns: summary.median,
                min_ns: summary.min,
                max_ns: summary.max,
                mean_ns: Available::from_option(mean(samples), "no samples"),
                standard_deviation_ns: Available::from_option(
                    stddev(samples),
                    "a sample standard deviation needs at least two samples",
                ),
                median_ci_ns: interval(marginal.median),
                mean_ci_ns: interval(marginal.mean),
                throughput_gib_s: Available::from_option(
                    throughput_gib_s(payload, summary.median),
                    "the median elapsed time is zero, so throughput is undefined",
                ),
                speedup: Available::from_option(
                    speedup(baseline, summary.median),
                    "a zero median makes the speedup undefined",
                ),
                speedup_ci,
            }
        })
        .collect();
    Ok(Statistics {
        method: method(resamples, seed),
        configurations: rows,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use std::env;
    use std::error::Error;
    use std::num::NonZeroU32;
    use std::path::PathBuf;
    use std::process::Command;

    use foundry_infer_core::ByteSize;

    use super::{
        measured,
        statistics,
    };
    use crate::bench::random::SplitMix64;
    use crate::bench::report::{
        ChecksumStatus,
        Execution,
        ExecutionOutcomeDto,
        PhaseDto,
    };
    use crate::bench::workload::round_order;

    pub(crate) fn synthetic_executions(
        warmup: u32,
        iterations: u32,
        seed: u64,
    ) -> Vec<Execution> {
        let mut generator = SplitMix64::new(seed);
        let mut executions = Vec::new();
        for round in 0..warmup.saturating_add(iterations) {
            for (position, buffers) in (0_u32..).zip(round_order(round)) {
                let jitter = generator.next_u64() % 5_000_000;
                let base = 200_000_000_u64.saturating_sub(u64::from(buffers) * 1_000_000);
                executions.push(Execution {
                    sequence: u32::try_from(executions.len()).unwrap_or(u32::MAX),
                    phase: if round < warmup {
                        PhaseDto::Warmup
                    } else {
                        PhaseDto::Measured
                    },
                    round,
                    position,
                    buffers,
                    elapsed_ns: Some(base.saturating_add(jitter)),
                    outcome: ExecutionOutcomeDto::Completed,
                    checksums: ChecksumStatus::Verified,
                });
            }
        }
        executions
    }

    fn resamples(count: u32) -> NonZeroU32 { NonZeroU32::new(count).unwrap_or(NonZeroU32::MIN) }

    #[test]
    fn warmup_executions_are_excluded() {
        let executions = synthetic_executions(2, 5, 1);
        let samples = measured(&executions);
        assert!(
            samples.iter().all(|(_, samples)| samples.len() == 5),
            "only measured rounds count"
        );
        let first_measured = executions
            .iter()
            .find(|execution| execution.phase == PhaseDto::Measured && execution.buffers == 1)
            .and_then(|execution| execution.elapsed_ns);
        assert_eq!(
            samples
                .first()
                .and_then(|(_, samples)| samples.first())
                .copied(),
            first_measured,
            "samples keep their chronological order"
        );
    }

    #[test]
    fn statistics_are_finite_and_reproducible() -> Result<(), Box<dyn Error>> {
        let executions = synthetic_executions(2, 30, 5);
        let payload = ByteSize::from_gib(3)?;
        let first = statistics(&executions, payload, resamples(1000), 0)?;
        assert_eq!(
            first,
            statistics(&executions, payload, resamples(1000), 0)?,
            "the same seed reproduces the statistics"
        );
        for row in &first.configurations {
            let values = [
                row.mean_ns.value,
                row.standard_deviation_ns.value,
                row.throughput_gib_s.value,
                row.speedup.value,
                row.median_ci_ns.value.map(|interval| interval.lower),
                row.mean_ci_ns.value.map(|interval| interval.upper),
            ];
            assert!(
                values.iter().flatten().all(|value| value.is_finite()),
                "{} buffers have finite statistics",
                row.buffers
            );
            assert_eq!(
                row.samples, 30,
                "{} buffers count measured samples",
                row.buffers
            );
        }
        let baseline = first.configurations.first().ok_or("no baseline")?;
        assert_eq!(baseline.speedup.value, Some(1.0), "the baseline is 1x");
        assert!(
            baseline.speedup_ci.value.is_none() && baseline.speedup_ci.reason.is_some(),
            "the baseline has no speedup interval"
        );
        Ok(())
    }

    #[test]
    fn singleton_and_zero_samples_stay_defined() -> Result<(), Box<dyn Error>> {
        let mut executions = synthetic_executions(0, 1, 2);
        for execution in &mut executions {
            execution.elapsed_ns = Some(0);
        }
        let result = statistics(&executions, ByteSize::from_gib(1)?, resamples(10), 0)?;
        for row in &result.configurations {
            assert!(
                row.standard_deviation_ns.value.is_none()
                    && row.median_ci_ns.value.is_none()
                    && row.throughput_gib_s.value.is_none()
                    && row.speedup.value.is_none(),
                "{} buffers: undefined values are unavailable",
                row.buffers
            );
            assert!(
                row.throughput_gib_s.reason.is_some() && row.speedup.reason.is_some(),
                "{} buffers: unavailable values are explained",
                row.buffers
            );
        }
        Ok(())
    }

    #[test]
    fn offline_recalculation_matches() -> Result<(), Box<dyn Error>> {
        let script =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/bench_summary.py");
        let required = env::var_os("FOUNDRY_REQUIRE_UV").is_some();
        if Command::new("uv").arg("--version").output().is_err() {
            assert!(!required, "uv is required but unavailable");
            eprintln!("uv: SKIPPED, offline recalculation not verified");
            return Ok(());
        }
        let executions = synthetic_executions(2, 12, 8);
        let payload = ByteSize::from_mib(48 * 16)?;
        let statistics = statistics(&executions, payload, resamples(2000), 17)?;
        let report = serde_json::json!({
            "schema_version": 1,
            "workload": { "value": { "payload_bytes": payload.bytes() } },
            "executions": executions,
            "statistics": { "value": statistics },
        });
        let path = PathBuf::from(env!("OUT_DIR")).join("offline-recalculation.json");
        std::fs::write(&path, serde_json::to_string(&report)?)?;
        let output = Command::new("uv")
            .args(["run", "--quiet", "--script"])
            .arg(&script)
            .arg(&path)
            .output()?;
        assert!(
            output.status.success(),
            "the offline recalculation matches, stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(())
    }
}
