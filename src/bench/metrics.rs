use std::collections::BTreeMap;

use serde::Serialize;

use crate::bench::stats::u64_to_f64;
use crate::bench::{
    BenchConfig,
    BenchError,
    DecodeSample,
    RunSamples,
    Summary,
};
use crate::nn::DeviceProfile;

const NS_PER_S: f64 = 1e9;

/// One set of metrics for Foundry; durations are host wall times, not GPU-only
/// times.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Metrics {
    pub prompt_step_ns: Option<Summary>,
    pub decode_step_ns: Option<Summary>,
    pub decode_by_position: Vec<DecodePositionMetrics>,
    pub inter_token_ns: Option<Summary>,
    pub tpot_ns: Option<Summary>,
    pub step_sum_ns: Summary,
    pub first_token_ns: Option<Summary>,
    pub generation_ns: Option<Summary>,
    pub prompt_tokens_per_s: Option<Summary>,
    pub decode_tokens_per_s: Option<Summary>,
    pub output_tokens_per_s: Option<Summary>,
    pub total_tokens_per_s: Option<Summary>,
    pub processed_tokens_per_s: Option<Summary>,
    pub decode_dispatch_count: Option<Summary>,
    pub decode_allocation_count: Option<Summary>,
    pub decode_allocated_bytes: Option<Summary>,
    pub decode_gpu_busy_ns: Option<Summary>,
    // Step wall time minus GPU busy time: encoding, allocation and waits.
    pub decode_gpu_idle_ns: Option<Summary>,
    pub decode_kernel_gpu_ns: BTreeMap<&'static str, Summary>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DecodePositionMetrics {
    pub position: usize,
    pub context_tokens: usize,
    pub step_ns: Summary,
    pub inter_token_ns: Option<Summary>,
}

struct DerivedMetrics {
    prompt: Option<f64>,
    decode: Option<f64>,
    output: Option<f64>,
    total: Option<f64>,
    processed: Option<f64>,
    tpot: Option<f64>,
}

impl Metrics {
    pub(crate) fn from_runs(
        runs: &[RunSamples],
        config: &BenchConfig,
    ) -> Result<Self, BenchError> {
        let prompt_tokens =
            f64::from(config.batch_size.get()) * f64::from(config.prompt_token_count);
        let derived = runs
            .iter()
            .map(|run| {
                let decode_tokens = f64::from(config.batch_size.get()) * count(run.decode.len())?;
                let output_tokens = run
                    .token_ids
                    .iter()
                    .map(|row| count(row.len()))
                    .sum::<Result<f64, _>>()?;
                let total_ns = run.step_sum_ns();
                Ok(DerivedMetrics {
                    prompt: run
                        .prompt_step_ns
                        .and_then(|ns| throughput(prompt_tokens, ns)),
                    decode: throughput(decode_tokens, sum_ns(run.decode_ns())),
                    output: run
                        .generation_ns
                        .and_then(|ns| throughput(output_tokens, ns)),
                    total: run
                        .generation_ns
                        .and_then(|ns| throughput(prompt_tokens + output_tokens, ns)),
                    processed: throughput(prompt_tokens + decode_tokens, total_ns),
                    tpot: tpot_ns(run)?,
                })
            })
            .collect::<Result<Vec<_>, BenchError>>()?;
        let derived_summary =
            |select: fn(&DerivedMetrics) -> Option<f64>| summary(derived.iter().filter_map(select));
        let step_sums = runs
            .iter()
            .map(|run| u64_to_f64(run.step_sum_ns()))
            .collect::<Vec<_>>();
        let decode_profile_summary = |select: fn(&DecodeSample, &DeviceProfile) -> Option<f64>| {
            summary(
                runs.iter()
                    .flat_map(|run| &run.decode)
                    .filter_map(|sample| select(sample, sample.profile.as_ref()?)),
            )
        };
        Ok(Self {
            prompt_step_ns: summary(
                runs.iter()
                    .filter_map(|run| run.prompt_step_ns.map(u64_to_f64)),
            )?,
            decode_step_ns: summary(runs.iter().flat_map(RunSamples::decode_ns).map(u64_to_f64))?,
            decode_by_position: decode_by_position(runs)?,
            inter_token_ns: summary(
                runs.iter()
                    .flat_map(|run| &run.decode)
                    .filter_map(|sample| sample.inter_token_ns.map(u64_to_f64)),
            )?,
            tpot_ns: derived_summary(|run| run.tpot)?,
            step_sum_ns: Summary::try_from(step_sums.as_slice())?,
            first_token_ns: summary(
                runs.iter()
                    .filter_map(|run| run.first_token_ns.map(u64_to_f64)),
            )?,
            generation_ns: summary(
                runs.iter()
                    .filter_map(|run| run.generation_ns.map(u64_to_f64)),
            )?,
            prompt_tokens_per_s: derived_summary(|run| run.prompt)?,
            decode_tokens_per_s: derived_summary(|run| run.decode)?,
            output_tokens_per_s: derived_summary(|run| run.output)?,
            total_tokens_per_s: derived_summary(|run| run.total)?,
            processed_tokens_per_s: derived_summary(|run| run.processed)?,
            decode_dispatch_count: decode_profile_summary(|_, profile| {
                Some(u64_to_f64(profile.dispatch_count()))
            })?,
            decode_allocation_count: decode_profile_summary(|_, profile| {
                Some(u64_to_f64(profile.allocation_count))
            })?,
            decode_allocated_bytes: decode_profile_summary(|_, profile| {
                Some(u64_to_f64(profile.allocated_bytes))
            })?,
            decode_gpu_busy_ns: decode_profile_summary(|_, profile| profile.gpu_busy_ns)?,
            decode_gpu_idle_ns: decode_profile_summary(|sample, profile| {
                profile
                    .gpu_busy_ns
                    .map(|busy_ns| u64_to_f64(sample.step_ns) - busy_ns)
            })?,
            decode_kernel_gpu_ns: decode_kernel_gpu_ns(runs)?,
        })
    }
}

fn decode_kernel_gpu_ns(
    runs: &[RunSamples]
) -> Result<BTreeMap<&'static str, Summary>, BenchError> {
    let mut samples: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    for profile in runs
        .iter()
        .flat_map(|run| &run.decode)
        .filter_map(|sample| sample.profile.as_ref())
    {
        for (&kernel, timing) in &profile.kernels {
            if let Some(gpu_ns) = timing.gpu_ns {
                samples.entry(kernel).or_default().push(gpu_ns);
            }
        }
    }
    samples
        .into_iter()
        .map(|(kernel, gpu_ns)| Ok((kernel, Summary::try_from(gpu_ns.as_slice())?)))
        .collect()
}

fn tpot_ns(run: &RunSamples) -> Result<Option<f64>, BenchError> {
    let intervals = run.token_ids.first().map_or(0, Vec::len).saturating_sub(1);
    match (run.first_token_ns, run.generation_ns) {
        (Some(first), Some(last)) if intervals > 0 => Ok(Some(
            u64_to_f64(last.saturating_sub(first)) / count(intervals)?,
        )),
        _ => Ok(None),
    }
}

fn decode_by_position(runs: &[RunSamples]) -> Result<Vec<DecodePositionMetrics>, BenchError> {
    let mut groups: BTreeMap<(usize, usize), Vec<&DecodeSample>> = BTreeMap::new();
    for sample in runs.iter().flat_map(|run| &run.decode) {
        groups
            .entry((sample.position, sample.context_tokens))
            .or_default()
            .push(sample);
    }
    groups
        .into_iter()
        .map(|((position, context_tokens), samples)| {
            let steps = samples
                .iter()
                .map(|sample| u64_to_f64(sample.step_ns))
                .collect::<Vec<_>>();
            Ok(DecodePositionMetrics {
                position,
                context_tokens,
                step_ns: Summary::try_from(steps.as_slice())?,
                inter_token_ns: summary(
                    samples
                        .iter()
                        .filter_map(|sample| sample.inter_token_ns.map(u64_to_f64)),
                )?,
            })
        })
        .collect()
}

fn count(value: usize) -> Result<f64, BenchError> {
    u64::try_from(value)
        .map(u64_to_f64)
        .map_err(|_| BenchError::PositionOverflow)
}

fn throughput(
    tokens: f64,
    ns: u64,
) -> Option<f64> {
    (tokens > 0.0 && ns > 0).then(|| NS_PER_S * tokens / u64_to_f64(ns))
}

fn summary(samples: impl Iterator<Item = f64>) -> Result<Option<Summary>, BenchError> {
    let samples = samples.collect::<Vec<_>>();
    (!samples.is_empty())
        .then(|| Summary::try_from(samples.as_slice()))
        .transpose()
}

pub(crate) fn sum_ns(samples: impl Iterator<Item = u64>) -> u64 {
    samples.fold(0, |total, sample| total.saturating_add(sample))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::num::NonZeroU32;

    use crate::bench::{
        BenchConfig,
        BenchError,
        DecodeSample,
        Metrics,
        RunSamples,
        TimingScope,
    };
    use crate::nn::{
        DeviceProfile,
        KernelProfile,
    };

    fn config(
        prompt: u32,
        batch: u32,
    ) -> BenchConfig {
        BenchConfig {
            prompt_token_count: prompt,
            new_token_count: 4,
            timing_scope: TimingScope::Generation,
            depth_token_count: 7,
            warmup_run_count: 0,
            warmup_ms: 0,
            run_count: NonZeroU32::MIN,
            batch_size: NonZeroU32::new(batch).expect("nonzero batch"),
            seed: 0,
            profile: None,
        }
    }

    fn run(
        prompt_step_ns: Option<u64>,
        decode_ns: Vec<u64>,
        tokens: usize,
        batch: usize,
    ) -> RunSamples {
        RunSamples {
            prompt_step_ns,
            first_token_ns: prompt_step_ns.or_else(|| decode_ns.first().copied()),
            generation_ns: Some(
                prompt_step_ns
                    .unwrap_or(0)
                    .saturating_add(super::sum_ns(decode_ns.iter().copied())),
            ),
            decode: decode_ns
                .into_iter()
                .enumerate()
                .map(|(position, step_ns)| DecodeSample {
                    position,
                    context_tokens: position.saturating_add(11),
                    step_ns,
                    inter_token_ns: (prompt_step_ns.is_some() || position > 0).then_some(step_ns),
                    profile: None,
                })
                .collect(),
            token_ids: vec![vec![1; tokens]; batch],
            peak_memory_bytes: Some(2_000_000_000),
            prompt_profile: None,
        }
    }

    #[test]
    fn rates_use_all_steps_and_actual_output_counts() {
        for batch in [1, 2] {
            let run = run(
                Some(1_000_000_000),
                vec![250_000_000, 250_000_000, 500_000_000],
                4,
                batch,
            );
            let metrics = Metrics::from_runs(
                &[run],
                &config(4, u32::try_from(batch).expect("small batch")),
            )
            .expect("valid metrics");
            let factor = f64::from(u32::try_from(batch).expect("small batch"));
            assert_eq!(
                metrics.prompt_tokens_per_s.map(|s| s.mean),
                Some(4.0 * factor),
                "prompt tokens / prompt step"
            );
            assert_eq!(
                metrics.decode_tokens_per_s.map(|s| s.mean),
                Some(3.0 * factor),
                "all three decode steps, including the last"
            );
            assert_eq!(
                metrics.output_tokens_per_s.map(|s| s.mean),
                Some(2.0 * factor),
                "four output tokens / two seconds"
            );
            assert_eq!(
                metrics.total_tokens_per_s.map(|s| s.mean),
                Some(4.0 * factor),
                "four input plus four output tokens"
            );
            assert_eq!(
                metrics.processed_tokens_per_s.map(|s| s.mean),
                Some(3.5 * factor),
                "four prompt plus three decode inputs; depth excluded"
            );
            assert_eq!(
                metrics.step_sum_ns.mean, 2e9,
                "batch does not multiply latency"
            );
            assert_eq!(
                metrics.decode_step_ns.map(|s| (s.count, s.p50)),
                Some((3, 250_000_000.0)),
                "raw decode distribution preserved"
            );
        }
    }

    #[test]
    fn prompt_only_and_decode_only_have_defined_metrics() {
        let prompt = Metrics::from_runs(
            &[RunSamples {
                prompt_step_ns: Some(250_000_000),
                token_ids: vec![vec![]],
                ..RunSamples::default()
            }],
            &config(2, 1),
        )
        .expect("prompt metrics");
        assert_eq!(
            prompt.prompt_tokens_per_s.map(|s| s.mean),
            Some(8.0),
            "two prompt tokens"
        );
        assert_eq!(
            prompt.output_tokens_per_s, None,
            "prompt-only processing produces no output"
        );
        assert_eq!(
            (prompt.decode_step_ns, prompt.decode_tokens_per_s),
            (None, None),
            "no decode"
        );
        let decode = Metrics::from_runs(&[run(None, vec![500_000_000; 2], 2, 1)], &config(0, 1))
            .expect("decode metrics");
        assert_eq!(decode.prompt_tokens_per_s, None, "no prompt");
        assert_eq!(
            decode.decode_tokens_per_s.map(|s| s.mean),
            Some(2.0),
            "two decoded tokens"
        );
    }

    #[test]
    fn summaries_average_per_run_rates_and_preserve_dispersion() {
        let runs = [
            run(None, vec![500_000_000; 2], 2, 1),
            run(None, vec![250_000_000; 2], 2, 1),
        ];
        let metrics = Metrics::from_runs(&runs, &config(0, 1)).expect("metrics");
        let rates = metrics.output_tokens_per_s.expect("output rates");
        assert_eq!(
            (rates.count, rates.mean, rates.p50, rates.variance),
            (2, 3.0, 3.0, 2.0),
            "mean of per-run rates, sample variance"
        );
    }

    #[test]
    fn zero_duration_retains_latency_without_infinite_throughput() {
        let zero = Metrics::from_runs(&[run(Some(0), vec![], 1, 1)], &config(2, 1))
            .expect("zero duration retained");
        assert_eq!(zero.step_sum_ns.mean, 0.0, "zero latency sample retained");
        assert_eq!(
            zero.output_tokens_per_s, None,
            "zero time is unavailable, never infinity"
        );
    }

    #[test]
    fn empty_measurements_are_rejected() {
        assert_eq!(
            Metrics::from_runs(&[], &config(0, 1)),
            Err(BenchError::NoSample),
            "empty measurements rejected"
        );
    }

    #[test]
    fn full_generation_rates_use_wall_time_and_are_unavailable_for_forward_passes() {
        let mut run = run(Some(100_000_000), vec![100_000_000], 2, 1);
        run.first_token_ns = Some(200_000_000);
        run.generation_ns = Some(500_000_000);
        let metrics =
            Metrics::from_runs(&[run.clone()], &config(4, 1)).expect("generation metrics");
        let measured = (
            metrics.step_sum_ns.mean,
            metrics.first_token_ns.map(|s| s.mean),
            metrics.generation_ns.map(|s| s.mean),
            metrics.output_tokens_per_s.map(|s| s.mean),
            metrics.total_tokens_per_s.map(|s| s.mean),
            metrics.processed_tokens_per_s.map(|s| s.mean),
        );
        let expected = (
            200_000_000.0,
            Some(200_000_000.0),
            Some(500_000_000.0),
            Some(4.0),
            Some(12.0),
            Some(25.0),
        );
        assert_eq!(
            measured, expected,
            "wall-time rates differ from step-sum rates"
        );
        run.first_token_ns = None;
        run.generation_ns = None;
        run.decode
            .iter_mut()
            .for_each(|sample| sample.inter_token_ns = None);
        let metrics = Metrics::from_runs(
            &[run],
            &BenchConfig {
                timing_scope: TimingScope::Forward,
                ..config(4, 1)
            },
        )
        .expect("forward metrics");
        assert_eq!(
            (
                metrics.first_token_ns,
                metrics.generation_ns,
                metrics.inter_token_ns,
                metrics.tpot_ns,
                metrics.output_tokens_per_s,
                metrics.total_tokens_per_s
            ),
            (None, None, None, None, None, None),
            "forward timing does not invent host latencies or full generation rates"
        );
    }

    #[test]
    fn inter_token_latency_and_tpot_use_host_availability_not_step_time() {
        let mut sample = run(Some(100_000_000), vec![90_000_000, 110_000_000], 3, 2);
        sample.generation_ns = Some(500_000_000);
        for (step, interval) in sample.decode.iter_mut().zip([150_000_000, 250_000_000]) {
            step.inter_token_ns = Some(interval);
        }
        let metrics = Metrics::from_runs(&[sample], &config(4, 2)).expect("metrics");
        let itl = metrics.inter_token_ns.expect("two host intervals");
        let tpot = metrics.tpot_ns.expect("one request mean");
        assert_eq!(
            (itl.count, itl.min, itl.max, itl.mean),
            (2, 150_000_000.0, 250_000_000.0, 200_000_000.0),
            "tokens available at 100, 250, 500 ms"
        );
        assert_eq!(
            (tpot.count, tpot.mean),
            (1, 200_000_000.0),
            "TPOT is per sequence, not divided by batch size"
        );
        assert_eq!(
            metrics.decode_step_ns.map(|s| s.mean),
            Some(100_000_000.0),
            "ITL also includes orchestration outside the step timer"
        );
    }

    #[test]
    fn position_statistics_keep_repetitions_and_contexts_separate() {
        let mut samples = [
            run(Some(100), vec![10, 900], 3, 1),
            run(Some(100), vec![30, 1100], 3, 1),
        ];
        let metrics = Metrics::from_runs(&samples, &config(4, 1)).expect("metrics");
        let positions = metrics
            .decode_by_position
            .iter()
            .map(|p| (p.position, p.context_tokens, p.step_ns.count, p.step_ns.p50))
            .collect::<Vec<_>>();
        assert_eq!(
            positions,
            vec![(0, 11, 2, 20.0), (1, 12, 2, 1000.0)],
            "compare repetitions at the same position and context"
        );
        assert_eq!(
            metrics.inter_token_ns.map(|s| s.count),
            Some(4),
            "four raw intervals"
        );
        assert_eq!(
            metrics.tpot_ns.map(|s| s.count),
            Some(2),
            "two request means, not four independent requests"
        );
        samples
            .last_mut()
            .expect("second run")
            .decode
            .first_mut()
            .expect("first step")
            .context_tokens = 111;
        let metrics = Metrics::from_runs(&samples, &config(4, 1)).expect("metrics");
        let groups = metrics
            .decode_by_position
            .iter()
            .map(|p| (p.position, p.context_tokens, p.step_ns.count))
            .collect::<Vec<_>>();
        assert_eq!(
            groups,
            vec![(0, 11, 1), (0, 111, 1), (1, 12, 2)],
            "different contexts never merged even at the same position"
        );
    }

    #[test]
    fn decode_profiles_summarize_each_step_and_exclude_the_prompt() {
        let profile =
            |dispatch_count, allocation_count, allocated_bytes, gpu_busy_ns| DeviceProfile {
                kernels: [
                    (
                        "binary_add",
                        KernelProfile {
                            dispatch_count,
                            gpu_ns: None,
                        },
                    ),
                    (
                        "matmul",
                        KernelProfile {
                            dispatch_count: 1,
                            gpu_ns: None,
                        },
                    ),
                ]
                .into(),
                allocation_count,
                allocated_bytes,
                gpu_busy_ns,
            };
        let mut sample = run(Some(100), vec![10, 10], 3, 1);
        sample.prompt_profile = Some(profile(99, 99, 99, Some(99.0)));
        sample
            .decode
            .iter_mut()
            .zip([profile(3, 4, 64, Some(4.0)), profile(5, 6, 96, Some(7.0))])
            .for_each(|(step, profile)| step.profile = Some(profile));
        let profile_metrics = |metrics: Metrics| {
            [
                metrics.decode_dispatch_count,
                metrics.decode_allocation_count,
                metrics.decode_allocated_bytes,
                metrics.decode_gpu_busy_ns,
                metrics.decode_gpu_idle_ns,
            ]
            .map(|summary| summary.map(|s| (s.count, s.min, s.max)))
        };
        assert_eq!(
            profile_metrics(Metrics::from_runs(&[sample.clone()], &config(4, 1)).expect("metrics")),
            [
                Some((2, 4.0, 6.0)),
                Some((2, 4.0, 6.0)),
                Some((2, 64.0, 96.0)),
                Some((2, 4.0, 7.0)),
                Some((2, 3.0, 6.0)),
            ],
            "one sample per decode step, dispatches summed over kernels, idle is the 10 ns step \
             minus busy time"
        );
        sample
            .decode
            .iter_mut()
            .for_each(|step| step.profile = None);
        assert_eq!(
            profile_metrics(Metrics::from_runs(&[sample], &config(4, 1)).expect("metrics")),
            [None; 5],
            "no profile metrics without profiling"
        );
    }

    #[test]
    fn kernel_gpu_time_is_summarized_per_kernel_over_decode_steps_only() {
        let profile = |kernels: &[(&'static str, f64)]| DeviceProfile {
            kernels: kernels
                .iter()
                .map(|&(kernel, gpu_ns)| {
                    (
                        kernel,
                        KernelProfile {
                            dispatch_count: 1,
                            gpu_ns: Some(gpu_ns),
                        },
                    )
                })
                .collect(),
            ..DeviceProfile::default()
        };
        let mut sample = run(Some(100), vec![10, 10], 3, 1);
        sample.prompt_profile = Some(profile(&[("binary_add", 100.0)]));
        sample
            .decode
            .iter_mut()
            .zip([
                profile(&[("binary_add", 2.0)]),
                profile(&[("binary_add", 4.0), ("matmul", 8.0)]),
            ])
            .for_each(|(step, profile)| step.profile = Some(profile));
        let metrics = Metrics::from_runs(&[sample], &config(4, 1)).expect("metrics");
        assert_eq!(
            metrics
                .decode_kernel_gpu_ns
                .iter()
                .map(|(&kernel, s)| (kernel, s.count, s.min, s.max))
                .collect::<Vec<_>>(),
            [("binary_add", 2, 2.0, 4.0), ("matmul", 1, 8.0, 8.0)],
            "one sample per decode step that ran the kernel; the prompt step is excluded"
        );
        let untimed =
            Metrics::from_runs(&[run(Some(100), vec![10], 2, 1)], &config(4, 1)).expect("metrics");
        assert_eq!(
            untimed.decode_kernel_gpu_ns,
            BTreeMap::new(),
            "no kernel time without kernel profiling"
        );
    }

    #[test]
    fn json_has_common_metrics_and_no_engine_names() {
        let config = config(2, 1);
        let samples = vec![run(Some(1), vec![], 1, 1)];
        let metrics = Metrics::from_runs(&samples, &config).expect("metrics");
        let report = crate::bench::Report {
            config,
            depth_ids: vec![vec![1; 7]],
            prompt_ids: vec![vec![1; 2]],
            start_ids: None,
            warmup_run_count: 0,
            runs: samples,
            metrics,
            peak_memory_bytes: Some(2_000_000_000),
            peak_memory_source: Some("test"),
            session_peak_memory_bytes: Some(3_000_000_000),
        };
        let json = serde_json::to_value(report).expect("serializable report");
        assert!(json.get("stats").is_none(), "statistics live in metrics");
        assert_eq!(
            json.pointer("/runs/0/token_ids"),
            Some(&serde_json::json!([[1]])),
            "raw generated IDs retained"
        );
        assert_eq!(
            json.pointer("/runs/0/prompt_step_ns"),
            Some(&serde_json::json!(1)),
            "raw timing retained"
        );
        let object = json
            .get("metrics")
            .and_then(serde_json::Value::as_object)
            .expect("metrics object");
        assert_eq!(object.len(), 19, "one field per canonical metric");
        for name in ["mlx", "vllm", "llama_cpp", "sglang", "candle", "zml"] {
            assert!(!object.contains_key(name), "no engine projection {name}");
        }
        assert_eq!(
            object.get("decode_tokens_per_s"),
            Some(&serde_json::Value::Null),
            "unavailable metrics are explicit"
        );
    }
}
