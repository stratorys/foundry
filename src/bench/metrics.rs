use serde::Serialize;

use crate::bench::stats::u64_to_f64;
use crate::bench::{
    BenchConfig,
    BenchError,
    RunSamples,
    Summary,
};

const NS_PER_S: f64 = 1e9;

/// One set of metrics for Foundry; durations are host wall times, not GPU-only
/// times.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Metrics {
    pub prompt_step_ns: Option<Summary>,
    pub decode_step_ns: Option<Summary>,
    pub step_sum_ns: Summary,
    pub first_token_ns: Option<Summary>,
    pub generation_ns: Option<Summary>,
    pub prompt_tokens_per_s: Option<Summary>,
    pub decode_tokens_per_s: Option<Summary>,
    pub output_tokens_per_s: Option<Summary>,
    pub total_tokens_per_s: Option<Summary>,
    pub processed_tokens_per_s: Option<Summary>,
}

struct Throughputs {
    prompt: Option<f64>,
    decode: Option<f64>,
    output: Option<f64>,
    total: Option<f64>,
    processed: Option<f64>,
}

impl Metrics {
    pub(crate) fn from_runs(
        runs: &[RunSamples],
        config: &BenchConfig,
    ) -> Result<Self, BenchError> {
        let prompt_tokens =
            f64::from(config.batch_size.get()) * f64::from(config.prompt_token_count);
        let rates = runs
            .iter()
            .map(|run| {
                let decode_tokens =
                    f64::from(config.batch_size.get()) * count(run.decode_ns.len())?;
                let output_tokens = run
                    .token_ids
                    .iter()
                    .map(|row| count(row.len()))
                    .sum::<Result<f64, _>>()?;
                let total_ns = run.step_sum_ns();
                Ok(Throughputs {
                    prompt: run
                        .prompt_step_ns
                        .and_then(|ns| throughput(prompt_tokens, ns)),
                    decode: throughput(decode_tokens, sum_ns(&run.decode_ns)),
                    output: run
                        .generation_ns
                        .and_then(|ns| throughput(output_tokens, ns)),
                    total: run
                        .generation_ns
                        .and_then(|ns| throughput(prompt_tokens + output_tokens, ns)),
                    processed: throughput(prompt_tokens + decode_tokens, total_ns),
                })
            })
            .collect::<Result<Vec<_>, BenchError>>()?;
        let rate_summary =
            |select: fn(&Throughputs) -> Option<f64>| summary(rates.iter().filter_map(select));
        let step_sums = runs
            .iter()
            .map(|run| u64_to_f64(run.step_sum_ns()))
            .collect::<Vec<_>>();
        Ok(Self {
            prompt_step_ns: summary(
                runs.iter()
                    .filter_map(|run| run.prompt_step_ns.map(u64_to_f64)),
            )?,
            decode_step_ns: summary(
                runs.iter()
                    .flat_map(|run| run.decode_ns.iter().copied().map(u64_to_f64)),
            )?,
            step_sum_ns: Summary::try_from(step_sums.as_slice())?,
            first_token_ns: summary(
                runs.iter()
                    .filter_map(|run| run.first_token_ns.map(u64_to_f64)),
            )?,
            generation_ns: summary(
                runs.iter()
                    .filter_map(|run| run.generation_ns.map(u64_to_f64)),
            )?,
            prompt_tokens_per_s: rate_summary(|rates| rates.prompt)?,
            decode_tokens_per_s: rate_summary(|rates| rates.decode)?,
            output_tokens_per_s: rate_summary(|rates| rates.output)?,
            total_tokens_per_s: rate_summary(|rates| rates.total)?,
            processed_tokens_per_s: rate_summary(|rates| rates.processed)?,
        })
    }
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

pub(crate) fn sum_ns(samples: &[u64]) -> u64 {
    samples
        .iter()
        .fold(0, |total, &sample| total.saturating_add(sample))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use crate::bench::{
        BenchConfig,
        BenchError,
        Metrics,
        RunSamples,
        TimingScope,
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
                    .saturating_add(super::sum_ns(&decode_ns)),
            ),
            decode_ns,
            token_ids: vec![vec![1; tokens]; batch],
            peak_memory_bytes: Some(2_000_000_000),
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
                metrics.output_tokens_per_s,
                metrics.total_tokens_per_s
            ),
            (None, None, None, None),
            "forward timing does not invent host latencies or full generation rates"
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
        assert_eq!(object.len(), 10, "one field per canonical metric");
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
