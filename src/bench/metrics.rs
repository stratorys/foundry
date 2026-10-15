use std::collections::BTreeMap;

use serde::Serialize;

use crate::bench::RunSamples;
use crate::bench::error::BenchError;
use crate::bench::stats::{
    mean,
    median,
    ns_to_f64,
    percentile,
};

const NS_PER_S: f64 = 1e9;
const VLLM_PERCENTS: [u32; 6] = [10, 25, 50, 75, 90, 99];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenCounts {
    pub prompt_count: u32,
    pub gen_count: u32,
    pub depth_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Metrics {
    pub llama_cpp: LlamaCpp,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mlx: Option<Mlx>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sglang: Option<Sglang>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vllm: Option<Vllm>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candle: Option<Candle>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LlamaCpp {
    pub n_prompt: u32,
    pub n_gen: u32,
    pub n_depth: u32,
    pub avg_ns: u64,
    pub stddev_ns: f64,
    pub avg_ts: f64,
    pub stddev_ts: f64,
    pub samples_ns: Vec<u64>,
    pub samples_ts: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Mlx {
    pub prompt_tps: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_tps: Option<f64>,
    pub trials: Vec<MlxTrial>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct MlxTrial {
    pub prompt_tps: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_tps: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sglang {
    pub runs: Vec<SglangRun>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SglangRun {
    pub prefill_latency: f64,
    pub prefill_throughput: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub median_decode_latency: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub median_decode_throughput: Option<f64>,
    pub total_latency: f64,
    pub overall_throughput: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Vllm {
    pub avg_latency: f64,
    pub latencies: Vec<f64>,
    pub percentiles: BTreeMap<u32, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Candle {
    pub tokens_per_s: Vec<f64>,
}

pub fn metrics(
    runs: &[RunSamples],
    counts: TokenCounts,
) -> Result<Metrics, BenchError> {
    let prefills_ns = runs
        .iter()
        .map(|run| run.prefill_ns.map(|prefill_ns| (prefill_ns, run)))
        .collect::<Option<Vec<(u64, &RunSamples)>>>();
    let measured_runs = prefills_ns.filter(|prefills_ns| !prefills_ns.is_empty());
    Ok(Metrics {
        llama_cpp: llama_cpp(runs, counts)?,
        mlx: measured_runs
            .as_deref()
            .map(|prefills_ns| mlx(prefills_ns, counts))
            .transpose()?,
        sglang: measured_runs
            .as_deref()
            .map(|prefills_ns| sglang(prefills_ns, counts)),
        vllm: measured_runs.as_deref().map(vllm).transpose()?,
        candle: measured_runs
            .as_deref()
            .filter(|_| counts.gen_count >= 2)
            .map(|prefills_ns| candle(prefills_ns, counts)),
    })
}

fn llama_cpp(
    runs: &[RunSamples],
    counts: TokenCounts,
) -> Result<LlamaCpp, BenchError> {
    let samples_ns = runs.iter().map(RunSamples::total_ns).collect::<Vec<u64>>();
    let token_count = f64::from(counts.prompt_count) + f64::from(counts.gen_count);
    let samples_ts = samples_ns
        .iter()
        .map(|&sample_ns| NS_PER_S * token_count / ns_to_f64(sample_ns))
        .collect::<Vec<f64>>();
    let run_count = u64::try_from(samples_ns.len()).map_err(|_| BenchError::SampleCountOverflow)?;
    let avg_ns = sum_ns(&samples_ns)
        .checked_div(run_count)
        .ok_or(BenchError::NoSample)?;
    let samples_ns_f64 = samples_ns
        .iter()
        .copied()
        .map(ns_to_f64)
        .collect::<Vec<f64>>();
    Ok(LlamaCpp {
        n_prompt: counts.prompt_count,
        n_gen: counts.gen_count,
        n_depth: counts.depth_count,
        avg_ns,
        stddev_ns: stdev_llama_cpp(&samples_ns_f64),
        avg_ts: mean(&samples_ts).ok_or(BenchError::NoSample)?,
        stddev_ts: stdev_llama_cpp(&samples_ts),
        samples_ns,
        samples_ts,
    })
}

fn stdev_llama_cpp(samples: &[f64]) -> f64 {
    let Some(count) = u32::try_from(samples.len())
        .ok()
        .filter(|&count| count > 1)
        .map(f64::from)
    else {
        return 0.0;
    };
    let mean = samples.iter().sum::<f64>() / count;
    let square_sum = samples.iter().map(|sample| sample * sample).sum::<f64>();
    (square_sum / (count - 1.0) - mean * mean * count / (count - 1.0)).sqrt()
}

fn mlx(
    prefills_ns: &[(u64, &RunSamples)],
    counts: TokenCounts,
) -> Result<Mlx, BenchError> {
    let trials = prefills_ns
        .iter()
        .map(|&(prefill_ns, run)| MlxTrial {
            prompt_tps: f64::from(counts.prompt_count) / seconds(prefill_ns),
            generation_tps: (counts.gen_count >= 2)
                .then(|| f64::from(counts.gen_count) / seconds(sum_ns(after_first_token(run)))),
        })
        .collect::<Vec<MlxTrial>>();
    let prompt_tps = trials
        .iter()
        .map(|trial| trial.prompt_tps)
        .collect::<Vec<f64>>();
    let generation_tps = trials
        .iter()
        .map(|trial| trial.generation_tps)
        .collect::<Option<Vec<f64>>>();
    Ok(Mlx {
        prompt_tps: mean(&prompt_tps).ok_or(BenchError::NoSample)?,
        generation_tps: generation_tps.as_deref().and_then(mean),
        trials,
    })
}

fn sglang(
    prefills_ns: &[(u64, &RunSamples)],
    counts: TokenCounts,
) -> Sglang {
    let runs = prefills_ns
        .iter()
        .map(|&(prefill_ns, run)| {
            let decode_ns = after_first_token(run);
            let prefill_latency = seconds(prefill_ns);
            let decode_latencies = decode_ns
                .iter()
                .map(|&step_ns| seconds(step_ns))
                .collect::<Vec<f64>>();
            let median_decode_latency = median(&decode_latencies);
            let total_latency = prefill_latency + decode_latencies.iter().sum::<f64>();
            SglangRun {
                prefill_latency,
                prefill_throughput: f64::from(counts.prompt_count) / prefill_latency,
                median_decode_latency,
                median_decode_throughput: median_decode_latency.map(|latency| 1.0 / latency),
                total_latency,
                overall_throughput: (f64::from(counts.prompt_count) + f64::from(counts.gen_count))
                    / total_latency,
            }
        })
        .collect();
    Sglang {
        runs,
    }
}

fn vllm(prefills_ns: &[(u64, &RunSamples)]) -> Result<Vllm, BenchError> {
    let latencies = prefills_ns
        .iter()
        .map(|&(prefill_ns, run)| {
            seconds(prefill_ns.saturating_add(sum_ns(after_first_token(run))))
        })
        .collect::<Vec<f64>>();
    let mut sorted = latencies.clone();
    sorted.sort_unstable_by(f64::total_cmp);
    let percentiles = VLLM_PERCENTS
        .into_iter()
        .map(|percent| {
            percentile(&sorted, percent)
                .map(|latency| (percent, latency))
                .ok_or(BenchError::NoSample)
        })
        .collect::<Result<BTreeMap<u32, f64>, BenchError>>()?;
    Ok(Vllm {
        avg_latency: mean(&latencies).ok_or(BenchError::NoSample)?,
        latencies,
        percentiles,
    })
}

fn candle(
    prefills_ns: &[(u64, &RunSamples)],
    counts: TokenCounts,
) -> Candle {
    let generated_after_first = f64::from(counts.gen_count) - 1.0;
    Candle {
        tokens_per_s: prefills_ns
            .iter()
            .map(|&(_, run)| generated_after_first / seconds(sum_ns(after_first_token(run))))
            .collect(),
    }
}

fn after_first_token(run: &RunSamples) -> &[u64] {
    run.decode_ns
        .split_last()
        .map_or(&[][..], |(_, decode_ns)| decode_ns)
}

pub fn sum_ns(samples_ns: &[u64]) -> u64 {
    samples_ns
        .iter()
        .fold(0, |total_ns, &sample_ns| total_ns.saturating_add(sample_ns))
}

fn seconds(ns: u64) -> f64 { ns_to_f64(ns) / NS_PER_S }

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::bench::RunSamples;
    use crate::bench::metrics::{
        Candle,
        LlamaCpp,
        Metrics,
        Mlx,
        MlxTrial,
        Sglang,
        SglangRun,
        TokenCounts,
        Vllm,
        metrics,
    };

    #[test]
    fn pg_runs_give_every_engine_metric() {
        let runs = [
            RunSamples {
                prefill_ns: Some(1_000_000_000),
                decode_ns: vec![250_000_000, 250_000_000, 500_000_000],
                token_ids: vec![1, 2, 3, 4],
            },
            RunSamples {
                prefill_ns: Some(500_000_000),
                decode_ns: vec![125_000_000, 375_000_000, 1_000_000_000],
                token_ids: vec![1, 2, 3, 4],
            },
        ];
        assert_eq!(
            metrics(
                &runs,
                TokenCounts {
                    prompt_count: 4,
                    gen_count: 3,
                    depth_count: 0,
                }
            ),
            Ok(Metrics {
                llama_cpp: LlamaCpp {
                    n_prompt: 4,
                    n_gen: 3,
                    n_depth: 0,
                    avg_ns: 2_000_000_000,
                    stddev_ns: 0.0,
                    avg_ts: 3.5,
                    stddev_ts: 0.0,
                    samples_ns: vec![2_000_000_000, 2_000_000_000],
                    samples_ts: vec![3.5, 3.5],
                },
                mlx: Some(Mlx {
                    prompt_tps: 6.0,
                    generation_tps: Some(6.0),
                    trials: vec![
                        MlxTrial {
                            prompt_tps: 4.0,
                            generation_tps: Some(6.0),
                        },
                        MlxTrial {
                            prompt_tps: 8.0,
                            generation_tps: Some(6.0),
                        },
                    ],
                }),
                sglang: Some(Sglang {
                    runs: vec![
                        SglangRun {
                            prefill_latency: 1.0,
                            prefill_throughput: 4.0,
                            median_decode_latency: Some(0.25),
                            median_decode_throughput: Some(4.0),
                            total_latency: 1.5,
                            overall_throughput: 7.0 / 1.5,
                        },
                        SglangRun {
                            prefill_latency: 0.5,
                            prefill_throughput: 8.0,
                            median_decode_latency: Some(0.25),
                            median_decode_throughput: Some(4.0),
                            total_latency: 1.0,
                            overall_throughput: 7.0,
                        },
                    ],
                }),
                vllm: Some(Vllm {
                    avg_latency: 1.25,
                    latencies: vec![1.5, 1.0],
                    percentiles: BTreeMap::from([
                        (10, 1.05),
                        (25, 1.125),
                        (50, 1.25),
                        (75, 1.375),
                        (90, 1.45),
                        (99, 1.495),
                    ]),
                }),
                candle: Some(Candle {
                    tokens_per_s: vec![4.0, 4.0],
                }),
            }),
            "prefill and three decode steps; the other engines skip the last step"
        );
    }

    #[test]
    fn pp_runs_skip_decode_metrics() {
        let runs = [RunSamples {
            prefill_ns: Some(250_000_000),
            decode_ns: Vec::new(),
            token_ids: vec![9],
        }];
        assert_eq!(
            metrics(
                &runs,
                TokenCounts {
                    prompt_count: 2,
                    gen_count: 0,
                    depth_count: 3,
                }
            ),
            Ok(Metrics {
                llama_cpp: LlamaCpp {
                    n_prompt: 2,
                    n_gen: 0,
                    n_depth: 3,
                    avg_ns: 250_000_000,
                    stddev_ns: 0.0,
                    avg_ts: 8.0,
                    stddev_ts: 0.0,
                    samples_ns: vec![250_000_000],
                    samples_ts: vec![8.0],
                },
                mlx: Some(Mlx {
                    prompt_tps: 8.0,
                    generation_tps: None,
                    trials: vec![MlxTrial {
                        prompt_tps: 8.0,
                        generation_tps: None,
                    }],
                }),
                sglang: Some(Sglang {
                    runs: vec![SglangRun {
                        prefill_latency: 0.25,
                        prefill_throughput: 8.0,
                        median_decode_latency: None,
                        median_decode_throughput: None,
                        total_latency: 0.25,
                        overall_throughput: 8.0,
                    }],
                }),
                vllm: Some(Vllm {
                    avg_latency: 0.25,
                    latencies: vec![0.25],
                    percentiles: [10, 25, 50, 75, 90, 99]
                        .into_iter()
                        .map(|percent| (percent, 0.25))
                        .collect(),
                }),
                candle: None,
            }),
            "prefill only"
        );
    }

    #[test]
    fn tg_runs_give_only_llama_cpp() {
        let runs = [
            RunSamples {
                prefill_ns: None,
                decode_ns: vec![500_000_000, 500_000_000],
                token_ids: vec![1, 2],
            },
            RunSamples {
                prefill_ns: None,
                decode_ns: vec![250_000_000, 250_000_000],
                token_ids: vec![1, 2],
            },
        ];
        assert_eq!(
            metrics(
                &runs,
                TokenCounts {
                    prompt_count: 0,
                    gen_count: 2,
                    depth_count: 0,
                }
            ),
            Ok(Metrics {
                llama_cpp: LlamaCpp {
                    n_prompt: 0,
                    n_gen: 2,
                    n_depth: 0,
                    avg_ns: 750_000_000,
                    stddev_ns: 500_000_000_f64 / 2_f64.sqrt(),
                    avg_ts: 3.0,
                    stddev_ts: 2_f64.sqrt(),
                    samples_ns: vec![1_000_000_000, 500_000_000],
                    samples_ts: vec![2.0, 4.0],
                },
                mlx: None,
                sglang: None,
                vllm: None,
                candle: None,
            }),
            "no prefill, so only llama.cpp t/s applies"
        );
    }
}
