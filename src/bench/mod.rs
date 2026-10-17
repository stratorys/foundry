mod error;
mod memory;
mod metadata;
mod metrics;
mod rng;
mod stats;

use std::num::{
    NonZeroU32,
    NonZeroUsize,
};
use std::time::{
    Duration,
    Instant,
};

use serde::Serialize;
use tracing::{
    error,
    info,
};

pub use self::error::BenchError;
pub use self::memory::PeakMemorySource;
pub use self::metadata::{
    Metadata,
    git_commit,
    rfc3339_utc,
};
pub use self::metrics::{
    Candle,
    LlamaCpp,
    Metrics,
    Mlx,
    MlxTrial,
    Sglang,
    SglangRun,
    TokenCounts,
    Vllm,
};
use self::metrics::{
    metrics,
    run_ts,
    sum_ns,
};
use self::rng::SplitMix64;
pub use self::stats::Summary;
use self::stats::{
    mean,
    ns_to_f64,
};
use crate::generate::Decoder;
use crate::models::CausalLm;
use crate::nn::Device;

const NS_PER_MS: f64 = 1e6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BenchConfig {
    pub prompt_token_count: u32,
    pub gen_token_count: u32,
    pub depth_token_count: u32,
    pub warmup_run_count: u32,
    pub warmup_ms: u64,
    pub run_count: NonZeroU32,
    pub batch_size: NonZeroU32,
    pub seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Pp,
    Tg,
    Pg,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RunSamples {
    pub prefill_ns: Option<u64>,
    pub decode_ns: Vec<u64>,
    pub token_ids: Vec<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Report {
    pub config: BenchConfig,
    pub mode: Mode,
    pub depth_ids: Vec<Vec<u32>>,
    pub prompt_ids: Vec<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_ids: Option<Vec<u32>>,
    pub warmup_run_count: u32,
    pub runs: Vec<RunSamples>,
    pub stats: Stats,
    pub metrics: Metrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_memory_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_memory_source: Option<PeakMemorySource>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Stats {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_ns: Option<Summary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_step_ns: Option<Summary>,
    pub run_ns: Summary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Inputs {
    batch_size: NonZeroUsize,
    depth_ids: Vec<Vec<u32>>,
    prompt_ids: Vec<Vec<u32>>,
    start_ids: Option<Vec<u32>>,
    gen_token_count: usize,
}

impl TryFrom<&BenchConfig> for Mode {
    type Error = BenchError;

    fn try_from(config: &BenchConfig) -> Result<Self, Self::Error> {
        match (config.prompt_token_count, config.gen_token_count) {
            (0, 0) => {
                error!(message = "Benchmark holds neither prompt nor generated tokens.");
                Err(BenchError::NoToken)
            }
            (_, 0) => Ok(Self::Pp),
            (0, _) => Ok(Self::Tg),
            _ => Ok(Self::Pg),
        }
    }
}

impl RunSamples {
    pub fn total_ns(&self) -> u64 {
        self.prefill_ns
            .unwrap_or(0)
            .saturating_add(sum_ns(&self.decode_ns))
    }
}

pub fn bench<D: Device, M: CausalLm<D>>(
    device: &mut D,
    model: &M,
    vocab_size: NonZeroU32,
    config: &BenchConfig,
) -> Result<Report, BenchError> {
    let mode = Mode::try_from(config)?;
    let inputs = draw_inputs(config, mode, vocab_size)?;
    let counts = TokenCounts {
        batch_size: config.batch_size.get(),
        prompt_count: config.prompt_token_count,
        gen_count: config.gen_token_count,
        depth_count: config.depth_token_count,
    };
    info!(
        message = "Starting the benchmark.",
        ?mode,
        prompt_tokens = config.prompt_token_count,
        gen_tokens = config.gen_token_count,
        depth = config.depth_token_count,
        warmup_runs = config.warmup_run_count,
        warmup_ms = config.warmup_ms,
        runs = config.run_count.get(),
        batch_size = config.batch_size.get(),
        seed = config.seed,
    );
    let warmup_start = Instant::now();
    let warmup = Duration::from_millis(config.warmup_ms);
    let mut warmup_run_count = 0_u32;
    while warmup_run_count < config.warmup_run_count || warmup_start.elapsed() < warmup {
        let warmup_run = run(device, model, &inputs)?;
        warmup_run_count = warmup_run_count.checked_add(1).ok_or_else(|| {
            error!(message = "Run count overflows u32.", warmup_run_count);
            BenchError::RunCountOverflow
        })?;
        info!(
            message = "Finished a warmup run.",
            warmup_run = warmup_run_count,
            run_ms = ns_to_ms(warmup_run.total_ns()),
        );
    }
    let runs = (1..=config.run_count.get())
        .map(|run_number| {
            let measured_run = run(device, model, &inputs)?;
            let decode_ms = measured_run
                .decode_ns
                .iter()
                .copied()
                .map(ns_to_ms)
                .collect::<Vec<f64>>();
            info!(
                message = "Finished a measured run.",
                run = run_number,
                runs = config.run_count.get(),
                prefill_ms = ?measured_run.prefill_ns.map(ns_to_ms),
                decode_ms_mean = ?mean(&decode_ms),
                ts = run_ts(&measured_run, counts),
            );
            Ok(measured_run)
        })
        .collect::<Result<Vec<RunSamples>, BenchError>>()?;
    let stats = stats(&runs)?;
    let metrics = metrics(&runs, counts)?;
    let peak_memory = device.current_allocated_memory();
    Ok(Report {
        config: *config,
        mode,
        depth_ids: inputs.depth_ids,
        prompt_ids: inputs.prompt_ids,
        start_ids: inputs.start_ids,
        warmup_run_count,
        runs,
        stats,
        metrics,
        peak_memory_bytes: peak_memory.map(|peak_memory| peak_memory.bytes),
        peak_memory_source: peak_memory.map(|peak_memory| peak_memory.source.into()),
    })
}

fn draw_inputs(
    config: &BenchConfig,
    mode: Mode,
    vocab_size: NonZeroU32,
) -> Result<Inputs, BenchError> {
    let batch_size = NonZeroUsize::try_from(config.batch_size).map_err(|_| {
        error!(
            message = "Batch size does not fit in usize.",
            batch_size = config.batch_size.get()
        );
        BenchError::BatchSizeOverflow
    })?;
    let mut rng = SplitMix64::new(config.seed);
    let mut rows = |count: u32| {
        let token_count = token_count(count)?;
        (0..batch_size.get())
            .map(|_| token_ids(&mut rng, token_count, vocab_size))
            .collect::<Result<Vec<Vec<u32>>, BenchError>>()
    };
    let depth_ids = rows(config.depth_token_count)?;
    let prompt_ids = rows(config.prompt_token_count)?;
    let start_ids = match mode {
        Mode::Tg => Some(token_ids(&mut rng, batch_size.get(), vocab_size)?),
        Mode::Pp | Mode::Pg => None,
    };
    Ok(Inputs {
        batch_size,
        depth_ids,
        prompt_ids,
        start_ids,
        gen_token_count: token_count(config.gen_token_count)?,
    })
}

fn token_ids(
    rng: &mut SplitMix64,
    count: usize,
    vocab_size: NonZeroU32,
) -> Result<Vec<u32>, BenchError> {
    (0..count).map(|_| token_id(rng, vocab_size)).collect()
}

fn token_id(
    rng: &mut SplitMix64,
    vocab_size: NonZeroU32,
) -> Result<u32, BenchError> {
    rng.below(vocab_size).ok_or_else(|| {
        error!(
            message = "Drawing a random token ID failed.",
            bound = vocab_size.get()
        );
        BenchError::Random
    })
}

fn token_count(count: u32) -> Result<usize, BenchError> {
    usize::try_from(count).map_err(|_| {
        error!(message = "Token position overflows usize.", count);
        BenchError::PositionOverflow
    })
}

fn run<D: Device, M: CausalLm<D>>(
    device: &mut D,
    model: &M,
    inputs: &Inputs,
) -> Result<RunSamples, BenchError> {
    let mut decoder = Decoder::new(device, model, inputs.batch_size).map_err(|error| {
        error!(message = "Creating the decoder failed.", %error);
        BenchError::Decoder
    })?;
    let depth_ids = inputs.depth_ids.concat();
    if !depth_ids.is_empty() {
        step(&mut decoder, &depth_ids, 0)?;
    }
    let mut past_seen_tokens = inputs.depth_ids.first().map_or(0, Vec::len);
    let mut token_ids =
        vec![Vec::with_capacity(inputs.gen_token_count.saturating_add(1)); inputs.batch_size.get()];
    let prompt_ids = inputs.prompt_ids.concat();
    let prompt_len = inputs.prompt_ids.first().map_or(0, Vec::len);
    let (prefill_ns, mut tokens) = match (NonZeroUsize::new(prompt_len), &inputs.start_ids) {
        (Some(prompt_len), _) => {
            let (tokens, prefill_ns) = timed_step(&mut decoder, &prompt_ids, past_seen_tokens)?;
            push_tokens(&mut token_ids, &tokens);
            past_seen_tokens = advance(past_seen_tokens, prompt_len.get())?;
            (Some(prefill_ns), tokens)
        }
        (None, Some(start_ids)) => (None, start_ids.clone()),
        (None, None) => {
            error!(message = "Benchmark holds neither prompt nor start token.");
            return Err(BenchError::NoToken);
        }
    };
    let decode_ns = (0..inputs.gen_token_count)
        .map(|_| {
            let (next_tokens, step_ns) = timed_step(&mut decoder, &tokens, past_seen_tokens)?;
            push_tokens(&mut token_ids, &next_tokens);
            tokens = next_tokens;
            past_seen_tokens = advance(past_seen_tokens, 1)?;
            Ok(step_ns)
        })
        .collect::<Result<Vec<u64>, BenchError>>()?;
    Ok(RunSamples {
        prefill_ns,
        decode_ns,
        token_ids,
        peak_memory_bytes: device
            .current_allocated_memory()
            .map(|peak_memory| peak_memory.bytes),
    })
}

fn push_tokens(
    token_ids: &mut [Vec<u32>],
    tokens: &[u32],
) {
    token_ids
        .iter_mut()
        .zip(tokens)
        .for_each(|(row, &token)| row.push(token));
}

fn timed_step<D: Device, M: CausalLm<D>>(
    decoder: &mut Decoder<'_, D, M>,
    input_ids: &[u32],
    past_seen_tokens: usize,
) -> Result<(Vec<u32>, u64), BenchError> {
    let start = Instant::now();
    let tokens = step(decoder, input_ids, past_seen_tokens)?;
    let elapsed = start.elapsed();
    let elapsed_ns = u64::try_from(elapsed.as_nanos()).map_err(|_| {
        error!(
            message = "Step duration does not fit in u64 nanoseconds.",
            ?elapsed
        );
        BenchError::DurationOverflow
    })?;
    Ok((tokens, elapsed_ns))
}

fn step<D: Device, M: CausalLm<D>>(
    decoder: &mut Decoder<'_, D, M>,
    input_ids: &[u32],
    past_seen_tokens: usize,
) -> Result<Vec<u32>, BenchError> {
    decoder.step(input_ids, past_seen_tokens).map_err(|error| {
        error!(
            message = "Benchmark step failed.",
            past_seen_tokens,
            token_count = input_ids.len(),
            %error,
        );
        BenchError::Step
    })
}

fn ns_to_ms(ns: u64) -> f64 { ns_to_f64(ns) / NS_PER_MS }

fn advance(
    past_seen_tokens: usize,
    seq_len: usize,
) -> Result<usize, BenchError> {
    past_seen_tokens.checked_add(seq_len).ok_or_else(|| {
        error!(
            message = "Token position overflows usize.",
            past_seen_tokens, seq_len
        );
        BenchError::PositionOverflow
    })
}

fn stats(runs: &[RunSamples]) -> Result<Stats, BenchError> {
    let prefill_ns = runs
        .iter()
        .map(|run| run.prefill_ns.map(ns_to_f64))
        .collect::<Option<Vec<f64>>>()
        .filter(|samples| !samples.is_empty());
    let decode_step_ns = runs
        .iter()
        .flat_map(|run| run.decode_ns.iter().copied().map(ns_to_f64))
        .collect::<Vec<f64>>();
    let run_ns = runs
        .iter()
        .map(|run| ns_to_f64(run.total_ns()))
        .collect::<Vec<f64>>();
    Ok(Stats {
        prefill_ns: prefill_ns.as_deref().map(Summary::try_from).transpose()?,
        decode_step_ns: (!decode_step_ns.is_empty())
            .then(|| Summary::try_from(decode_step_ns.as_slice()))
            .transpose()?,
        run_ns: Summary::try_from(run_ns.as_slice())?,
    })
}

#[cfg(test)]
mod tests {
    use std::num::{
        NonZeroU32,
        NonZeroUsize,
    };

    use crate::bench::rng::SplitMix64;
    use crate::bench::{
        BenchConfig,
        BenchError,
        Inputs,
        Mode,
        draw_inputs,
        token_ids,
    };

    const CONFIG: BenchConfig = BenchConfig {
        prompt_token_count: 0,
        gen_token_count: 0,
        depth_token_count: 0,
        warmup_run_count: 2,
        warmup_ms: 0,
        run_count: NonZeroU32::MIN.saturating_add(2),
        batch_size: NonZeroU32::MIN,
        seed: 7,
    };

    #[test]
    fn mode_follows_the_token_counts() {
        assert_eq!(
            [
                BenchConfig {
                    prompt_token_count: 4,
                    gen_token_count: 0,
                    depth_token_count: 0,
                    ..CONFIG
                },
                BenchConfig {
                    prompt_token_count: 0,
                    gen_token_count: 4,
                    depth_token_count: 0,
                    ..CONFIG
                },
                BenchConfig {
                    prompt_token_count: 4,
                    gen_token_count: 4,
                    depth_token_count: 0,
                    ..CONFIG
                },
                BenchConfig {
                    prompt_token_count: 0,
                    gen_token_count: 0,
                    depth_token_count: 4,
                    ..CONFIG
                }
            ]
            .iter()
            .map(Mode::try_from)
            .collect::<Vec<Result<Mode, BenchError>>>(),
            vec![
                Ok(Mode::Pp),
                Ok(Mode::Tg),
                Ok(Mode::Pg),
                Err(BenchError::NoToken)
            ],
            "pp without generation, tg without prompt, pg with both"
        );
    }

    #[test]
    fn draw_inputs_gives_each_sequence_its_own_rows() {
        let vocab_size = NonZeroU32::new(64).expect("64 is not zero");
        let config = BenchConfig {
            prompt_token_count: 3,
            gen_token_count: 2,
            depth_token_count: 2,
            batch_size: NonZeroU32::MIN.saturating_add(1),
            ..CONFIG
        };
        let shape = |inputs: Inputs| {
            (
                inputs.batch_size,
                inputs
                    .depth_ids
                    .iter()
                    .map(Vec::len)
                    .collect::<Vec<usize>>(),
                inputs
                    .prompt_ids
                    .iter()
                    .map(Vec::len)
                    .collect::<Vec<usize>>(),
                inputs.start_ids.as_ref().map(Vec::len),
                inputs.gen_token_count,
            )
        };
        let batch_size = NonZeroUsize::new(2).expect("2 is not zero");
        assert_eq!(
            [Mode::Pg, Mode::Tg].map(|mode| draw_inputs(&config, mode, vocab_size).map(shape)),
            [
                Ok((batch_size, vec![2, 2], vec![3, 3], None, 2)),
                Ok((batch_size, vec![2, 2], vec![3, 3], Some(2), 2)),
            ],
            "two sequences of depth 2 and prompt 3; only tg draws start tokens"
        );
    }

    #[test]
    fn token_ids_are_below_the_vocab_size_and_repeat_with_the_seed() {
        let vocab_size = NonZeroU32::new(7).expect("7 is not zero");
        let token_ids_drawn =
            token_ids(&mut SplitMix64::new(42), 1000, vocab_size).expect("drawing succeeds");
        assert_eq!(token_ids_drawn.len(), 1000, "one ID per requested token");
        assert!(
            token_ids_drawn.iter().all(|&token_id| token_id < 7),
            "every ID is below 7"
        );
        assert_eq!(
            token_ids(&mut SplitMix64::new(42), 1000, vocab_size),
            Ok(token_ids_drawn),
            "the same seed gives the same IDs"
        );
    }

    #[test]
    fn same_seed_draws_the_same_inputs() {
        let config = BenchConfig {
            prompt_token_count: 4,
            gen_token_count: 1,
            depth_token_count: 4,
            ..CONFIG
        };
        let draw = || {
            draw_inputs(
                &config,
                Mode::Pg,
                NonZeroU32::new(64).expect("64 is not zero"),
            )
        };
        assert_eq!(draw(), draw(), "seed 7 gives the same depth and prompt IDs");
    }
}
