mod error;
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
pub use self::metadata::{
    Metadata,
    git_commit,
};
use self::metrics::sum_ns;
pub use self::metrics::{
    DecodePositionMetrics,
    Metrics,
};
use self::rng::SplitMix64;
pub use self::stats::Summary;
use self::stats::u64_to_f64;
use crate::generate::{
    Decoder,
    GenerateError,
};
use crate::models::CausalLm;
use crate::nn::Device;

const NS_PER_MS: f64 = 1e6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BenchConfig {
    pub prompt_token_count: u32,
    pub new_token_count: u32,
    pub timing_scope: TimingScope,
    pub depth_token_count: u32,
    pub warmup_run_count: u32,
    pub warmup_ms: u64,
    pub run_count: NonZeroU32,
    pub batch_size: NonZeroU32,
    pub seed: u64,
}

/// Separate invocations prevent forward synchronization from perturbing
/// generation timing.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum TimingScope {
    Forward,
    #[default]
    Generation,
}

impl BenchConfig {
    pub fn decode_step_count(&self) -> u32 {
        self.new_token_count
            .saturating_sub(u32::from(self.prompt_token_count > 0))
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub struct RunSamples {
    pub prompt_step_ns: Option<u64>,
    pub decode: Vec<DecodeSample>,
    pub first_token_ns: Option<u64>,
    pub generation_ns: Option<u64>,
    pub token_ids: Vec<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_memory_bytes: Option<u64>,
}

/// One single-token forward; context is the cache length before processing its
/// input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DecodeSample {
    pub position: usize,
    pub context_tokens: usize,
    pub step_ns: u64,
    pub inter_token_ns: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Report {
    pub config: BenchConfig,
    pub depth_ids: Vec<Vec<u32>>,
    pub prompt_ids: Vec<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_ids: Option<Vec<u32>>,
    pub warmup_run_count: u32,
    pub runs: Vec<RunSamples>,
    pub metrics: Metrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_memory_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_memory_source: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Inputs {
    batch_size: NonZeroUsize,
    depth_ids: Vec<Vec<u32>>,
    prompt_ids: Vec<Vec<u32>>,
    start_ids: Option<Vec<u32>>,
    new_token_count: usize,
    timing_scope: TimingScope,
}

impl RunSamples {
    pub fn decode_ns(&self) -> impl Iterator<Item = u64> + '_ {
        self.decode.iter().map(|sample| sample.step_ns)
    }

    pub fn step_sum_ns(&self) -> u64 {
        self.prompt_step_ns
            .unwrap_or(0)
            .saturating_add(sum_ns(self.decode_ns()))
    }
}

pub fn bench<D: Device, M: CausalLm<D>>(
    device: &mut D,
    model: &M,
    vocab_size: NonZeroU32,
    config: &BenchConfig,
) -> Result<Report, BenchError> {
    if config.prompt_token_count == 0 && config.new_token_count == 0 {
        return Err(BenchError::NoToken);
    }
    let inputs = draw_inputs(config, vocab_size)?;
    info!(message = "Starting the benchmark.", ?config);
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
            run_ms = ns_to_ms(warmup_run.step_sum_ns()),
        );
    }
    let runs = (1..=config.run_count.get())
        .map(|run_number| {
            let measured_run = run(device, model, &inputs)?;
            info!(
                message = "Finished a measured run.",
                run = run_number,
                runs = config.run_count.get(),
                prefill_ms = ?measured_run.prompt_step_ns.map(ns_to_ms),
                decode_steps = measured_run.decode.len(),
                decode_ms_total = ns_to_ms(sum_ns(measured_run.decode_ns())),
            );
            Ok(measured_run)
        })
        .collect::<Result<Vec<RunSamples>, BenchError>>()?;
    let metrics = Metrics::from_runs(&runs, config)?;
    let peak_memory = device.max_memory_allocated();
    Ok(Report {
        config: *config,
        depth_ids: inputs.depth_ids,
        prompt_ids: inputs.prompt_ids,
        start_ids: inputs.start_ids,
        warmup_run_count,
        runs,
        metrics,
        peak_memory_bytes: peak_memory.map(|peak_memory| peak_memory.bytes),
        peak_memory_source: peak_memory.map(|peak_memory| peak_memory.source),
    })
}

fn draw_inputs(
    config: &BenchConfig,
    vocab_size: NonZeroU32,
) -> Result<Inputs, BenchError> {
    let batch_size =
        NonZeroUsize::try_from(config.batch_size).map_err(|_| BenchError::BatchSizeOverflow)?;
    let mut rng = SplitMix64::new(config.seed);
    let mut rows = |count: u32| {
        let token_count = token_count(count)?;
        (0..batch_size.get())
            .map(|_| token_ids(&mut rng, token_count, vocab_size))
            .collect::<Result<Vec<Vec<u32>>, BenchError>>()
    };
    let depth_ids = rows(config.depth_token_count)?;
    let prompt_ids = rows(config.prompt_token_count)?;
    let start_ids = if config.prompt_token_count == 0 {
        Some(token_ids(&mut rng, batch_size.get(), vocab_size)?)
    } else {
        None
    };
    Ok(Inputs {
        batch_size,
        depth_ids,
        prompt_ids,
        start_ids,
        new_token_count: token_count(config.new_token_count)?,
        timing_scope: config.timing_scope,
    })
}

fn token_ids(
    rng: &mut SplitMix64,
    count: usize,
    vocab_size: NonZeroU32,
) -> Result<Vec<u32>, BenchError> {
    (0..count)
        .map(|_| rng.below(vocab_size).ok_or(BenchError::Random))
        .collect()
}

fn token_count(count: u32) -> Result<usize, BenchError> {
    usize::try_from(count).map_err(|_| BenchError::PositionOverflow)
}

fn run<D: Device, M: CausalLm<D>>(
    device: &mut D,
    model: &M,
    inputs: &Inputs,
) -> Result<RunSamples, BenchError> {
    let prompt_len = inputs.prompt_ids.first().map_or(0, Vec::len);
    let mut tokens = inputs
        .start_ids
        .clone()
        .unwrap_or_else(|| inputs.prompt_ids.concat());
    let depth_ids = inputs.depth_ids.concat();
    let mut past_seen_tokens = inputs.depth_ids.first().map_or(0, Vec::len);
    let mut samples = RunSamples {
        token_ids: vec![Vec::with_capacity(inputs.new_token_count); inputs.batch_size.get()],
        decode: Vec::with_capacity(inputs.new_token_count),
        ..RunSamples::default()
    };
    device.synchronize().map_err(|error| {
        error!(message = "Synchronizing before the benchmark failed.", %error);
        BenchError::Step
    })?;
    let start = Instant::now();
    let mut decoder = Decoder::new(device, model, inputs.batch_size).map_err(|error| {
        error!(message = "Creating the decoder failed.", %error);
        BenchError::Decoder
    })?;
    if !depth_ids.is_empty() {
        decoder.step(&depth_ids, 0).map_err(step_error)?;
    }
    if inputs.timing_scope == TimingScope::Forward {
        decoder.synchronize().map_err(step_error)?;
    }
    for position in 0..inputs.new_token_count.max(usize::from(prompt_len > 0)) {
        let is_prompt = position == 0 && prompt_len > 0;
        let step = timed_step(
            &mut decoder,
            &tokens,
            past_seen_tokens,
            inputs.timing_scope,
            inputs.new_token_count > 0,
        )?;
        let inter_token_ns =
            if inputs.new_token_count > 0 && inputs.timing_scope == TimingScope::Generation {
                let elapsed = duration_ns(step.completed_at.duration_since(start))?;
                let interval = samples
                    .generation_ns
                    .map(|previous| elapsed.saturating_sub(previous));
                samples.first_token_ns.get_or_insert(elapsed);
                samples.generation_ns = Some(elapsed);
                interval
            } else {
                None
            };
        if is_prompt {
            samples.prompt_step_ns = Some(step.ns);
        } else {
            samples.decode.push(DecodeSample {
                position: samples.decode.len(),
                context_tokens: past_seen_tokens,
                step_ns: step.ns,
                inter_token_ns,
            });
        }
        samples
            .token_ids
            .iter_mut()
            .zip(&step.tokens)
            .for_each(|(row, &token)| row.push(token));
        tokens = step.tokens;
        past_seen_tokens = advance(past_seen_tokens, if is_prompt { prompt_len } else { 1 })?;
    }
    samples.peak_memory_bytes = device.max_memory_allocated().map(|peak| peak.bytes);
    Ok(samples)
}

struct TimedStep {
    tokens: Vec<u32>,
    ns: u64,
    completed_at: Instant,
}

fn timed_step<D: Device, M: CausalLm<D>>(
    decoder: &mut Decoder<'_, D, M>,
    input_ids: &[u32],
    past_seen_tokens: usize,
    scope: TimingScope,
    select_token: bool,
) -> Result<TimedStep, BenchError> {
    let start = Instant::now();
    let logits = decoder
        .forward_logits(input_ids, past_seen_tokens)
        .map_err(step_error)?;
    let forward_end = if scope == TimingScope::Forward || !select_token {
        decoder.synchronize().map_err(step_error)?;
        Some(Instant::now())
    } else {
        None
    };
    let tokens = if select_token {
        decoder.next_tokens(&logits).map_err(step_error)?
    } else {
        Vec::new()
    };
    let completed_at = forward_end.unwrap_or_else(Instant::now);
    Ok(TimedStep {
        tokens,
        ns: duration_ns(completed_at.duration_since(start))?,
        completed_at,
    })
}

fn duration_ns(elapsed: Duration) -> Result<u64, BenchError> {
    u64::try_from(elapsed.as_nanos()).map_err(|_| BenchError::DurationOverflow)
}

fn step_error(error: GenerateError) -> BenchError {
    error!(message = "Benchmark step failed.", %error);
    BenchError::Step
}

fn ns_to_ms(ns: u64) -> f64 { u64_to_f64(ns) / NS_PER_MS }

fn advance(
    past_seen_tokens: usize,
    seq_len: usize,
) -> Result<usize, BenchError> {
    past_seen_tokens
        .checked_add(seq_len)
        .ok_or(BenchError::PositionOverflow)
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
        TimingScope,
        bench,
        draw_inputs,
        token_ids,
    };
    use crate::models::CausalLm;
    use crate::nn::primitive::{
        Op,
        ReduceOp,
    };
    use crate::nn::{
        AllocatedMemory,
        DType,
        Device,
        Operand,
        Shape,
        Tensor,
        TensorError,
    };

    const CONFIG: BenchConfig = BenchConfig {
        prompt_token_count: 0,
        new_token_count: 0,
        timing_scope: TimingScope::Generation,
        depth_token_count: 0,
        warmup_run_count: 2,
        warmup_ms: 0,
        run_count: NonZeroU32::MIN.saturating_add(2),
        batch_size: NonZeroU32::MIN,
        seed: 7,
    };

    #[test]
    fn cache_capacity_follows_exact_token_counts() {
        for (prompt, output, decode) in [(4, 0, 0), (0, 4, 4), (4, 1, 0), (4, 128, 127)] {
            let config = BenchConfig {
                prompt_token_count: prompt,
                new_token_count: output,
                ..CONFIG
            };
            assert_eq!(
                config.decode_step_count(),
                decode,
                "only processed decode inputs occupy cache"
            );
        }
        let mut device = RecordingDevice::default();
        assert!(
            matches!(
                bench(&mut device, &TestModel, NonZeroU32::MIN, &CONFIG),
                Err(BenchError::NoToken)
            ),
            "empty benchmark rejected"
        );
        assert!(device.events.is_empty(), "reject before device work");
    }

    #[test]
    fn draw_inputs_gives_each_sequence_its_own_rows() {
        for prompt in [0, 3] {
            let config = BenchConfig {
                prompt_token_count: prompt,
                new_token_count: 2,
                depth_token_count: 2,
                batch_size: NonZeroU32::MIN.saturating_add(1),
                ..CONFIG
            };
            let inputs =
                draw_inputs(&config, NonZeroU32::new(64).expect("vocabulary")).expect("inputs");
            let lengths = |rows: &[Vec<u32>]| rows.iter().map(Vec::len).collect::<Vec<_>>();
            assert_eq!(
                lengths(&inputs.depth_ids),
                vec![2, 2],
                "independent depth rows"
            );
            assert_eq!(
                lengths(&inputs.prompt_ids),
                vec![usize::try_from(prompt).expect("small prompt"); 2],
                "independent prompt rows"
            );
            let start = if prompt == 0 { Some(2) } else { None };
            assert_eq!(
                inputs.start_ids.as_ref().map(Vec::len),
                start,
                "start inputs only without a prompt"
            );
            assert_eq!(
                (inputs.batch_size.get(), inputs.new_token_count),
                (2, 2),
                "batch and exact output count"
            );
        }
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
            new_token_count: 1,
            depth_token_count: 4,
            ..CONFIG
        };
        let draw = || draw_inputs(&config, NonZeroU32::new(64).expect("64 is not zero"));
        assert_eq!(draw(), draw(), "seed 7 gives the same depth and prompt IDs");
    }

    #[derive(Default)]
    struct RecordingDevice {
        events: Vec<&'static str>,
        forwards: Vec<(usize, usize)>,
        sync_count: usize,
        selected_at: Option<std::time::Instant>,
        read_at: Option<std::time::Instant>,
        fail_on_sync: Option<usize>,
    }

    impl Device for RecordingDevice {
        type Storage = Vec<u8>;
        type Error = std::io::Error;
        fn device_name(&self) -> String { "recording".to_owned() }
        fn max_memory_allocated(&self) -> Option<AllocatedMemory> { None }
        fn synchronize(&mut self) -> Result<(), Self::Error> {
            self.events.push("ready");
            self.sync_count = self.sync_count.saturating_add(1);
            if self.fail_on_sync == Some(self.sync_count) {
                return Err(std::io::Error::other("synchronization failed"));
            }
            Ok(())
        }
        fn storage_from_bytes(
            &mut self,
            bytes: &[u8],
        ) -> Result<Vec<u8>, Self::Error> {
            Ok(bytes.to_vec())
        }
        fn zeros(
            &mut self,
            nbytes: usize,
        ) -> Result<Vec<u8>, Self::Error> {
            self.events.push("cache");
            Ok(vec![0; nbytes])
        }
        fn storage_to_bytes(
            &mut self,
            input: &Operand<&Vec<u8>>,
        ) -> Result<Vec<u8>, Self::Error> {
            self.synchronize()?;
            self.events.push("read");
            let bytes = input.storage().to_vec();
            self.read_at = Some(std::time::Instant::now());
            Ok(bytes)
        }
        fn run(
            &mut self,
            op: Op,
            _inputs: &[Operand<&Vec<u8>>],
            output: &Shape,
            _dtype: DType,
        ) -> Result<Vec<u8>, Self::Error> {
            assert_eq!(
                op,
                Op::Reduce {
                    op: ReduceOp::Argmax,
                    dim: 2
                },
                "only token selection is needed by the test model"
            );
            self.selected_at = Some(std::time::Instant::now());
            self.events.push("select");
            Ok(1_u32.to_le_bytes().repeat(output.numel()))
        }
        fn run_in_place(
            &mut self,
            _op: Op,
            _target: Operand<&mut Vec<u8>>,
            _inputs: &[Operand<&Vec<u8>>],
        ) -> Result<(), Self::Error> {
            Err(std::io::Error::other("unexpected in-place operation"))
        }
    }

    struct TestModel;
    impl CausalLm<RecordingDevice> for TestModel {
        type Cache = Tensor<RecordingDevice>;
        type Error = TensorError;
        fn cache(
            &self,
            device: &mut RecordingDevice,
            batch: NonZeroUsize,
        ) -> Result<Self::Cache, Self::Error> {
            Tensor::zeros(device, DType::F32, Shape::try_from([batch.get(), 1])?)
        }
        fn forward(
            &self,
            device: &mut RecordingDevice,
            ids: &Tensor<RecordingDevice>,
            _cache: &mut Self::Cache,
            past: usize,
            _keep: usize,
        ) -> Result<Tensor<RecordingDevice>, Self::Error> {
            device.events.push("forward");
            device
                .forwards
                .push((past, ids.shape().dims().get(1).copied().unwrap_or(0)));
            let batch = ids.shape().dims().first().copied().unwrap_or(0);
            let bytes = [0_f32, 1.0].map(f32::to_le_bytes).concat().repeat(batch);
            Tensor::from_bytes(device, &bytes, DType::F32, Shape::try_from([batch, 1, 2])?)
        }
    }

    fn measured_config(
        prompt: u32,
        output: u32,
        scope: TimingScope,
    ) -> BenchConfig {
        BenchConfig {
            prompt_token_count: prompt,
            new_token_count: output,
            timing_scope: scope,
            warmup_run_count: 0,
            run_count: NonZeroU32::MIN,
            ..CONFIG
        }
    }

    #[test]
    fn exact_generation_in_both_scopes_and_batches() {
        for scope in [TimingScope::Generation, TimingScope::Forward] {
            for output in [1, 128] {
                for batch in [1, 2] {
                    let config = BenchConfig {
                        batch_size: NonZeroU32::new(batch).expect("positive batch"),
                        depth_token_count: 7,
                        ..measured_config(4, output, scope)
                    };
                    let mut device = RecordingDevice::default();
                    let report = bench(&mut device, &TestModel, NonZeroU32::MIN, &config)
                        .expect("benchmark");
                    let run = report.runs.first().expect("one run");
                    let output = usize::try_from(output).expect("small output");
                    let batch = usize::try_from(batch).expect("small batch");
                    assert_eq!(
                        run.token_ids,
                        vec![vec![1; output]; batch],
                        "exact greedy output per sequence"
                    );
                    assert_eq!(
                        run.decode.len(),
                        output.saturating_sub(1),
                        "prefill produces the first token"
                    );
                    assert!(
                        run.decode
                            .iter()
                            .enumerate()
                            .all(|(i, s)| s.position == i
                                && s.context_tokens == i.saturating_add(11)),
                        "context before each one-token input"
                    );
                    assert_eq!(
                        device.forwards.get(..2),
                        Some([(0, 7), (7, 4)].as_slice()),
                        "depth then prompt"
                    );
                    let host = scope == TimingScope::Generation;
                    assert_eq!(
                        (
                            run.first_token_ns.is_some(),
                            run.generation_ns.is_some(),
                            report.metrics.output_tokens_per_s.is_some()
                        ),
                        (host, host, host),
                        "host metrics only in generation pass"
                    );
                    if output == 1 {
                        assert_eq!(
                            run.first_token_ns, run.generation_ns,
                            "same completion event"
                        );
                        assert_eq!(
                            (report.metrics.inter_token_ns, report.metrics.tpot_ns),
                            (None, None),
                            "one output has no interval"
                        );
                    }
                    let intervals = run
                        .decode
                        .iter()
                        .filter_map(|s| s.inter_token_ns)
                        .collect::<Vec<_>>();
                    assert_eq!(
                        intervals.len(),
                        if host { output.saturating_sub(1) } else { 0 },
                        "forward pass does not invent host intervals"
                    );
                    if let (Some(first), Some(last)) = (run.first_token_ns, run.generation_ns) {
                        assert_eq!(
                            intervals.iter().sum::<u64>(),
                            last.saturating_sub(first),
                            "intervals reconstruct the host timeline"
                        );
                    }
                    if let Some(ns) = run.generation_ns {
                        assert!(ns >= run.step_sum_ns(), "wall time includes setup");
                    }
                }
            }
        }
    }

    #[test]
    fn prefill_only_produces_no_tokens_and_decode_only_counts_every_step() {
        for scope in [TimingScope::Generation, TimingScope::Forward] {
            for (prompt, output) in [(4, 0), (0, 1), (0, 128)] {
                let mut device = RecordingDevice::default();
                let config = BenchConfig {
                    depth_token_count: if prompt == 0 { 7 } else { 0 },
                    ..measured_config(prompt, output, scope)
                };
                let report =
                    bench(&mut device, &TestModel, NonZeroU32::MIN, &config).expect("benchmark");
                let run = report.runs.first().expect("one run");
                let output = usize::try_from(output).expect("small output");
                assert_eq!(run.token_ids, vec![vec![1; output]], "exact output count");
                assert_eq!(
                    run.prompt_step_ns.is_some(),
                    prompt > 0,
                    "prompt measurement only with prompt"
                );
                assert_eq!(run.decode.len(), output, "all decode-only steps counted");
                assert!(
                    run.decode
                        .iter()
                        .enumerate()
                        .all(|(i, s)| s.position == i && s.context_tokens == i.saturating_add(7)),
                    "decode-only starts at the initial cache depth"
                );
                assert_eq!(
                    run.decode.first().and_then(|s| s.inter_token_ns),
                    None,
                    "first decode-only token has no predecessor"
                );
                if output == 0 {
                    assert_eq!(
                        (run.first_token_ns, run.generation_ns),
                        (None, None),
                        "no token to time"
                    );
                    assert!(
                        !device.events.contains(&"select"),
                        "no selection in prefill-only pass"
                    );
                }
            }
        }
    }

    #[test]
    fn forward_waits_before_selection_and_generation_waits_at_readback() {
        for (scope, expected) in [
            (
                TimingScope::Generation,
                vec!["ready", "cache", "forward", "select", "ready", "read"],
            ),
            (
                TimingScope::Forward,
                vec![
                    "ready", "cache", "ready", "forward", "ready", "select", "ready", "read",
                ],
            ),
        ] {
            let mut device = RecordingDevice::default();
            let config = measured_config(4, 1, scope);
            bench(&mut device, &TestModel, NonZeroU32::MIN, &config).expect("benchmark");
            assert_eq!(
                device.events, expected,
                "synchronization boundaries for {scope:?}"
            );
        }
    }

    #[test]
    fn completion_clock_stops_at_the_selected_boundary() {
        for scope in [TimingScope::Forward, TimingScope::Generation] {
            let mut device = RecordingDevice::default();
            let mut decoder =
                crate::generate::Decoder::new(&mut device, &TestModel, NonZeroUsize::MIN)
                    .expect("decoder");
            decoder.synchronize().expect("cache ready");
            let step = super::timed_step(&mut decoder, &[0; 4], 0, scope, true).expect("step");
            drop(decoder);
            match scope {
                TimingScope::Forward => assert!(
                    step.completed_at <= device.selected_at.expect("selection"),
                    "forward clock stops before selection"
                ),
                TimingScope::Generation => assert!(
                    step.completed_at >= device.read_at.expect("readback"),
                    "generation clock includes readback"
                ),
            }
        }
    }

    #[test]
    fn synchronization_failure_does_not_produce_a_report() {
        for failure in [1, 2, 3] {
            let mut device = RecordingDevice {
                fail_on_sync: Some(failure),
                ..RecordingDevice::default()
            };
            assert!(
                bench(
                    &mut device,
                    &TestModel,
                    NonZeroU32::MIN,
                    &measured_config(4, 1, TimingScope::Forward)
                )
                .is_err(),
                "sync failure {failure} propagates"
            );
        }
    }
}
