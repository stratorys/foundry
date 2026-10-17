use std::fmt;
use std::fs::File;
use std::io::{
    BufWriter,
    Write,
};
use std::num::NonZeroU32;
use std::path::{
    Path,
    PathBuf,
};
use std::process::ExitCode;
use std::str::FromStr;
use std::time::{
    Duration,
    Instant,
    SystemTime,
    UNIX_EPOCH,
};

use clap::{
    Args,
    Parser,
    Subcommand,
};
#[cfg(target_os = "macos")]
use foundry::backend::metal::MetalBackend;
use foundry::bench::{
    BenchConfig,
    Metadata,
    Report,
    bench,
    git_commit,
    rfc3339_utc,
};
use foundry::generate::{
    Generation,
    generate,
};
use foundry::models::llama::{
    LlamaConfig,
    LlamaForCausalLm,
};
use foundry::nn::Backend;
use foundry::weights::Weights;
use hf_hub::HFClientSync;
use serde::Serialize;
use tracing::{
    error,
    info,
    warn,
};

const BACKEND_NAME: &str = "metal";

#[derive(Parser)]
#[command(about = "Foundry LLM inference engine.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
enum Command {
    #[command(
        about = "Generate token IDs greedily from prompt token IDs on Metal and print them as \
                 JSON."
    )]
    Run {
        #[arg(long, help = "Hugging Face repository id, as owner/name.")]
        model: ModelId,
        #[arg(
            long,
            value_delimiter = ',',
            required = true,
            help = "Prompt token IDs, comma-separated."
        )]
        prompt_ids: Vec<u32>,
        #[arg(long, help = "Maximum number of generated tokens.")]
        max_new_tokens: usize,
    },
    #[command(
        about = "Benchmark prefill and greedy decode on random token IDs and write report.json."
    )]
    Bench(BenchArgs),
}

#[derive(Debug, PartialEq, Eq, Args)]
struct BenchArgs {
    #[arg(long, help = "Hugging Face repository id, as owner/name.")]
    model: ModelId,
    #[arg(
        long,
        help = "Prompt tokens processed in one prefill; 0 benchmarks generation only."
    )]
    prompt_tokens: u32,
    #[arg(
        long,
        help = "Decode steps after the prefill; 0 benchmarks the prefill only."
    )]
    gen_tokens: u32,
    #[arg(
        long,
        default_value_t = 0,
        help = "Tokens filled into the cache before each measured run."
    )]
    depth: u32,
    #[arg(long, default_value_t = 1, help = "Minimum number of warmup runs.")]
    warmup_runs: u32,
    #[arg(
        long,
        default_value_t = 0,
        help = "Minimum warmup duration in milliseconds."
    )]
    warmup_ms: u64,
    #[arg(long, default_value = "10", help = "Number of measured runs.")]
    runs: NonZeroU32,
    #[arg(
        long,
        default_value = "1",
        help = "Sequences generated together, each with its own random prompt of the same length."
    )]
    batch_size: NonZeroU32,
    #[arg(long, default_value_t = 0, help = "Seed of the random token IDs.")]
    seed: u64,
    #[arg(long, help = "Directory that receives report.json.")]
    out: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelId {
    owner: String,
    name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("Model id is not of the form owner/name.")]
struct ModelIdError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
enum CliError {
    #[error("Model cannot be resolved from the Hugging Face cache.")]
    Resolve,

    #[error("Opening the model weights failed.")]
    Weights,

    #[error("Opening the model config failed.")]
    Config,

    #[error("Total tensor byte count overflows usize.")]
    TotalByteCountOverflow,

    #[error("Tensor byte count does not match the index declaration.")]
    BytesMismatch,

    #[error("Writing the report to stdout failed.")]
    Output,

    #[cfg(target_os = "macos")]
    #[error("Creating the Metal backend failed.")]
    Backend,

    #[cfg(not(target_os = "macos"))]
    #[error("No backend is available on this platform.")]
    BackendUnavailable,

    #[error("Loading the model failed.")]
    Load,

    #[error("Prompt and generated token counts overflow usize.")]
    SeqLenOverflow,

    #[error("Generation failed.")]
    Generate,

    #[error("Vocabulary size is zero or does not fit in u32.")]
    VocabSize,

    #[error("Benchmark failed.")]
    Bench,

    #[error("Writing the benchmark report failed.")]
    Report,

    #[error("System clock cannot be formatted as an RFC 3339 time.")]
    Timestamp,

    #[error("Parameter count overflows u64.")]
    ParameterCountOverflow,
}

#[derive(Debug, Serialize)]
struct BenchRecord {
    metadata: Metadata,
    #[serde(flatten)]
    report: Report,
}

impl FromStr for ModelId {
    type Err = ModelIdError;

    fn from_str(model: &str) -> Result<Self, Self::Err> {
        match model.split_once('/') {
            Some((owner, name)) if !owner.is_empty() && !name.is_empty() && !name.contains('/') => {
                Ok(Self {
                    owner: owner.to_owned(),
                    name: name.to_owned(),
                })
            }
            _ => {
                error!(message = "Model id is not of the form owner/name.", model);
                Err(ModelIdError)
            }
        }
    }
}

impl fmt::Display for ModelId {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(formatter, "{}/{}", self.owner, self.name)
    }
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();
    let result = match Cli::parse().command {
        Command::Run {
            model,
            prompt_ids,
            max_new_tokens,
        } => run(&model, &prompt_ids, max_new_tokens),
        Command::Bench(args) => bench_command(&args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            error!(message = "Command failed.", %error);
            ExitCode::FAILURE
        }
    }
}

fn resolve_snapshot(model: &ModelId) -> Result<PathBuf, CliError> {
    let resolve_error = |error| {
        error!(
            message = "Model cannot be resolved from the Hugging Face cache.",
            %model,
            %error,
        );
        CliError::Resolve
    };
    HFClientSync::new()
        .map_err(resolve_error)?
        .model(model.owner.as_str(), model.name.as_str())
        .snapshot_download()
        .local_files_only(true)
        .send()
        .map_err(resolve_error)
}

fn open_weights(directory: &Path) -> Result<Weights, CliError> {
    Weights::open(directory).map_err(|error| {
        error!(
            message = "Opening the model weights failed.",
            directory = %directory.display(),
            %error,
        );
        CliError::Weights
    })
}

fn run(
    model: &ModelId,
    prompt_ids: &[u32],
    max_new_tokens: usize,
) -> Result<(), CliError> {
    let directory = resolve_snapshot(model)?;
    let config = open_config(&directory)?;
    let weights = open_weights(&directory)?;
    let generation = generate_on_platform(model, &config, &weights, prompt_ids, max_new_tokens)?;
    let token_ids = generation.token_ids.concat();
    info!(
        message = "Generated tokens.",
        backend = BACKEND_NAME,
        prompt_tokens = prompt_ids.len(),
        generated_tokens = token_ids.len(),
        prefill_ms = %generation.prefill.as_millis(),
        decode_steps = generation.decode_steps.len(),
        decode_ms_total = %generation.decode_steps.iter().sum::<Duration>().as_millis(),
    );
    write_token_ids(&mut std::io::stdout().lock(), &token_ids).map_err(|error| {
        error!(message = "Writing the token IDs to stdout failed.", %error);
        CliError::Output
    })
}

fn open_config(directory: &Path) -> Result<LlamaConfig, CliError> {
    LlamaConfig::open(directory).map_err(|error| {
        error!(
            message = "Opening the model config failed.",
            directory = %directory.display(),
            %error,
        );
        CliError::Config
    })
}

fn bench_command(args: &BenchArgs) -> Result<(), CliError> {
    let test_time = test_time()?;
    let git_commit = read_git_commit();
    let directory = resolve_snapshot(&args.model)?;
    let config = open_config(&directory)?;
    let weights = open_weights(&directory)?;
    let model_bytes = model_bytes(&args.model, &weights)?;
    let model_parameter_count = parameter_count(&args.model, &weights)?;
    let vocab_size = u32::try_from(config.vocab_size())
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| {
            error!(
                message = "Vocabulary size is zero or does not fit in u32.",
                vocab_size = config.vocab_size()
            );
            CliError::VocabSize
        })?;
    let bench_config = BenchConfig {
        prompt_token_count: args.prompt_tokens,
        gen_token_count: args.gen_tokens,
        depth_token_count: args.depth,
        warmup_run_count: args.warmup_runs,
        warmup_ms: args.warmup_ms,
        run_count: args.runs,
        batch_size: args.batch_size,
        seed: args.seed,
    };
    let (device, report) =
        bench_on_platform(&args.model, &config, &weights, vocab_size, &bench_config)?;
    let record = BenchRecord {
        metadata: Metadata {
            git_commit,
            test_time,
            backend: BACKEND_NAME.to_owned(),
            device,
            model_id: args.model.to_string(),
            model_bytes,
            model_parameter_count,
        },
        report,
    };
    let report_path = write_bench_report(&args.out, &record)?;
    info!(
        message = "Benchmarked the model.",
        backend = BACKEND_NAME,
        device = %record.metadata.device,
        git_commit = ?record.metadata.git_commit,
        mode = ?record.report.mode,
        llama_cpp_avg_ts = record.report.metrics.llama_cpp.avg_ts,
        mlx_prompt_tps = ?record.report.metrics.mlx.as_ref().map(|mlx| mlx.prompt_tps),
        mlx_generation_tps = ?record.report.metrics.mlx.as_ref().and_then(|mlx| mlx.generation_tps),
        peak_memory_bytes = ?record.report.peak_memory_bytes,
        peak_memory_source = ?record.report.peak_memory_source,
        report = %report_path.display(),
    );
    Ok(())
}

fn test_time() -> Result<String, CliError> {
    let since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            error!(message = "System clock is before the Unix epoch.", %error);
            CliError::Timestamp
        })?;
    rfc3339_utc(since_epoch.as_secs()).ok_or_else(|| {
        error!(
            message = "System clock cannot be formatted as an RFC 3339 time.",
            unix_s = since_epoch.as_secs()
        );
        CliError::Timestamp
    })
}

fn read_git_commit() -> Option<String> {
    let output = match std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            warn!(message = "Running git failed; the commit is not recorded.", %error);
            return None;
        }
    };
    let commit = output
        .status
        .success()
        .then(|| git_commit(&output.stdout))
        .flatten();
    if commit.is_none() {
        warn!(
            message = "Git returned no commit; the commit is not recorded.",
            status = %output.status,
        );
    }
    commit
}

fn model_bytes<S: AsRef<[u8]>>(
    model: &ModelId,
    weights: &Weights<S>,
) -> Result<u64, CliError> {
    let bytes = weights.tensors().try_fold(0_u64, |bytes, view| {
        u64::try_from(view.bytes.len())
            .ok()
            .and_then(|tensor_bytes| bytes.checked_add(tensor_bytes))
            .ok_or_else(|| {
                error!(message = "Total tensor byte count overflows u64.", %model);
                CliError::TotalByteCountOverflow
            })
    })?;
    if let Some(total_size) = weights.total_size()
        && bytes != total_size
    {
        error!(
            message = "Tensor byte count does not match the index declaration.",
            %model,
            bytes,
            total_size,
        );
        return Err(CliError::BytesMismatch);
    }
    Ok(bytes)
}

fn parameter_count<S: AsRef<[u8]>>(
    model: &ModelId,
    weights: &Weights<S>,
) -> Result<u64, CliError> {
    weights.tensors().try_fold(0_u64, |parameter_count, view| {
        u64::try_from(view.shape.numel())
            .ok()
            .and_then(|numel| parameter_count.checked_add(numel))
            .ok_or_else(|| {
                error!(message = "Parameter count overflows u64.", %model);
                CliError::ParameterCountOverflow
            })
    })
}

#[cfg(target_os = "macos")]
fn metal_backend() -> Result<MetalBackend, CliError> {
    MetalBackend::new().map_err(|error| {
        error!(message = "Creating the Metal backend failed.", %error);
        CliError::Backend
    })
}

#[cfg(target_os = "macos")]
fn generate_on_platform(
    model: &ModelId,
    config: &LlamaConfig,
    weights: &Weights,
    prompt_ids: &[u32],
    max_new_tokens: usize,
) -> Result<Generation, CliError> {
    generate_on(
        &mut metal_backend()?,
        model,
        config,
        weights,
        prompt_ids,
        max_new_tokens,
    )
}

#[cfg(not(target_os = "macos"))]
fn generate_on_platform(
    _model: &ModelId,
    _config: &LlamaConfig,
    _weights: &Weights,
    _prompt_ids: &[u32],
    _max_new_tokens: usize,
) -> Result<Generation, CliError> {
    error!(message = "No backend is available on this platform.");
    Err(CliError::BackendUnavailable)
}

#[cfg(target_os = "macos")]
fn bench_on_platform(
    model: &ModelId,
    config: &LlamaConfig,
    weights: &Weights,
    vocab_size: NonZeroU32,
    bench_config: &BenchConfig,
) -> Result<(String, Report), CliError> {
    bench_on(
        &mut metal_backend()?,
        model,
        config,
        weights,
        vocab_size,
        bench_config,
    )
}

#[cfg(not(target_os = "macos"))]
fn bench_on_platform(
    _model: &ModelId,
    _config: &LlamaConfig,
    _weights: &Weights,
    _vocab_size: NonZeroU32,
    _bench_config: &BenchConfig,
) -> Result<(String, Report), CliError> {
    error!(message = "No backend is available on this platform.");
    Err(CliError::BackendUnavailable)
}

#[cfg_attr(
    not(target_os = "macos"),
    expect(dead_code, reason = "no backend on this platform")
)]
fn bench_on<B: Backend>(
    backend: &mut B,
    model: &ModelId,
    config: &LlamaConfig,
    weights: &Weights,
    vocab_size: NonZeroU32,
    bench_config: &BenchConfig,
) -> Result<(String, Report), CliError> {
    let max_cache_len = [
        bench_config.depth_token_count,
        bench_config.prompt_token_count,
        bench_config.gen_token_count,
    ]
    .into_iter()
    .try_fold(0_usize, |max_cache_len, token_count| {
        usize::try_from(token_count)
            .ok()
            .and_then(|token_count| max_cache_len.checked_add(token_count))
    })
    .ok_or_else(|| {
        error!(
            message = "Benchmark token counts overflow usize.",
            ?bench_config
        );
        CliError::SeqLenOverflow
    })?;
    let device = backend.device_name();
    info!(message = "Selected the device.", %device);
    let llama = load_llama(backend, model, config, weights, max_cache_len)?;
    let report = bench(backend, &llama, vocab_size, bench_config).map_err(|error| {
        error!(message = "Benchmark failed.", %model, %error);
        CliError::Bench
    })?;
    Ok((device, report))
}

fn write_bench_report(
    out: &Path,
    record: &BenchRecord,
) -> Result<PathBuf, CliError> {
    let report_path = out.join("report.json");
    let write_error = |error: &dyn std::error::Error| {
        error!(
            message = "Writing the benchmark report failed.",
            path = %report_path.display(),
            %error,
        );
        CliError::Report
    };
    std::fs::create_dir_all(out).map_err(|error| write_error(&error))?;
    let mut writer =
        BufWriter::new(File::create(&report_path).map_err(|error| write_error(&error))?);
    serde_json::to_writer_pretty(&mut writer, record).map_err(|error| write_error(&error))?;
    writeln!(writer)
        .and_then(|()| writer.flush())
        .map_err(|error| write_error(&error))?;
    Ok(report_path)
}

#[cfg_attr(
    not(target_os = "macos"),
    expect(dead_code, reason = "no backend on this platform")
)]
fn load_llama<B: Backend>(
    backend: &mut B,
    model: &ModelId,
    config: &LlamaConfig,
    weights: &Weights,
    max_cache_len: usize,
) -> Result<LlamaForCausalLm<B>, CliError> {
    info!(message = "Loading the model.", %model);
    let load_start = Instant::now();
    let llama =
        LlamaForCausalLm::load(backend, config, weights, max_cache_len).map_err(|error| {
            error!(message = "Loading the model failed.", %model, max_cache_len, %error);
            CliError::Load
        })?;
    info!(
        message = "Loaded the model.",
        %model,
        max_cache_len,
        load_ms = %load_start.elapsed().as_millis(),
    );
    Ok(llama)
}

#[cfg_attr(
    not(target_os = "macos"),
    expect(dead_code, reason = "no backend on this platform")
)]
fn generate_on<B: Backend>(
    backend: &mut B,
    model: &ModelId,
    config: &LlamaConfig,
    weights: &Weights,
    prompt_ids: &[u32],
    max_new_tokens: usize,
) -> Result<Generation, CliError> {
    let max_cache_len = prompt_ids
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| {
            error!(
                message = "Prompt and generated token counts overflow usize.",
                prompt_tokens = prompt_ids.len(),
                max_new_tokens,
            );
            CliError::SeqLenOverflow
        })?;
    let llama = load_llama(backend, model, config, weights, max_cache_len)?;
    generate(
        backend,
        &llama,
        &[prompt_ids],
        max_new_tokens,
        config.eos_token_id(),
    )
    .map_err(|error| {
        error!(message = "Generation failed.", %model, %error);
        CliError::Generate
    })
}

fn write_token_ids(
    out: &mut impl Write,
    token_ids: &[u32],
) -> std::io::Result<()> {
    let token_ids = token_ids
        .iter()
        .map(u32::to_string)
        .collect::<Vec<String>>()
        .join(",");
    writeln!(out, "[{token_ids}]")
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;
    use std::path::PathBuf;

    use clap::error::ErrorKind;
    use clap::{
        CommandFactory,
        Parser,
    };
    use foundry::weights::{
        Shard,
        Weights,
    };
    use serde_json::json;

    use crate::{
        BenchArgs,
        Cli,
        CliError,
        Command,
        ModelId,
        ModelIdError,
        model_bytes,
        parameter_count,
        write_token_ids,
    };

    #[test]
    fn model_id_parses_owner_and_name() {
        let model = "a/b".parse::<ModelId>();
        assert_eq!(
            model,
            Ok(ModelId {
                owner: "a".to_owned(),
                name: "b".to_owned(),
            }),
            "owner and name"
        );
        assert_eq!(
            model.map(|model| model.to_string()),
            Ok("a/b".to_owned()),
            "display gives back owner/name"
        );
    }

    #[test]
    fn model_id_rejects_malformed_values() {
        ["ab", "/b", "a/", "a/b/c", ""]
            .into_iter()
            .for_each(|model| {
                assert_eq!(
                    model.parse::<ModelId>(),
                    Err(ModelIdError),
                    "model id {model:?} is rejected"
                );
            });
    }

    #[test]
    fn cli_definition_is_consistent() { Cli::command().debug_assert(); }

    #[test]
    fn missing_subcommand_displays_the_help() {
        assert_eq!(
            Cli::try_parse_from(["foundry"])
                .map(|cli| cli.command)
                .map_err(|error| error.kind()),
            Err(ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand),
            "without a subcommand the help is shown"
        );
    }

    #[test]
    fn unknown_subcommand_is_rejected() {
        assert_eq!(
            Cli::try_parse_from(["foundry", "load"])
                .map(|cli| cli.command)
                .map_err(|error| error.kind()),
            Err(ErrorKind::InvalidSubcommand),
            "load is not a subcommand"
        );
    }

    #[test]
    fn run_parses_model_prompt_ids_and_max_new_tokens() {
        assert_eq!(
            Cli::try_parse_from([
                "foundry",
                "run",
                "--model",
                "a/b",
                "--prompt-ids",
                "1,2,3",
                "--max-new-tokens",
                "4",
            ])
            .map(|cli| cli.command)
            .map_err(|error| error.kind()),
            Ok(Command::Run {
                model: ModelId {
                    owner: "a".to_owned(),
                    name: "b".to_owned(),
                },
                prompt_ids: vec![1, 2, 3],
                max_new_tokens: 4,
            }),
            "comma-separated prompt IDs"
        );
    }

    #[test]
    fn run_rejects_the_backend_option() {
        assert_eq!(
            Cli::try_parse_from([
                "foundry",
                "run",
                "--model",
                "a/b",
                "--backend",
                "metal",
                "--prompt-ids",
                "1",
                "--max-new-tokens",
                "1",
            ])
            .map(|cli| cli.command)
            .map_err(|error| error.kind()),
            Err(ErrorKind::UnknownArgument),
            "the backend is not selectable"
        );
    }

    #[test]
    fn run_rejects_missing_or_malformed_prompt_ids() {
        struct ArgsKindCase<T0, T1> {
            args: T0,
            kind: T1,
        }
        [
            ArgsKindCase {
                args: vec!["foundry", "run", "--model", "a/b", "--max-new-tokens", "4"],
                kind: ErrorKind::MissingRequiredArgument,
            },
            ArgsKindCase {
                args: vec![
                    "foundry",
                    "run",
                    "--model",
                    "a/b",
                    "--prompt-ids",
                    "1,x",
                    "--max-new-tokens",
                    "4",
                ],
                kind: ErrorKind::ValueValidation,
            },
        ]
        .into_iter()
        .for_each(
            |ArgsKindCase {
                 args,
                 kind,
             }| {
                assert_eq!(
                    Cli::try_parse_from(&args)
                        .map(|cli| cli.command)
                        .map_err(|error| error.kind()),
                    Err(kind),
                    "args {args:?}"
                );
            },
        );
    }

    #[test]
    fn bench_parses_counts_with_defaults() {
        assert_eq!(
            Cli::try_parse_from([
                "foundry",
                "bench",
                "--model",
                "a/b",
                "--prompt-tokens",
                "512",
                "--gen-tokens",
                "128",
                "--out",
                "reports",
            ])
            .map(|cli| cli.command)
            .map_err(|error| error.kind()),
            Ok(Command::Bench(BenchArgs {
                model: ModelId {
                    owner: "a".to_owned(),
                    name: "b".to_owned(),
                },
                prompt_tokens: 512,
                gen_tokens: 128,
                depth: 0,
                warmup_runs: 1,
                warmup_ms: 0,
                runs: NonZeroU32::new(10).expect("10 is not zero"),
                batch_size: NonZeroU32::MIN,
                seed: 0,
                out: PathBuf::from("reports"),
            })),
            "pg 512/128 with one warmup run and ten measured runs"
        );
    }

    #[test]
    fn bench_parses_the_batch_size() {
        assert_eq!(
            Cli::try_parse_from([
                "foundry",
                "bench",
                "--model",
                "a/b",
                "--prompt-tokens",
                "512",
                "--gen-tokens",
                "128",
                "--batch-size",
                "8",
                "--out",
                "reports",
            ])
            .map(|cli| match cli.command {
                Command::Bench(args) => Some(args.batch_size),
                Command::Run {
                    ..
                } => None,
            })
            .map_err(|error| error.kind()),
            Ok(NonZeroU32::new(8)),
            "a batch of eight sequences"
        );
    }

    #[test]
    fn bench_rejects_zero_runs_zero_batch_and_missing_arguments() {
        struct ArgsKindCase<T0, T1> {
            args: T0,
            kind: T1,
        }
        let base = [
            "foundry",
            "bench",
            "--model",
            "a/b",
            "--prompt-tokens",
            "1",
            "--gen-tokens",
            "1",
        ];
        [
            ArgsKindCase {
                args: [base.as_slice(), &["--out", "r", "--runs", "0"]].concat(),
                kind: ErrorKind::ValueValidation,
            },
            ArgsKindCase {
                args: [base.as_slice(), &["--out", "r", "--batch-size", "0"]].concat(),
                kind: ErrorKind::ValueValidation,
            },
            ArgsKindCase {
                args: base.to_vec(),
                kind: ErrorKind::MissingRequiredArgument,
            },
            ArgsKindCase {
                args: vec!["foundry", "bench", "--model", "a/b", "--out", "r"],
                kind: ErrorKind::MissingRequiredArgument,
            },
        ]
        .into_iter()
        .for_each(
            |ArgsKindCase {
                 args,
                 kind,
             }| {
                assert_eq!(
                    Cli::try_parse_from(&args)
                        .map(|cli| cli.command)
                        .map_err(|error| error.kind()),
                    Err(kind),
                    "args {args:?}"
                );
            },
        );
    }

    #[test]
    fn write_token_ids_prints_a_json_array() {
        let mut out = Vec::new();
        write_token_ids(&mut out, &[1, 2, 3]).expect("writing to a vector succeeds");
        assert_eq!(
            String::from_utf8(out),
            Ok("[1,2,3]\n".to_owned()),
            "token IDs as a JSON array"
        );
    }

    #[test]
    fn model_bytes_sums_tensor_bytes() {
        let header = serde_json::to_vec(&json!({
            "a": { "dtype": "BF16", "shape": [2], "data_offsets": [0, 4] },
            "b": { "dtype": "F32", "shape": [1], "data_offsets": [4, 8] },
        }))
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..8)
            .collect();
        let index = serde_json::to_vec(&json!({
            "metadata": { "total_size": 8 },
            "weight_map": { "a": "s1.safetensors", "b": "s1.safetensors" },
        }))
        .expect("the index serializes");
        let weights = Weights::from_shards(
            Some(&index),
            vec![Shard {
                name: "s1.safetensors".to_owned(),
                bytes: file,
            }],
        )
        .expect("the shard is valid");
        assert_eq!(
            model_bytes(
                &ModelId {
                    owner: "a".to_owned(),
                    name: "b".to_owned(),
                },
                &weights,
            ),
            Ok(8),
            "two tensors of four bytes matching the declared size"
        );
    }

    #[test]
    fn model_bytes_rejects_a_declared_size_mismatch() {
        let header = serde_json::to_vec(
            &json!({ "a": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] } }),
        )
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..4)
            .collect();
        let index = serde_json::to_vec(&json!({
            "metadata": { "total_size": 5 },
            "weight_map": { "a": "s1.safetensors" },
        }))
        .expect("the index serializes");
        let weights = Weights::from_shards(
            Some(&index),
            vec![Shard {
                name: "s1.safetensors".to_owned(),
                bytes: file,
            }],
        )
        .expect("the shard is valid");
        assert_eq!(
            model_bytes(
                &ModelId {
                    owner: "a".to_owned(),
                    name: "b".to_owned(),
                },
                &weights,
            ),
            Err(CliError::BytesMismatch),
            "the index declares one byte more than the tensors hold"
        );
    }

    #[test]
    fn parameter_count_sums_the_elements_of_every_tensor() {
        let header = serde_json::to_vec(&json!({
            "a": { "dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12] },
            "b": { "dtype": "F32", "shape": [2], "data_offsets": [12, 20] },
        }))
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..20)
            .collect();
        let weights = Weights::from_shards(
            None,
            vec![Shard {
                name: "model.safetensors".to_owned(),
                bytes: file,
            }],
        )
        .expect("the file is valid");
        assert_eq!(
            parameter_count(
                &ModelId {
                    owner: "a".to_owned(),
                    name: "b".to_owned(),
                },
                &weights,
            ),
            Ok(8),
            "six bf16 and two f32 elements"
        );
    }
}
