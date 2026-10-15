use std::collections::BTreeSet;
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
    ValueEnum,
};
use foundry::backend::cpu::CpuBackend;
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
use foundry::core::Backend;
use foundry::generate::{
    Generation,
    generate,
};
use foundry::models::llama::{
    Llama,
    LlamaConfig,
    LlamaConfigError,
    LlamaWeights,
};
use foundry::weights::Weights;
use hf_hub::HFClientSync;
use serde::Serialize;
use tracing::{
    error,
    info,
    warn,
};

const GIB_BYTES: usize = 1024 * 1024 * 1024;
const LABEL_WIDTH: usize = 14;
const FIELD_WIDTH: usize = 12;

#[derive(Parser)]
#[command(about = "Foundry LLM inference engine.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
enum Command {
    #[command(
        about = "Open a model from the Hugging Face cache and summarize its weights and \
                 architecture."
    )]
    Inspect {
        #[arg(long, help = "Hugging Face repository id, as owner/name.")]
        model: ModelId,
        #[arg(long, help = "List every tensor with its dtype, shape and size.")]
        tensors: bool,
    },
    #[command(
        about = "Generate token IDs greedily from prompt token IDs on the selected backend and \
                 print them as JSON."
    )]
    Run {
        #[arg(long, help = "Hugging Face repository id, as owner/name.")]
        model: ModelId,
        #[arg(
            long,
            value_enum,
            default_value_t = BackendKind::Cpu,
            help = "Backend that runs the model."
        )]
        backend: BackendKind,
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
        about = "Benchmark prefill and greedy decode at batch 1 on random token IDs and write \
                 report.json."
    )]
    Bench(BenchArgs),
}

#[derive(Debug, PartialEq, Eq, Args)]
struct BenchArgs {
    #[arg(long, help = "Hugging Face repository id, as owner/name.")]
    model: ModelId,
    #[arg(
        long,
        value_enum,
        default_value_t = BackendKind::Cpu,
        help = "Backend that runs the model."
    )]
    backend: BackendKind,
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
    #[arg(long, default_value_t = 0, help = "Seed of the random token IDs.")]
    seed: u64,
    #[arg(long, help = "Directory that receives report.json.")]
    out: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendKind {
    Cpu,
    #[cfg(target_os = "macos")]
    Metal,
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

    #[error("Loading the model weights failed.")]
    Load,

    #[error("Building the model failed.")]
    Model,

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

#[derive(Debug, PartialEq, Eq)]
struct WeightsSummary {
    shard_count: usize,
    tensor_count: usize,
    bytes: usize,
    bytes_declared: Option<u64>,
    dtypes: BTreeSet<String>,
}

#[derive(Debug, PartialEq, Eq)]
struct TensorRow {
    name: String,
    dtype: String,
    shape: String,
    bytes: usize,
}

impl BackendKind {
    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(target_os = "macos")]
            Self::Metal => "metal",
        }
    }
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
        Command::Inspect {
            model,
            tensors,
        } => inspect(&model, tensors),
        Command::Run {
            model,
            backend,
            prompt_ids,
            max_new_tokens,
        } => run(&model, backend, &prompt_ids, max_new_tokens),
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

fn inspect(
    model: &ModelId,
    tensors: bool,
) -> Result<(), CliError> {
    let directory = resolve_snapshot(model)?;
    let weights = open_weights(&directory)?;
    let summary = summarize_weights(model, &weights)?;
    let config = load_architecture(&directory)?;
    let tensor_rows = tensors.then(|| tensor_rows(&weights));
    write_report(
        &mut std::io::stdout().lock(),
        model,
        &directory,
        &summary,
        config.as_ref(),
        tensor_rows.as_deref(),
    )
    .map_err(|error| {
        error!(message = "Writing the report to stdout failed.", %error);
        CliError::Output
    })
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
    backend: BackendKind,
    prompt_ids: &[u32],
    max_new_tokens: usize,
) -> Result<(), CliError> {
    let directory = resolve_snapshot(model)?;
    let config = open_config(&directory)?;
    let weights = open_weights(&directory)?;
    let generation = match backend {
        BackendKind::Cpu => generate_on(
            &mut CpuBackend::new(),
            model,
            &config,
            &weights,
            prompt_ids,
            max_new_tokens,
        ),
        #[cfg(target_os = "macos")]
        BackendKind::Metal => generate_on(
            &mut MetalBackend::new().map_err(|error| {
                error!(message = "Creating the Metal backend failed.", %error);
                CliError::Backend
            })?,
            model,
            &config,
            &weights,
            prompt_ids,
            max_new_tokens,
        ),
    }?;
    info!(
        message = "Generated tokens.",
        ?backend,
        prompt_tokens = prompt_ids.len(),
        generated_tokens = generation.token_ids.len(),
        prefill_ms = %generation.prefill.as_millis(),
        decode_steps = generation.decode_steps.len(),
        decode_ms_total = %generation.decode_steps.iter().sum::<Duration>().as_millis(),
    );
    write_token_ids(&mut std::io::stdout().lock(), &generation.token_ids).map_err(|error| {
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
        seed: args.seed,
    };
    let (device, report) = match args.backend {
        BackendKind::Cpu => bench_on(
            &mut CpuBackend::new(),
            &args.model,
            &config,
            &weights,
            vocab_size,
            &bench_config,
        ),
        #[cfg(target_os = "macos")]
        BackendKind::Metal => bench_on(
            &mut MetalBackend::new().map_err(|error| {
                error!(message = "Creating the Metal backend failed.", %error);
                CliError::Backend
            })?,
            &args.model,
            &config,
            &weights,
            vocab_size,
            &bench_config,
        ),
    }?;
    let record = BenchRecord {
        metadata: Metadata {
            git_commit,
            test_time,
            backend: args.backend.name().to_owned(),
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
        backend = args.backend.name(),
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
    let bytes = summarize_weights(model, weights)?.bytes;
    u64::try_from(bytes).map_err(|_| {
        error!(message = "Total tensor byte count overflows u64.", %model, bytes);
        CliError::TotalByteCountOverflow
    })
}

fn parameter_count<S: AsRef<[u8]>>(
    model: &ModelId,
    weights: &Weights<S>,
) -> Result<u64, CliError> {
    weights.tensors().try_fold(0_u64, |parameter_count, view| {
        u64::try_from(view.shape.element_count())
            .ok()
            .and_then(|element_count| parameter_count.checked_add(element_count))
            .ok_or_else(|| {
                error!(message = "Parameter count overflows u64.", %model);
                CliError::ParameterCountOverflow
            })
    })
}

fn bench_on<B: Backend>(
    backend: &mut B,
    model: &ModelId,
    config: &LlamaConfig,
    weights: &Weights,
    vocab_size: NonZeroU32,
    bench_config: &BenchConfig,
) -> Result<(String, Report), CliError> {
    let seq_len_max = [
        bench_config.depth_token_count,
        bench_config.prompt_token_count,
        bench_config.gen_token_count,
    ]
    .into_iter()
    .try_fold(0_usize, |seq_len_max, token_count| {
        usize::try_from(token_count)
            .ok()
            .and_then(|token_count| seq_len_max.checked_add(token_count))
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
    let llama = load_llama(backend, model, config, weights, seq_len_max)?;
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

fn load_llama<B: Backend>(
    backend: &mut B,
    model: &ModelId,
    config: &LlamaConfig,
    weights: &Weights,
    seq_len_max: usize,
) -> Result<Llama<B>, CliError> {
    info!(message = "Loading the model.", %model);
    let load_start = Instant::now();
    let llama_weights = LlamaWeights::load(backend, config, weights).map_err(|error| {
        error!(message = "Loading the model weights failed.", %model, %error);
        CliError::Load
    })?;
    let llama = Llama::new(backend, config, llama_weights, seq_len_max).map_err(|error| {
        error!(message = "Building the model failed.", %model, seq_len_max, %error);
        CliError::Model
    })?;
    info!(
        message = "Loaded the model.",
        %model,
        seq_len_max,
        load_ms = %load_start.elapsed().as_millis(),
    );
    Ok(llama)
}

fn generate_on<B: Backend>(
    backend: &mut B,
    model: &ModelId,
    config: &LlamaConfig,
    weights: &Weights,
    prompt_ids: &[u32],
    max_new_tokens: usize,
) -> Result<Generation, CliError> {
    let seq_len_max = prompt_ids
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
    let llama = load_llama(backend, model, config, weights, seq_len_max)?;
    generate(
        backend,
        &llama,
        prompt_ids,
        max_new_tokens,
        config.eos_token_ids(),
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

fn write_report(
    out: &mut impl Write,
    model: &ModelId,
    directory: &Path,
    summary: &WeightsSummary,
    config: Option<&LlamaConfig>,
    tensor_rows: Option<&[TensorRow]>,
) -> std::io::Result<()> {
    write_header(out, model, directory)?;
    write_weights(out, summary)?;
    if let Some(config) = config {
        write_architecture(out, config)?;
    }
    if let Some(rows) = tensor_rows {
        write_tensors(out, rows)?;
    }
    Ok(())
}

fn summarize_weights<S: AsRef<[u8]>>(
    model: &ModelId,
    weights: &Weights<S>,
) -> Result<WeightsSummary, CliError> {
    let bytes = weights.tensors().try_fold(0_usize, |bytes, view| {
        bytes.checked_add(view.bytes.len()).ok_or_else(|| {
            error!(message = "Total tensor byte count overflows usize.", %model);
            CliError::TotalByteCountOverflow
        })
    })?;
    let bytes_declared = weights.bytes_declared();
    if let Some(bytes_declared) = bytes_declared
        && u64::try_from(bytes).ok() != Some(bytes_declared)
    {
        error!(
            message = "Tensor byte count does not match the index declaration.",
            %model,
            bytes,
            bytes_declared,
        );
        return Err(CliError::BytesMismatch);
    }
    let dtypes = weights
        .tensors()
        .map(|view| {
            let dtype = view.dtype;
            format!("{dtype:?}")
        })
        .collect::<BTreeSet<String>>();
    Ok(WeightsSummary {
        shard_count: weights.shard_count(),
        tensor_count: weights.tensors().count(),
        bytes,
        bytes_declared,
        dtypes,
    })
}

fn load_architecture(directory: &Path) -> Result<Option<LlamaConfig>, CliError> {
    match LlamaConfig::open(directory) {
        Ok(config) => Ok(Some(config)),
        Err(LlamaConfigError::ModelType) => {
            warn!(message = "Model type is not supported; architecture is skipped.");
            Ok(None)
        }
        Err(error) => {
            error!(
                message = "Opening the model config failed.",
                directory = %directory.display(),
                %error,
            );
            Err(CliError::Config)
        }
    }
}

fn tensor_rows<S: AsRef<[u8]>>(weights: &Weights<S>) -> Vec<TensorRow> {
    let mut rows: Vec<TensorRow> = weights
        .tensors()
        .map(|view| {
            let dtype = view.dtype;
            let dims = view.shape.dims();
            TensorRow {
                name: view.name.to_owned(),
                dtype: format!("{dtype:?}"),
                shape: format!("{dims:?}"),
                bytes: view.bytes.len(),
            }
        })
        .collect();
    rows.sort_unstable_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
    rows
}

fn write_header(
    out: &mut impl Write,
    model: &ModelId,
    directory: &Path,
) -> std::io::Result<()> {
    writeln!(out, "{:<LABEL_WIDTH$}{model}", "Model")?;
    writeln!(out, "{:<LABEL_WIDTH$}{}", "Snapshot", directory.display())
}

fn write_weights(
    out: &mut impl Write,
    summary: &WeightsSummary,
) -> std::io::Result<()> {
    let size_gib = format_gib(summary.bytes);
    let dtypes = summary
        .dtypes
        .iter()
        .map(String::as_str)
        .collect::<Vec<&str>>()
        .join(", ");
    writeln!(out)?;
    writeln!(out, "Weights")?;
    write_field(out, "Shards", summary.shard_count)?;
    write_field(out, "Tensors", summary.tensor_count)?;
    write_field(
        out,
        "Size",
        format_args!(
            "{} bytes ({} GiB)",
            summary.bytes,
            size_gib.as_deref().unwrap_or("?")
        ),
    )?;
    match summary.bytes_declared {
        Some(bytes_declared) => write_field(
            out,
            "Declared",
            format_args!("{bytes_declared} bytes (match)"),
        )?,
        None => write_field(out, "Declared", "no index")?,
    }
    write_field(out, "Dtypes", dtypes)
}

fn write_architecture(
    out: &mut impl Write,
    config: &LlamaConfig,
) -> std::io::Result<()> {
    let rope = config.rope_scaling().map_or_else(
        || format!("theta {}, no scaling", config.rope_theta()),
        |scaling| {
            format!(
                "theta {}, llama3 factor {}, low {}, high {}, original {}",
                config.rope_theta(),
                scaling.factor,
                scaling.low_freq_factor,
                scaling.high_freq_factor,
                scaling.original_max_position_embeddings
            )
        },
    );
    let eos_token_ids = config
        .eos_token_ids()
        .iter()
        .map(u32::to_string)
        .collect::<Vec<String>>()
        .join(", ");
    writeln!(out)?;
    writeln!(out, "{:<LABEL_WIDTH$}llama", "Architecture")?;
    write_field(out, "Layers", config.num_hidden_layers())?;
    write_field(out, "Hidden", config.hidden_size())?;
    write_field(out, "MLP", config.intermediate_size())?;
    write_field(
        out,
        "Heads",
        format_args!(
            "{} query, {} key-value, head dim {}",
            config.num_attention_heads(),
            config.num_key_value_heads(),
            config.head_dim()
        ),
    )?;
    write_field(out, "Vocab", config.vocab_size())?;
    write_field(out, "Positions", config.max_position_embeddings())?;
    write_field(out, "RMS eps", config.rms_norm_eps())?;
    write_field(out, "RoPE", rope)?;
    write_field(out, "EOS", eos_token_ids)
}

fn write_tensors(
    out: &mut impl Write,
    rows: &[TensorRow],
) -> std::io::Result<()> {
    let name_width = rows.iter().map(|row| row.name.len()).max().unwrap_or(0);
    let dtype_width = rows.iter().map(|row| row.dtype.len()).max().unwrap_or(0);
    let shape_width = rows.iter().map(|row| row.shape.len()).max().unwrap_or(0);
    let bytes_width = rows
        .iter()
        .map(|row| row.bytes.to_string().len())
        .max()
        .unwrap_or(0);
    writeln!(out)?;
    writeln!(out, "Tensors")?;
    rows.iter().try_for_each(|row| {
        writeln!(
            out,
            "  {:<name_width$}  {:<dtype_width$}  {:<shape_width$}  {:>bytes_width$}",
            row.name, row.dtype, row.shape, row.bytes
        )
    })
}

fn write_field(
    out: &mut impl Write,
    label: &str,
    value: impl fmt::Display,
) -> std::io::Result<()> {
    writeln!(out, "  {label:<FIELD_WIDTH$}{value}")
}

fn format_gib(bytes: usize) -> Option<String> {
    let whole = bytes.checked_div(GIB_BYTES)?;
    let hundredths = bytes
        .checked_rem(GIB_BYTES)?
        .checked_mul(100)?
        .checked_div(GIB_BYTES)?;
    Some(format!("{whole}.{hundredths:02}"))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
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
        BackendKind,
        BenchArgs,
        Cli,
        CliError,
        Command,
        GIB_BYTES,
        ModelId,
        ModelIdError,
        TensorRow,
        WeightsSummary,
        format_gib,
        parameter_count,
        summarize_weights,
        tensor_rows,
        write_tensors,
        write_token_ids,
        write_weights,
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
    fn inspect_parses_model_and_tensors_flag() {
        struct ArgsTensorsCase<T0, T1> {
            args: T0,
            tensors: T1,
        }
        [
            ArgsTensorsCase {
                args: vec!["foundry", "inspect", "--model", "a/b"],
                tensors: false,
            },
            ArgsTensorsCase {
                args: vec!["foundry", "inspect", "--model", "a/b", "--tensors"],
                tensors: true,
            },
        ]
        .into_iter()
        .for_each(
            |ArgsTensorsCase {
                 args,
                 tensors,
             }| {
                assert_eq!(
                    Cli::try_parse_from(&args)
                        .map(|cli| cli.command)
                        .map_err(|error| error.kind()),
                    Ok(Command::Inspect {
                        model: ModelId {
                            owner: "a".to_owned(),
                            name: "b".to_owned(),
                        },
                        tensors,
                    }),
                    "args {args:?}"
                );
            },
        );
    }

    #[test]
    fn inspect_without_model_is_rejected() {
        assert_eq!(
            Cli::try_parse_from(["foundry", "inspect"])
                .map(|cli| cli.command)
                .map_err(|error| error.kind()),
            Err(ErrorKind::MissingRequiredArgument),
            "--model is required"
        );
    }

    #[test]
    fn inspect_with_invalid_model_is_rejected() {
        assert_eq!(
            Cli::try_parse_from(["foundry", "inspect", "--model", "ab"])
                .map(|cli| cli.command)
                .map_err(|error| error.kind()),
            Err(ErrorKind::ValueValidation),
            "a model id without owner is rejected"
        );
    }

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
                backend: BackendKind::Cpu,
                prompt_ids: vec![1, 2, 3],
                max_new_tokens: 4,
            }),
            "comma-separated prompt IDs on the default cpu backend"
        );
    }

    #[test]
    fn run_parses_the_cpu_backend() {
        assert_eq!(
            Cli::try_parse_from([
                "foundry",
                "run",
                "--model",
                "a/b",
                "--backend",
                "cpu",
                "--prompt-ids",
                "1",
                "--max-new-tokens",
                "1",
            ])
            .map(|cli| cli.command)
            .map_err(|error| error.kind()),
            Ok(Command::Run {
                model: ModelId {
                    owner: "a".to_owned(),
                    name: "b".to_owned(),
                },
                backend: BackendKind::Cpu,
                prompt_ids: vec![1],
                max_new_tokens: 1,
            }),
            "explicit cpu backend"
        );
    }

    #[test]
    fn run_rejects_an_unknown_backend() {
        assert_eq!(
            Cli::try_parse_from([
                "foundry",
                "run",
                "--model",
                "a/b",
                "--backend",
                "gpu",
                "--prompt-ids",
                "1",
                "--max-new-tokens",
                "1",
            ])
            .map(|cli| cli.command)
            .map_err(|error| error.kind()),
            Err(ErrorKind::InvalidValue),
            "gpu is not a backend"
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
                backend: BackendKind::Cpu,
                prompt_tokens: 512,
                gen_tokens: 128,
                depth: 0,
                warmup_runs: 1,
                warmup_ms: 0,
                runs: NonZeroU32::new(10).expect("10 is not zero"),
                seed: 0,
                out: PathBuf::from("reports"),
            })),
            "pg 512/128 with one warmup run and ten measured runs"
        );
    }

    #[test]
    fn bench_rejects_zero_runs_and_missing_arguments() {
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
    fn format_gib_truncates_to_two_decimals() {
        struct BytesExpectedCase<T0, T1> {
            bytes: T0,
            expected: T1,
        }
        [
            BytesExpectedCase {
                bytes: 0,
                expected: "0.00",
            },
            BytesExpectedCase {
                bytes: GIB_BYTES,
                expected: "1.00",
            },
            BytesExpectedCase {
                bytes: 1_610_612_736,
                expected: "1.50",
            },
            BytesExpectedCase {
                bytes: 1_127_428_916,
                expected: "1.05",
            },
            BytesExpectedCase {
                bytes: 1_073_741_823,
                expected: "0.99",
            },
        ]
        .into_iter()
        .for_each(
            |BytesExpectedCase {
                 bytes,
                 expected,
             }| {
                assert_eq!(
                    format_gib(bytes),
                    Some(expected.to_owned()),
                    "{bytes} bytes"
                );
            },
        );
    }

    #[test]
    fn write_weights_prints_declared_size_when_present() {
        struct BytesDeclaredDeclaredLineCase<T0, T1> {
            bytes_declared: T0,
            declared_line: T1,
        }
        [
            BytesDeclaredDeclaredLineCase {
                bytes_declared: Some(16),
                declared_line: "  Declared    16 bytes (match)\n",
            },
            BytesDeclaredDeclaredLineCase {
                bytes_declared: None,
                declared_line: "  Declared    no index\n",
            },
        ]
        .into_iter()
        .for_each(
            |BytesDeclaredDeclaredLineCase {
                 bytes_declared,
                 declared_line,
             }| {
                let mut out = Vec::new();
                write_weights(
                    &mut out,
                    &WeightsSummary {
                        shard_count: 1,
                        tensor_count: 2,
                        bytes: 16,
                        bytes_declared,
                        dtypes: BTreeSet::from(["BF16".to_owned(), "F32".to_owned()]),
                    },
                )
                .expect("writing to a vector succeeds");
                assert_eq!(
                    String::from_utf8(out),
                    Ok(format!(
                        "\nWeights\n  Shards      1\n  Tensors     2\n  Size        16 bytes \
                         (0.00 GiB)\n{declared_line}  Dtypes      BF16, F32\n"
                    )),
                    "declared size {bytes_declared:?}"
                );
            },
        );
    }

    #[test]
    fn write_tensors_aligns_columns() {
        let mut empty = Vec::new();
        write_tensors(&mut empty, &[]).expect("writing to a vector succeeds");
        assert_eq!(
            String::from_utf8(empty),
            Ok("\nTensors\n".to_owned()),
            "no tensor"
        );
        let mut out = Vec::new();
        write_tensors(
            &mut out,
            &[
                TensorRow {
                    name: "a.long".to_owned(),
                    dtype: "BF16".to_owned(),
                    shape: "[2, 3]".to_owned(),
                    bytes: 12,
                },
                TensorRow {
                    name: "b".to_owned(),
                    dtype: "F32".to_owned(),
                    shape: "[2]".to_owned(),
                    bytes: 8,
                },
            ],
        )
        .expect("writing to a vector succeeds");
        assert_eq!(
            String::from_utf8(out),
            Ok("\nTensors\n  a.long  BF16  [2, 3]  12\n  b       F32   [2]      8\n".to_owned()),
            "columns are padded to the widest value"
        );
    }

    #[test]
    fn summarize_weights_counts_shards_tensors_bytes_and_dtypes() {
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
            summarize_weights(
                &ModelId {
                    owner: "a".to_owned(),
                    name: "b".to_owned(),
                },
                &weights,
            ),
            Ok(WeightsSummary {
                shard_count: 1,
                tensor_count: 2,
                bytes: 8,
                bytes_declared: Some(8),
                dtypes: BTreeSet::from(["BF16".to_owned(), "F32".to_owned()]),
            }),
            "summary of one shard with two tensors"
        );
    }

    #[test]
    fn summarize_weights_rejects_a_declared_size_mismatch() {
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
            summarize_weights(
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

    #[test]
    fn tensor_rows_are_sorted_by_name() {
        let header = serde_json::to_vec(&json!({
            "b": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] },
            "a": { "dtype": "BF16", "shape": [2, 1], "data_offsets": [4, 8] },
        }))
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..8)
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
            tensor_rows(&weights),
            vec![
                TensorRow {
                    name: "a".to_owned(),
                    dtype: "BF16".to_owned(),
                    shape: "[2, 1]".to_owned(),
                    bytes: 4,
                },
                TensorRow {
                    name: "b".to_owned(),
                    dtype: "F32".to_owned(),
                    shape: "[1]".to_owned(),
                    bytes: 4,
                },
            ],
            "rows in name order"
        );
    }
}
