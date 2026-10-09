use std::collections::BTreeSet;
use std::fmt;
use std::io::{
    self,
    Write,
};
use std::path::{
    Path,
    PathBuf,
};
use std::process::ExitCode;
use std::str::FromStr;

use clap::{
    Parser,
    Subcommand,
};
use foundry::models::llama::{
    LlamaConfig,
    LlamaConfigError,
};
use foundry::weights::{
    Weights,
    WeightsError,
};
use hf_hub::{
    HFClientSync,
    HFError,
};
use tracing::{
    error,
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

#[derive(Subcommand)]
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
}

#[derive(Debug, Clone)]
struct ModelId {
    owner: String,
    name: String,
}

#[derive(Debug, thiserror::Error)]
#[error("Model id {model:?} is not of the form owner/name.")]
struct ModelIdError {
    model: String,
}

#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error("Model {model} cannot be resolved from the Hugging Face cache: {error}")]
    Resolve {
        model: String,
        #[source]
        error: HFError,
    },

    #[error(transparent)]
    Weights(#[from] WeightsError),

    #[error(transparent)]
    Config(#[from] LlamaConfigError),

    #[error("Total byte count of model {model} overflows usize.")]
    ByteCountOverflow { model: String },

    #[error("Model {model} has {bytes} bytes of tensors; its index declares {bytes_declared}.")]
    BytesMismatch {
        model: String,
        bytes: usize,
        bytes_declared: u64,
    },

    #[error("Writing the report to stdout failed.")]
    Output(#[from] io::Error),
}

struct WeightsSummary {
    shard_count: usize,
    tensor_count: usize,
    bytes: usize,
    bytes_declared: Option<u64>,
    dtypes: BTreeSet<String>,
}

struct TensorRow {
    name: String,
    dtype: String,
    shape: String,
    bytes: usize,
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
            _ => Err(ModelIdError {
                model: model.to_owned(),
            }),
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
        .with_writer(io::stderr)
        .with_target(false)
        .init();
    let result = match Cli::parse().command {
        Command::Inspect {
            model,
            tensors,
        } => inspect(&model, tensors),
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
    let resolve_error = |error| CliError::Resolve {
        model: model.to_string(),
        error,
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
    let weights = Weights::open(&directory)?;
    let summary = summarize_weights(model, &weights)?;
    let config = load_architecture(&directory)?;
    let tensor_rows = if tensors {
        Some(tensor_rows(&weights)?)
    } else {
        None
    };
    let mut out = io::stdout().lock();
    write_header(&mut out, model, &directory)?;
    write_weights(&mut out, &summary)?;
    if let Some(config) = &config {
        write_architecture(&mut out, config)?;
    }
    if let Some(rows) = &tensor_rows {
        write_tensors(&mut out, rows)?;
    }
    Ok(())
}

fn summarize_weights(
    model: &ModelId,
    weights: &Weights,
) -> Result<WeightsSummary, CliError> {
    let bytes = weights.names().try_fold(0_usize, |bytes, name| {
        let tensor_bytes = weights.get(name)?.bytes.len();
        bytes
            .checked_add(tensor_bytes)
            .ok_or_else(|| CliError::ByteCountOverflow {
                model: model.to_string(),
            })
    })?;
    let bytes_declared = weights.bytes_declared();
    if let Some(bytes_declared) = bytes_declared
        && u64::try_from(bytes).ok() != Some(bytes_declared)
    {
        return Err(CliError::BytesMismatch {
            model: model.to_string(),
            bytes,
            bytes_declared,
        });
    }
    let dtypes = weights
        .names()
        .map(|name| {
            weights.get(name).map(|view| {
                let dtype = view.dtype;
                format!("{dtype:?}")
            })
        })
        .collect::<Result<BTreeSet<String>, WeightsError>>()?;
    Ok(WeightsSummary {
        shard_count: weights.shard_count(),
        tensor_count: weights.names().count(),
        bytes,
        bytes_declared,
        dtypes,
    })
}

fn load_architecture(directory: &Path) -> Result<Option<LlamaConfig>, CliError> {
    match LlamaConfig::open(directory) {
        Ok(config) => Ok(Some(config)),
        Err(LlamaConfigError::ModelType {
            model_type,
        }) => {
            warn!(
                message = "Model type is not supported; architecture is skipped.",
                %model_type,
            );
            Ok(None)
        }
        Err(error) => Err(error.into()),
    }
}

fn tensor_rows(weights: &Weights) -> Result<Vec<TensorRow>, WeightsError> {
    let mut names: Vec<&str> = weights.names().collect();
    names.sort_unstable();
    names
        .into_iter()
        .map(|name| {
            let view = weights.get(name)?;
            let dtype = view.dtype;
            let dims = view.shape.dims();
            Ok(TensorRow {
                name: name.to_owned(),
                dtype: format!("{dtype:?}"),
                shape: format!("{dims:?}"),
                bytes: view.bytes.len(),
            })
        })
        .collect()
}

fn write_header(
    out: &mut impl Write,
    model: &ModelId,
    directory: &Path,
) -> io::Result<()> {
    writeln!(out, "{:<LABEL_WIDTH$}{model}", "Model")?;
    writeln!(out, "{:<LABEL_WIDTH$}{}", "Snapshot", directory.display())
}

fn write_weights(
    out: &mut impl Write,
    summary: &WeightsSummary,
) -> io::Result<()> {
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
) -> io::Result<()> {
    let rope = config.rope_scaling();
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
    write_field(
        out,
        "RoPE",
        format_args!(
            "theta {}, llama3 factor {}, low {}, high {}, original {}",
            config.rope_theta(),
            rope.factor,
            rope.low_freq_factor,
            rope.high_freq_factor,
            rope.original_max_position_embeddings
        ),
    )?;
    write_field(out, "EOS", eos_token_ids)
}

fn write_tensors(
    out: &mut impl Write,
    rows: &[TensorRow],
) -> io::Result<()> {
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
) -> io::Result<()> {
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
