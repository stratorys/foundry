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
    LlamaError,
};
use foundry::weights::Weights;
use hf_hub::HFClientSync;
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
    let weights = Weights::open(&directory).map_err(|error| {
        error!(
            message = "Opening the model weights failed.",
            directory = %directory.display(),
            %error,
        );
        CliError::Weights
    })?;
    let summary = summarize_weights(model, &weights)?;
    let config = load_architecture(&directory)?;
    let tensor_rows = tensors.then(|| tensor_rows(&weights));
    write_report(
        &mut io::stdout().lock(),
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

fn write_report(
    out: &mut impl Write,
    model: &ModelId,
    directory: &Path,
    summary: &WeightsSummary,
    config: Option<&LlamaConfig>,
    tensor_rows: Option<&[TensorRow]>,
) -> io::Result<()> {
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
        Err(LlamaError::ModelType) => {
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

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
        Cli,
        CliError,
        Command,
        GIB_BYTES,
        ModelId,
        ModelIdError,
        TensorRow,
        WeightsSummary,
        format_gib,
        summarize_weights,
        tensor_rows,
        write_tensors,
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
