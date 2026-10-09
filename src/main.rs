use std::collections::BTreeSet;
use std::fmt;
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;

use clap::{
    Parser,
    Subcommand,
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
    info,
};

#[derive(Parser)]
#[command(about = "Foundry LLM inference engine.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    #[command(about = "Open a model from the Hugging Face cache and summarize its weights.")]
    Inspect {
        #[arg(long, help = "Hugging Face repository id, as owner/name.")]
        model: ModelId,
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

    #[error("Total byte count of model {model} overflows usize.")]
    ByteCountOverflow { model: String },
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
    tracing_subscriber::fmt().init();
    let result = match Cli::parse().command {
        Command::Inspect {
            model,
        } => inspect(&model),
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

fn inspect(model: &ModelId) -> Result<(), CliError> {
    let directory = resolve_snapshot(model)?;
    let weights = Weights::open(&directory)?;
    let bytes = weights.names().try_fold(0_usize, |bytes, name| {
        let tensor_bytes = weights.get(name)?.bytes.len();
        bytes
            .checked_add(tensor_bytes)
            .ok_or_else(|| CliError::ByteCountOverflow {
                model: model.to_string(),
            })
    })?;
    let dtypes = weights
        .names()
        .map(|name| {
            weights.get(name).map(|view| {
                let dtype = view.dtype;
                format!("{dtype:?}")
            })
        })
        .collect::<Result<BTreeSet<String>, WeightsError>>()?;
    info!(
        message = "Inspected model.",
        %model,
        directory = %directory.display(),
        tensor_count = weights.names().count(),
        bytes,
        ?dtypes,
    );
    Ok(())
}
