use std::path::{
    Path,
    PathBuf,
};
use std::time::Instant;

use foundry::models::llama::{
    LlamaConfig,
    LlamaForCausalLm,
};
use foundry::nn::Device;
use foundry::weights::Weights;
use hf_hub::HFClientSync;
use tracing::{
    error,
    info,
};

use crate::cli::ModelId;
use crate::cli::error::CliError;

pub struct Snapshot {
    pub config: LlamaConfig,
    pub weights: Weights,
}

impl Snapshot {
    pub fn open(model: &ModelId) -> Result<Self, CliError> {
        let directory = resolve_snapshot(model)?;
        let config = open_config(&directory)?;
        let weights = open_weights(&directory)?;
        Ok(Self {
            config,
            weights,
        })
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

pub fn load_llama<D: Device>(
    device: &mut D,
    model: &ModelId,
    snapshot: &Snapshot,
    max_cache_len: usize,
) -> Result<LlamaForCausalLm<D>, CliError> {
    info!(message = "Loading the model.", %model);
    let load_start = Instant::now();
    let llama = LlamaForCausalLm::load(device, &snapshot.config, &snapshot.weights, max_cache_len)
        .map_err(|error| {
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
