use std::fs;
use std::path::Path;

use crate::checkpoint::Checkpoint;
use crate::config::LlamaConfig;
use crate::error::LlamaError;
use crate::manifest::{
    CONFIG_FILE,
    Manifest,
    WEIGHTS_FILE,
};
use crate::metal::context::{
    Buffer,
    Context,
    allocated_size,
};
use crate::metal::session::Session;
use crate::metal::weights::ResidentWeights;
use crate::rope::llama3_frequencies;
use crate::safetensors::{
    Header,
    read_header,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelMemory {
    pub weight_tensor_bytes: u64,
    pub weight_buffer_bytes: usize,
    pub rope_buffer_bytes: usize,
}

pub struct LlamaModel {
    context: Context,
    config: LlamaConfig,
    weights: ResidentWeights,
    frequencies: Buffer,
}

fn read_text(path: &Path) -> Result<String, LlamaError> {
    fs::read_to_string(path).map_err(|error| LlamaError::io(path, error))
}

fn file_bytes(path: &Path) -> Result<u64, LlamaError> {
    fs::metadata(path)
        .map(|metadata| metadata.len())
        .map_err(|error| LlamaError::io(path, error))
}

impl LlamaModel {
    pub fn load(
        snapshot: &Path,
        manifest: &Path,
    ) -> Result<Self, LlamaError> {
        let manifest = Manifest::parse(&read_text(manifest)?)?;
        let config_path = snapshot.join(CONFIG_FILE);
        let weights_path = snapshot.join(WEIGHTS_FILE);
        manifest.check_file_size(CONFIG_FILE, file_bytes(&config_path)?)?;
        manifest.check_file_size(WEIGHTS_FILE, file_bytes(&weights_path)?)?;
        let header = read_header(&weights_path)?;
        manifest.check_header(&header)?;
        Self::from_parts(&read_text(&config_path)?, &weights_path, &header)
    }

    pub fn load_unpinned(
        config: &Path,
        weights: &Path,
    ) -> Result<Self, LlamaError> {
        let header = read_header(weights)?;
        Self::from_parts(&read_text(config)?, weights, &header)
    }

    fn from_parts(
        config: &str,
        weights_path: &Path,
        header: &Header,
    ) -> Result<Self, LlamaError> {
        let config = LlamaConfig::parse(config)?;
        let checkpoint = Checkpoint::from_entries(&config, &header.tensors)?;
        let context = Context::new()?;
        let weights = ResidentWeights::load(&context, weights_path, header, &checkpoint)?;
        let frequencies =
            context.shared_from(&llama3_frequencies(&config.rope, config.head_dim))?;
        Ok(Self {
            context,
            config,
            weights,
            frequencies,
        })
    }

    pub fn config(&self) -> &LlamaConfig { &self.config }

    pub fn session(&self) -> Result<Session<'_>, LlamaError> { Session::new(self) }

    pub fn device_name(&self) -> String { self.context.device_name() }

    pub fn device_allocated_bytes(&self) -> usize { self.context.allocated_bytes() }

    pub fn synchronize(&self) -> Result<(), LlamaError> { Ok(self.context.synchronize()?) }

    pub fn memory(&self) -> ModelMemory {
        ModelMemory {
            weight_tensor_bytes: self.weights.tensor_bytes,
            weight_buffer_bytes: self.weights.allocated_bytes(),
            rope_buffer_bytes: allocated_size(&self.frequencies),
        }
    }

    pub(crate) fn context(&self) -> &Context { &self.context }

    pub(crate) fn weights(&self) -> &ResidentWeights { &self.weights }

    pub(crate) fn frequencies(&self) -> &Buffer { &self.frequencies }
}
