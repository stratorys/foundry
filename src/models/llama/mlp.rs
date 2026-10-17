use crate::models::llama::error::tensor_failed;
use crate::models::llama::weights::load_tensor;
use crate::models::llama::{
    LlamaConfig,
    LlamaError,
};
use crate::nn::{
    Device,
    Linear,
    Silu,
    Tensor,
    TensorError,
};
use crate::weights::Weights;

pub struct LlamaMlp<D: Device> {
    gate_proj: Linear<D>,
    up_proj: Linear<D>,
    down_proj: Linear<D>,
    act_fn: Silu<D>,
}

impl<D: Device> LlamaMlp<D> {
    pub fn load<S: AsRef<[u8]>>(
        device: &mut D,
        config: &LlamaConfig,
        weights: &Weights<S>,
        prefix: &str,
    ) -> Result<Self, LlamaError> {
        let hidden_size = config.hidden_size();
        let intermediate_size = config.intermediate_size();
        let mut linear = |name: &str, dims: [usize; 2]| {
            load_tensor(device, weights, &format!("{prefix}.{name}.weight"), &dims).map(Linear::new)
        };
        let gate_proj = linear("gate_proj", [intermediate_size, hidden_size])?;
        let up_proj = linear("up_proj", [intermediate_size, hidden_size])?;
        let down_proj = linear("down_proj", [hidden_size, intermediate_size])?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: Silu::new(device).map_err(tensor_failed)?,
        })
    }

    pub fn forward(
        &self,
        device: &mut D,
        x: &Tensor<D>,
    ) -> Result<Tensor<D>, TensorError> {
        let gate = self.gate_proj.forward(device, x)?;
        let up = self.up_proj.forward(device, x)?;
        let hidden = self.act_fn.forward(device, &gate)?.mul(device, &up)?;
        self.down_proj.forward(device, &hidden)
    }
}
