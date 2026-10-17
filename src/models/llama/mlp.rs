use crate::core::{
    Backend,
    Tensor,
    TensorError,
};
use crate::models::llama::weights::load_tensor;
use crate::models::llama::{
    LlamaConfig,
    LlamaError,
};
use crate::nn::{
    Linear,
    Silu,
};
use crate::weights::Weights;

pub struct LlamaMlp<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
    act_fn: Silu<B>,
}

impl<B: Backend> LlamaMlp<B> {
    pub fn load<S: AsRef<[u8]>>(
        backend: &mut B,
        config: &LlamaConfig,
        weights: &Weights<S>,
        prefix: &str,
    ) -> Result<Self, LlamaError<B::Error>> {
        let hidden_size = config.hidden_size();
        let intermediate_size = config.intermediate_size();
        let mut linear = |name: &str, dims: [usize; 2]| {
            load_tensor(backend, weights, &format!("{prefix}.{name}.weight"), &dims)
                .map(Linear::new)
        };
        let gate_proj = linear("gate_proj", [intermediate_size, hidden_size])?;
        let up_proj = linear("up_proj", [intermediate_size, hidden_size])?;
        let down_proj = linear("down_proj", [hidden_size, intermediate_size])?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: Silu::new(backend)?,
        })
    }

    pub fn forward(
        &self,
        backend: &mut B,
        x: &Tensor<B>,
    ) -> Result<Tensor<B>, TensorError<B::Error>> {
        let gate = self.gate_proj.forward(backend, x)?;
        let up = self.up_proj.forward(backend, x)?;
        let hidden = self.act_fn.forward(backend, &gate)?.mul(backend, &up)?;
        self.down_proj.forward(backend, &hidden)
    }
}
