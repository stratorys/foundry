use crate::core::{
    Backend,
    CoreError,
    Tensor,
};
use crate::nn::{
    Linear,
    Silu,
};

pub struct SwigluMlp<B: Backend> {
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
    silu: Silu<B>,
}

impl<B: Backend> SwigluMlp<B> {
    pub fn new(
        backend: &mut B,
        gate: Linear<B>,
        up: Linear<B>,
        down: Linear<B>,
    ) -> Result<Self, CoreError> {
        Ok(Self {
            gate,
            up,
            down,
            silu: Silu::new(backend)?,
        })
    }

    pub fn forward(
        &self,
        backend: &mut B,
        x: &Tensor<B>,
    ) -> Result<Tensor<B>, CoreError> {
        let gate = self.gate.forward(backend, x)?;
        let up = self.up.forward(backend, x)?;
        let hidden = self.silu.forward(backend, &gate)?.mul(backend, &up)?;
        self.down.forward(backend, &hidden)
    }
}
