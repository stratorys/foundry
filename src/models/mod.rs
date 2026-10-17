pub mod llama;

use std::num::NonZeroUsize;

use crate::nn::{
    Device,
    Tensor,
};

pub trait CausalLm<D: Device> {
    type Cache;
    type Error: std::error::Error;

    fn cache(
        &self,
        device: &mut D,
        batch_size: NonZeroUsize,
    ) -> Result<Self::Cache, Self::Error>;

    fn forward(
        &self,
        device: &mut D,
        input_ids: &Tensor<D>,
        past_key_values: &mut Self::Cache,
        past_seen_tokens: usize,
    ) -> Result<Tensor<D>, Self::Error>;
}

pub trait Submodules<D: Device> {
    type Error: std::error::Error;

    fn forward_submodule(
        &self,
        device: &mut D,
        target: &str,
        input: &Tensor<D>,
    ) -> Result<Option<Tensor<D>>, Self::Error>;
}
