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
        logits_to_keep: usize,
    ) -> Result<Tensor<D>, Self::Error>;
}
