use crate::nn::{
    Device,
    Tensor,
    TensorError,
};

pub trait Module<D: Device> {
    fn forward(
        &self,
        device: &mut D,
        input: &Tensor<D>,
    ) -> Result<Tensor<D>, TensorError>;
}
