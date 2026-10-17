use crate::nn::{
    Device,
    Shape,
    Tensor,
    TensorError,
};

pub struct Linear<D: Device> {
    weight: Tensor<D>,
}

impl<D: Device> Linear<D> {
    pub fn new(weight: Tensor<D>) -> Self {
        Self {
            weight,
        }
    }

    pub fn forward(
        &self,
        device: &mut D,
        x: &Tensor<D>,
    ) -> Result<Tensor<D>, TensorError> {
        let weight_t = self.weight.permute(&[1, 0])?;
        let dims = x.shape().dims();
        let in_features = x.shape().size(x.shape().last_dim()?)?;
        let rows = x.shape().numel().checked_div(in_features).unwrap_or(0);
        let out_features = weight_t.shape().size(1)?;
        let output_dims = dims
            .iter()
            .take(dims.len().saturating_sub(1))
            .copied()
            .chain([out_features])
            .collect::<Vec<usize>>();
        x.reshape(Shape::try_from([rows, in_features].as_slice())?)?
            .matmul(device, &weight_t)?
            .reshape(Shape::try_from(output_dims.as_slice())?)
    }
}

pub struct Embedding<D: Device> {
    weight: Tensor<D>,
}

impl<D: Device> Embedding<D> {
    pub fn new(weight: Tensor<D>) -> Self {
        Self {
            weight,
        }
    }

    pub fn forward(
        &self,
        device: &mut D,
        input_ids: &Tensor<D>,
    ) -> Result<Tensor<D>, TensorError> {
        self.weight.index_select(device, 0, input_ids)
    }

    pub fn weight(&self) -> &Tensor<D> { &self.weight }
}
