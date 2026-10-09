use crate::core::{
    Backend,
    CoreError,
    DType,
    Shape,
    Tensor,
};

const SEQ_AXIS: usize = 1;

pub struct KvCache<B: Backend> {
    keys: Tensor<B>,
    values: Tensor<B>,
    length: usize,
}

impl<B: Backend> KvCache<B> {
    pub fn new(
        backend: &mut B,
        dtype: DType,
        kv_heads: usize,
        seq_len_max: usize,
        head_dim: usize,
    ) -> Result<Self, B::Error> {
        let shape = Shape::try_from([kv_heads, seq_len_max, head_dim].as_slice())?;
        Ok(Self {
            keys: Tensor::zeros(backend, dtype, shape)?,
            values: Tensor::zeros(backend, dtype, shape)?,
            length: 0,
        })
    }

    pub fn length(&self) -> usize { self.length }

    pub fn append(
        &mut self,
        backend: &mut B,
        keys: &Tensor<B>,
        values: &Tensor<B>,
    ) -> Result<(), B::Error> {
        let seq_len =
            keys.shape()
                .dims()
                .get(SEQ_AXIS)
                .copied()
                .ok_or(CoreError::AxisOutOfRange {
                    axis: SEQ_AXIS,
                    rank: keys.shape().rank(),
                })?;
        self.keys
            .slice_update(backend, keys, SEQ_AXIS, self.length)?;
        self.values
            .slice_update(backend, values, SEQ_AXIS, self.length)?;
        self.length = self.length.saturating_add(seq_len);
        Ok(())
    }

    pub fn keys(&self) -> Result<Tensor<B>, CoreError> {
        self.keys.narrow(SEQ_AXIS, 0, self.length)
    }

    pub fn values(&self) -> Result<Tensor<B>, CoreError> {
        self.values.narrow(SEQ_AXIS, 0, self.length)
    }
}
