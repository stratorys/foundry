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
        })
    }

    pub fn update(
        &mut self,
        backend: &mut B,
        keys: &Tensor<B>,
        values: &Tensor<B>,
        position: usize,
    ) -> Result<(Tensor<B>, Tensor<B>), B::Error> {
        if keys.shape() != values.shape() {
            return Err(CoreError::KvIncompatible {
                keys: keys.shape().dims().to_vec(),
                values: values.shape().dims().to_vec(),
            }
            .into());
        }
        if keys.dtype() != values.dtype() {
            return Err(CoreError::DTypeMismatch {
                lhs: keys.dtype(),
                rhs: values.dtype(),
            }
            .into());
        }
        let seq_len =
            keys.shape()
                .dims()
                .get(SEQ_AXIS)
                .copied()
                .ok_or(CoreError::AxisOutOfRange {
                    axis: SEQ_AXIS,
                    rank: keys.shape().rank(),
                })?;
        let length =
            position
                .checked_add(seq_len)
                .ok_or_else(|| CoreError::SliceUpdateOutOfBounds {
                    axis: SEQ_AXIS,
                    start: position,
                    target: self.keys.shape().dims().to_vec(),
                    update: keys.shape().dims().to_vec(),
                })?;
        self.keys.slice_update(backend, keys, SEQ_AXIS, position)?;
        self.values
            .slice_update(backend, values, SEQ_AXIS, position)?;
        Ok((
            self.keys.narrow(SEQ_AXIS, 0, length)?,
            self.values.narrow(SEQ_AXIS, 0, length)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::cpu::{
        CpuBackend,
        CpuError,
    };
    use crate::core::{
        CoreError,
        DType,
        Shape,
        Tensor,
    };
    use crate::nn::KvCache;

    #[test]
    fn kv_cache_update_beyond_capacity_is_rejected() {
        let mut backend = CpuBackend::new();
        let [key, value] = [[1, 3, 2], [1, 3, 2]].map(|dims| {
            Tensor::zeros(
                &mut backend,
                DType::BF16,
                Shape::try_from(dims.as_slice()).expect("the shape is valid"),
            )
            .expect("the zeros succeed")
        });
        let mut cache =
            KvCache::new(&mut backend, DType::BF16, 1, 2, 2).expect("the cache is built");
        assert!(
            matches!(
                cache.update(&mut backend, &key, &value, 0),
                Err(CpuError::Core(CoreError::SliceUpdateOutOfBounds { .. }))
            ),
            "writing three positions to a cache of two is rejected"
        );
    }

    #[test]
    fn kv_cache_update_rejects_keys_and_values_of_different_shapes() {
        let mut backend = CpuBackend::new();
        let [key, value] = [[1, 2, 2], [1, 1, 2]].map(|dims| {
            Tensor::zeros(
                &mut backend,
                DType::BF16,
                Shape::try_from(dims.as_slice()).expect("the shape is valid"),
            )
            .expect("the zeros succeed")
        });
        let mut cache =
            KvCache::new(&mut backend, DType::BF16, 1, 4, 2).expect("the cache is built");
        assert!(
            matches!(
                cache.update(&mut backend, &key, &value, 0),
                Err(CpuError::Core(CoreError::KvIncompatible { .. }))
            ),
            "keys of two positions with values of one position are rejected"
        );
    }
}
