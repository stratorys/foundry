use std::num::NonZeroUsize;

use tracing::error;

use crate::core::{
    Backend,
    CoreError,
    DType,
    Shape,
    Tensor,
    exact_f32,
};
use crate::nn::{
    KvCache,
    softmax_last_axis,
};

const SEQ_AXIS: usize = 1;

pub struct Attention<B: Backend> {
    head_dim: NonZeroUsize,
    scale: Tensor<B>,
}

#[derive(Debug, PartialEq, Eq)]
struct AttentionDims {
    heads: usize,
    kv_heads: usize,
    group: usize,
    seq_len: usize,
    seq_len_total: usize,
    head_dim: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        backend: &mut B,
        head_dim: NonZeroUsize,
    ) -> Result<Self, CoreError> {
        let head_dim_f32 = exact_f32(head_dim.get())?;
        Ok(Self {
            head_dim,
            scale: Tensor::upload(
                backend,
                &head_dim_f32.sqrt().recip().to_le_bytes(),
                DType::F32,
                Shape::try_from([1].as_slice())?,
            )?,
        })
    }

    pub fn forward(
        &self,
        backend: &mut B,
        query: &Tensor<B>,
        cache: &KvCache<B>,
        seq_len_total: usize,
        mask: &Tensor<B>,
    ) -> Result<Tensor<B>, CoreError> {
        let keys = &cache.keys().narrow(SEQ_AXIS, 0, seq_len_total)?;
        let values = &cache.values().narrow(SEQ_AXIS, 0, seq_len_total)?;
        let AttentionDims {
            heads,
            kv_heads,
            group,
            seq_len,
            seq_len_total,
            head_dim,
        } = attention_dims(self.head_dim, query.shape(), keys.shape(), values.shape())?;
        let rows = group.saturating_mul(seq_len);
        let keys_t = keys.permute(&[0, 2, 1])?;
        let mask_f32 = mask.cast(backend, DType::F32)?;
        let scores = query
            .contiguous(backend)?
            .reshape(Shape::try_from([kv_heads, rows, head_dim].as_slice())?)?
            .matmul(backend, &keys_t)?
            .reshape(Shape::try_from(
                [kv_heads, group, seq_len, seq_len_total].as_slice(),
            )?)?
            .cast(backend, DType::F32)?
            .mul(backend, &self.scale)?
            .add(backend, &mask_f32)?;
        softmax_last_axis(backend, &scores)?
            .cast(backend, query.dtype())?
            .reshape(Shape::try_from([kv_heads, rows, seq_len_total].as_slice())?)?
            .matmul(backend, values)?
            .reshape(Shape::try_from([heads, seq_len, head_dim].as_slice())?)?
            .permute(&[1, 0, 2])?
            .contiguous(backend)?
            .reshape(Shape::try_from(
                [seq_len, heads.saturating_mul(head_dim)].as_slice(),
            )?)
    }
}

fn attention_dims(
    head_dim_expected: NonZeroUsize,
    query: &Shape,
    keys: &Shape,
    values: &Shape,
) -> Result<AttentionDims, CoreError> {
    let incompatible = || {
        error!(
            message = "Query, keys and values are incompatible for attention.",
            query = ?query.dims(),
            keys = ?keys.dims(),
            values = ?values.dims(),
        );
        CoreError::AttentionIncompatible
    };
    let (&[heads, seq_len, head_dim], &[kv_heads, seq_len_total, key_head_dim]) =
        (query.dims(), keys.dims())
    else {
        return Err(incompatible());
    };
    let is_grouped = heads.checked_rem(kv_heads) == Some(0);
    let is_compatible =
        is_grouped && seq_len <= seq_len_total && head_dim == key_head_dim && keys == values;
    if !is_compatible {
        return Err(incompatible());
    }
    if head_dim != head_dim_expected.get() {
        error!(
            message = "Attention head dim does not match.",
            head_dim,
            head_dim_expected = head_dim_expected.get(),
        );
        return Err(CoreError::AttentionHeadDimMismatch);
    }
    Ok(AttentionDims {
        heads,
        kv_heads,
        group: heads.checked_div(kv_heads).ok_or_else(incompatible)?,
        seq_len,
        seq_len_total,
        head_dim,
    })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::core::{
        CoreError,
        Shape,
    };
    use crate::nn::attention::{
        AttentionDims,
        attention_dims,
    };

    #[test]
    fn attention_dims_computes_the_group() {
        let [query, keys] = [[4, 1, 2], [2, 3, 2]]
            .map(|dims| Shape::try_from(dims.as_slice()).expect("the shape is valid"));
        assert_eq!(
            attention_dims(
                NonZeroUsize::new(2).expect("the head dim is non-zero"),
                &query,
                &keys,
                &keys,
            ),
            Ok(AttentionDims {
                heads: 4,
                kv_heads: 2,
                group: 2,
                seq_len: 1,
                seq_len_total: 3,
                head_dim: 2,
            }),
            "four query heads share two key-value heads"
        );
    }

    #[test]
    fn attention_dims_rejects_rank_other_than_three() {
        assert_eq!(
            attention_dims(
                NonZeroUsize::new(2).expect("the head dim is non-zero"),
                &Shape::try_from([4, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(CoreError::AttentionIncompatible),
            "a rank 2 query"
        );
        assert_eq!(
            attention_dims(
                NonZeroUsize::new(2).expect("the head dim is non-zero"),
                &Shape::try_from([4, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(CoreError::AttentionIncompatible),
            "rank 2 keys"
        );
    }

    #[test]
    fn attention_dims_rejects_zero_kv_heads() {
        assert_eq!(
            attention_dims(
                NonZeroUsize::new(2).expect("the head dim is non-zero"),
                &Shape::try_from([4, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([0, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([0, 2, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(CoreError::AttentionIncompatible),
            "zero key-value heads"
        );
    }

    #[test]
    fn attention_dims_rejects_heads_not_a_multiple_of_kv_heads() {
        assert_eq!(
            attention_dims(
                NonZeroUsize::new(2).expect("the head dim is non-zero"),
                &Shape::try_from([3, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(CoreError::AttentionIncompatible),
            "three query heads over two key-value heads"
        );
    }

    #[test]
    fn attention_dims_rejects_query_longer_than_keys() {
        assert_eq!(
            attention_dims(
                NonZeroUsize::new(2).expect("the head dim is non-zero"),
                &Shape::try_from([4, 3, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(CoreError::AttentionIncompatible),
            "three query positions over two key positions"
        );
    }

    #[test]
    fn attention_dims_rejects_key_head_dim_mismatch() {
        assert_eq!(
            attention_dims(
                NonZeroUsize::new(2).expect("the head dim is non-zero"),
                &Shape::try_from([4, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 4].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 4].as_slice()).expect("the shape is valid"),
            ),
            Err(CoreError::AttentionIncompatible),
            "head dims 2 and 4 differ"
        );
    }

    #[test]
    fn attention_dims_rejects_values_shaped_unlike_keys() {
        assert_eq!(
            attention_dims(
                NonZeroUsize::new(2).expect("the head dim is non-zero"),
                &Shape::try_from([4, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 3, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(CoreError::AttentionIncompatible),
            "values hold one more position than keys"
        );
    }
}
