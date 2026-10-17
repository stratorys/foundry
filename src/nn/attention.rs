use tracing::error;

use crate::nn::{
    DType,
    Device,
    Shape,
    Tensor,
    TensorError,
    softmax,
};

#[derive(Debug, PartialEq, Eq)]
struct AttentionDims {
    batch: usize,
    heads: usize,
    kv_heads: usize,
    group: usize,
    seq_len: usize,
    kv_length: usize,
    head_dim: usize,
}

pub fn scaled_dot_product_attention<D: Device>(
    device: &mut D,
    query: &Tensor<D>,
    key: &Tensor<D>,
    value: &Tensor<D>,
    attn_mask: &Tensor<D>,
    scale: &Tensor<D>,
) -> Result<Tensor<D>, TensorError> {
    let AttentionDims {
        batch,
        heads,
        kv_heads,
        group,
        seq_len,
        kv_length,
        head_dim,
    } = attention_dims(query.shape(), key.shape(), value.shape())?;
    let rows = group.saturating_mul(seq_len);
    let key_t = key.permute(&[0, 1, 3, 2])?;
    let attn_mask_f32 = attn_mask.to_dtype(device, DType::F32)?;
    let scores = query
        .contiguous(device)?
        .reshape(Shape::try_from(
            [batch, kv_heads, rows, head_dim].as_slice(),
        )?)?
        .matmul(device, &key_t)?
        .reshape(Shape::try_from(
            [batch.saturating_mul(kv_heads), group, seq_len, kv_length].as_slice(),
        )?)?
        .to_dtype(device, DType::F32)?
        .mul(device, scale)?
        .add(device, &attn_mask_f32)?;
    softmax(device, &scores, scores.shape().last_dim()?)?
        .to_dtype(device, query.dtype())?
        .reshape(Shape::try_from(
            [batch, kv_heads, rows, kv_length].as_slice(),
        )?)?
        .matmul(device, value)?
        .reshape(Shape::try_from(
            [batch, heads, seq_len, head_dim].as_slice(),
        )?)
}

fn attention_dims(
    query: &Shape,
    keys: &Shape,
    values: &Shape,
) -> Result<AttentionDims, TensorError> {
    let incompatible = || {
        error!(
            message = "Query, keys and values are incompatible for attention.",
            query = ?query.dims(),
            keys = ?keys.dims(),
            values = ?values.dims(),
        );
        TensorError::AttentionIncompatible
    };
    let &[batch, heads, seq_len, head_dim] = query.dims() else {
        return Err(incompatible());
    };
    let &[key_batch, kv_heads, kv_length, key_head_dim] = keys.dims() else {
        return Err(incompatible());
    };
    let is_grouped = heads.checked_rem(kv_heads) == Some(0);
    let is_compatible = batch == key_batch
        && is_grouped
        && seq_len <= kv_length
        && head_dim == key_head_dim
        && keys == values;
    if !is_compatible {
        return Err(incompatible());
    }
    Ok(AttentionDims {
        batch,
        heads,
        kv_heads,
        group: heads.checked_div(kv_heads).ok_or_else(incompatible)?,
        seq_len,
        kv_length,
        head_dim,
    })
}

#[cfg(test)]
mod tests {
    use crate::nn::attention::{
        AttentionDims,
        attention_dims,
    };
    use crate::nn::{
        Shape,
        TensorError,
    };

    #[test]
    fn attention_dims_computes_the_group() {
        let [query, keys] = [[2, 4, 1, 2], [2, 2, 3, 2]]
            .map(|dims| Shape::try_from(dims.as_slice()).expect("the shape is valid"));
        assert_eq!(
            attention_dims(&query, &keys, &keys,),
            Ok(AttentionDims {
                batch: 2,
                heads: 4,
                kv_heads: 2,
                group: 2,
                seq_len: 1,
                kv_length: 3,
                head_dim: 2,
            }),
            "in each of two sequences, four query heads share two key-value heads"
        );
    }

    #[test]
    fn attention_dims_rejects_rank_other_than_four() {
        assert_eq!(
            attention_dims(
                &Shape::try_from([4, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([1, 2, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([1, 2, 2, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(TensorError::AttentionIncompatible),
            "a rank 3 query"
        );
        assert_eq!(
            attention_dims(
                &Shape::try_from([1, 4, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([2, 2, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(TensorError::AttentionIncompatible),
            "rank 3 keys"
        );
    }

    #[test]
    fn attention_dims_rejects_zero_kv_heads() {
        assert_eq!(
            attention_dims(
                &Shape::try_from([1, 4, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([1, 0, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([1, 0, 2, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(TensorError::AttentionIncompatible),
            "zero key-value heads"
        );
    }

    #[test]
    fn attention_dims_rejects_heads_not_a_multiple_of_kv_heads() {
        assert_eq!(
            attention_dims(
                &Shape::try_from([1, 3, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([1, 2, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([1, 2, 2, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(TensorError::AttentionIncompatible),
            "three query heads over two key-value heads"
        );
    }

    #[test]
    fn attention_dims_rejects_query_longer_than_keys() {
        assert_eq!(
            attention_dims(
                &Shape::try_from([1, 4, 3, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([1, 2, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([1, 2, 2, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(TensorError::AttentionIncompatible),
            "three query positions over two key positions"
        );
    }

    #[test]
    fn attention_dims_rejects_key_head_dim_mismatch() {
        assert_eq!(
            attention_dims(
                &Shape::try_from([1, 4, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([1, 2, 2, 4].as_slice()).expect("the shape is valid"),
                &Shape::try_from([1, 2, 2, 4].as_slice()).expect("the shape is valid"),
            ),
            Err(TensorError::AttentionIncompatible),
            "head dims 2 and 4 differ"
        );
    }

    #[test]
    fn attention_dims_rejects_values_shaped_unlike_keys() {
        assert_eq!(
            attention_dims(
                &Shape::try_from([1, 4, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([1, 2, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([1, 2, 3, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(TensorError::AttentionIncompatible),
            "values hold one more position than keys"
        );
    }

    #[test]
    fn attention_dims_rejects_query_and_keys_of_different_batches() {
        assert_eq!(
            attention_dims(
                &Shape::try_from([2, 4, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([1, 2, 2, 2].as_slice()).expect("the shape is valid"),
                &Shape::try_from([1, 2, 2, 2].as_slice()).expect("the shape is valid"),
            ),
            Err(TensorError::AttentionIncompatible),
            "a query of two sequences over keys of one sequence"
        );
    }
}
