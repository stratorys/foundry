use std::num::NonZeroUsize;

use crate::core::{
    Backend,
    CoreError,
    DType,
    Shape,
    Tensor,
};
use crate::nn::softmax_last_axis;

pub struct Attention<B: Backend> {
    head_dim: NonZeroUsize,
    scale: Tensor<B>,
}

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
    ) -> Result<Self, B::Error> {
        let head_dim_f32 = u16::try_from(head_dim.get()).map(f32::from).map_err(|_| {
            CoreError::DimensionTooLargeForF32 {
                dim: head_dim.get(),
                dim_max: u16::MAX.into(),
            }
        })?;
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
        keys: &Tensor<B>,
        values: &Tensor<B>,
        mask: &Tensor<B>,
    ) -> Result<Tensor<B>, B::Error> {
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
            .map_err(Into::into)
    }
}

fn attention_dims(
    head_dim_expected: NonZeroUsize,
    query: &Shape,
    keys: &Shape,
    values: &Shape,
) -> Result<AttentionDims, CoreError> {
    let incompatible = || CoreError::AttentionIncompatible {
        query: query.dims().to_vec(),
        key: keys.dims().to_vec(),
        value: values.dims().to_vec(),
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
        return Err(CoreError::AttentionHeadDimMismatch {
            head_dim,
            head_dim_expected: head_dim_expected.get(),
        });
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
