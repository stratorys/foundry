use crate::core::{
    Backend,
    CoreError,
    DType,
    Shape,
    Tensor,
};
use crate::nn::{
    KvCache,
    softmax_last_axis,
};

pub struct Attention<B: Backend> {
    scale: Tensor<B>,
}

struct AttentionDims {
    heads: usize,
    kv_heads: usize,
    group: usize,
    seq_len: usize,
    head_dim: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        backend: &mut B,
        head_dim: usize,
    ) -> Result<Self, B::Error> {
        let head_dim_f32 = u16::try_from(head_dim).map(f32::from).map_err(|_| {
            CoreError::DimensionTooLargeForF32 {
                dim: head_dim,
                dim_max: u16::MAX.into(),
            }
        })?;
        Ok(Self {
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
        cache: &mut KvCache<B>,
        query: &Tensor<B>,
        key: &Tensor<B>,
        value: &Tensor<B>,
        mask: &Tensor<B>,
    ) -> Result<Tensor<B>, B::Error> {
        let AttentionDims {
            heads,
            kv_heads,
            group,
            seq_len,
            head_dim,
        } = attention_dims(query.shape(), key.shape(), value.shape())?;
        cache.append(backend, key, value)?;
        let seq_len_total = cache.length();
        let rows = group.saturating_mul(seq_len);
        let keys_t = cache.keys()?.permute(&[0, 2, 1])?;
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
            .matmul(backend, &cache.values()?)?
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
    query: &Shape,
    key: &Shape,
    value: &Shape,
) -> Result<AttentionDims, CoreError> {
    let incompatible = || CoreError::AttentionIncompatible {
        query: query.dims().to_vec(),
        key: key.dims().to_vec(),
        value: value.dims().to_vec(),
    };
    let (&[heads, seq_len, head_dim], &[kv_heads, key_seq_len, key_head_dim]) =
        (query.dims(), key.dims())
    else {
        return Err(incompatible());
    };
    let is_grouped = heads.checked_rem(kv_heads) == Some(0);
    let is_compatible =
        is_grouped && seq_len == key_seq_len && head_dim == key_head_dim && key == value;
    if !is_compatible {
        return Err(incompatible());
    }
    Ok(AttentionDims {
        heads,
        kv_heads,
        group: heads.checked_div(kv_heads).ok_or_else(incompatible)?,
        seq_len,
        head_dim,
    })
}
