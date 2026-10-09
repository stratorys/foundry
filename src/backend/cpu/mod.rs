mod error;

use std::iter;

use half::{
    bf16,
    f16,
};

pub use self::error::CpuError;
use crate::core::primitive::{
    BinaryOp,
    ReduceOp,
    UnaryOp,
    concat_rule,
    gather_rule,
    matmul_rule,
    reduce_rule,
    slice_update_rule,
};
use crate::core::{
    Backend,
    CoreError,
    DType,
    Layout,
    Operand,
    OperandMut,
    Shape,
};

pub struct CpuStorage {
    bytes: Vec<u8>,
}

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self { Self }
}

impl Backend for CpuBackend {
    type Error = CpuError;
    type Storage = CpuStorage;

    fn upload(
        &mut self,
        bytes: &[u8],
        dtype: DType,
        shape: &Shape,
    ) -> Result<CpuStorage, CpuError> {
        let bytes_expected = byte_len(dtype, shape)?;
        if bytes.len() != bytes_expected {
            return Err(CpuError::ByteLengthMismatch {
                bytes: bytes.len(),
                bytes_expected,
            });
        }
        Ok(CpuStorage {
            bytes: bytes.to_vec(),
        })
    }

    fn zeros(
        &mut self,
        dtype: DType,
        shape: &Shape,
    ) -> Result<CpuStorage, CpuError> {
        Ok(CpuStorage {
            bytes: vec![0; byte_len(dtype, shape)?],
        })
    }

    fn download(
        &mut self,
        input: Operand<'_, CpuStorage>,
    ) -> Result<Vec<u8>, CpuError> {
        let layout = input.layout();
        if !layout.is_contiguous() {
            return Err(CpuError::DownloadNonContiguous);
        }
        let bytes = &input.storage().bytes;
        let out_of_storage = || CpuError::IndexOutOfStorage {
            index: layout.offset(),
            bytes_len: bytes.len(),
        };
        let start = layout
            .offset()
            .checked_mul(input.dtype().size_bytes())
            .ok_or_else(out_of_storage)?;
        let end = start
            .checked_add(byte_len(input.dtype(), layout.shape())?)
            .ok_or_else(out_of_storage)?;
        bytes
            .get(start..end)
            .map(<[u8]>::to_vec)
            .ok_or_else(out_of_storage)
    }

    fn unary(
        &mut self,
        op: UnaryOp,
        input: Operand<'_, CpuStorage>,
    ) -> Result<CpuStorage, CpuError> {
        reject_u32("unary", input.dtype())?;
        let values: Vec<f32> = read_floats(&input, "unary")?
            .into_iter()
            .map(|value| match op {
                UnaryOp::Neg => -value,
                UnaryOp::Exp => value.exp(),
                UnaryOp::Sqrt => value.sqrt(),
                UnaryOp::Recip => 1.0 / value,
            })
            .collect();
        encode(input.dtype(), &values, "unary")
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        lhs: Operand<'_, CpuStorage>,
        rhs: Operand<'_, CpuStorage>,
    ) -> Result<CpuStorage, CpuError> {
        reject_u32("binary", lhs.dtype())?;
        reject_u32("binary", rhs.dtype())?;
        let values: Vec<f32> = read_floats(&lhs, "binary")?
            .into_iter()
            .zip(read_floats(&rhs, "binary")?)
            .map(|(lhs, rhs)| match op {
                BinaryOp::Add => lhs + rhs,
                BinaryOp::Sub => lhs - rhs,
                BinaryOp::Mul => lhs * rhs,
                BinaryOp::Div => lhs / rhs,
            })
            .collect();
        encode(lhs.dtype(), &values, "binary")
    }

    fn reduce(
        &mut self,
        op: ReduceOp,
        input: Operand<'_, CpuStorage>,
        axis: usize,
    ) -> Result<CpuStorage, CpuError> {
        let dtype = input.dtype();
        reject_u32("reduce", dtype)?;
        let layout = input.layout();
        let (_, shape) = reduce_rule(op, dtype, layout.shape(), axis)?;
        let axis_out_of_range = || CoreError::AxisOutOfRange {
            axis,
            rank: layout.shape().rank(),
        };
        let axis_len = layout
            .shape()
            .dims()
            .get(axis)
            .copied()
            .ok_or_else(axis_out_of_range)?;
        let axis_stride = layout
            .strides()
            .get(axis)
            .copied()
            .ok_or_else(axis_out_of_range)?;
        let rows = (0..shape.element_count())
            .map(|row| {
                let base = strided_index(shape.dims(), layout.strides(), layout.offset(), row)?;
                (0..axis_len)
                    .map(|position| {
                        let index = offset_by(base, &[(position, axis_stride)]).ok_or(
                            CpuError::IndexOverflow {
                                linear: row,
                            },
                        )?;
                        read_float(&input.storage().bytes, dtype, index, "reduce")
                    })
                    .collect::<Result<Vec<f32>, CpuError>>()
            })
            .collect::<Result<Vec<Vec<f32>>, CpuError>>()?;
        match op {
            ReduceOp::Sum => {
                let sums: Vec<f32> = rows
                    .iter()
                    .map(|row| row.iter().fold(0.0, |sum, &value| sum + value))
                    .collect();
                encode(dtype, &sums, "reduce")
            }
            ReduceOp::Max => {
                let maxima: Vec<f32> = rows
                    .iter()
                    .map(|row| row.iter().copied().fold(f32::NEG_INFINITY, f32::max))
                    .collect();
                encode(dtype, &maxima, "reduce")
            }
            ReduceOp::Argmax => {
                let indices = rows
                    .iter()
                    .map(|row| {
                        let (_, position) = row.iter().enumerate().fold(
                            (f32::NEG_INFINITY, 0_usize),
                            |(best, best_position), (position, &value)| {
                                if value > best {
                                    (value, position)
                                } else {
                                    (best, best_position)
                                }
                            },
                        );
                        u32::try_from(position).map_err(|_| CpuError::IndexOverflow {
                            linear: position,
                        })
                    })
                    .collect::<Result<Vec<u32>, CpuError>>()?;
                Ok(CpuStorage {
                    bytes: indices
                        .iter()
                        .flat_map(|index| index.to_le_bytes())
                        .collect(),
                })
            }
        }
    }

    fn matmul(
        &mut self,
        lhs: Operand<'_, CpuStorage>,
        rhs: Operand<'_, CpuStorage>,
    ) -> Result<CpuStorage, CpuError> {
        let dtype = lhs.dtype();
        reject_u32("matmul", dtype)?;
        let (_, shape) = matmul_rule(
            dtype,
            lhs.layout().shape(),
            rhs.dtype(),
            rhs.layout().shape(),
        )?;
        let incompatible = || CoreError::MatmulIncompatible {
            lhs: lhs.layout().shape().dims().to_vec(),
            rhs: rhs.layout().shape().dims().to_vec(),
        };
        let (lhs_batch_strides, [lhs_row_stride, lhs_col_stride]) = lhs
            .layout()
            .strides()
            .split_last_chunk::<2>()
            .ok_or_else(incompatible)?;
        let (rhs_batch_strides, [rhs_row_stride, rhs_col_stride]) = rhs
            .layout()
            .strides()
            .split_last_chunk::<2>()
            .ok_or_else(incompatible)?;
        let (batch_dims, [m, n]) = shape
            .dims()
            .split_last_chunk::<2>()
            .ok_or_else(incompatible)?;
        let k = lhs
            .layout()
            .shape()
            .dims()
            .last()
            .copied()
            .ok_or_else(incompatible)?;
        let values = (0..shape.element_count())
            .map(|linear| {
                let overflow = || CpuError::IndexOverflow {
                    linear,
                };
                let col = linear.checked_rem(*n).ok_or_else(overflow)?;
                let row = linear
                    .checked_div(*n)
                    .and_then(|rows| rows.checked_rem(*m))
                    .ok_or_else(overflow)?;
                let batch = m
                    .checked_mul(*n)
                    .and_then(|matrix| linear.checked_div(matrix))
                    .ok_or_else(overflow)?;
                let lhs_base =
                    strided_index(batch_dims, lhs_batch_strides, lhs.layout().offset(), batch)?;
                let rhs_base =
                    strided_index(batch_dims, rhs_batch_strides, rhs.layout().offset(), batch)?;
                (0..k).try_fold(0.0_f32, |sum, inner| {
                    let lhs_index = offset_by(
                        lhs_base,
                        &[(row, *lhs_row_stride), (inner, *lhs_col_stride)],
                    )
                    .ok_or_else(overflow)?;
                    let rhs_index = offset_by(
                        rhs_base,
                        &[(inner, *rhs_row_stride), (col, *rhs_col_stride)],
                    )
                    .ok_or_else(overflow)?;
                    let lhs_value = read_float(&lhs.storage().bytes, dtype, lhs_index, "matmul")?;
                    let rhs_value = read_float(&rhs.storage().bytes, dtype, rhs_index, "matmul")?;
                    Ok(sum + lhs_value * rhs_value)
                })
            })
            .collect::<Result<Vec<f32>, CpuError>>()?;
        encode(dtype, &values, "matmul")
    }

    fn copy(
        &mut self,
        input: Operand<'_, CpuStorage>,
    ) -> Result<CpuStorage, CpuError> {
        let layout = input.layout();
        let size = input.dtype().size_bytes();
        let bytes = (0..layout.shape().element_count()).try_fold(
            Vec::with_capacity(byte_len(input.dtype(), layout.shape())?),
            |mut bytes, linear| {
                let index = strided_index(
                    layout.shape().dims(),
                    layout.strides(),
                    layout.offset(),
                    linear,
                )?;
                bytes.extend_from_slice(element_bytes(&input.storage().bytes, size, index)?);
                Ok::<Vec<u8>, CpuError>(bytes)
            },
        )?;
        Ok(CpuStorage {
            bytes,
        })
    }

    fn cast(
        &mut self,
        input: Operand<'_, CpuStorage>,
        dtype: DType,
    ) -> Result<CpuStorage, CpuError> {
        reject_u32("cast", input.dtype())?;
        reject_u32("cast", dtype)?;
        encode(dtype, &read_floats(&input, "cast")?, "cast")
    }

    fn gather(
        &mut self,
        table: Operand<'_, CpuStorage>,
        indices: Operand<'_, CpuStorage>,
    ) -> Result<CpuStorage, CpuError> {
        let dtype = table.dtype();
        reject_u32("gather", dtype)?;
        let (_, shape) = gather_rule(
            dtype,
            table.layout().shape(),
            indices.dtype(),
            indices.layout().shape(),
        )?;
        let incompatible = || CoreError::GatherIncompatible {
            table: table.layout().shape().dims().to_vec(),
            indices: indices.layout().shape().dims().to_vec(),
        };
        let ([rows, cols], [row_stride, col_stride]) =
            (table.layout().shape().dims(), table.layout().strides())
        else {
            return Err(incompatible().into());
        };
        let size = dtype.size_bytes();
        let indices_layout = indices.layout();
        let bytes = (0..shape.element_count()).try_fold(
            Vec::with_capacity(byte_len(dtype, &shape)?),
            |mut bytes, linear| {
                let overflow = || CpuError::IndexOverflow {
                    linear,
                };
                let position = linear.checked_div(*cols).ok_or_else(overflow)?;
                let col = linear.checked_rem(*cols).ok_or_else(overflow)?;
                let index_position = strided_index(
                    indices_layout.shape().dims(),
                    indices_layout.strides(),
                    indices_layout.offset(),
                    position,
                )?;
                let row = usize::try_from(read_u32(&indices.storage().bytes, index_position)?)
                    .map_err(|_| overflow())?;
                if row >= *rows {
                    bytes.extend(iter::repeat_n(0, size));
                } else {
                    let index = offset_by(
                        table.layout().offset(),
                        &[(row, *row_stride), (col, *col_stride)],
                    )
                    .ok_or_else(overflow)?;
                    bytes.extend_from_slice(element_bytes(&table.storage().bytes, size, index)?);
                }
                Ok::<Vec<u8>, CpuError>(bytes)
            },
        )?;
        Ok(CpuStorage {
            bytes,
        })
    }

    fn concat(
        &mut self,
        lhs: Operand<'_, CpuStorage>,
        rhs: Operand<'_, CpuStorage>,
        axis: usize,
    ) -> Result<CpuStorage, CpuError> {
        let dtype = lhs.dtype();
        reject_u32("concat", dtype)?;
        let (_, shape) = concat_rule(
            dtype,
            lhs.layout().shape(),
            rhs.dtype(),
            rhs.layout().shape(),
            axis,
        )?;
        let lhs_len =
            lhs.layout()
                .shape()
                .dims()
                .get(axis)
                .copied()
                .ok_or(CoreError::AxisOutOfRange {
                    axis,
                    rank: lhs.layout().shape().rank(),
                })?;
        let layout = Layout::contiguous(shape);
        let mut bytes = vec![0; byte_len(dtype, &shape)?];
        write_window(&mut bytes, &layout, axis, 0, &lhs)?;
        write_window(&mut bytes, &layout, axis, lhs_len, &rhs)?;
        Ok(CpuStorage {
            bytes,
        })
    }

    fn slice_update(
        &mut self,
        mut target: OperandMut<'_, CpuStorage>,
        update: Operand<'_, CpuStorage>,
        axis: usize,
        start: usize,
    ) -> Result<(), CpuError> {
        let layout = target.layout();
        let dtype = target.dtype();
        reject_u32("slice_update", dtype)?;
        slice_update_rule(
            dtype,
            layout.shape(),
            update.dtype(),
            update.layout().shape(),
            axis,
            start,
        )?;
        write_window(
            &mut target.storage_mut().bytes,
            layout,
            axis,
            start,
            &update,
        )
    }
}

fn byte_len(
    dtype: DType,
    shape: &Shape,
) -> Result<usize, CpuError> {
    shape
        .element_count()
        .checked_mul(dtype.size_bytes())
        .ok_or_else(|| CpuError::ByteCountOverflow {
            dims: shape.dims().to_vec(),
        })
}

fn reject_u32(
    primitive: &'static str,
    dtype: DType,
) -> Result<(), CpuError> {
    match dtype {
        DType::U32 => Err(CpuError::UnsupportedDType {
            primitive,
            dtype,
        }),
        DType::F32 | DType::F16 | DType::BF16 => Ok(()),
    }
}

fn strided_index(
    dims: &[usize],
    strides: &[usize],
    offset: usize,
    linear: usize,
) -> Result<usize, CpuError> {
    dims.iter()
        .zip(strides)
        .rev()
        .try_fold((offset, linear), |(index, rest), (&dim, &stride)| {
            let index = rest
                .checked_rem(dim)?
                .checked_mul(stride)?
                .checked_add(index)?;
            Some((index, rest.checked_div(dim)?))
        })
        .map(|(index, _)| index)
        .ok_or(CpuError::IndexOverflow {
            linear,
        })
}

fn offset_by(
    base: usize,
    terms: &[(usize, usize)],
) -> Option<usize> {
    terms.iter().try_fold(base, |index, &(coordinate, stride)| {
        coordinate.checked_mul(stride)?.checked_add(index)
    })
}

fn element_bytes(
    bytes: &[u8],
    size: usize,
    index: usize,
) -> Result<&[u8], CpuError> {
    let out_of_storage = || CpuError::IndexOutOfStorage {
        index,
        bytes_len: bytes.len(),
    };
    let start = index.checked_mul(size).ok_or_else(out_of_storage)?;
    let end = start.checked_add(size).ok_or_else(out_of_storage)?;
    bytes.get(start..end).ok_or_else(out_of_storage)
}

fn read_float(
    bytes: &[u8],
    dtype: DType,
    index: usize,
    primitive: &'static str,
) -> Result<f32, CpuError> {
    match (dtype, element_bytes(bytes, dtype.size_bytes(), index)?) {
        (DType::F32, &[b0, b1, b2, b3]) => Ok(f32::from_le_bytes([b0, b1, b2, b3])),
        (DType::F16, &[b0, b1]) => Ok(f16::from_le_bytes([b0, b1]).to_f32()),
        (DType::BF16, &[b0, b1]) => Ok(bf16::from_le_bytes([b0, b1]).to_f32()),
        (DType::U32, _) => Err(CpuError::UnsupportedDType {
            primitive,
            dtype,
        }),
        (DType::F32 | DType::F16 | DType::BF16, _) => Err(CpuError::IndexOutOfStorage {
            index,
            bytes_len: bytes.len(),
        }),
    }
}

fn read_u32(
    bytes: &[u8],
    index: usize,
) -> Result<u32, CpuError> {
    match element_bytes(bytes, DType::U32.size_bytes(), index)? {
        &[b0, b1, b2, b3] => Ok(u32::from_le_bytes([b0, b1, b2, b3])),
        _ => Err(CpuError::IndexOutOfStorage {
            index,
            bytes_len: bytes.len(),
        }),
    }
}

fn read_floats(
    input: &Operand<'_, CpuStorage>,
    primitive: &'static str,
) -> Result<Vec<f32>, CpuError> {
    let layout = input.layout();
    (0..layout.shape().element_count())
        .map(|linear| {
            let index = strided_index(
                layout.shape().dims(),
                layout.strides(),
                layout.offset(),
                linear,
            )?;
            read_float(&input.storage().bytes, input.dtype(), index, primitive)
        })
        .collect()
}

fn encode(
    dtype: DType,
    values: &[f32],
    primitive: &'static str,
) -> Result<CpuStorage, CpuError> {
    let bytes = match dtype {
        DType::F32 => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        DType::F16 => values
            .iter()
            .flat_map(|&value| f16::from_f32(value).to_le_bytes())
            .collect(),
        DType::BF16 => values
            .iter()
            .flat_map(|&value| bf16::from_f32(value).to_le_bytes())
            .collect(),
        DType::U32 => {
            return Err(CpuError::UnsupportedDType {
                primitive,
                dtype,
            });
        }
    };
    Ok(CpuStorage {
        bytes,
    })
}

fn write_window(
    target: &mut [u8],
    target_layout: &Layout,
    axis: usize,
    start: usize,
    update: &Operand<'_, CpuStorage>,
) -> Result<(), CpuError> {
    let size = update.dtype().size_bytes();
    let axis_stride =
        target_layout
            .strides()
            .get(axis)
            .copied()
            .ok_or(CoreError::AxisOutOfRange {
                axis,
                rank: target_layout.shape().rank(),
            })?;
    let offset = offset_by(target_layout.offset(), &[(start, axis_stride)]).ok_or(
        CpuError::IndexOverflow {
            linear: start,
        },
    )?;
    let update_layout = update.layout();
    let dims = update_layout.shape().dims();
    (0..update_layout.shape().element_count()).try_for_each(|linear| {
        let source = strided_index(
            dims,
            update_layout.strides(),
            update_layout.offset(),
            linear,
        )?;
        let destination = strided_index(dims, target_layout.strides(), offset, linear)?;
        let value = element_bytes(&update.storage().bytes, size, source)?;
        let bytes_len = target.len();
        let out_of_storage = || CpuError::IndexOutOfStorage {
            index: destination,
            bytes_len,
        };
        let begin = destination.checked_mul(size).ok_or_else(out_of_storage)?;
        let end = begin.checked_add(size).ok_or_else(out_of_storage)?;
        target
            .get_mut(begin..end)
            .ok_or_else(out_of_storage)?
            .copy_from_slice(value);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use crate::backend::cpu::{
        CpuBackend,
        CpuError,
    };
    use crate::core::{
        DType,
        Shape,
        Tensor,
    };

    #[test]
    fn upload_then_download_returns_the_same_bytes_for_each_dtype() {
        let mut backend = CpuBackend::new();
        [
            (DType::F32, 2),
            (DType::F16, 4),
            (DType::BF16, 4),
            (DType::U32, 2),
        ]
        .into_iter()
        .for_each(|(dtype, len)| {
            let bytes: Vec<u8> = (1_u8..=8).collect();
            let downloaded = Tensor::upload(
                &mut backend,
                &bytes,
                dtype,
                Shape::try_from([len].as_slice()).expect("the shape is valid"),
            )
            .expect("the upload succeeds")
            .download(&mut backend)
            .expect("the download succeeds");
            assert_eq!(downloaded, bytes, "{dtype:?} round trip");
        });
    }

    #[test]
    fn zeros_download_only_zero_bytes() {
        let mut backend = CpuBackend::new();
        let downloaded = Tensor::zeros(
            &mut backend,
            DType::BF16,
            Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
        )
        .expect("the zeros succeed")
        .download(&mut backend)
        .expect("the download succeeds");
        assert_eq!(downloaded, vec![0; 12], "six bf16 zeros");
    }

    #[test]
    fn upload_with_wrong_byte_length_is_rejected() {
        let mut backend = CpuBackend::new();
        assert!(
            matches!(
                Tensor::upload(
                    &mut backend,
                    &[0; 6],
                    DType::F32,
                    Shape::try_from([2].as_slice()).expect("the shape is valid"),
                ),
                Err(CpuError::ByteLengthMismatch {
                    bytes: 6,
                    bytes_expected: 8,
                })
            ),
            "six bytes for two f32 are rejected"
        );
    }

    #[test]
    fn download_of_a_permuted_tensor_is_rejected() {
        let mut backend = CpuBackend::new();
        let tensor = Tensor::zeros(
            &mut backend,
            DType::F32,
            Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
        )
        .expect("the zeros succeed")
        .permute(&[1, 0])
        .expect("the permutation is valid");
        assert!(
            matches!(
                tensor.download(&mut backend),
                Err(CpuError::DownloadNonContiguous)
            ),
            "a permuted view is not downloaded"
        );
    }

    #[test]
    fn copy_of_a_permuted_u32_tensor_is_contiguous() {
        let mut backend = CpuBackend::new();
        let actual: Vec<u32> = Tensor::upload(
            &mut backend,
            &[1_u32, 2, 3, 4, 5, 6]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::U32,
            Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid")
        .contiguous(&mut backend)
        .expect("the copy succeeds")
        .download(&mut backend)
        .expect("the download succeeds")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&chunk| u32::from_le_bytes(chunk))
        .collect();
        assert_eq!(actual, vec![1, 4, 2, 5, 3, 6], "transposed values");
    }

    #[test]
    fn copy_of_a_broadcast_tensor_repeats_values() {
        let mut backend = CpuBackend::new();
        let actual: Vec<f32> = Tensor::upload(
            &mut backend,
            &[1.0_f32, 2.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .broadcast_as(Shape::try_from([3, 2].as_slice()).expect("the shape is valid"))
        .expect("the broadcast is valid")
        .contiguous(&mut backend)
        .expect("the copy succeeds")
        .download(&mut backend)
        .expect("the download succeeds")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&chunk| f32::from_le_bytes(chunk))
        .collect();
        assert_eq!(actual, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0], "repeated row");
    }

    #[test]
    fn copy_of_a_narrowed_tensor_keeps_the_sub_range() {
        let mut backend = CpuBackend::new();
        let actual: Vec<f32> = Tensor::upload(
            &mut backend,
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .narrow(1, 1, 2)
        .expect("the narrow is in bounds")
        .contiguous(&mut backend)
        .expect("the copy succeeds")
        .download(&mut backend)
        .expect("the download succeeds")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&chunk| f32::from_le_bytes(chunk))
        .collect();
        assert_eq!(actual, vec![2.0, 3.0, 5.0, 6.0], "last two columns");
    }

    #[test]
    fn cast_f32_to_bf16_rounds_to_nearest_even_and_keeps_nan() {
        let mut backend = CpuBackend::new();
        let actual: Vec<u16> = Tensor::upload(
            &mut backend,
            &[
                f32::from_bits(0x3F80_8000),
                f32::from_bits(0x3F81_8000),
                f32::NAN,
            ]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .cast(&mut backend, DType::BF16)
        .expect("the cast succeeds")
        .download(&mut backend)
        .expect("the download succeeds")
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&chunk| u16::from_le_bytes(chunk))
        .collect();
        assert_eq!(
            actual.get(..2),
            Some([0x3F80_u16, 0x3F82].as_slice()),
            "ties round to the even bf16"
        );
        assert!(
            actual
                .get(2)
                .is_some_and(|&bits| bits & 0x7F80 == 0x7F80 && bits & 0x007F != 0),
            "NaN stays NaN"
        );
    }

    #[test]
    fn cast_f32_to_f16_and_back_rounds_known_values() {
        let mut backend = CpuBackend::new();
        let actual: Vec<f32> = Tensor::upload(
            &mut backend,
            &[0.1_f32, 65_504.0, -2.5]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .cast(&mut backend, DType::F16)
        .expect("the cast succeeds")
        .cast(&mut backend, DType::F32)
        .expect("the cast succeeds")
        .download(&mut backend)
        .expect("the download succeeds")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&chunk| f32::from_le_bytes(chunk))
        .collect();
        assert_eq!(
            actual,
            vec![0.099_975_586, 65_504.0, -2.5],
            "nearest f16 values"
        );
    }

    #[test]
    fn unary_ops_match_hand_computed_values_for_each_float_dtype() {
        let mut backend = CpuBackend::new();
        [
            (DType::F32, 1e-5_f32),
            (DType::F16, 1e-2),
            (DType::BF16, 1e-2),
        ]
        .into_iter()
        .for_each(|(dtype, tolerance)| {
            let input = Tensor::upload(
                &mut backend,
                &[1.0_f32, 4.0]
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<u8>>(),
                DType::F32,
                Shape::try_from([2].as_slice()).expect("the shape is valid"),
            )
            .expect("the upload succeeds")
            .cast(&mut backend, dtype)
            .expect("the cast succeeds");
            [
                ("neg", input.neg(&mut backend), [-1.0_f32, -4.0]),
                ("exp", input.exp(&mut backend), [2.718_281_7, 54.598_15]),
                ("sqrt", input.sqrt(&mut backend), [1.0, 2.0]),
                ("recip", input.recip(&mut backend), [1.0, 0.25]),
            ]
            .into_iter()
            .for_each(|(name, output, expected)| {
                let actual: Vec<f32> = output
                    .expect("the unary succeeds")
                    .cast(&mut backend, DType::F32)
                    .expect("the cast succeeds")
                    .download(&mut backend)
                    .expect("the download succeeds")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&chunk| f32::from_le_bytes(chunk))
                    .collect();
                assert_eq!(actual.len(), expected.len(), "{name} element count");
                actual.iter().zip(expected).for_each(|(&actual, expected)| {
                    assert!(
                        (actual - expected).abs() <= tolerance * expected.abs(),
                        "{name} {dtype:?}: {actual} is not within {tolerance} of {expected}"
                    );
                });
            });
        });
    }

    #[test]
    fn binary_ops_match_hand_computed_values_for_each_float_dtype() {
        let mut backend = CpuBackend::new();
        [DType::F32, DType::F16, DType::BF16]
            .into_iter()
            .for_each(|dtype| {
                let [lhs, rhs] = [[6.0_f32, 8.0], [2.0, 4.0]].map(|values| {
                    Tensor::upload(
                        &mut backend,
                        &values
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<u8>>(),
                        DType::F32,
                        Shape::try_from([2].as_slice()).expect("the shape is valid"),
                    )
                    .expect("the upload succeeds")
                    .cast(&mut backend, dtype)
                    .expect("the cast succeeds")
                });
                [
                    ("add", lhs.add(&mut backend, &rhs), [8.0_f32, 12.0]),
                    ("sub", lhs.sub(&mut backend, &rhs), [4.0, 4.0]),
                    ("mul", lhs.mul(&mut backend, &rhs), [12.0, 32.0]),
                    ("div", lhs.div(&mut backend, &rhs), [3.0, 2.0]),
                ]
                .into_iter()
                .for_each(|(name, output, expected)| {
                    let actual: Vec<f32> = output
                        .expect("the binary succeeds")
                        .cast(&mut backend, DType::F32)
                        .expect("the cast succeeds")
                        .download(&mut backend)
                        .expect("the download succeeds")
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|&chunk| f32::from_le_bytes(chunk))
                        .collect();
                    assert_eq!(actual, expected.to_vec(), "{name} {dtype:?}");
                });
            });
    }

    #[test]
    fn add_broadcasts_a_row_over_a_matrix() {
        let mut backend = CpuBackend::new();
        let [matrix, row] = [
            (vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            (vec![10.0, 20.0, 30.0], vec![3]),
        ]
        .map(|(values, dims)| {
            Tensor::upload(
                &mut backend,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<u8>>(),
                DType::F32,
                Shape::try_from(dims.as_slice()).expect("the shape is valid"),
            )
            .expect("the upload succeeds")
        });
        let actual: Vec<f32> = matrix
            .add(&mut backend, &row)
            .expect("the add succeeds")
            .download(&mut backend)
            .expect("the download succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(
            actual,
            vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
            "the row is added to each matrix row"
        );
    }

    #[test]
    fn binary_on_u32_is_rejected() {
        let mut backend = CpuBackend::new();
        let tensor = Tensor::zeros(
            &mut backend,
            DType::U32,
            Shape::try_from([2].as_slice()).expect("the shape is valid"),
        )
        .expect("the zeros succeed");
        assert!(
            matches!(
                tensor.add(&mut backend, &tensor),
                Err(CpuError::UnsupportedDType {
                    primitive: "binary",
                    dtype: DType::U32,
                })
            ),
            "add rejects u32"
        );
    }

    #[test]
    fn sum_max_and_argmax_over_each_axis_match_hand_computed_values() {
        let mut backend = CpuBackend::new();
        let matrix = Tensor::upload(
            &mut backend,
            &[1.0_f32, 5.0, 3.0, 4.0, 2.0, 6.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds");
        [
            (
                "sum 0",
                matrix.sum(&mut backend, 0),
                vec![5.0_f32, 7.0, 9.0],
            ),
            ("sum 1", matrix.sum(&mut backend, 1), vec![9.0, 12.0]),
            ("max 0", matrix.max(&mut backend, 0), vec![4.0, 5.0, 6.0]),
            ("max 1", matrix.max(&mut backend, 1), vec![5.0, 6.0]),
        ]
        .into_iter()
        .for_each(|(name, output, expected)| {
            let actual: Vec<f32> = output
                .expect("the reduction succeeds")
                .download(&mut backend)
                .expect("the download succeeds")
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&chunk| f32::from_le_bytes(chunk))
                .collect();
            assert_eq!(actual, expected, "{name}");
        });
        [
            (
                "argmax 0",
                matrix.argmax(&mut backend, 0),
                vec![1_u32, 0, 1],
            ),
            ("argmax 1", matrix.argmax(&mut backend, 1), vec![1, 2]),
        ]
        .into_iter()
        .for_each(|(name, output, expected)| {
            let output = output.expect("the argmax succeeds");
            assert_eq!(output.dtype(), DType::U32, "{name} dtype");
            let actual: Vec<u32> = output
                .download(&mut backend)
                .expect("the download succeeds")
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&chunk| u32::from_le_bytes(chunk))
                .collect();
            assert_eq!(actual, expected, "{name}");
        });
    }

    #[test]
    fn argmax_returns_the_first_index_on_ties() {
        let mut backend = CpuBackend::new();
        let actual = Tensor::upload(
            &mut backend,
            &[3.0_f32, 7.0, 7.0, 1.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([4].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .argmax(&mut backend, 0)
        .expect("the argmax succeeds")
        .download(&mut backend)
        .expect("the download succeeds");
        assert_eq!(
            actual,
            1_u32.to_le_bytes().to_vec(),
            "first of the tied maxima"
        );
    }

    #[test]
    fn sum_over_an_empty_axis_is_zero() {
        let mut backend = CpuBackend::new();
        let actual: Vec<f32> = Tensor::zeros(
            &mut backend,
            DType::F32,
            Shape::try_from([2, 0].as_slice()).expect("the shape is valid"),
        )
        .expect("the zeros succeed")
        .sum(&mut backend, 1)
        .expect("the sum succeeds")
        .download(&mut backend)
        .expect("the download succeeds")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&chunk| f32::from_le_bytes(chunk))
        .collect();
        assert_eq!(actual, vec![0.0, 0.0], "one zero per row");
    }

    #[test]
    fn matmul_matches_hand_computed_values_for_each_float_dtype() {
        let mut backend = CpuBackend::new();
        [DType::F32, DType::F16, DType::BF16]
            .into_iter()
            .for_each(|dtype| {
                let [lhs, rhs] = [
                    (vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
                    (vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]),
                ]
                .map(|(values, dims)| {
                    Tensor::upload(
                        &mut backend,
                        &values
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<u8>>(),
                        DType::F32,
                        Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                    )
                    .expect("the upload succeeds")
                    .cast(&mut backend, dtype)
                    .expect("the cast succeeds")
                });
                let actual: Vec<f32> = lhs
                    .matmul(&mut backend, &rhs)
                    .expect("the matmul succeeds")
                    .cast(&mut backend, DType::F32)
                    .expect("the cast succeeds")
                    .download(&mut backend)
                    .expect("the download succeeds")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&chunk| f32::from_le_bytes(chunk))
                    .collect();
                assert_eq!(actual, vec![58.0, 64.0, 139.0, 154.0], "{dtype:?} product");
            });
    }

    #[test]
    fn matmul_reads_a_transposed_right_operand() {
        let mut backend = CpuBackend::new();
        let [lhs, rhs] = [
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0],
        ]
        .map(|values| {
            Tensor::upload(
                &mut backend,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<u8>>(),
                DType::F32,
                Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
            )
            .expect("the upload succeeds")
        });
        let actual: Vec<f32> = lhs
            .matmul(
                &mut backend,
                &rhs.permute(&[1, 0]).expect("the permutation is valid"),
            )
            .expect("the matmul succeeds")
            .download(&mut backend)
            .expect("the download succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(
            actual,
            vec![58.0, 64.0, 139.0, 154.0],
            "x times the transpose"
        );
    }

    #[test]
    fn batched_matmul_reads_permuted_batch_axes() {
        let mut backend = CpuBackend::new();
        let [lhs, rhs] = [
            (vec![1.0_f32, 2.0, 3.0, 4.0], vec![1, 2, 2]),
            (vec![5.0, 6.0, 7.0, 8.0], vec![2, 2, 1]),
        ]
        .map(|(values, dims)| {
            Tensor::upload(
                &mut backend,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<u8>>(),
                DType::F32,
                Shape::try_from(dims.as_slice()).expect("the shape is valid"),
            )
            .expect("the upload succeeds")
        });
        let output = lhs
            .permute(&[1, 0, 2])
            .expect("the permutation is valid")
            .matmul(&mut backend, &rhs)
            .expect("the matmul succeeds");
        assert_eq!(output.shape().dims(), &[2, 1, 1], "one product per batch");
        let actual: Vec<f32> = output
            .download(&mut backend)
            .expect("the download succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(actual, vec![17.0, 53.0], "1·5 + 2·6 and 3·7 + 4·8");
    }

    #[test]
    fn gather_selects_rows_and_writes_zeros_for_out_of_range_indices() {
        let mut backend = CpuBackend::new();
        let table = Tensor::upload(
            &mut backend,
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([3, 2].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds");
        let indices = Tensor::upload(
            &mut backend,
            &[2_u32, 0, 7]
                .iter()
                .flat_map(|index| index.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::U32,
            Shape::try_from([3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds");
        let actual: Vec<f32> = table
            .gather(&mut backend, &indices)
            .expect("the gather succeeds")
            .download(&mut backend)
            .expect("the download succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(
            actual,
            vec![5.0, 6.0, 1.0, 2.0, 0.0, 0.0],
            "rows 2 and 0, then a zero row"
        );
    }

    #[test]
    fn concat_on_first_and_last_axis_matches_hand_computed_values() {
        let mut backend = CpuBackend::new();
        let [lhs, rhs] = [[1.0_f32, 2.0], [3.0, 4.0]].map(|values| {
            Tensor::upload(
                &mut backend,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<u8>>(),
                DType::F32,
                Shape::try_from([1, 2].as_slice()).expect("the shape is valid"),
            )
            .expect("the upload succeeds")
        });
        [(0, [2, 2]), (1, [1, 4])]
            .into_iter()
            .for_each(|(axis, dims)| {
                let output = lhs
                    .concat(&mut backend, &rhs, axis)
                    .expect("the concat succeeds");
                assert_eq!(output.shape().dims(), &dims, "concat dims on axis {axis}");
                let actual: Vec<f32> = output
                    .download(&mut backend)
                    .expect("the download succeeds")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&chunk| f32::from_le_bytes(chunk))
                    .collect();
                assert_eq!(actual, vec![1.0, 2.0, 3.0, 4.0], "concat on axis {axis}");
            });
    }

    #[test]
    fn concat_of_u32_is_rejected() {
        let mut backend = CpuBackend::new();
        let tensor = Tensor::zeros(
            &mut backend,
            DType::U32,
            Shape::try_from([2].as_slice()).expect("the shape is valid"),
        )
        .expect("the zeros succeed");
        assert!(
            matches!(
                tensor.concat(&mut backend, &tensor, 0),
                Err(CpuError::UnsupportedDType {
                    primitive: "concat",
                    dtype: DType::U32,
                })
            ),
            "concat rejects u32"
        );
    }

    #[test]
    fn slice_update_writes_one_position_and_keeps_the_others() {
        let mut backend = CpuBackend::new();
        let mut target = Tensor::zeros(
            &mut backend,
            DType::F32,
            Shape::try_from([3, 2].as_slice()).expect("the shape is valid"),
        )
        .expect("the zeros succeed");
        let update = Tensor::upload(
            &mut backend,
            &[7.0_f32, 8.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([1, 2].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds");
        target
            .slice_update(&mut backend, &update, 0, 1)
            .expect("the slice update succeeds");
        let actual: Vec<f32> = target
            .download(&mut backend)
            .expect("the download succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(
            actual,
            vec![0.0, 0.0, 7.0, 8.0, 0.0, 0.0],
            "row 1 is written"
        );
    }
}
