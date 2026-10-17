mod error;

use std::iter;

use half::{
    bf16,
    f16,
};
use tracing::{
    error,
    warn,
};

pub use self::error::CpuError;
use crate::core::primitive::{
    BinaryOp,
    ConcatSpec,
    GatherSpec,
    MatmulSpec,
    ReduceOp,
    ReduceSpec,
    SliceUpdateSpec,
    UnaryOp,
};
use crate::core::{
    Backend,
    FloatDType,
    Layout,
    Operand,
    OperandMut,
    PeakMemory,
};

const CPU_NAME_FALLBACK: &str = "CPU";

pub struct CpuStorage {
    bytes: Vec<u8>,
}

pub struct CpuBackend;

struct StrideTerm {
    coordinate: usize,
    stride: usize,
}

struct StridedCursor {
    index: usize,
    rest: usize,
}

struct ArgmaxFold {
    value: f32,
    position: usize,
}

impl CpuBackend {
    pub fn new() -> Self { Self }
}

impl Backend for CpuBackend {
    type Error = CpuError;
    type Storage = CpuStorage;

    fn device_name(&self) -> String {
        cpu_brand().unwrap_or_else(|| {
            warn!(message = "CPU model name is unavailable; reporting the generic name.");
            CPU_NAME_FALLBACK.to_owned()
        })
    }

    fn peak_memory(&self) -> Option<PeakMemory> { None }

    fn upload(
        &mut self,
        bytes: &[u8],
    ) -> Result<CpuStorage, CpuError> {
        Ok(CpuStorage {
            bytes: bytes.to_vec(),
        })
    }

    fn zeros(
        &mut self,
        byte_len: usize,
    ) -> Result<CpuStorage, CpuError> {
        Ok(CpuStorage {
            bytes: vec![0; byte_len],
        })
    }

    fn download(
        &mut self,
        input: Operand<'_, CpuStorage>,
    ) -> Result<Vec<u8>, CpuError> {
        let layout = input.layout();
        let size = input.dtype().size_bytes();
        let bytes = &input.storage().bytes;
        let start = layout.offset().saturating_mul(size);
        let end = start.saturating_add(layout.shape().byte_len(input.dtype()));
        bytes
            .get(start..end)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| out_of_storage(start, bytes.len()))
    }

    fn unary(
        &mut self,
        op: UnaryOp,
        input: Operand<'_, CpuStorage, FloatDType>,
    ) -> Result<CpuStorage, CpuError> {
        let values: Vec<f32> = read_floats(&input)?
            .into_iter()
            .map(|value| match op {
                UnaryOp::Neg => -value,
                UnaryOp::Exp => value.exp(),
                UnaryOp::Sqrt => value.sqrt(),
                UnaryOp::Recip => 1.0 / value,
            })
            .collect();
        Ok(encode(input.dtype(), &values))
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        lhs: Operand<'_, CpuStorage, FloatDType>,
        rhs: Operand<'_, CpuStorage, FloatDType>,
    ) -> Result<CpuStorage, CpuError> {
        let values: Vec<f32> = read_floats(&lhs)?
            .into_iter()
            .zip(read_floats(&rhs)?)
            .map(|(lhs, rhs)| match op {
                BinaryOp::Add => lhs + rhs,
                BinaryOp::Sub => lhs - rhs,
                BinaryOp::Mul => lhs * rhs,
                BinaryOp::Div => lhs / rhs,
            })
            .collect();
        Ok(encode(lhs.dtype(), &values))
    }

    fn reduce(
        &mut self,
        input: Operand<'_, CpuStorage, FloatDType>,
        spec: &ReduceSpec,
    ) -> Result<CpuStorage, CpuError> {
        let layout = input.layout();
        let rows = (0..spec.output().element_count())
            .map(|row| {
                let base =
                    strided_index(spec.output().dims(), layout.strides(), layout.offset(), row)?;
                (0..spec.axis_len())
                    .map(|position| {
                        let index = offset_by(
                            base,
                            &[StrideTerm {
                                coordinate: position,
                                stride: spec.axis_stride(),
                            }],
                        )
                        .ok_or_else(|| index_overflow(row))?;
                        read_float(&input.storage().bytes, input.dtype(), index)
                    })
                    .collect::<Result<Vec<f32>, CpuError>>()
            })
            .collect::<Result<Vec<Vec<f32>>, CpuError>>()?;
        match spec.op() {
            ReduceOp::Sum => {
                let sums: Vec<f32> = rows
                    .iter()
                    .map(|row| row.iter().fold(0.0, |sum, &value| sum + value))
                    .collect();
                Ok(encode(input.dtype(), &sums))
            }
            ReduceOp::Max => {
                let maxima: Vec<f32> = rows
                    .iter()
                    .map(|row| row.iter().copied().fold(f32::NEG_INFINITY, f32::max))
                    .collect();
                Ok(encode(input.dtype(), &maxima))
            }
            ReduceOp::Argmax => {
                let indices = rows
                    .iter()
                    .map(|row| {
                        let best = row.iter().enumerate().fold(
                            ArgmaxFold {
                                value: f32::NEG_INFINITY,
                                position: 0,
                            },
                            |best, (position, &value)| {
                                if value > best.value {
                                    ArgmaxFold {
                                        value,
                                        position,
                                    }
                                } else {
                                    best
                                }
                            },
                        );
                        u32::try_from(best.position).map_err(|_| index_overflow(best.position))
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
        lhs: Operand<'_, CpuStorage, FloatDType>,
        rhs: Operand<'_, CpuStorage, FloatDType>,
        spec: &MatmulSpec,
    ) -> Result<CpuStorage, CpuError> {
        let m = spec.m();
        let k = spec.k();
        let n = spec.n();
        let lhs_matrix = spec.lhs();
        let rhs_matrix = spec.rhs();
        let batch_dims = spec.batch().dims();
        let values = (0..spec.output().element_count())
            .map(|linear| {
                let overflow = || index_overflow(linear);
                let col = linear.checked_rem(n).ok_or_else(overflow)?;
                let row = linear
                    .checked_div(n)
                    .and_then(|rows| rows.checked_rem(m))
                    .ok_or_else(overflow)?;
                let batch = m
                    .checked_mul(n)
                    .and_then(|matrix| linear.checked_div(matrix))
                    .ok_or_else(overflow)?;
                let lhs_base = strided_index(
                    batch_dims,
                    lhs_matrix.batch_strides(),
                    lhs_matrix.offset(),
                    batch,
                )?;
                let rhs_base = strided_index(
                    batch_dims,
                    rhs_matrix.batch_strides(),
                    rhs_matrix.offset(),
                    batch,
                )?;
                (0..k).try_fold(0.0_f32, |sum, inner| {
                    let lhs_index = offset_by(
                        lhs_base,
                        &[
                            StrideTerm {
                                coordinate: row,
                                stride: lhs_matrix.row_stride(),
                            },
                            StrideTerm {
                                coordinate: inner,
                                stride: lhs_matrix.col_stride(),
                            },
                        ],
                    )
                    .ok_or_else(overflow)?;
                    let rhs_index = offset_by(
                        rhs_base,
                        &[
                            StrideTerm {
                                coordinate: inner,
                                stride: rhs_matrix.row_stride(),
                            },
                            StrideTerm {
                                coordinate: col,
                                stride: rhs_matrix.col_stride(),
                            },
                        ],
                    )
                    .ok_or_else(overflow)?;
                    let lhs_value = read_float(&lhs.storage().bytes, spec.dtype(), lhs_index)?;
                    let rhs_value = read_float(&rhs.storage().bytes, spec.dtype(), rhs_index)?;
                    Ok(sum + lhs_value * rhs_value)
                })
            })
            .collect::<Result<Vec<f32>, CpuError>>()?;
        Ok(encode(spec.dtype(), &values))
    }

    fn copy(
        &mut self,
        input: Operand<'_, CpuStorage>,
    ) -> Result<CpuStorage, CpuError> {
        let layout = input.layout();
        let size = input.dtype().size_bytes();
        let bytes = (0..layout.shape().element_count()).try_fold(
            Vec::with_capacity(layout.shape().byte_len(input.dtype())),
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
        input: Operand<'_, CpuStorage, FloatDType>,
        dtype: FloatDType,
    ) -> Result<CpuStorage, CpuError> {
        Ok(encode(dtype, &read_floats(&input)?))
    }

    fn gather(
        &mut self,
        table: Operand<'_, CpuStorage, FloatDType>,
        indices: Operand<'_, CpuStorage>,
        spec: &GatherSpec,
    ) -> Result<CpuStorage, CpuError> {
        let size = spec.dtype().size_bytes();
        let cols = spec.cols();
        let indices_layout = indices.layout();
        let bytes = (0..spec.output().element_count()).try_fold(
            Vec::with_capacity(spec.output().byte_len(spec.dtype().into())),
            |mut bytes, linear| {
                let overflow = || index_overflow(linear);
                let position = linear.checked_div(cols).ok_or_else(overflow)?;
                let col = linear.checked_rem(cols).ok_or_else(overflow)?;
                let index_position = strided_index(
                    indices_layout.shape().dims(),
                    indices_layout.strides(),
                    indices_layout.offset(),
                    position,
                )?;
                let row = usize::try_from(read_u32(&indices.storage().bytes, index_position)?)
                    .map_err(|_| overflow())?;
                if row >= spec.rows() {
                    bytes.extend(iter::repeat_n(0, size));
                } else {
                    let index = offset_by(
                        table.layout().offset(),
                        &[
                            StrideTerm {
                                coordinate: row,
                                stride: spec.row_stride(),
                            },
                            StrideTerm {
                                coordinate: col,
                                stride: spec.col_stride(),
                            },
                        ],
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
        lhs: Operand<'_, CpuStorage, FloatDType>,
        rhs: Operand<'_, CpuStorage, FloatDType>,
        spec: &ConcatSpec,
    ) -> Result<CpuStorage, CpuError> {
        let layout = Layout::contiguous(*spec.output());
        let mut bytes = vec![0; spec.output().byte_len(spec.dtype().into())];
        write_window(&mut bytes, &layout, 0, &lhs)?;
        write_window(&mut bytes, &layout, spec.rhs_offset(), &rhs)?;
        Ok(CpuStorage {
            bytes,
        })
    }

    fn slice_update(
        &mut self,
        mut target: OperandMut<'_, CpuStorage, FloatDType>,
        update: Operand<'_, CpuStorage, FloatDType>,
        spec: &SliceUpdateSpec,
    ) -> Result<(), CpuError> {
        let layout = target.layout();
        write_window(
            &mut target.storage_mut().bytes,
            layout,
            spec.offset(),
            &update,
        )
    }
}

fn index_overflow(linear: usize) -> CpuError {
    error!(message = "Element index overflows usize.", linear);
    CpuError::IndexOverflow
}

fn out_of_storage(
    index: usize,
    bytes_len: usize,
) -> CpuError {
    error!(
        message = "Element is outside its storage.",
        index, bytes_len
    );
    CpuError::IndexOutOfStorage
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
        .try_fold(
            StridedCursor {
                index: offset,
                rest: linear,
            },
            |cursor, (&dim, &stride)| {
                Some(StridedCursor {
                    index: cursor
                        .rest
                        .checked_rem(dim)?
                        .checked_mul(stride)?
                        .checked_add(cursor.index)?,
                    rest: cursor.rest.checked_div(dim)?,
                })
            },
        )
        .map(|cursor| cursor.index)
        .ok_or_else(|| index_overflow(linear))
}

fn offset_by(
    base: usize,
    terms: &[StrideTerm],
) -> Option<usize> {
    terms.iter().try_fold(base, |index, term| {
        term.coordinate.checked_mul(term.stride)?.checked_add(index)
    })
}

fn element_bytes(
    bytes: &[u8],
    size: usize,
    index: usize,
) -> Result<&[u8], CpuError> {
    let start = index
        .checked_mul(size)
        .ok_or_else(|| out_of_storage(index, bytes.len()))?;
    let end = start
        .checked_add(size)
        .ok_or_else(|| out_of_storage(index, bytes.len()))?;
    bytes
        .get(start..end)
        .ok_or_else(|| out_of_storage(index, bytes.len()))
}

fn read_float(
    bytes: &[u8],
    dtype: FloatDType,
    index: usize,
) -> Result<f32, CpuError> {
    match (dtype, element_bytes(bytes, dtype.size_bytes(), index)?) {
        (FloatDType::F32, &[b0, b1, b2, b3]) => Ok(f32::from_le_bytes([b0, b1, b2, b3])),
        (FloatDType::F16, &[b0, b1]) => Ok(f16::from_le_bytes([b0, b1]).to_f32()),
        (FloatDType::BF16, &[b0, b1]) => Ok(bf16::from_le_bytes([b0, b1]).to_f32()),
        (FloatDType::F32 | FloatDType::F16 | FloatDType::BF16, _) => {
            Err(out_of_storage(index, bytes.len()))
        }
    }
}

fn read_u32(
    bytes: &[u8],
    index: usize,
) -> Result<u32, CpuError> {
    match element_bytes(bytes, size_of::<u32>(), index)? {
        &[b0, b1, b2, b3] => Ok(u32::from_le_bytes([b0, b1, b2, b3])),
        _ => Err(out_of_storage(index, bytes.len())),
    }
}

fn read_floats(input: &Operand<'_, CpuStorage, FloatDType>) -> Result<Vec<f32>, CpuError> {
    let layout = input.layout();
    (0..layout.shape().element_count())
        .map(|linear| {
            let index = strided_index(
                layout.shape().dims(),
                layout.strides(),
                layout.offset(),
                linear,
            )?;
            read_float(&input.storage().bytes, input.dtype(), index)
        })
        .collect()
}

fn encode(
    dtype: FloatDType,
    values: &[f32],
) -> CpuStorage {
    let bytes = match dtype {
        FloatDType::F32 => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        FloatDType::F16 => values
            .iter()
            .flat_map(|&value| f16::from_f32(value).to_le_bytes())
            .collect(),
        FloatDType::BF16 => values
            .iter()
            .flat_map(|&value| bf16::from_f32(value).to_le_bytes())
            .collect(),
    };
    CpuStorage {
        bytes,
    }
}

fn write_window(
    target: &mut [u8],
    target_layout: &Layout,
    offset: usize,
    update: &Operand<'_, CpuStorage, FloatDType>,
) -> Result<(), CpuError> {
    let size = update.dtype().size_bytes();
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
        let begin = destination
            .checked_mul(size)
            .ok_or_else(|| out_of_storage(destination, bytes_len))?;
        let end = begin
            .checked_add(size)
            .ok_or_else(|| out_of_storage(destination, bytes_len))?;
        target
            .get_mut(begin..end)
            .ok_or_else(|| out_of_storage(destination, bytes_len))?
            .copy_from_slice(value);
        Ok(())
    })
}

#[cfg(target_os = "macos")]
fn cpu_brand() -> Option<String> {
    let output = std::process::Command::new("sysctl")
        .arg("machdep.cpu.brand_string")
        .output()
        .ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    output
        .status
        .success()
        .then(|| field_value(&text, "machdep.cpu.brand_string"))
        .flatten()
}

#[cfg(target_os = "linux")]
fn cpu_brand() -> Option<String> {
    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    field_value(&text, "model name")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn cpu_brand() -> Option<String> { None }

#[cfg(any(target_os = "macos", target_os = "linux", test))]
fn field_value(
    text: &str,
    key: &str,
) -> Option<String> {
    text.lines()
        .find_map(|line| {
            line.split_once(':')
                .filter(|(line_key, _)| line_key.trim() == key)
                .map(|(_, value)| value.trim().to_owned())
        })
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use crate::backend::cpu::field_value;

    #[test]
    fn field_value_reads_the_first_matching_key() {
        struct TextKeyExpectedCase<T0, T1, T2> {
            text: T0,
            key: T1,
            expected: T2,
        }
        [
            TextKeyExpectedCase {
                text: "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Xeon(R) \
                       CPU @ 2.20GHz\nprocessor\t: 1\nmodel name\t: other\n",
                key: "model name",
                expected: Some("Intel(R) Xeon(R) CPU @ 2.20GHz".to_owned()),
            },
            TextKeyExpectedCase {
                text: "machdep.cpu.brand_string: Apple M4 Pro\n",
                key: "machdep.cpu.brand_string",
                expected: Some("Apple M4 Pro".to_owned()),
            },
            TextKeyExpectedCase {
                text: "processor\t: 0\nCPU implementer\t: 0x41\n",
                key: "model name",
                expected: None,
            },
            TextKeyExpectedCase {
                text: "model name\t:   \n",
                key: "model name",
                expected: None,
            },
        ]
        .into_iter()
        .for_each(
            |TextKeyExpectedCase {
                 text,
                 key,
                 expected,
             }| {
                assert_eq!(field_value(text, key), expected, "key {key:?} in {text:?}");
            },
        );
    }
}
