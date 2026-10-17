use std::error::Error;

use tracing::error;

use crate::nn::{
    AllocatedMemory,
    CoreError,
    DType,
    Operand,
    Shape,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// Flips the sign of each element.
    Neg,
    /// Computes e raised to each element.
    Exp,
    /// Computes the square root of each element.
    Sqrt,
    /// Computes one divided by each element.
    Reciprocal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// Adds two elements.
    Add,
    /// Subtracts the right element from the left one.
    Sub,
    /// Multiplies two elements.
    Mul,
    /// Divides the left element by the right one.
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Adds all elements along a dimension.
    Sum,
    /// Keeps the largest element along a dimension.
    Amax,
    /// Keeps the index of the largest element along a dimension.
    Argmax,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    /// One input, same shape and float dtype.
    Unary(UnaryOp),
    /// Two inputs of the same float dtype, broadcast together.
    Binary(BinaryOp),
    /// One input reduced along `dim`, kept with size one.
    Reduce { op: ReduceOp, dim: usize },
    /// Two inputs multiplied over their last two dimensions.
    Matmul,
    /// One input copied to a contiguous layout.
    Copy,
    /// One float input converted to the float dtype.
    ToDtype(DType),
    /// A 2D table and 1D `u32` indices selecting rows along `dim`.
    IndexSelect { dim: usize },
    /// Two inputs joined along `dim`.
    Cat { dim: usize },
    /// A target and a source written into it from `start` along `dim`.
    SliceScatter { dim: usize, start: usize },
}

pub trait Backend {
    type Storage;
    type Error: Error + 'static;

    fn device_name(&self) -> String;

    fn current_allocated_memory(&self) -> Option<AllocatedMemory>;

    fn storage_from_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<Self::Storage, Self::Error>;

    fn zeros(
        &mut self,
        nbytes: usize,
    ) -> Result<Self::Storage, Self::Error>;

    fn storage_to_bytes(
        &mut self,
        input: &Operand<&Self::Storage>,
    ) -> Result<Vec<u8>, Self::Error>;

    fn run(
        &mut self,
        op: Op,
        inputs: &[Operand<&Self::Storage>],
        output: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage, Self::Error>;

    fn run_in_place(
        &mut self,
        op: Op,
        target: Operand<&mut Self::Storage>,
        inputs: &[Operand<&Self::Storage>],
    ) -> Result<(), Self::Error>;
}

impl Op {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Unary(_) => "unary",
            Self::Binary(_) => "binary",
            Self::Reduce {
                ..
            } => "reduce",
            Self::Matmul => "matmul",
            Self::Copy => "copy",
            Self::ToDtype(_) => "to_dtype",
            Self::IndexSelect {
                ..
            } => "index_select",
            Self::Cat {
                ..
            } => "cat",
            Self::SliceScatter {
                ..
            } => "slice_scatter",
        }
    }
}

pub fn output(
    op: Op,
    inputs: &[(DType, Shape)],
) -> Result<(Shape, DType), CoreError> {
    match (op, inputs) {
        (Op::Unary(_), &[(dtype, shape)]) => Ok((shape, require_float(dtype)?)),
        (Op::Binary(_), &[(lhs_dtype, lhs), (rhs_dtype, rhs)]) => {
            let dtype = require_same_float(lhs_dtype, rhs_dtype)?;
            Ok((Shape::broadcast(&lhs, &rhs)?, dtype))
        }
        (
            Op::Reduce {
                op,
                dim,
            },
            &[(dtype, shape)],
        ) => reduce_output(op, dim, dtype, &shape),
        (Op::Matmul, &[(lhs_dtype, lhs), (rhs_dtype, rhs)]) => {
            let dtype = require_same_float(lhs_dtype, rhs_dtype)?;
            Ok((matmul_output(&lhs, &rhs)?, dtype))
        }
        (Op::Copy, &[(dtype, shape)]) => Ok((shape, dtype)),
        (Op::ToDtype(to), &[(from, shape)]) => {
            require_float(from)?;
            Ok((shape, require_float(to)?))
        }
        (
            Op::IndexSelect {
                dim,
            },
            &[(table_dtype, table), (indices_dtype, indices)],
        ) => index_select_output(dim, table_dtype, &table, indices_dtype, &indices),
        (
            Op::Cat {
                dim,
            },
            &[(lhs_dtype, lhs), (rhs_dtype, rhs)],
        ) => {
            let dtype = require_same_float(lhs_dtype, rhs_dtype)?;
            Ok((cat_output(dim, &lhs, &rhs)?, dtype))
        }
        (
            Op::SliceScatter {
                dim,
                start,
            },
            &[(target_dtype, target), (src_dtype, src)],
        ) => {
            let dtype = require_same_float(target_dtype, src_dtype)?;
            check_slice_scatter(dim, start, &target, &src)?;
            Ok((target, dtype))
        }
        _ => {
            error!(
                message = "Operand count does not match the operation.",
                ?op,
                operand_count = inputs.len(),
            );
            Err(CoreError::OperandCount)
        }
    }
}

fn reduce_output(
    op: ReduceOp,
    dim: usize,
    dtype: DType,
    shape: &Shape,
) -> Result<(Shape, DType), CoreError> {
    let dtype = require_float(dtype)?;
    let dim_size = shape.size(dim)?;
    let needs_element = matches!(op, ReduceOp::Amax | ReduceOp::Argmax);
    if needs_element && dim_size == 0 {
        error!(
            message = "Reduction over an empty dimension has no result.",
            ?op,
            dim,
            dims = ?shape.dims(),
        );
        return Err(CoreError::EmptyReduction);
    }
    let dtype_output = match op {
        ReduceOp::Sum | ReduceOp::Amax => dtype,
        ReduceOp::Argmax => DType::U32,
    };
    Ok((with_dim(shape, dim, 1)?, dtype_output))
}

fn matmul_output(
    lhs: &Shape,
    rhs: &Shape,
) -> Result<Shape, CoreError> {
    let incompatible = || {
        error!(
            message = "Shapes cannot be multiplied.",
            lhs = ?lhs.dims(),
            rhs = ?rhs.dims(),
        );
        CoreError::MatmulIncompatible
    };
    let Some((lhs_batch, &[m, lhs_k])) = lhs.dims().split_last_chunk::<2>() else {
        return Err(incompatible());
    };
    let Some((rhs_batch, &[rhs_k, n])) = rhs.dims().split_last_chunk::<2>() else {
        return Err(incompatible());
    };
    if lhs_batch != rhs_batch || lhs_k != rhs_k {
        return Err(incompatible());
    }
    let dims: Vec<usize> = lhs_batch.iter().copied().chain([m, n]).collect();
    Shape::try_from(dims.as_slice())
}

fn index_select_output(
    dim: usize,
    table_dtype: DType,
    table: &Shape,
    indices_dtype: DType,
    indices: &Shape,
) -> Result<(Shape, DType), CoreError> {
    let dtype = require_float(table_dtype)?;
    if indices_dtype != DType::U32 {
        error!(message = "Index select indices are not u32.", dtype = ?indices_dtype);
        return Err(CoreError::IndicesNotU32);
    }
    let incompatible = || {
        error!(
            message = "Indices cannot select rows of the table.",
            dim,
            table = ?table.dims(),
            indices = ?indices.dims(),
        );
        CoreError::IndexSelectIncompatible
    };
    if dim != 0 {
        return Err(incompatible());
    }
    let &[_, cols] = table.dims() else {
        return Err(incompatible());
    };
    let &[index_count] = indices.dims() else {
        return Err(incompatible());
    };
    Ok((Shape::try_from([index_count, cols].as_slice())?, dtype))
}

fn cat_output(
    dim: usize,
    lhs: &Shape,
    rhs: &Shape,
) -> Result<Shape, CoreError> {
    let incompatible = || {
        error!(
            message = "Shapes cannot be concatenated.",
            dim,
            lhs = ?lhs.dims(),
            rhs = ?rhs.dims(),
        );
        CoreError::CatIncompatible
    };
    let lhs_size = lhs.size(dim)?;
    let rhs_size = rhs.size(dim)?;
    if !dims_equal_except(lhs, rhs, dim) {
        return Err(incompatible());
    }
    let size_output = lhs_size.checked_add(rhs_size).ok_or_else(incompatible)?;
    with_dim(lhs, dim, size_output)
}

fn check_slice_scatter(
    dim: usize,
    start: usize,
    target: &Shape,
    src: &Shape,
) -> Result<(), CoreError> {
    let target_size = target.size(dim)?;
    let src_size = src.size(dim)?;
    let fits = start
        .checked_add(src_size)
        .is_some_and(|end| end <= target_size);
    if !fits || !dims_equal_except(target, src, dim) {
        error!(
            message = "Slice scatter does not fit the target.",
            dim,
            start,
            target = ?target.dims(),
            src = ?src.dims(),
        );
        return Err(CoreError::SliceScatterOutOfBounds);
    }
    Ok(())
}

fn require_float(dtype: DType) -> Result<DType, CoreError> {
    match dtype.float() {
        Some(_) => Ok(dtype),
        None => {
            error!(message = "DType is not a float dtype.", ?dtype);
            Err(CoreError::DTypeNotFloat)
        }
    }
}

fn require_same_float(
    lhs: DType,
    rhs: DType,
) -> Result<DType, CoreError> {
    if lhs != rhs {
        error!(message = "DTypes do not match.", ?lhs, ?rhs);
        return Err(CoreError::DTypeMismatch);
    }
    require_float(lhs)
}

fn with_dim(
    shape: &Shape,
    dim: usize,
    size: usize,
) -> Result<Shape, CoreError> {
    shape.size(dim)?;
    let dims: Vec<usize> = shape
        .dims()
        .iter()
        .enumerate()
        .map(|(index, &current)| if index == dim { size } else { current })
        .collect();
    Shape::try_from(dims.as_slice())
}

fn dims_equal_except(
    lhs: &Shape,
    rhs: &Shape,
    dim: usize,
) -> bool {
    lhs.ndim() == rhs.ndim()
        && lhs
            .dims()
            .iter()
            .zip(rhs.dims())
            .enumerate()
            .all(|(index, (lhs_size, rhs_size))| index == dim || lhs_size == rhs_size)
}

#[cfg(test)]
mod tests {
    use crate::nn::primitive::{
        BinaryOp,
        Op,
        ReduceOp,
        UnaryOp,
        output,
    };
    use crate::nn::{
        CoreError,
        DType,
        Layout,
        Shape,
    };

    #[test]
    fn unary_keeps_shape_and_float_dtype() {
        assert_eq!(
            output(
                Op::Unary(UnaryOp::Exp),
                &[(
                    DType::BF16,
                    Shape::try_from([2, 3].as_slice()).expect("valid shape")
                )]
            ),
            Ok((
                Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                DType::BF16
            )),
            "a float dtype is kept"
        );
    }

    #[test]
    fn unary_rejects_integer_dtype() {
        assert_eq!(
            output(
                Op::Unary(UnaryOp::Neg),
                &[(
                    DType::U32,
                    Shape::try_from([2].as_slice()).expect("valid shape")
                )]
            ),
            Err(CoreError::DTypeNotFloat),
            "unary needs a float dtype"
        );
    }

    #[test]
    fn operand_count_must_match_the_operation() {
        assert_eq!(
            [
                output(Op::Unary(UnaryOp::Neg), &[]),
                output(
                    Op::Matmul,
                    &[(
                        DType::F32,
                        Shape::try_from([2, 3].as_slice()).expect("valid shape")
                    )]
                ),
                output(
                    Op::Copy,
                    &[
                        (
                            DType::F32,
                            Shape::try_from([2].as_slice()).expect("valid shape")
                        ),
                        (
                            DType::F32,
                            Shape::try_from([2].as_slice()).expect("valid shape")
                        )
                    ]
                ),
            ],
            [
                Err(CoreError::OperandCount),
                Err(CoreError::OperandCount),
                Err(CoreError::OperandCount),
            ],
            "no operand, one operand for matmul, two operands for copy"
        );
    }

    #[test]
    fn binary_broadcasts_shapes() {
        assert_eq!(
            output(
                Op::Binary(BinaryOp::Add),
                &[
                    (
                        DType::BF16,
                        Shape::try_from([2, 3].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::BF16,
                        Shape::try_from([3].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Ok((
                Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                DType::BF16
            )),
            "dtype and broadcast shape"
        );
    }

    #[test]
    fn binary_rejects_dtype_mismatch() {
        assert_eq!(
            output(
                Op::Binary(BinaryOp::Mul),
                &[
                    (
                        DType::BF16,
                        Shape::try_from([2, 3].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::F32,
                        Shape::try_from([2, 3].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Err(CoreError::DTypeMismatch),
            "binary needs equal dtypes"
        );
    }

    #[test]
    fn binary_rejects_integer_dtype() {
        assert_eq!(
            output(
                Op::Binary(BinaryOp::Sub),
                &[
                    (
                        DType::U32,
                        Shape::try_from([2, 3].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::U32,
                        Shape::try_from([2, 3].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Err(CoreError::DTypeNotFloat),
            "binary needs a float dtype"
        );
    }

    #[test]
    fn binary_rejects_incompatible_shapes() {
        assert_eq!(
            output(
                Op::Binary(BinaryOp::Div),
                &[
                    (
                        DType::BF16,
                        Shape::try_from([2, 3].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::BF16,
                        Shape::try_from([4, 3].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Err(CoreError::BroadcastIncompatible),
            "dims 2 and 4 do not broadcast"
        );
    }

    #[test]
    fn reduce_keeps_dim_with_size_one() {
        assert_eq!(
            output(
                Op::Reduce {
                    op: ReduceOp::Sum,
                    dim: 1,
                },
                &[(
                    DType::BF16,
                    Shape::try_from([2, 3, 4].as_slice()).expect("valid shape")
                )]
            ),
            Ok((
                Shape::try_from([2, 1, 4].as_slice()).expect("valid shape"),
                DType::BF16
            )),
            "a sum keeps the dtype and the dim with size one"
        );
    }

    #[test]
    fn reduce_argmax_outputs_u32() {
        assert_eq!(
            output(
                Op::Reduce {
                    op: ReduceOp::Argmax,
                    dim: 1,
                },
                &[(
                    DType::F32,
                    Shape::try_from([2, 3].as_slice()).expect("valid shape")
                )]
            ),
            Ok((
                Shape::try_from([2, 1].as_slice()).expect("valid shape"),
                DType::U32
            )),
            "argmax outputs u32 indices"
        );
    }

    #[test]
    fn reduce_rejects_integer_dtype() {
        assert_eq!(
            output(
                Op::Reduce {
                    op: ReduceOp::Sum,
                    dim: 1,
                },
                &[(
                    DType::U32,
                    Shape::try_from([2, 3].as_slice()).expect("valid shape")
                )]
            ),
            Err(CoreError::DTypeNotFloat),
            "reduce needs a float dtype"
        );
    }

    #[test]
    fn reduce_rejects_dim_out_of_range() {
        assert_eq!(
            output(
                Op::Reduce {
                    op: ReduceOp::Sum,
                    dim: 2,
                },
                &[(
                    DType::F32,
                    Shape::try_from([2, 3].as_slice()).expect("valid shape")
                )]
            ),
            Err(CoreError::DimOutOfRange),
            "dim 2 does not exist"
        );
    }

    #[test]
    fn reduce_amax_and_argmax_over_empty_dim_are_rejected() {
        [ReduceOp::Amax, ReduceOp::Argmax]
            .into_iter()
            .for_each(|op| {
                assert_eq!(
                    output(
                        Op::Reduce {
                            op,
                            dim: 1,
                        },
                        &[(
                            DType::F32,
                            Shape::try_from([2, 0].as_slice()).expect("valid shape")
                        )]
                    ),
                    Err(CoreError::EmptyReduction),
                    "{op:?} over an empty dim"
                );
            });
    }

    #[test]
    fn reduce_sum_over_empty_dim_keeps_dim_with_size_one() {
        assert_eq!(
            output(
                Op::Reduce {
                    op: ReduceOp::Sum,
                    dim: 1,
                },
                &[(
                    DType::F32,
                    Shape::try_from([2, 0].as_slice()).expect("valid shape")
                )]
            ),
            Ok((
                Shape::try_from([2, 1].as_slice()).expect("valid shape"),
                DType::F32
            )),
            "a sum over an empty dim is zero"
        );
    }

    #[test]
    fn matmul_multiplies_last_two_dims() {
        assert_eq!(
            output(
                Op::Matmul,
                &[
                    (
                        DType::BF16,
                        Shape::try_from([5, 2, 3].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::BF16,
                        Shape::try_from([5, 3, 4].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Ok((
                Shape::try_from([5, 2, 4].as_slice()).expect("valid shape"),
                DType::BF16
            )),
            "batch kept, m by n output"
        );
    }

    #[test]
    fn matmul_rejects_inner_dim_mismatch() {
        assert_eq!(
            output(
                Op::Matmul,
                &[
                    (
                        DType::F32,
                        Shape::try_from([2, 3].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::F32,
                        Shape::try_from([4, 5].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Err(CoreError::MatmulIncompatible),
            "inner dims 3 and 4 differ"
        );
    }

    #[test]
    fn matmul_rejects_batch_mismatch() {
        assert_eq!(
            output(
                Op::Matmul,
                &[
                    (
                        DType::F32,
                        Shape::try_from([2, 2, 3].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::F32,
                        Shape::try_from([5, 3, 4].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Err(CoreError::MatmulIncompatible),
            "batch dims 2 and 5 differ"
        );
    }

    #[test]
    fn matmul_rejects_rank_below_two() {
        assert_eq!(
            [
                output(
                    Op::Matmul,
                    &[
                        (
                            DType::F32,
                            Shape::try_from([3].as_slice()).expect("valid shape")
                        ),
                        (
                            DType::F32,
                            Shape::try_from([3, 4].as_slice()).expect("valid shape")
                        )
                    ]
                ),
                output(
                    Op::Matmul,
                    &[
                        (
                            DType::F32,
                            Shape::try_from([2, 3].as_slice()).expect("valid shape")
                        ),
                        (
                            DType::F32,
                            Shape::try_from([3].as_slice()).expect("valid shape")
                        )
                    ]
                ),
            ],
            [
                Err(CoreError::MatmulIncompatible),
                Err(CoreError::MatmulIncompatible)
            ],
            "a rank 1 lhs, then a rank 1 rhs"
        );
    }

    #[test]
    fn matmul_rejects_integer_dtype() {
        assert_eq!(
            output(
                Op::Matmul,
                &[
                    (
                        DType::U32,
                        Shape::try_from([2, 3].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::U32,
                        Shape::try_from([3, 4].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Err(CoreError::DTypeNotFloat),
            "matmul needs a float dtype"
        );
    }

    #[test]
    fn matmul_rejects_dtype_mismatch() {
        assert_eq!(
            output(
                Op::Matmul,
                &[
                    (
                        DType::BF16,
                        Shape::try_from([2, 3].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::F32,
                        Shape::try_from([3, 4].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Err(CoreError::DTypeMismatch),
            "matmul needs equal dtypes"
        );
    }

    #[test]
    fn matmul_rejects_output_beyond_the_element_limit() {
        assert_eq!(
            output(
                Op::Matmul,
                &[
                    (
                        DType::F32,
                        Shape::try_from([1 << 16, 1].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::F32,
                        Shape::try_from([1, 1 << 16].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Err(CoreError::NumelOverflow),
            "the output holds 2^32 elements"
        );
    }

    #[test]
    fn copy_keeps_shape_and_any_dtype() {
        assert_eq!(
            output(
                Op::Copy,
                &[(
                    DType::U32,
                    Shape::try_from([2, 3].as_slice()).expect("valid shape")
                )]
            ),
            Ok((
                Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                DType::U32
            )),
            "copy accepts integers"
        );
    }

    #[test]
    fn to_dtype_returns_the_target_dtype() {
        assert_eq!(
            output(
                Op::ToDtype(DType::BF16),
                &[(
                    DType::F32,
                    Shape::try_from([3].as_slice()).expect("valid shape")
                )]
            ),
            Ok((
                Shape::try_from([3].as_slice()).expect("valid shape"),
                DType::BF16
            )),
            "shape kept, target dtype"
        );
    }

    #[test]
    fn to_dtype_rejects_integer_source_and_target() {
        assert_eq!(
            [
                output(
                    Op::ToDtype(DType::F32),
                    &[(
                        DType::U32,
                        Shape::try_from([3].as_slice()).expect("valid shape")
                    )]
                ),
                output(
                    Op::ToDtype(DType::U32),
                    &[(
                        DType::F32,
                        Shape::try_from([3].as_slice()).expect("valid shape")
                    )]
                ),
            ],
            [Err(CoreError::DTypeNotFloat), Err(CoreError::DTypeNotFloat)],
            "from u32, then to u32"
        );
    }

    #[test]
    fn index_select_selects_table_rows() {
        assert_eq!(
            output(
                Op::IndexSelect {
                    dim: 0,
                },
                &[
                    (
                        DType::BF16,
                        Shape::try_from([3, 4].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::U32,
                        Shape::try_from([5].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Ok((
                Shape::try_from([5, 4].as_slice()).expect("valid shape"),
                DType::BF16
            )),
            "one row of the table per index"
        );
    }

    #[test]
    fn index_select_rejects_integer_table() {
        assert_eq!(
            output(
                Op::IndexSelect {
                    dim: 0,
                },
                &[
                    (
                        DType::U32,
                        Shape::try_from([4, 3].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::U32,
                        Shape::try_from([2].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Err(CoreError::DTypeNotFloat),
            "the table needs a float dtype"
        );
    }

    #[test]
    fn index_select_rejects_non_u32_indices() {
        assert_eq!(
            output(
                Op::IndexSelect {
                    dim: 0,
                },
                &[
                    (
                        DType::F32,
                        Shape::try_from([4, 3].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::F32,
                        Shape::try_from([2].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Err(CoreError::IndicesNotU32),
            "indices must be u32"
        );
    }

    #[test]
    fn index_select_rejects_invalid_ranks() {
        [
            (
                Shape::try_from([4].as_slice()).expect("valid shape"),
                Shape::try_from([2].as_slice()).expect("valid shape"),
            ),
            (
                Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"),
                Shape::try_from([2].as_slice()).expect("valid shape"),
            ),
            (
                Shape::try_from([4, 3].as_slice()).expect("valid shape"),
                Shape::try_from([2, 1].as_slice()).expect("valid shape"),
            ),
        ]
        .into_iter()
        .for_each(|(table, indices)| {
            assert_eq!(
                output(
                    Op::IndexSelect {
                        dim: 0,
                    },
                    &[(DType::F32, table), (DType::U32, indices)]
                ),
                Err(CoreError::IndexSelectIncompatible),
                "table {:?} and indices {:?}",
                table.dims(),
                indices.dims()
            );
        });
    }

    #[test]
    fn index_select_rejects_a_dim_other_than_zero() {
        assert_eq!(
            output(
                Op::IndexSelect {
                    dim: 1,
                },
                &[
                    (
                        DType::F32,
                        Shape::try_from([4, 3].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::U32,
                        Shape::try_from([2].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Err(CoreError::IndexSelectIncompatible),
            "only rows of the table are selected"
        );
    }

    #[test]
    fn index_select_rejects_output_beyond_the_element_limit() {
        assert_eq!(
            output(
                Op::IndexSelect {
                    dim: 0,
                },
                &[
                    (
                        DType::F32,
                        Shape::try_from([1, 1 << 16].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::U32,
                        Shape::try_from([1 << 16].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Err(CoreError::NumelOverflow),
            "the output holds 2^32 elements"
        );
    }

    #[test]
    fn cat_adds_the_sizes_along_dim() {
        assert_eq!(
            output(
                Op::Cat {
                    dim: 1,
                },
                &[
                    (
                        DType::BF16,
                        Shape::try_from([2, 3, 4].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::BF16,
                        Shape::try_from([2, 5, 4].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Ok((
                Shape::try_from([2, 8, 4].as_slice()).expect("valid shape"),
                DType::BF16
            )),
            "sizes 3 and 5 add along dim 1"
        );
    }

    #[test]
    fn cat_rejects_mismatch_outside_dim() {
        assert_eq!(
            output(
                Op::Cat {
                    dim: 1,
                },
                &[
                    (
                        DType::F32,
                        Shape::try_from([2, 3].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::F32,
                        Shape::try_from([4, 5].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Err(CoreError::CatIncompatible),
            "dims 2 and 4 differ outside the dim"
        );
    }

    #[test]
    fn cat_rejects_rank_mismatch() {
        assert_eq!(
            output(
                Op::Cat {
                    dim: 0,
                },
                &[
                    (
                        DType::F32,
                        Shape::try_from([2, 3, 4].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::F32,
                        Shape::try_from([2, 3].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Err(CoreError::CatIncompatible),
            "ranks 3 and 2 differ"
        );
    }

    #[test]
    fn cat_rejects_dtype_mismatch_and_integer_dtype() {
        let cat = Op::Cat {
            dim: 0,
        };
        assert_eq!(
            [
                output(
                    cat,
                    &[
                        (
                            DType::F32,
                            Shape::try_from([2, 3].as_slice()).expect("valid shape")
                        ),
                        (
                            DType::BF16,
                            Shape::try_from([2, 3].as_slice()).expect("valid shape")
                        )
                    ]
                ),
                output(
                    cat,
                    &[
                        (
                            DType::U32,
                            Shape::try_from([2, 3].as_slice()).expect("valid shape")
                        ),
                        (
                            DType::U32,
                            Shape::try_from([2, 3].as_slice()).expect("valid shape")
                        )
                    ]
                ),
            ],
            [Err(CoreError::DTypeMismatch), Err(CoreError::DTypeNotFloat)],
            "cat needs equal float dtypes"
        );
    }

    #[test]
    fn cat_rejects_dim_out_of_range() {
        [
            (
                Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                Shape::try_from([2, 3].as_slice()).expect("valid shape"),
            ),
            (
                Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"),
                Shape::try_from([2, 3].as_slice()).expect("valid shape"),
            ),
        ]
        .into_iter()
        .for_each(|(lhs, rhs)| {
            assert_eq!(
                output(
                    Op::Cat {
                        dim: 2,
                    },
                    &[(DType::F32, lhs), (DType::F32, rhs)]
                ),
                Err(CoreError::DimOutOfRange),
                "lhs {:?} and rhs {:?}",
                lhs.dims(),
                rhs.dims()
            );
        });
    }

    #[test]
    fn cat_rejects_output_beyond_the_element_limit() {
        assert_eq!(
            output(
                Op::Cat {
                    dim: 0,
                },
                &[
                    (
                        DType::F32,
                        Shape::try_from([1 << 30].as_slice()).expect("valid shape")
                    ),
                    (
                        DType::F32,
                        Shape::try_from([1 << 30].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Err(CoreError::NumelOverflow),
            "the output holds 2^31 elements"
        );
    }

    #[test]
    fn slice_scatter_keeps_the_target() {
        let target =
            *Layout::contiguous(Shape::try_from([2, 8, 4].as_slice()).expect("valid shape"))
                .narrow(0, 1, 1)
                .expect("valid narrow")
                .shape();
        assert_eq!(
            output(
                Op::SliceScatter {
                    dim: 1,
                    start: 3,
                },
                &[
                    (DType::BF16, target),
                    (
                        DType::BF16,
                        Shape::try_from([1, 2, 4].as_slice()).expect("valid shape")
                    )
                ]
            ),
            Ok((
                Shape::try_from([1, 8, 4].as_slice()).expect("valid shape"),
                DType::BF16
            )),
            "the target shape and dtype"
        );
    }

    #[test]
    fn slice_scatter_rejects_out_of_bounds_and_mismatches() {
        [
            (
                Shape::try_from([1, 3].as_slice()).expect("valid shape"),
                0,
                usize::MAX,
            ),
            (
                Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                0,
                1,
            ),
            (
                Shape::try_from([1, 2].as_slice()).expect("valid shape"),
                0,
                0,
            ),
            (
                Shape::try_from([1, 3, 1].as_slice()).expect("valid shape"),
                0,
                0,
            ),
        ]
        .into_iter()
        .for_each(|(src, dim, start)| {
            assert_eq!(
                output(
                    Op::SliceScatter {
                        dim,
                        start,
                    },
                    &[
                        (
                            DType::F32,
                            Shape::try_from([2, 3].as_slice()).expect("valid shape")
                        ),
                        (DType::F32, src)
                    ]
                ),
                Err(CoreError::SliceScatterOutOfBounds),
                "src {:?} at start {start}",
                src.dims()
            );
        });
    }

    #[test]
    fn slice_scatter_rejects_dtype_and_dim() {
        let target = Shape::try_from([2, 3].as_slice()).expect("valid shape");
        assert_eq!(
            [
                output(
                    Op::SliceScatter {
                        dim: 0,
                        start: 0,
                    },
                    &[(DType::F32, target), (DType::BF16, target)]
                ),
                output(
                    Op::SliceScatter {
                        dim: 2,
                        start: 0,
                    },
                    &[(DType::F32, target), (DType::F32, target)]
                ),
            ],
            [Err(CoreError::DTypeMismatch), Err(CoreError::DimOutOfRange)],
            "unequal dtypes, then dim 2 out of range"
        );
    }
}
