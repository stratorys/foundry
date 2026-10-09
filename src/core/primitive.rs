use crate::core::{
    CoreError,
    DType,
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
    Recip,
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
    /// Adds all elements along an axis.
    Sum,
    /// Keeps the largest element along an axis.
    Max,
    /// Keeps the index of the largest element along an axis.
    Argmax,
}

pub fn unary_rule(
    dtype: DType,
    shape: &Shape,
) -> Result<(DType, Shape), CoreError> {
    require_float(dtype)?;
    Ok((dtype, *shape))
}

pub fn binary_rule(
    lhs_dtype: DType,
    lhs_shape: &Shape,
    rhs_dtype: DType,
    rhs_shape: &Shape,
) -> Result<(DType, Shape), CoreError> {
    require_same_dtype(lhs_dtype, rhs_dtype)?;
    Ok((lhs_dtype, Shape::broadcast(lhs_shape, rhs_shape)?))
}

pub fn reduce_rule(
    op: ReduceOp,
    dtype: DType,
    shape: &Shape,
    axis: usize,
) -> Result<(DType, Shape), CoreError> {
    let dtype_output = match op {
        ReduceOp::Sum | ReduceOp::Max => dtype,
        ReduceOp::Argmax => DType::U32,
    };
    let axis_len = dim(shape, axis)?;
    let needs_element = matches!(op, ReduceOp::Max | ReduceOp::Argmax);
    if needs_element && axis_len == 0 {
        return Err(CoreError::EmptyReduction {
            op,
            axis,
            dims: shape.dims().to_vec(),
        });
    }
    Ok((dtype_output, with_dim(shape, axis, 1)?))
}

pub fn matmul_rule(
    lhs_dtype: DType,
    lhs_shape: &Shape,
    rhs_dtype: DType,
    rhs_shape: &Shape,
) -> Result<(DType, Shape), CoreError> {
    require_float(lhs_dtype)?;
    require_same_dtype(lhs_dtype, rhs_dtype)?;
    let incompatible = || CoreError::MatmulIncompatible {
        lhs: lhs_shape.dims().to_vec(),
        rhs: rhs_shape.dims().to_vec(),
    };
    let (Some((lhs_batch, [m, lhs_k])), Some((rhs_batch, [rhs_k, n]))) = (
        lhs_shape.dims().split_last_chunk::<2>(),
        rhs_shape.dims().split_last_chunk::<2>(),
    ) else {
        return Err(incompatible());
    };
    if lhs_batch != rhs_batch || lhs_k != rhs_k {
        return Err(incompatible());
    }
    let dims: Vec<usize> = lhs_batch.iter().copied().chain([*m, *n]).collect();
    Ok((lhs_dtype, Shape::try_from(dims.as_slice())?))
}

pub fn copy_rule(
    dtype: DType,
    shape: &Shape,
) -> Result<(DType, Shape), CoreError> {
    Ok((dtype, *shape))
}

pub fn cast_rule(
    dtype_from: DType,
    dtype_to: DType,
    shape: &Shape,
) -> Result<(DType, Shape), CoreError> {
    require_float(dtype_from)?;
    require_float(dtype_to)?;
    Ok((dtype_to, *shape))
}

pub fn gather_rule(
    table_dtype: DType,
    table_shape: &Shape,
    indices_dtype: DType,
    indices_shape: &Shape,
) -> Result<(DType, Shape), CoreError> {
    if indices_dtype != DType::U32 {
        return Err(CoreError::DTypeUnexpected {
            dtype: indices_dtype,
            dtype_expected: DType::U32,
        });
    }
    let ([_, table_cols], [index_count]) = (table_shape.dims(), indices_shape.dims()) else {
        return Err(CoreError::GatherIncompatible {
            table: table_shape.dims().to_vec(),
            indices: indices_shape.dims().to_vec(),
        });
    };
    Ok((
        table_dtype,
        Shape::try_from([*index_count, *table_cols].as_slice())?,
    ))
}

pub fn concat_rule(
    lhs_dtype: DType,
    lhs_shape: &Shape,
    rhs_dtype: DType,
    rhs_shape: &Shape,
    axis: usize,
) -> Result<(DType, Shape), CoreError> {
    require_same_dtype(lhs_dtype, rhs_dtype)?;
    let incompatible = || CoreError::ConcatIncompatible {
        axis,
        lhs: lhs_shape.dims().to_vec(),
        rhs: rhs_shape.dims().to_vec(),
    };
    let (lhs_dim, rhs_dim) = (dim(lhs_shape, axis)?, dim(rhs_shape, axis)?);
    if !dims_equal_except(lhs_shape, rhs_shape, axis) {
        return Err(incompatible());
    }
    let dim_output = lhs_dim.checked_add(rhs_dim).ok_or_else(incompatible)?;
    Ok((lhs_dtype, with_dim(lhs_shape, axis, dim_output)?))
}

pub fn slice_update_rule(
    target_dtype: DType,
    target_shape: &Shape,
    update_dtype: DType,
    update_shape: &Shape,
    axis: usize,
    start: usize,
) -> Result<(DType, Shape), CoreError> {
    require_same_dtype(target_dtype, update_dtype)?;
    let (target_dim, update_dim) = (dim(target_shape, axis)?, dim(update_shape, axis)?);
    let fits = start
        .checked_add(update_dim)
        .is_some_and(|end| end <= target_dim);
    if !fits || !dims_equal_except(target_shape, update_shape, axis) {
        return Err(CoreError::SliceUpdateOutOfBounds {
            axis,
            start,
            target: target_shape.dims().to_vec(),
            update: update_shape.dims().to_vec(),
        });
    }
    Ok((target_dtype, *target_shape))
}

fn require_float(dtype: DType) -> Result<(), CoreError> {
    if dtype.is_float() {
        Ok(())
    } else {
        Err(CoreError::DTypeNotFloat {
            dtype,
        })
    }
}

fn require_same_dtype(
    lhs: DType,
    rhs: DType,
) -> Result<(), CoreError> {
    if lhs == rhs {
        Ok(())
    } else {
        Err(CoreError::DTypeMismatch {
            lhs,
            rhs,
        })
    }
}

fn dim(
    shape: &Shape,
    axis: usize,
) -> Result<usize, CoreError> {
    shape
        .dims()
        .get(axis)
        .copied()
        .ok_or(CoreError::AxisOutOfRange {
            axis,
            rank: shape.rank(),
        })
}

fn with_dim(
    shape: &Shape,
    axis: usize,
    size: usize,
) -> Result<Shape, CoreError> {
    dim(shape, axis)?;
    let dims: Vec<usize> = shape
        .dims()
        .iter()
        .enumerate()
        .map(|(index, &dim)| if index == axis { size } else { dim })
        .collect();
    Shape::try_from(dims.as_slice())
}

fn dims_equal_except(
    lhs: &Shape,
    rhs: &Shape,
    axis: usize,
) -> bool {
    lhs.rank() == rhs.rank()
        && lhs
            .dims()
            .iter()
            .zip(rhs.dims())
            .enumerate()
            .all(|(index, (lhs_dim, rhs_dim))| index == axis || lhs_dim == rhs_dim)
}

#[cfg(test)]
mod tests {
    use crate::core::primitive::{
        ReduceOp,
        binary_rule,
        cast_rule,
        concat_rule,
        gather_rule,
        matmul_rule,
        reduce_rule,
        slice_update_rule,
        unary_rule,
    };
    use crate::core::{
        CoreError,
        DType,
        Shape,
    };

    #[test]
    fn unary_keeps_dtype_and_shape() {
        let (dtype, output) = unary_rule(
            DType::BF16,
            &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
        )
        .expect("float input");
        assert_eq!(dtype, DType::BF16, "dtype");
        assert_eq!(output.dims(), &[2, 3], "dims");
    }

    #[test]
    fn unary_rejects_integer_dtype() {
        assert_eq!(
            unary_rule(
                DType::U32,
                &Shape::try_from([2].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::DTypeNotFloat {
                dtype: DType::U32,
            }),
            "unary needs a float dtype"
        );
    }

    #[test]
    fn binary_broadcasts_shapes() {
        let (dtype, output) = binary_rule(
            DType::BF16,
            &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
            DType::BF16,
            &Shape::try_from([3].as_slice()).expect("valid shape"),
        )
        .expect("broadcastable");
        assert_eq!(dtype, DType::BF16, "dtype");
        assert_eq!(output.dims(), &[2, 3], "dims");
    }

    #[test]
    fn binary_rejects_dtype_mismatch() {
        assert_eq!(
            binary_rule(
                DType::BF16,
                &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                DType::F32,
                &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::DTypeMismatch {
                lhs: DType::BF16,
                rhs: DType::F32,
            }),
            "binary needs equal dtypes"
        );
    }

    #[test]
    fn binary_rejects_incompatible_shapes() {
        assert_eq!(
            binary_rule(
                DType::BF16,
                &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                DType::BF16,
                &Shape::try_from([4].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::BroadcastIncompatible {
                lhs: vec![2, 3],
                rhs: vec![4],
            }),
            "3 and 4 do not broadcast"
        );
    }

    #[test]
    fn reduce_keeps_axis_with_size_one() {
        let (dtype, output) = reduce_rule(
            ReduceOp::Max,
            DType::F32,
            &Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"),
            2,
        )
        .expect("valid axis");
        assert_eq!(dtype, DType::F32, "dtype");
        assert_eq!(output.dims(), &[2, 3, 1], "dims");
    }

    #[test]
    fn reduce_argmax_outputs_u32() {
        let (dtype, output) = reduce_rule(
            ReduceOp::Argmax,
            DType::BF16,
            &Shape::try_from([1, 5].as_slice()).expect("valid shape"),
            1,
        )
        .expect("valid axis");
        assert_eq!(dtype, DType::U32, "dtype");
        assert_eq!(output.dims(), &[1, 1], "dims");
    }

    #[test]
    fn reduce_rejects_axis_out_of_range() {
        assert_eq!(
            reduce_rule(
                ReduceOp::Sum,
                DType::F32,
                &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                2,
            ),
            Err(CoreError::AxisOutOfRange {
                axis: 2,
                rank: 2,
            }),
            "axis 2 of a rank 2 shape"
        );
    }

    #[test]
    fn reduce_max_and_argmax_over_empty_axis_are_rejected() {
        [ReduceOp::Max, ReduceOp::Argmax]
            .into_iter()
            .for_each(|op| {
                assert_eq!(
                    reduce_rule(
                        op,
                        DType::F32,
                        &Shape::try_from([2, 0].as_slice()).expect("valid shape"),
                        1,
                    ),
                    Err(CoreError::EmptyReduction {
                        op,
                        axis: 1,
                        dims: vec![2, 0],
                    }),
                    "{op:?} over an empty axis"
                );
            });
    }

    #[test]
    fn reduce_sum_over_empty_axis_keeps_axis_with_size_one() {
        let (dtype, output) = reduce_rule(
            ReduceOp::Sum,
            DType::F32,
            &Shape::try_from([2, 0].as_slice()).expect("valid shape"),
            1,
        )
        .expect("sum of nothing");
        assert_eq!(dtype, DType::F32, "dtype");
        assert_eq!(output.dims(), &[2, 1], "dims");
    }

    #[test]
    fn matmul_multiplies_last_two_axes() {
        let (dtype, output) = matmul_rule(
            DType::BF16,
            &Shape::try_from([2, 3, 4, 5].as_slice()).expect("valid shape"),
            DType::BF16,
            &Shape::try_from([2, 3, 5, 4].as_slice()).expect("valid shape"),
        )
        .expect("compatible");
        assert_eq!(dtype, DType::BF16, "dtype");
        assert_eq!(output.dims(), &[2, 3, 4, 4], "dims");
    }

    #[test]
    fn matmul_rejects_inner_dim_mismatch() {
        assert_eq!(
            matmul_rule(
                DType::BF16,
                &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                DType::BF16,
                &Shape::try_from([4, 3].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::MatmulIncompatible {
                lhs: vec![2, 3],
                rhs: vec![4, 3],
            }),
            "inner dims 3 and 4 differ"
        );
    }

    #[test]
    fn matmul_rejects_batch_mismatch() {
        assert_eq!(
            matmul_rule(
                DType::BF16,
                &Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"),
                DType::BF16,
                &Shape::try_from([5, 4, 3].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::MatmulIncompatible {
                lhs: vec![2, 3, 4],
                rhs: vec![5, 4, 3],
            }),
            "batch dims 2 and 5 differ"
        );
    }

    #[test]
    fn matmul_rejects_rank_below_two() {
        assert_eq!(
            matmul_rule(
                DType::BF16,
                &Shape::try_from([3].as_slice()).expect("valid shape"),
                DType::BF16,
                &Shape::try_from([3, 4].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::MatmulIncompatible {
                lhs: vec![3],
                rhs: vec![3, 4],
            }),
            "a rank 1 lhs is rejected"
        );
    }

    #[test]
    fn matmul_rejects_integer_dtype() {
        assert_eq!(
            matmul_rule(
                DType::U32,
                &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                DType::U32,
                &Shape::try_from([3, 4].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::DTypeNotFloat {
                dtype: DType::U32,
            }),
            "matmul needs a float dtype"
        );
    }

    #[test]
    fn cast_changes_dtype_and_keeps_shape() {
        let (dtype, output) = cast_rule(
            DType::BF16,
            DType::F32,
            &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
        )
        .expect("float to float");
        assert_eq!(dtype, DType::F32, "dtype");
        assert_eq!(output.dims(), &[2, 3], "dims");
    }

    #[test]
    fn cast_rejects_integer_dtype() {
        assert_eq!(
            cast_rule(
                DType::U32,
                DType::F32,
                &Shape::try_from([2].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::DTypeNotFloat {
                dtype: DType::U32,
            }),
            "cast needs a float source"
        );
    }

    #[test]
    fn gather_selects_table_rows() {
        let (dtype, output) = gather_rule(
            DType::BF16,
            &Shape::try_from([5, 3].as_slice()).expect("valid shape"),
            DType::U32,
            &Shape::try_from([2].as_slice()).expect("valid shape"),
        )
        .expect("valid gather");
        assert_eq!(dtype, DType::BF16, "dtype");
        assert_eq!(output.dims(), &[2, 3], "dims");
    }

    #[test]
    fn gather_rejects_non_u32_indices() {
        assert_eq!(
            gather_rule(
                DType::BF16,
                &Shape::try_from([5, 3].as_slice()).expect("valid shape"),
                DType::F32,
                &Shape::try_from([2].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::DTypeUnexpected {
                dtype: DType::F32,
                dtype_expected: DType::U32,
            }),
            "indices must be u32"
        );
    }

    #[test]
    fn gather_rejects_invalid_ranks() {
        assert_eq!(
            gather_rule(
                DType::BF16,
                &Shape::try_from([5, 3].as_slice()).expect("valid shape"),
                DType::U32,
                &Shape::try_from([1, 2].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::GatherIncompatible {
                table: vec![5, 3],
                indices: vec![1, 2],
            }),
            "indices must be rank 1"
        );
    }

    #[test]
    fn concat_adds_axis_dims() {
        let (dtype, output) = concat_rule(
            DType::BF16,
            &Shape::try_from([2, 3, 4, 5].as_slice()).expect("valid shape"),
            DType::BF16,
            &Shape::try_from([2, 3, 1, 5].as_slice()).expect("valid shape"),
            2,
        )
        .expect("compatible");
        assert_eq!(dtype, DType::BF16, "dtype");
        assert_eq!(output.dims(), &[2, 3, 5, 5], "dims");
    }

    #[test]
    fn concat_rejects_mismatch_outside_axis() {
        assert_eq!(
            concat_rule(
                DType::BF16,
                &Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"),
                DType::BF16,
                &Shape::try_from([2, 3, 5].as_slice()).expect("valid shape"),
                1,
            ),
            Err(CoreError::ConcatIncompatible {
                axis: 1,
                lhs: vec![2, 3, 4],
                rhs: vec![2, 3, 5],
            }),
            "dims 4 and 5 differ outside the axis"
        );
    }

    #[test]
    fn slice_update_returns_target_shape() {
        let (dtype, output) = slice_update_rule(
            DType::BF16,
            &Shape::try_from([2, 3, 8, 4].as_slice()).expect("valid shape"),
            DType::BF16,
            &Shape::try_from([2, 3, 2, 4].as_slice()).expect("valid shape"),
            2,
            6,
        )
        .expect("in bounds");
        assert_eq!(dtype, DType::BF16, "dtype");
        assert_eq!(output.dims(), &[2, 3, 8, 4], "dims");
    }

    #[test]
    fn slice_update_rejects_out_of_bounds() {
        assert_eq!(
            slice_update_rule(
                DType::BF16,
                &Shape::try_from([2, 3, 8, 4].as_slice()).expect("valid shape"),
                DType::BF16,
                &Shape::try_from([2, 3, 2, 4].as_slice()).expect("valid shape"),
                2,
                7,
            ),
            Err(CoreError::SliceUpdateOutOfBounds {
                axis: 2,
                start: 7,
                target: vec![2, 3, 8, 4],
                update: vec![2, 3, 2, 4],
            }),
            "7 plus 2 exceeds 8"
        );
    }

    #[test]
    fn slice_update_rejects_mismatch_outside_axis() {
        assert_eq!(
            slice_update_rule(
                DType::BF16,
                &Shape::try_from([2, 3, 8, 4].as_slice()).expect("valid shape"),
                DType::BF16,
                &Shape::try_from([2, 1, 2, 4].as_slice()).expect("valid shape"),
                2,
                0,
            ),
            Err(CoreError::SliceUpdateOutOfBounds {
                axis: 2,
                start: 0,
                target: vec![2, 3, 8, 4],
                update: vec![2, 1, 2, 4],
            }),
            "dims 3 and 1 differ outside the axis"
        );
    }

    #[test]
    fn reduce_and_binary_accept_u32() {
        let shape = Shape::try_from([2, 3].as_slice()).expect("valid shape");
        assert_eq!(
            reduce_rule(ReduceOp::Sum, DType::U32, &shape, 1),
            Ok((
                DType::U32,
                Shape::try_from([2, 1].as_slice()).expect("valid shape")
            )),
            "the reduce rule does not check the dtype"
        );
        assert_eq!(
            binary_rule(DType::U32, &shape, DType::U32, &shape),
            Ok((DType::U32, shape)),
            "the binary rule does not check the dtype"
        );
    }

    #[test]
    fn matmul_rejects_dtype_mismatch() {
        assert_eq!(
            matmul_rule(
                DType::BF16,
                &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                DType::F32,
                &Shape::try_from([3, 4].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::DTypeMismatch {
                lhs: DType::BF16,
                rhs: DType::F32,
            }),
            "matmul needs equal dtypes"
        );
    }

    #[test]
    fn matmul_rejects_rhs_rank_below_two() {
        assert_eq!(
            matmul_rule(
                DType::F32,
                &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                DType::F32,
                &Shape::try_from([3].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::MatmulIncompatible {
                lhs: vec![2, 3],
                rhs: vec![3],
            }),
            "a rank 1 rhs is rejected"
        );
    }

    #[test]
    fn matmul_rejects_output_beyond_the_element_limit() {
        assert_eq!(
            matmul_rule(
                DType::F32,
                &Shape::try_from([1 << 16, 1].as_slice()).expect("valid shape"),
                DType::F32,
                &Shape::try_from([1, 1 << 16].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::ElementCountOverflow {
                dims: vec![1 << 16, 1 << 16],
                element_count_max: 0x7FFF_FFFF,
            }),
            "the output holds 2^32 elements"
        );
    }

    #[test]
    fn cast_rejects_integer_target() {
        assert_eq!(
            cast_rule(
                DType::F32,
                DType::U32,
                &Shape::try_from([2].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::DTypeNotFloat {
                dtype: DType::U32,
            }),
            "cast needs a float target"
        );
    }

    #[test]
    fn gather_rejects_table_rank_other_than_two() {
        [[4].as_slice(), &[2, 3, 4]].into_iter().for_each(|table| {
            assert_eq!(
                gather_rule(
                    DType::F32,
                    &Shape::try_from(table).expect("valid shape"),
                    DType::U32,
                    &Shape::try_from([2].as_slice()).expect("valid shape"),
                ),
                Err(CoreError::GatherIncompatible {
                    table: table.to_vec(),
                    indices: vec![2],
                }),
                "table {table:?} is not rank 2"
            );
        });
    }

    #[test]
    fn gather_rejects_output_beyond_the_element_limit() {
        assert_eq!(
            gather_rule(
                DType::F32,
                &Shape::try_from([1, 1 << 16].as_slice()).expect("valid shape"),
                DType::U32,
                &Shape::try_from([1 << 16].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::ElementCountOverflow {
                dims: vec![1 << 16, 1 << 16],
                element_count_max: 0x7FFF_FFFF,
            }),
            "the output holds 2^32 elements"
        );
    }

    #[test]
    fn concat_rejects_dtype_mismatch() {
        let shape = Shape::try_from([2, 3].as_slice()).expect("valid shape");
        assert_eq!(
            concat_rule(DType::F32, &shape, DType::BF16, &shape, 0),
            Err(CoreError::DTypeMismatch {
                lhs: DType::F32,
                rhs: DType::BF16,
            }),
            "concat needs equal dtypes"
        );
    }

    #[test]
    fn concat_rejects_axis_out_of_range() {
        assert_eq!(
            concat_rule(
                DType::F32,
                &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                DType::F32,
                &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                2,
            ),
            Err(CoreError::AxisOutOfRange {
                axis: 2,
                rank: 2,
            }),
            "axis 2 does not exist on the lhs"
        );
        assert_eq!(
            concat_rule(
                DType::F32,
                &Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"),
                DType::F32,
                &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                2,
            ),
            Err(CoreError::AxisOutOfRange {
                axis: 2,
                rank: 2,
            }),
            "axis 2 does not exist on the rhs"
        );
    }

    #[test]
    fn concat_rejects_rank_mismatch() {
        assert_eq!(
            concat_rule(
                DType::F32,
                &Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"),
                DType::F32,
                &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                0,
            ),
            Err(CoreError::ConcatIncompatible {
                axis: 0,
                lhs: vec![2, 3, 4],
                rhs: vec![2, 3],
            }),
            "ranks 3 and 2 differ"
        );
    }

    #[test]
    fn concat_rejects_output_beyond_the_element_limit() {
        let shape = Shape::try_from([1 << 30].as_slice()).expect("valid shape");
        assert_eq!(
            concat_rule(DType::F32, &shape, DType::F32, &shape, 0),
            Err(CoreError::ElementCountOverflow {
                dims: vec![1 << 31],
                element_count_max: 0x7FFF_FFFF,
            }),
            "the output holds 2^31 elements"
        );
    }

    #[test]
    fn slice_update_rejects_dtype_mismatch() {
        let shape = Shape::try_from([2, 3].as_slice()).expect("valid shape");
        assert_eq!(
            slice_update_rule(DType::F32, &shape, DType::BF16, &shape, 0, 0),
            Err(CoreError::DTypeMismatch {
                lhs: DType::F32,
                rhs: DType::BF16,
            }),
            "slice update needs equal dtypes"
        );
    }

    #[test]
    fn slice_update_rejects_axis_out_of_range() {
        let shape = Shape::try_from([2, 3].as_slice()).expect("valid shape");
        assert_eq!(
            slice_update_rule(DType::F32, &shape, DType::F32, &shape, 2, 0),
            Err(CoreError::AxisOutOfRange {
                axis: 2,
                rank: 2,
            }),
            "axis 2 does not exist"
        );
    }

    #[test]
    fn slice_update_rejects_overflowing_start() {
        assert_eq!(
            slice_update_rule(
                DType::F32,
                &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                DType::F32,
                &Shape::try_from([1, 3].as_slice()).expect("valid shape"),
                0,
                usize::MAX,
            ),
            Err(CoreError::SliceUpdateOutOfBounds {
                axis: 0,
                start: usize::MAX,
                target: vec![2, 3],
                update: vec![1, 3],
            }),
            "start plus the update length overflows usize"
        );
    }

    #[test]
    fn slice_update_rejects_rank_mismatch() {
        assert_eq!(
            slice_update_rule(
                DType::F32,
                &Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"),
                DType::F32,
                &Shape::try_from([1, 3].as_slice()).expect("valid shape"),
                0,
                0,
            ),
            Err(CoreError::SliceUpdateOutOfBounds {
                axis: 0,
                start: 0,
                target: vec![2, 3, 4],
                update: vec![1, 3],
            }),
            "ranks 3 and 2 differ"
        );
    }
}
