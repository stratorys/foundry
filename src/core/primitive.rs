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
        copy_rule,
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

    const T: usize = 7;

    #[test]
    fn unary_keeps_dtype_and_shape() {
        let (dtype, output) = unary_rule(
            DType::BF16,
            &Shape::try_from([T, 3072].as_slice()).expect("valid shape"),
        )
        .expect("float input");
        assert_eq!(dtype, DType::BF16, "dtype");
        assert_eq!(output.dims(), &[T, 3072], "dims");
    }

    #[test]
    fn unary_rejects_integer_dtype() {
        let result = unary_rule(
            DType::U32,
            &Shape::try_from([T].as_slice()).expect("valid shape"),
        );
        assert!(
            matches!(
                result,
                Err(CoreError::DTypeNotFloat {
                    dtype: DType::U32
                })
            ),
            "got {result:?}"
        );
    }

    #[test]
    fn binary_broadcasts_shapes() {
        let (dtype, output) = binary_rule(
            DType::BF16,
            &Shape::try_from([T, 3072].as_slice()).expect("valid shape"),
            DType::BF16,
            &Shape::try_from([3072].as_slice()).expect("valid shape"),
        )
        .expect("broadcastable");
        assert_eq!(dtype, DType::BF16, "dtype");
        assert_eq!(output.dims(), &[T, 3072], "dims");
    }

    #[test]
    fn binary_rejects_dtype_mismatch() {
        let result = binary_rule(
            DType::BF16,
            &Shape::try_from([T, 3072].as_slice()).expect("valid shape"),
            DType::F32,
            &Shape::try_from([T, 3072].as_slice()).expect("valid shape"),
        );
        assert!(
            matches!(result, Err(CoreError::DTypeMismatch { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn binary_rejects_incompatible_shapes() {
        let result = binary_rule(
            DType::BF16,
            &Shape::try_from([T, 3072].as_slice()).expect("valid shape"),
            DType::BF16,
            &Shape::try_from([128].as_slice()).expect("valid shape"),
        );
        assert!(
            matches!(result, Err(CoreError::BroadcastIncompatible { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn reduce_keeps_axis_with_size_one() {
        let (dtype, output) = reduce_rule(
            ReduceOp::Max,
            DType::F32,
            &Shape::try_from([24, T, T].as_slice()).expect("valid shape"),
            2,
        )
        .expect("valid axis");
        assert_eq!(dtype, DType::F32, "dtype");
        assert_eq!(output.dims(), &[24, T, 1], "dims");
    }

    #[test]
    fn reduce_argmax_outputs_u32() {
        let (dtype, output) = reduce_rule(
            ReduceOp::Argmax,
            DType::BF16,
            &Shape::try_from([1, 128256].as_slice()).expect("valid shape"),
            1,
        )
        .expect("valid axis");
        assert_eq!(dtype, DType::U32, "dtype");
        assert_eq!(output.dims(), &[1, 1], "dims");
    }

    #[test]
    fn reduce_rejects_axis_out_of_range() {
        let result = reduce_rule(
            ReduceOp::Sum,
            DType::F32,
            &Shape::try_from([T, 3072].as_slice()).expect("valid shape"),
            2,
        );
        assert!(
            matches!(
                result,
                Err(CoreError::AxisOutOfRange {
                    axis: 2,
                    rank: 2
                })
            ),
            "got {result:?}"
        );
    }

    #[test]
    fn reduce_max_and_argmax_over_empty_axis_are_rejected() {
        [ReduceOp::Max, ReduceOp::Argmax]
            .into_iter()
            .for_each(|op| {
                let result = reduce_rule(
                    op,
                    DType::F32,
                    &Shape::try_from([2, 0].as_slice()).expect("valid shape"),
                    1,
                );
                assert!(
                    matches!(
                        result,
                        Err(CoreError::EmptyReduction {
                            axis: 1,
                            ..
                        })
                    ),
                    "{op:?} got {result:?}"
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
            &Shape::try_from([8, 3, T, 128].as_slice()).expect("valid shape"),
            DType::BF16,
            &Shape::try_from([8, 3, 128, T].as_slice()).expect("valid shape"),
        )
        .expect("compatible");
        assert_eq!(dtype, DType::BF16, "dtype");
        assert_eq!(output.dims(), &[8, 3, T, T], "dims");
    }

    #[test]
    fn matmul_rejects_inner_dim_mismatch() {
        let result = matmul_rule(
            DType::BF16,
            &Shape::try_from([T, 3072].as_slice()).expect("valid shape"),
            DType::BF16,
            &Shape::try_from([128256, 3072].as_slice()).expect("valid shape"),
        );
        assert!(
            matches!(result, Err(CoreError::MatmulIncompatible { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn matmul_rejects_batch_mismatch() {
        let result = matmul_rule(
            DType::BF16,
            &Shape::try_from([24, T, 128].as_slice()).expect("valid shape"),
            DType::BF16,
            &Shape::try_from([8, 128, T].as_slice()).expect("valid shape"),
        );
        assert!(
            matches!(result, Err(CoreError::MatmulIncompatible { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn matmul_rejects_rank_below_two() {
        let result = matmul_rule(
            DType::BF16,
            &Shape::try_from([3072].as_slice()).expect("valid shape"),
            DType::BF16,
            &Shape::try_from([3072, 128256].as_slice()).expect("valid shape"),
        );
        assert!(
            matches!(result, Err(CoreError::MatmulIncompatible { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn matmul_rejects_integer_dtype() {
        let result = matmul_rule(
            DType::U32,
            &Shape::try_from([T, 3072].as_slice()).expect("valid shape"),
            DType::U32,
            &Shape::try_from([3072, 128].as_slice()).expect("valid shape"),
        );
        assert!(
            matches!(result, Err(CoreError::DTypeNotFloat { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn copy_keeps_dtype_and_shape() {
        let (dtype, output) = copy_rule(
            DType::BF16,
            &Shape::try_from([24, T, 128].as_slice()).expect("valid shape"),
        )
        .expect("any input");
        assert_eq!(dtype, DType::BF16, "dtype");
        assert_eq!(output.dims(), &[24, T, 128], "dims");
    }

    #[test]
    fn cast_changes_dtype_and_keeps_shape() {
        let (dtype, output) = cast_rule(
            DType::BF16,
            DType::F32,
            &Shape::try_from([T, 3072].as_slice()).expect("valid shape"),
        )
        .expect("float to float");
        assert_eq!(dtype, DType::F32, "dtype");
        assert_eq!(output.dims(), &[T, 3072], "dims");
    }

    #[test]
    fn cast_rejects_integer_dtype() {
        let result = cast_rule(
            DType::U32,
            DType::F32,
            &Shape::try_from([T].as_slice()).expect("valid shape"),
        );
        assert!(
            matches!(result, Err(CoreError::DTypeNotFloat { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn gather_selects_table_rows() {
        let (dtype, output) = gather_rule(
            DType::BF16,
            &Shape::try_from([128256, 3072].as_slice()).expect("valid shape"),
            DType::U32,
            &Shape::try_from([T].as_slice()).expect("valid shape"),
        )
        .expect("valid gather");
        assert_eq!(dtype, DType::BF16, "dtype");
        assert_eq!(output.dims(), &[T, 3072], "dims");
    }

    #[test]
    fn gather_rejects_non_u32_indices() {
        let result = gather_rule(
            DType::BF16,
            &Shape::try_from([128256, 3072].as_slice()).expect("valid shape"),
            DType::F32,
            &Shape::try_from([T].as_slice()).expect("valid shape"),
        );
        assert!(
            matches!(result, Err(CoreError::DTypeUnexpected { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn gather_rejects_invalid_ranks() {
        let result = gather_rule(
            DType::BF16,
            &Shape::try_from([128256, 3072].as_slice()).expect("valid shape"),
            DType::U32,
            &Shape::try_from([1, T].as_slice()).expect("valid shape"),
        );
        assert!(
            matches!(result, Err(CoreError::GatherIncompatible { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn concat_adds_axis_dims() {
        let (dtype, output) = concat_rule(
            DType::BF16,
            &Shape::try_from([8, 3, T, 128].as_slice()).expect("valid shape"),
            DType::BF16,
            &Shape::try_from([8, 3, 1, 128].as_slice()).expect("valid shape"),
            2,
        )
        .expect("compatible");
        assert_eq!(dtype, DType::BF16, "dtype");
        assert_eq!(output.dims(), &[8, 3, T + 1, 128], "dims");
    }

    #[test]
    fn concat_rejects_mismatch_outside_axis() {
        let result = concat_rule(
            DType::BF16,
            &Shape::try_from([24, T, 128].as_slice()).expect("valid shape"),
            DType::BF16,
            &Shape::try_from([24, T, 64].as_slice()).expect("valid shape"),
            1,
        );
        assert!(
            matches!(result, Err(CoreError::ConcatIncompatible { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn slice_update_returns_target_shape() {
        let (dtype, output) = slice_update_rule(
            DType::BF16,
            &Shape::try_from([8, 3, 512, 128].as_slice()).expect("valid shape"),
            DType::BF16,
            &Shape::try_from([8, 3, T, 128].as_slice()).expect("valid shape"),
            2,
            505,
        )
        .expect("in bounds");
        assert_eq!(dtype, DType::BF16, "dtype");
        assert_eq!(output.dims(), &[8, 3, 512, 128], "dims");
    }

    #[test]
    fn slice_update_rejects_out_of_bounds() {
        let result = slice_update_rule(
            DType::BF16,
            &Shape::try_from([8, 3, 512, 128].as_slice()).expect("valid shape"),
            DType::BF16,
            &Shape::try_from([8, 3, T, 128].as_slice()).expect("valid shape"),
            2,
            506,
        );
        assert!(
            matches!(result, Err(CoreError::SliceUpdateOutOfBounds { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn slice_update_rejects_mismatch_outside_axis() {
        let result = slice_update_rule(
            DType::BF16,
            &Shape::try_from([8, 3, 512, 128].as_slice()).expect("valid shape"),
            DType::BF16,
            &Shape::try_from([8, 1, T, 128].as_slice()).expect("valid shape"),
            2,
            0,
        );
        assert!(
            matches!(result, Err(CoreError::SliceUpdateOutOfBounds { .. })),
            "got {result:?}"
        );
    }
}
