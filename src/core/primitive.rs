use std::array;

use tracing::error;

use crate::core::{
    CoreError,
    DType,
    FloatDType,
    Layout,
    RANK_MAX,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinarySpec {
    dtype: FloatDType,
    output: Shape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CastSpec {
    from: FloatDType,
    to: FloatDType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceSpec {
    op: ReduceOp,
    dtype: FloatDType,
    output: Shape,
    axis_len: usize,
    axis_stride: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixLayout {
    offset: usize,
    row_stride: usize,
    col_stride: usize,
    batch_strides: [usize; RANK_MAX],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulSpec {
    dtype: FloatDType,
    output: Shape,
    batch: Shape,
    m: usize,
    k: usize,
    n: usize,
    lhs: MatrixLayout,
    rhs: MatrixLayout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatherSpec {
    dtype: FloatDType,
    output: Shape,
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConcatSpec {
    dtype: FloatDType,
    output: Shape,
    rhs_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceUpdateSpec {
    dtype: FloatDType,
    offset: usize,
}

struct StrideFold {
    strides: Vec<usize>,
    stride_next: usize,
}

impl BinarySpec {
    pub fn dtype(&self) -> FloatDType { self.dtype }

    pub fn output(&self) -> &Shape { &self.output }
}

impl CastSpec {
    pub fn from(&self) -> FloatDType { self.from }

    pub fn to(&self) -> FloatDType { self.to }
}

impl ReduceSpec {
    pub fn op(&self) -> ReduceOp { self.op }

    pub fn dtype(&self) -> FloatDType { self.dtype }

    pub fn dtype_output(&self) -> DType {
        match self.op {
            ReduceOp::Sum | ReduceOp::Max => self.dtype.into(),
            ReduceOp::Argmax => DType::U32,
        }
    }

    pub fn output(&self) -> &Shape { &self.output }

    pub fn axis_len(&self) -> usize { self.axis_len }

    pub fn axis_stride(&self) -> usize { self.axis_stride }
}

impl MatrixLayout {
    pub fn contiguous(
        batch: &Shape,
        rows: usize,
        cols: usize,
    ) -> Self {
        let matrix_len = rows.saturating_mul(cols);
        let batch_strides_reversed = batch.dims().iter().rev().fold(
            StrideFold {
                strides: Vec::with_capacity(RANK_MAX),
                stride_next: matrix_len,
            },
            |mut fold, &dim| {
                fold.strides.push(fold.stride_next);
                StrideFold {
                    stride_next: fold.stride_next.saturating_mul(dim),
                    strides: fold.strides,
                }
            },
        );
        let batch_strides: Vec<usize> = batch_strides_reversed.strides.into_iter().rev().collect();
        Self {
            offset: 0,
            row_stride: cols,
            col_stride: 1,
            batch_strides: padded(&batch_strides),
        }
    }

    pub fn offset(&self) -> usize { self.offset }

    pub fn row_stride(&self) -> usize { self.row_stride }

    pub fn col_stride(&self) -> usize { self.col_stride }

    pub fn batch_strides(&self) -> &[usize; RANK_MAX] { &self.batch_strides }

    fn of(layout: &Layout) -> Option<Self> {
        let (batch_strides, [row_stride, col_stride]) = layout.strides().split_last_chunk::<2>()?;
        Some(Self {
            offset: layout.offset(),
            row_stride: *row_stride,
            col_stride: *col_stride,
            batch_strides: padded(batch_strides),
        })
    }
}

impl MatmulSpec {
    pub fn dtype(&self) -> FloatDType { self.dtype }

    pub fn output(&self) -> &Shape { &self.output }

    pub fn batch(&self) -> &Shape { &self.batch }

    pub fn m(&self) -> usize { self.m }

    pub fn k(&self) -> usize { self.k }

    pub fn n(&self) -> usize { self.n }

    pub fn lhs(&self) -> &MatrixLayout { &self.lhs }

    pub fn rhs(&self) -> &MatrixLayout { &self.rhs }
}

impl GatherSpec {
    pub fn dtype(&self) -> FloatDType { self.dtype }

    pub fn output(&self) -> &Shape { &self.output }

    pub fn rows(&self) -> usize { self.rows }

    pub fn cols(&self) -> usize { self.cols }

    pub fn row_stride(&self) -> usize { self.row_stride }

    pub fn col_stride(&self) -> usize { self.col_stride }
}

impl ConcatSpec {
    pub fn dtype(&self) -> FloatDType { self.dtype }

    pub fn output(&self) -> &Shape { &self.output }

    pub fn rhs_offset(&self) -> usize { self.rhs_offset }
}

impl SliceUpdateSpec {
    pub fn dtype(&self) -> FloatDType { self.dtype }

    pub fn offset(&self) -> usize { self.offset }
}

pub fn unary_rule(dtype: DType) -> Result<FloatDType, CoreError> { require_float(dtype) }

pub fn binary_rule(
    lhs_dtype: DType,
    lhs_shape: &Shape,
    rhs_dtype: DType,
    rhs_shape: &Shape,
) -> Result<BinarySpec, CoreError> {
    Ok(BinarySpec {
        dtype: require_same_float(lhs_dtype, rhs_dtype)?,
        output: Shape::broadcast(lhs_shape, rhs_shape)?,
    })
}

pub fn reduce_rule(
    op: ReduceOp,
    dtype: DType,
    layout: &Layout,
    axis: usize,
) -> Result<ReduceSpec, CoreError> {
    let dtype = require_float(dtype)?;
    let shape = layout.shape();
    let axis_len = shape.dim(axis)?;
    let needs_element = matches!(op, ReduceOp::Max | ReduceOp::Argmax);
    if needs_element && axis_len == 0 {
        error!(
            message = "Reduction over an empty axis has no result.",
            ?op,
            axis,
            dims = ?shape.dims(),
        );
        return Err(CoreError::EmptyReduction);
    }
    Ok(ReduceSpec {
        op,
        dtype,
        output: with_dim(shape, axis, 1)?,
        axis_len,
        axis_stride: layout.stride(axis)?,
    })
}

pub fn matmul_rule(
    lhs_dtype: DType,
    lhs: &Layout,
    rhs_dtype: DType,
    rhs: &Layout,
) -> Result<MatmulSpec, CoreError> {
    let dtype = require_same_float(lhs_dtype, rhs_dtype)?;
    let incompatible = || {
        error!(
            message = "Shapes cannot be multiplied.",
            lhs = ?lhs.shape().dims(),
            rhs = ?rhs.shape().dims(),
        );
        CoreError::MatmulIncompatible
    };
    let Some((lhs_batch, &[m, lhs_k])) = lhs.shape().dims().split_last_chunk::<2>() else {
        return Err(incompatible());
    };
    let Some((rhs_batch, &[rhs_k, n])) = rhs.shape().dims().split_last_chunk::<2>() else {
        return Err(incompatible());
    };
    if lhs_batch != rhs_batch || lhs_k != rhs_k {
        return Err(incompatible());
    }
    let Some(lhs_matrix) = MatrixLayout::of(lhs) else {
        return Err(incompatible());
    };
    let Some(rhs_matrix) = MatrixLayout::of(rhs) else {
        return Err(incompatible());
    };
    let dims: Vec<usize> = lhs_batch.iter().copied().chain([m, n]).collect();
    Ok(MatmulSpec {
        dtype,
        output: Shape::try_from(dims.as_slice())?,
        batch: Shape::try_from(lhs_batch)?,
        m,
        k: lhs_k,
        n,
        lhs: lhs_matrix,
        rhs: rhs_matrix,
    })
}

pub fn cast_rule(
    dtype_from: DType,
    dtype_to: DType,
) -> Result<CastSpec, CoreError> {
    Ok(CastSpec {
        from: require_float(dtype_from)?,
        to: require_float(dtype_to)?,
    })
}

pub fn gather_rule(
    table_dtype: DType,
    table: &Layout,
    indices_dtype: DType,
    indices_shape: &Shape,
) -> Result<GatherSpec, CoreError> {
    let dtype = require_float(table_dtype)?;
    if indices_dtype != DType::U32 {
        error!(message = "Gather indices are not u32.", dtype = ?indices_dtype);
        return Err(CoreError::IndicesNotU32);
    }
    let incompatible = || {
        error!(
            message = "Table and indices cannot be gathered.",
            table = ?table.shape().dims(),
            indices = ?indices_shape.dims(),
        );
        CoreError::GatherIncompatible
    };
    let &[rows, cols] = table.shape().dims() else {
        return Err(incompatible());
    };
    let &[row_stride, col_stride] = table.strides() else {
        return Err(incompatible());
    };
    let &[index_count] = indices_shape.dims() else {
        return Err(incompatible());
    };
    Ok(GatherSpec {
        dtype,
        output: Shape::try_from([index_count, cols].as_slice())?,
        rows,
        cols,
        row_stride,
        col_stride,
    })
}

pub fn concat_rule(
    lhs_dtype: DType,
    lhs_shape: &Shape,
    rhs_dtype: DType,
    rhs_shape: &Shape,
    axis: usize,
) -> Result<ConcatSpec, CoreError> {
    let dtype = require_same_float(lhs_dtype, rhs_dtype)?;
    let incompatible = || {
        error!(
            message = "Shapes cannot be concatenated.",
            axis,
            lhs = ?lhs_shape.dims(),
            rhs = ?rhs_shape.dims(),
        );
        CoreError::ConcatIncompatible
    };
    let lhs_dim = lhs_shape.dim(axis)?;
    let rhs_dim = rhs_shape.dim(axis)?;
    if !dims_equal_except(lhs_shape, rhs_shape, axis) {
        return Err(incompatible());
    }
    let dim_output = lhs_dim.checked_add(rhs_dim).ok_or_else(incompatible)?;
    let output = with_dim(lhs_shape, axis, dim_output)?;
    let rhs_offset = lhs_dim
        .checked_mul(Layout::contiguous(output).stride(axis)?)
        .ok_or_else(incompatible)?;
    Ok(ConcatSpec {
        dtype,
        output,
        rhs_offset,
    })
}

pub fn slice_update_rule(
    target_dtype: DType,
    target: &Layout,
    update_dtype: DType,
    update_shape: &Shape,
    axis: usize,
    start: usize,
) -> Result<SliceUpdateSpec, CoreError> {
    let dtype = require_same_float(target_dtype, update_dtype)?;
    let target_shape = target.shape();
    let out_of_bounds = || {
        error!(
            message = "Slice update does not fit the target.",
            axis,
            start,
            target = ?target_shape.dims(),
            update = ?update_shape.dims(),
        );
        CoreError::SliceUpdateOutOfBounds
    };
    let target_dim = target_shape.dim(axis)?;
    let update_dim = update_shape.dim(axis)?;
    let fits = start
        .checked_add(update_dim)
        .is_some_and(|end| end <= target_dim);
    if !fits || !dims_equal_except(target_shape, update_shape, axis) {
        return Err(out_of_bounds());
    }
    let offset = start
        .checked_mul(target.stride(axis)?)
        .and_then(|skipped| skipped.checked_add(target.offset()))
        .ok_or_else(out_of_bounds)?;
    Ok(SliceUpdateSpec {
        dtype,
        offset,
    })
}

fn require_float(dtype: DType) -> Result<FloatDType, CoreError> {
    dtype.float().ok_or_else(|| {
        error!(message = "DType is not a float dtype.", ?dtype);
        CoreError::DTypeNotFloat
    })
}

fn require_same_float(
    lhs: DType,
    rhs: DType,
) -> Result<FloatDType, CoreError> {
    if lhs != rhs {
        error!(message = "DTypes do not match.", ?lhs, ?rhs);
        return Err(CoreError::DTypeMismatch);
    }
    require_float(lhs)
}

fn with_dim(
    shape: &Shape,
    axis: usize,
    size: usize,
) -> Result<Shape, CoreError> {
    shape.dim(axis)?;
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

fn padded(values: &[usize]) -> [usize; RANK_MAX] {
    array::from_fn(|axis| values.get(axis).copied().unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use crate::core::primitive::{
        BinarySpec,
        CastSpec,
        ConcatSpec,
        GatherSpec,
        MatmulSpec,
        MatrixLayout,
        ReduceOp,
        ReduceSpec,
        SliceUpdateSpec,
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
        FloatDType,
        Layout,
        Shape,
    };

    #[test]
    fn unary_keeps_float_dtype() {
        assert_eq!(
            unary_rule(DType::BF16),
            Ok(FloatDType::BF16),
            "a float dtype is kept"
        );
    }

    #[test]
    fn unary_rejects_integer_dtype() {
        assert_eq!(
            unary_rule(DType::U32),
            Err(CoreError::DTypeNotFloat),
            "unary needs a float dtype"
        );
    }

    #[test]
    fn binary_broadcasts_shapes() {
        assert_eq!(
            binary_rule(
                DType::BF16,
                &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                DType::BF16,
                &Shape::try_from([3].as_slice()).expect("valid shape"),
            ),
            Ok(BinarySpec {
                dtype: FloatDType::BF16,
                output: Shape::try_from([2, 3].as_slice()).expect("valid shape"),
            }),
            "dtype and broadcast shape"
        );
    }

    #[test]
    fn binary_rejects_dtype_mismatch() {
        let shape = Shape::try_from([2, 3].as_slice()).expect("valid shape");
        assert_eq!(
            binary_rule(DType::BF16, &shape, DType::F32, &shape),
            Err(CoreError::DTypeMismatch),
            "binary needs equal dtypes"
        );
    }

    #[test]
    fn binary_rejects_integer_dtype() {
        let shape = Shape::try_from([2, 3].as_slice()).expect("valid shape");
        assert_eq!(
            binary_rule(DType::U32, &shape, DType::U32, &shape),
            Err(CoreError::DTypeNotFloat),
            "binary needs a float dtype"
        );
    }

    #[test]
    fn binary_rejects_incompatible_shapes() {
        assert_eq!(
            binary_rule(
                DType::BF16,
                &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                DType::BF16,
                &Shape::try_from([4, 3].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::BroadcastIncompatible),
            "dims 2 and 4 do not broadcast"
        );
    }

    #[test]
    fn reduce_keeps_axis_with_size_one() {
        let spec = reduce_rule(
            ReduceOp::Sum,
            DType::BF16,
            &Layout::contiguous(Shape::try_from([2, 3, 4].as_slice()).expect("valid shape")),
            1,
        )
        .expect("valid reduction");
        assert_eq!(
            spec,
            ReduceSpec {
                op: ReduceOp::Sum,
                dtype: FloatDType::BF16,
                output: Shape::try_from([2, 1, 4].as_slice()).expect("valid shape"),
                axis_len: 3,
                axis_stride: 4,
            },
            "dtype, output dims, axis length and stride"
        );
        assert_eq!(spec.dtype_output(), DType::BF16, "a sum keeps the dtype");
    }

    #[test]
    fn reduce_argmax_outputs_u32() {
        let spec = reduce_rule(
            ReduceOp::Argmax,
            DType::F32,
            &Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape")),
            1,
        )
        .expect("valid reduction");
        assert_eq!(
            spec,
            ReduceSpec {
                op: ReduceOp::Argmax,
                dtype: FloatDType::F32,
                output: Shape::try_from([2, 1].as_slice()).expect("valid shape"),
                axis_len: 3,
                axis_stride: 1,
            },
            "argmax over axis 1"
        );
        assert_eq!(
            spec.dtype_output(),
            DType::U32,
            "argmax outputs u32 indices"
        );
    }

    #[test]
    fn reduce_rejects_integer_dtype() {
        assert_eq!(
            reduce_rule(
                ReduceOp::Sum,
                DType::U32,
                &Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape")),
                1,
            ),
            Err(CoreError::DTypeNotFloat),
            "reduce needs a float dtype"
        );
    }

    #[test]
    fn reduce_rejects_axis_out_of_range() {
        assert_eq!(
            reduce_rule(
                ReduceOp::Sum,
                DType::F32,
                &Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape")),
                2,
            ),
            Err(CoreError::AxisOutOfRange),
            "axis 2 does not exist"
        );
    }

    #[test]
    fn reduce_max_and_argmax_over_empty_axis_are_rejected() {
        let layout = Layout::contiguous(Shape::try_from([2, 0].as_slice()).expect("valid shape"));
        [ReduceOp::Max, ReduceOp::Argmax]
            .into_iter()
            .for_each(|op| {
                assert_eq!(
                    reduce_rule(op, DType::F32, &layout, 1),
                    Err(CoreError::EmptyReduction),
                    "{op:?} over an empty axis"
                );
            });
    }

    #[test]
    fn reduce_sum_over_empty_axis_keeps_axis_with_size_one() {
        let spec = reduce_rule(
            ReduceOp::Sum,
            DType::F32,
            &Layout::contiguous(Shape::try_from([2, 0].as_slice()).expect("valid shape")),
            1,
        )
        .expect("a sum over an empty axis is zero");
        assert_eq!(spec.output().dims(), &[2, 1], "output dims");
    }

    #[test]
    fn matmul_multiplies_last_two_axes() {
        let spec = matmul_rule(
            DType::BF16,
            &Layout::contiguous(Shape::try_from([5, 2, 3].as_slice()).expect("valid shape")),
            DType::BF16,
            &Layout::contiguous(Shape::try_from([5, 3, 4].as_slice()).expect("valid shape")),
        )
        .expect("compatible shapes");
        assert_eq!(
            spec,
            MatmulSpec {
                dtype: FloatDType::BF16,
                output: Shape::try_from([5, 2, 4].as_slice()).expect("valid shape"),
                batch: Shape::try_from([5].as_slice()).expect("valid shape"),
                m: 2,
                k: 3,
                n: 4,
                lhs: MatrixLayout {
                    offset: 0,
                    row_stride: 3,
                    col_stride: 1,
                    batch_strides: [6, 0, 0, 0],
                },
                rhs: MatrixLayout {
                    offset: 0,
                    row_stride: 4,
                    col_stride: 1,
                    batch_strides: [12, 0, 0, 0],
                },
            },
            "dtype, output, batch, matrix dims and operand strides"
        );
    }

    #[test]
    fn matmul_reads_transposed_operand_strides() {
        let rhs = Layout::contiguous(Shape::try_from([4, 3].as_slice()).expect("valid shape"))
            .permute(&[1, 0])
            .expect("valid permutation");
        let spec = matmul_rule(
            DType::F32,
            &Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape")),
            DType::F32,
            &rhs,
        )
        .expect("compatible shapes");
        assert_eq!(
            spec.rhs(),
            &MatrixLayout {
                offset: 0,
                row_stride: 1,
                col_stride: 3,
                batch_strides: [0, 0, 0, 0],
            },
            "the transposed rhs walks rows with stride 1"
        );
    }

    #[test]
    fn matmul_rejects_inner_dim_mismatch() {
        assert_eq!(
            matmul_rule(
                DType::F32,
                &Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape")),
                DType::F32,
                &Layout::contiguous(Shape::try_from([4, 5].as_slice()).expect("valid shape")),
            ),
            Err(CoreError::MatmulIncompatible),
            "inner dims 3 and 4 differ"
        );
    }

    #[test]
    fn matmul_rejects_batch_mismatch() {
        assert_eq!(
            matmul_rule(
                DType::F32,
                &Layout::contiguous(Shape::try_from([2, 2, 3].as_slice()).expect("valid shape")),
                DType::F32,
                &Layout::contiguous(Shape::try_from([5, 3, 4].as_slice()).expect("valid shape")),
            ),
            Err(CoreError::MatmulIncompatible),
            "batch dims 2 and 5 differ"
        );
    }

    #[test]
    fn matmul_rejects_rank_below_two() {
        struct LhsRhsCase<T0, T1> {
            lhs: T0,
            rhs: T1,
        }
        [
            LhsRhsCase {
                lhs: [3].as_slice(),
                rhs: [3, 4].as_slice(),
            },
            LhsRhsCase {
                lhs: &[2, 3],
                rhs: &[3],
            },
        ]
        .into_iter()
        .for_each(
            |LhsRhsCase {
                 lhs,
                 rhs,
             }| {
                assert_eq!(
                    matmul_rule(
                        DType::F32,
                        &Layout::contiguous(Shape::try_from(lhs).expect("valid shape")),
                        DType::F32,
                        &Layout::contiguous(Shape::try_from(rhs).expect("valid shape")),
                    ),
                    Err(CoreError::MatmulIncompatible),
                    "lhs {lhs:?} and rhs {rhs:?}"
                );
            },
        );
    }

    #[test]
    fn matmul_rejects_integer_dtype() {
        assert_eq!(
            matmul_rule(
                DType::U32,
                &Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape")),
                DType::U32,
                &Layout::contiguous(Shape::try_from([3, 4].as_slice()).expect("valid shape")),
            ),
            Err(CoreError::DTypeNotFloat),
            "matmul needs a float dtype"
        );
    }

    #[test]
    fn matmul_rejects_dtype_mismatch() {
        assert_eq!(
            matmul_rule(
                DType::BF16,
                &Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape")),
                DType::F32,
                &Layout::contiguous(Shape::try_from([3, 4].as_slice()).expect("valid shape")),
            ),
            Err(CoreError::DTypeMismatch),
            "matmul needs equal dtypes"
        );
    }

    #[test]
    fn matmul_rejects_output_beyond_the_element_limit() {
        assert_eq!(
            matmul_rule(
                DType::F32,
                &Layout::contiguous(Shape::try_from([1 << 16, 1].as_slice()).expect("valid shape")),
                DType::F32,
                &Layout::contiguous(Shape::try_from([1, 1 << 16].as_slice()).expect("valid shape")),
            ),
            Err(CoreError::ElementCountOverflow),
            "the output holds 2^32 elements"
        );
    }

    #[test]
    fn cast_returns_both_float_dtypes() {
        assert_eq!(
            cast_rule(DType::F32, DType::BF16),
            Ok(CastSpec {
                from: FloatDType::F32,
                to: FloatDType::BF16,
            }),
            "source and target dtypes"
        );
    }

    #[test]
    fn cast_rejects_integer_source_and_target() {
        struct FromToCase<T0, T1> {
            from: T0,
            to: T1,
        }
        [
            FromToCase {
                from: DType::U32,
                to: DType::F32,
            },
            FromToCase {
                from: DType::F32,
                to: DType::U32,
            },
        ]
        .into_iter()
        .for_each(
            |FromToCase {
                 from,
                 to,
             }| {
                assert_eq!(
                    cast_rule(from, to),
                    Err(CoreError::DTypeNotFloat),
                    "cast from {from:?} to {to:?}"
                );
            },
        );
    }

    #[test]
    fn gather_selects_table_rows() {
        let table = Layout::contiguous(Shape::try_from([4, 3].as_slice()).expect("valid shape"))
            .permute(&[1, 0])
            .expect("valid permutation");
        let spec = gather_rule(
            DType::BF16,
            &table,
            DType::U32,
            &Shape::try_from([5].as_slice()).expect("valid shape"),
        )
        .expect("valid gather");
        assert_eq!(
            spec,
            GatherSpec {
                dtype: FloatDType::BF16,
                output: Shape::try_from([5, 4].as_slice()).expect("valid shape"),
                rows: 3,
                cols: 4,
                row_stride: 1,
                col_stride: 3,
            },
            "dtype, output dims, table dims and strides"
        );
    }

    #[test]
    fn gather_rejects_integer_table() {
        assert_eq!(
            gather_rule(
                DType::U32,
                &Layout::contiguous(Shape::try_from([4, 3].as_slice()).expect("valid shape")),
                DType::U32,
                &Shape::try_from([2].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::DTypeNotFloat),
            "the table needs a float dtype"
        );
    }

    #[test]
    fn gather_rejects_non_u32_indices() {
        assert_eq!(
            gather_rule(
                DType::F32,
                &Layout::contiguous(Shape::try_from([4, 3].as_slice()).expect("valid shape")),
                DType::F32,
                &Shape::try_from([2].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::IndicesNotU32),
            "indices must be u32"
        );
    }

    #[test]
    fn gather_rejects_invalid_ranks() {
        struct TableIndicesCase<T0, T1> {
            table: T0,
            indices: T1,
        }
        [
            TableIndicesCase {
                table: [4].as_slice(),
                indices: [2].as_slice(),
            },
            TableIndicesCase {
                table: &[2, 3, 4],
                indices: &[2],
            },
            TableIndicesCase {
                table: &[4, 3],
                indices: &[2, 1],
            },
        ]
        .into_iter()
        .for_each(
            |TableIndicesCase {
                 table,
                 indices,
             }| {
                assert_eq!(
                    gather_rule(
                        DType::F32,
                        &Layout::contiguous(Shape::try_from(table).expect("valid shape")),
                        DType::U32,
                        &Shape::try_from(indices).expect("valid shape"),
                    ),
                    Err(CoreError::GatherIncompatible),
                    "table {table:?} and indices {indices:?}"
                );
            },
        );
    }

    #[test]
    fn gather_rejects_output_beyond_the_element_limit() {
        assert_eq!(
            gather_rule(
                DType::F32,
                &Layout::contiguous(Shape::try_from([1, 1 << 16].as_slice()).expect("valid shape")),
                DType::U32,
                &Shape::try_from([1 << 16].as_slice()).expect("valid shape"),
            ),
            Err(CoreError::ElementCountOverflow),
            "the output holds 2^32 elements"
        );
    }

    #[test]
    fn concat_adds_axis_dims() {
        let spec = concat_rule(
            DType::BF16,
            &Shape::try_from([2, 3, 4].as_slice()).expect("valid shape"),
            DType::BF16,
            &Shape::try_from([2, 5, 4].as_slice()).expect("valid shape"),
            1,
        )
        .expect("compatible shapes");
        assert_eq!(
            spec,
            ConcatSpec {
                dtype: FloatDType::BF16,
                output: Shape::try_from([2, 8, 4].as_slice()).expect("valid shape"),
                rhs_offset: 12,
            },
            "dtype, output dims and rhs offset"
        );
    }

    #[test]
    fn concat_rejects_mismatch_outside_axis() {
        assert_eq!(
            concat_rule(
                DType::F32,
                &Shape::try_from([2, 3].as_slice()).expect("valid shape"),
                DType::F32,
                &Shape::try_from([4, 5].as_slice()).expect("valid shape"),
                1,
            ),
            Err(CoreError::ConcatIncompatible),
            "dims 2 and 4 differ outside the axis"
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
            Err(CoreError::ConcatIncompatible),
            "ranks 3 and 2 differ"
        );
    }

    #[test]
    fn concat_rejects_dtype_mismatch_and_integer_dtype() {
        let shape = Shape::try_from([2, 3].as_slice()).expect("valid shape");
        assert_eq!(
            concat_rule(DType::F32, &shape, DType::BF16, &shape, 0),
            Err(CoreError::DTypeMismatch),
            "concat needs equal dtypes"
        );
        assert_eq!(
            concat_rule(DType::U32, &shape, DType::U32, &shape, 0),
            Err(CoreError::DTypeNotFloat),
            "concat needs a float dtype"
        );
    }

    #[test]
    fn concat_rejects_axis_out_of_range() {
        struct LhsRhsCase<T0, T1> {
            lhs: T0,
            rhs: T1,
        }
        [
            LhsRhsCase {
                lhs: [2, 3].as_slice(),
                rhs: [2, 3].as_slice(),
            },
            LhsRhsCase {
                lhs: &[2, 3, 4],
                rhs: &[2, 3],
            },
        ]
        .into_iter()
        .for_each(
            |LhsRhsCase {
                 lhs,
                 rhs,
             }| {
                assert_eq!(
                    concat_rule(
                        DType::F32,
                        &Shape::try_from(lhs).expect("valid shape"),
                        DType::F32,
                        &Shape::try_from(rhs).expect("valid shape"),
                        2,
                    ),
                    Err(CoreError::AxisOutOfRange),
                    "lhs {lhs:?} and rhs {rhs:?}"
                );
            },
        );
    }

    #[test]
    fn concat_rejects_output_beyond_the_element_limit() {
        let shape = Shape::try_from([1 << 30].as_slice()).expect("valid shape");
        assert_eq!(
            concat_rule(DType::F32, &shape, DType::F32, &shape, 0),
            Err(CoreError::ElementCountOverflow),
            "the output holds 2^31 elements"
        );
    }

    #[test]
    fn slice_update_returns_window_offset() {
        let target =
            Layout::contiguous(Shape::try_from([2, 8, 4].as_slice()).expect("valid shape"))
                .narrow(0, 1, 1)
                .expect("valid narrow");
        assert_eq!(
            slice_update_rule(
                DType::BF16,
                &target,
                DType::BF16,
                &Shape::try_from([1, 2, 4].as_slice()).expect("valid shape"),
                1,
                3,
            ),
            Ok(SliceUpdateSpec {
                dtype: FloatDType::BF16,
                offset: 44,
            }),
            "target offset 32 plus start 3 times stride 4"
        );
    }

    #[test]
    fn slice_update_rejects_out_of_bounds_and_mismatches() {
        struct UpdateAxisStartCase<T0, T1, T2> {
            update: T0,
            axis: T1,
            start: T2,
        }
        let target = Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape"));
        [
            UpdateAxisStartCase {
                update: [1, 3].as_slice(),
                axis: 0,
                start: usize::MAX,
            },
            UpdateAxisStartCase {
                update: &[2, 3],
                axis: 0,
                start: 1,
            },
            UpdateAxisStartCase {
                update: &[1, 2],
                axis: 0,
                start: 0,
            },
            UpdateAxisStartCase {
                update: &[1, 3, 1],
                axis: 0,
                start: 0,
            },
        ]
        .into_iter()
        .for_each(
            |UpdateAxisStartCase {
                 update,
                 axis,
                 start,
             }| {
                assert_eq!(
                    slice_update_rule(
                        DType::F32,
                        &target,
                        DType::F32,
                        &Shape::try_from(update).expect("valid shape"),
                        axis,
                        start,
                    ),
                    Err(CoreError::SliceUpdateOutOfBounds),
                    "update {update:?} at start {start}"
                );
            },
        );
    }

    #[test]
    fn slice_update_rejects_dtype_and_axis() {
        let target = Layout::contiguous(Shape::try_from([2, 3].as_slice()).expect("valid shape"));
        let update = Shape::try_from([2, 3].as_slice()).expect("valid shape");
        assert_eq!(
            slice_update_rule(DType::F32, &target, DType::BF16, &update, 0, 0),
            Err(CoreError::DTypeMismatch),
            "slice update needs equal dtypes"
        );
        assert_eq!(
            slice_update_rule(DType::F32, &target, DType::F32, &update, 2, 0),
            Err(CoreError::AxisOutOfRange),
            "axis 2 does not exist"
        );
    }
}
