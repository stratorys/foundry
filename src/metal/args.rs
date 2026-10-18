use std::array;

use objc2_metal::MTLSize;
use tracing::error;

use crate::metal::error::{
    MetalError,
    invalid_operation,
};
use crate::nn::primitive::{
    Op,
    ReduceOp,
};
use crate::nn::{
    Layout,
    MAX_RANK,
    Shape,
};

pub const MATMUL_TILE: usize = 16;

#[repr(C)]
pub struct StridedArgs {
    count: u32,
    ndim: u32,
    dims: [u32; MAX_RANK],
    strides: [u32; MAX_RANK],
    offset: u32,
}

#[repr(C)]
pub struct ReduceArgs {
    rows: StridedArgs,
    dim_size: u32,
    dim_stride: u32,
}

#[repr(C)]
struct MatmulOperand {
    offset: u32,
    row_stride: u32,
    col_stride: u32,
    batch_strides: [u32; MAX_RANK],
}

#[repr(C)]
pub struct MatmulArgs {
    m: u32,
    n: u32,
    k: u32,
    batch_ndim: u32,
    batch_dims: [u32; MAX_RANK],
    lhs: MatmulOperand,
    rhs: MatmulOperand,
}

pub struct MatmulDispatch {
    pub args: MatmulArgs,
    pub grid: MTLSize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixLayout {
    offset: usize,
    row_stride: usize,
    col_stride: usize,
    batch_strides: [usize; MAX_RANK],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulGeometry {
    batch: Shape,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub lhs: MatrixLayout,
    pub rhs: MatrixLayout,
}

#[repr(C)]
pub struct IndexSelectArgs {
    count: u32,
    rows: u32,
    cols: u32,
    table_offset: u32,
    table_row_stride: u32,
    table_col_stride: u32,
    indices: StridedArgs,
}

impl StridedArgs {
    pub fn new(layout: &Layout) -> Result<Self, MetalError> {
        Self::with_shape(layout, layout.shape())
    }

    fn with_shape(
        layout: &Layout,
        shape: &Shape,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            count: index_u32(shape.numel())?,
            ndim: index_u32(shape.ndim())?,
            dims: padded_u32(shape.dims())?,
            strides: padded_u32(layout.strides())?,
            offset: index_u32(layout.storage_offset())?,
        })
    }

    pub fn window(
        target: &Layout,
        shape: &Shape,
        offset: usize,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            offset: index_u32(offset)?,
            ..Self::with_shape(target, shape)?
        })
    }
}

impl IndexSelectArgs {
    pub fn new(
        table: &Layout,
        indices: &Layout,
        output: &Shape,
    ) -> Result<Self, MetalError> {
        let (&[rows, cols], &[row_stride, col_stride]) = (table.shape().dims(), table.strides())
        else {
            return Err(invalid_operation(Op::IndexSelect {
                dim: 0,
            }));
        };
        Ok(Self {
            count: index_u32(output.numel())?,
            rows: index_u32(rows)?,
            cols: index_u32(cols)?,
            table_offset: index_u32(table.storage_offset())?,
            table_row_stride: index_u32(row_stride)?,
            table_col_stride: index_u32(col_stride)?,
            indices: StridedArgs::new(indices)?,
        })
    }
}

impl ReduceArgs {
    pub fn new(
        input: &Layout,
        output: &Shape,
        dim: usize,
    ) -> Result<Self, MetalError> {
        let (Some(&dim_size), Some(&dim_stride)) =
            (input.shape().dims().get(dim), input.strides().get(dim))
        else {
            return Err(invalid_operation(Op::Reduce {
                op: ReduceOp::Sum,
                dim,
            }));
        };
        Ok(Self {
            rows: StridedArgs::with_shape(input, output)?,
            dim_size: index_u32(dim_size)?,
            dim_stride: index_u32(dim_stride)?,
        })
    }
}

impl MatrixLayout {
    fn of(layout: &Layout) -> Option<Self> {
        let (batch_strides, [row_stride, col_stride]) = layout.strides().split_last_chunk::<2>()?;
        Some(Self {
            offset: layout.storage_offset(),
            row_stride: *row_stride,
            col_stride: *col_stride,
            batch_strides: padded(batch_strides),
        })
    }
}

impl MatmulGeometry {
    pub fn of(
        lhs: &Layout,
        rhs: &Layout,
    ) -> Option<Self> {
        let (batch, &[m, k]) = lhs.shape().dims().split_last_chunk::<2>()?;
        let &[_, n] = rhs.shape().dims().split_last_chunk::<2>()?.1;
        Some(Self {
            batch: Shape::try_from(batch).ok()?,
            m,
            k,
            n,
            lhs: MatrixLayout::of(lhs)?,
            rhs: MatrixLayout::of(rhs)?,
        })
    }
}

impl MatmulOperand {
    fn new(matrix: &MatrixLayout) -> Result<Self, MetalError> {
        Ok(Self {
            offset: index_u32(matrix.offset)?,
            row_stride: index_u32(matrix.row_stride)?,
            col_stride: index_u32(matrix.col_stride)?,
            batch_strides: padded_u32(&matrix.batch_strides)?,
        })
    }
}

impl MatmulDispatch {
    pub fn new(geometry: &MatmulGeometry) -> Result<Self, MetalError> {
        let batch_dims = geometry.batch.dims();
        let args = MatmulArgs {
            m: index_u32(geometry.m)?,
            n: index_u32(geometry.n)?,
            k: index_u32(geometry.k)?,
            batch_ndim: index_u32(batch_dims.len())?,
            batch_dims: padded_u32(batch_dims)?,
            lhs: MatmulOperand::new(&geometry.lhs)?,
            rhs: MatmulOperand::new(&geometry.rhs)?,
        };
        let grid = MTLSize {
            width: geometry.n.div_ceil(MATMUL_TILE),
            height: geometry.m.div_ceil(MATMUL_TILE),
            depth: geometry.batch.numel(),
        };
        Ok(Self {
            args,
            grid,
        })
    }
}

pub fn is_matmul_ready(
    matrix: &MatrixLayout,
    rows: usize,
    cols: usize,
) -> bool {
    let is_row_major = matrix.col_stride == 1 || cols == 1;
    let is_transposed = matrix.row_stride == 1 || rows == 1;
    is_row_major || is_transposed
}

pub fn slice_offset(
    layout: &Layout,
    dim: usize,
    start: Option<usize>,
) -> Option<usize> {
    start?
        .checked_mul(*layout.strides().get(dim)?)?
        .checked_add(layout.storage_offset())
}

fn padded(values: &[usize]) -> [usize; MAX_RANK] {
    array::from_fn(|index| values.get(index).copied().unwrap_or(0))
}

fn index_u32(value: usize) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| {
        error!(
            message = "Value does not fit in a 32-bit kernel index.",
            value
        );
        MetalError::KernelIndexTooLarge
    })
}

fn padded_u32(values: &[usize]) -> Result<[u32; MAX_RANK], MetalError> {
    values
        .iter()
        .zip(0..MAX_RANK)
        .try_fold([0_u32; MAX_RANK], |mut padded, (&value, dim)| {
            if let Some(slot) = padded.get_mut(dim) {
                *slot = index_u32(value)?;
            }
            Ok(padded)
        })
}

#[cfg(test)]
mod tests {
    use crate::metal::args::{
        MatmulGeometry,
        MatrixLayout,
        slice_offset,
    };
    use crate::nn::{
        Layout,
        Shape,
    };

    #[test]
    fn matmul_geometry_reads_dims_and_operand_strides() {
        let geometry = MatmulGeometry::of(
            &Layout::contiguous(Shape::try_from([5, 2, 3]).expect("valid shape")),
            &Layout::contiguous(Shape::try_from([5, 3, 4]).expect("valid shape")),
        );
        assert_eq!(
            geometry,
            Some(MatmulGeometry {
                batch: Shape::try_from([5]).expect("valid shape"),
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
            }),
            "batch, matrix dims and operand strides"
        );
    }

    #[test]
    fn matmul_geometry_reads_transposed_operand_strides() {
        let rhs = Layout::contiguous(Shape::try_from([4, 3]).expect("valid shape"))
            .permute(&[1, 0])
            .expect("valid permutation");
        assert_eq!(
            MatmulGeometry::of(
                &Layout::contiguous(Shape::try_from([2, 3]).expect("valid shape")),
                &rhs,
            )
            .map(|geometry| geometry.rhs),
            Some(MatrixLayout {
                offset: 0,
                row_stride: 1,
                col_stride: 3,
                batch_strides: [0, 0, 0, 0],
            }),
            "the transposed rhs walks rows with stride 1"
        );
    }

    #[test]
    fn slice_offset_adds_start_times_stride_to_the_layout_offset() {
        let target = Layout::contiguous(Shape::try_from([2, 8, 4]).expect("valid shape"))
            .narrow(0, 1, 1)
            .expect("valid narrow");
        let cat_output = Layout::contiguous(Shape::try_from([2, 8, 4]).expect("valid shape"));
        assert_eq!(
            [
                slice_offset(&target, 1, Some(3)),
                slice_offset(&cat_output, 1, Some(3)),
                slice_offset(&target, 3, Some(0)),
                slice_offset(&target, 1, Some(usize::MAX)),
            ],
            [Some(44), Some(12), None, None],
            "offset 32 plus 3 times 4, then 3 times 4, then a missing dim, then an overflow"
        );
    }
}
