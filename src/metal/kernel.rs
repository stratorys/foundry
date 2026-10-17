use std::ffi::c_void;
use std::ptr::NonNull;

use objc2_metal::MTLSize;

use crate::metal::args::{
    IndexSelectArgs,
    MatmulArgs,
    ReduceArgs,
    StridedArgs,
};
use crate::metal::error::{
    MetalError,
    not_float,
};
use crate::nn::DType;
use crate::nn::primitive::{
    BinaryOp,
    ReduceOp,
    UnaryOp,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Kernel {
    op: &'static str,
    dtype: DType,
}

pub enum KernelArgs<'args> {
    Strided(&'args StridedArgs),
    Reduce(&'args ReduceArgs),
    Matmul(&'args MatmulArgs),
    IndexSelect(&'args IndexSelectArgs),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Grid {
    Threads(usize),
    Groups { grid: MTLSize, group: MTLSize },
}

impl Kernel {
    pub fn new(
        op: &'static str,
        dtype: DType,
    ) -> Self {
        Self {
            op,
            dtype,
        }
    }

    pub fn float(
        op: &'static str,
        dtype: DType,
    ) -> Result<Self, MetalError> {
        match dtype {
            DType::F32 | DType::BF16 => Ok(Self::new(op, dtype)),
            DType::U32 => Err(not_float(dtype)),
        }
    }

    pub fn name(&self) -> String {
        let suffix = match self.dtype {
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            DType::U32 => "u32",
        };
        format!("{}_{suffix}", self.op)
    }
}

impl KernelArgs<'_> {
    pub fn bytes(&self) -> (NonNull<c_void>, usize) {
        match self {
            Self::Strided(args) => (NonNull::from(*args).cast(), size_of::<StridedArgs>()),
            Self::Reduce(args) => (NonNull::from(*args).cast(), size_of::<ReduceArgs>()),
            Self::Matmul(args) => (NonNull::from(*args).cast(), size_of::<MatmulArgs>()),
            Self::IndexSelect(args) => (NonNull::from(*args).cast(), size_of::<IndexSelectArgs>()),
        }
    }
}

impl Grid {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Threads(count) => *count == 0,
            Self::Groups {
                grid, ..
            } => grid.width == 0 || grid.height == 0 || grid.depth == 0,
        }
    }

    pub fn threads_required(&self) -> usize {
        match self {
            Self::Threads(_) => 1,
            Self::Groups {
                group, ..
            } => group
                .width
                .saturating_mul(group.height)
                .saturating_mul(group.depth),
        }
    }

    pub fn group(
        &self,
        threads_max: usize,
    ) -> MTLSize {
        match self {
            Self::Threads(count) => MTLSize {
                width: threads_max.min(*count),
                height: 1,
                depth: 1,
            },
            Self::Groups {
                group, ..
            } => *group,
        }
    }
}

pub fn to_dtype_kernel(
    dtype_from: DType,
    dtype: DType,
) -> Result<Kernel, MetalError> {
    let op = match dtype_from {
        DType::U32 => return Err(not_float(dtype_from)),
        _ if dtype_from == dtype => "copy",
        DType::F32 => "to_dtype_f32",
        DType::BF16 => "to_dtype_bf16",
    };
    Kernel::float(op, dtype)
}

pub fn unary_kernel(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "unary_neg",
        UnaryOp::Exp => "unary_exp",
        UnaryOp::Sqrt => "unary_sqrt",
        UnaryOp::Reciprocal => "unary_reciprocal",
    }
}

pub fn binary_kernel(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "binary_add",
        BinaryOp::Sub => "binary_sub",
        BinaryOp::Mul => "binary_mul",
        BinaryOp::Div => "binary_div",
    }
}

pub fn reduce_kernel(op: ReduceOp) -> &'static str {
    match op {
        ReduceOp::Sum => "reduce_sum",
        ReduceOp::Amax => "reduce_amax",
        ReduceOp::Argmax => "reduce_argmax",
    }
}

#[cfg(test)]
mod tests {
    use objc2_metal::MTLSize;

    use crate::metal::MetalError;
    use crate::metal::kernel::{
        Grid,
        Kernel,
        to_dtype_kernel,
    };
    use crate::nn::DType;

    #[test]
    fn kernel_name_joins_op_prefix_and_dtype_suffix() {
        assert_eq!(
            [
                Kernel::new("copy", DType::U32).name(),
                Kernel::new("unary_neg", DType::F32).name(),
                Kernel::new("matmul", DType::BF16).name(),
            ],
            ["copy_u32", "unary_neg_f32", "matmul_bf16"],
            "op prefix, underscore, dtype suffix"
        );
    }

    #[test]
    fn to_dtype_kernel_names_source_then_target_dtype() {
        assert_eq!(
            [
                to_dtype_kernel(DType::F32, DType::BF16).map(|kernel| kernel.name()),
                to_dtype_kernel(DType::BF16, DType::F32).map(|kernel| kernel.name()),
                to_dtype_kernel(DType::BF16, DType::BF16).map(|kernel| kernel.name()),
            ],
            [
                Ok(String::from("to_dtype_f32_bf16")),
                Ok(String::from("to_dtype_bf16_f32")),
                Ok(String::from("copy_bf16")),
            ],
            "the prefix names the source dtype, the suffix the target dtype, the same dtype copies"
        );
    }

    #[test]
    fn float_kernel_rejects_u32() {
        assert_eq!(
            [
                Kernel::float("matmul", DType::U32),
                to_dtype_kernel(DType::U32, DType::F32),
                to_dtype_kernel(DType::F32, DType::U32),
            ],
            [
                Err(MetalError::DTypeNotFloat),
                Err(MetalError::DTypeNotFloat),
                Err(MetalError::DTypeNotFloat),
            ],
            "float kernels have no u32 instance"
        );
    }

    #[test]
    fn grid_without_threads_or_groups_is_empty() {
        let group = MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        assert_eq!(
            [
                Grid::Threads(0).is_empty(),
                Grid::Threads(1).is_empty(),
                Grid::Groups {
                    grid: MTLSize {
                        width: 4,
                        height: 0,
                        depth: 1,
                    },
                    group,
                }
                .is_empty(),
                Grid::Groups {
                    grid: MTLSize {
                        width: 4,
                        height: 2,
                        depth: 1,
                    },
                    group,
                }
                .is_empty(),
            ],
            [true, false, true, false],
            "a grid is empty when it has no thread or a zero group dimension"
        );
    }

    #[test]
    fn grid_threads_required_is_the_group_volume() {
        assert_eq!(
            [
                Grid::Threads(5000).threads_required(),
                Grid::Groups {
                    grid: MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    group: MTLSize {
                        width: 16,
                        height: 16,
                        depth: 1,
                    },
                }
                .threads_required(),
            ],
            [1, 256],
            "a thread grid needs one thread per group, a group grid its group volume"
        );
    }

    #[test]
    fn grid_group_clamps_threads_to_the_pipeline_limit() {
        let group = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        assert_eq!(
            [
                Grid::Threads(10).group(1024),
                Grid::Threads(5000).group(1024),
                Grid::Groups {
                    grid: group,
                    group,
                }
                .group(1024),
            ],
            [
                MTLSize {
                    width: 10,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
                group,
            ],
            "a thread grid uses at most the pipeline limit, a group grid keeps its group"
        );
    }
}
