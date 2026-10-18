mod args;
mod error;
mod kernel;
mod stream;

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer,
    MTLCommandBuffer,
    MTLCommandQueue,
    MTLComputeCommandEncoder,
    MTLComputePipelineState,
    MTLCreateSystemDefaultDevice,
    MTLDevice,
    MTLLibrary,
    MTLResourceOptions,
    MTLSize,
};
use tracing::error;

use self::args::{
    IndexSelectArgs,
    MATMUL_TILE,
    MatmulDispatch,
    MatmulGeometry,
    MatrixLayout,
    ReduceArgs,
    StridedArgs,
    is_matmul_ready,
    slice_offset,
};
pub use self::error::MetalError;
use self::error::invalid_operation;
use self::kernel::{
    Grid,
    Kernel,
    KernelArgs,
    binary_kernel,
    reduce_kernel,
    to_dtype_kernel,
    unary_kernel,
};
use self::stream::CommandStream;
use crate::nn::primitive::{
    Op,
    ReduceOp,
};
use crate::nn::{
    AllocatedMemory,
    DType,
    Device,
    Layout,
    MAX_ITEMSIZE,
    MAX_RANK,
    Operand,
    Shape,
};

const KERNEL_SOURCES: &[&str] = &[
    include_str!("kernels/common.metal"),
    include_str!("kernels/copy.metal"),
    include_str!("kernels/unary.metal"),
    include_str!("kernels/binary.metal"),
    include_str!("kernels/reduce.metal"),
    include_str!("kernels/matmul.metal"),
    include_str!("kernels/index_select.metal"),
    include_str!("kernels/slice_scatter.metal"),
];

const REDUCE_THREADS: usize = 256;

type MtlDevice = Retained<ProtocolObject<dyn MTLDevice>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type ComputeEncoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

pub struct MetalStorage {
    buffer: Buffer,
    nbytes: usize,
    queue: Queue,
}

pub struct MetalDevice {
    device: MtlDevice,
    library: Library,
    pipelines: HashMap<Kernel, Pipeline>,
    stream: CommandStream,
    allocated_bytes_baseline: u64,
    allocated_bytes_peak: u64,
}

struct BufferView<'buffer> {
    storage: &'buffer MetalStorage,
    layout: &'buffer Layout,
}

impl MetalDevice {
    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceUnavailable)?;
        let queue = device.newCommandQueue().ok_or(MetalError::QueueCreation)?;
        let prelude = [
            ("MAX_RANK", MAX_RANK),
            ("REDUCE_THREADS", REDUCE_THREADS),
            ("MATMUL_TILE", MATMUL_TILE),
        ]
        .map(|(name, value)| format!("#define {name} {value}\n"))
        .concat();
        let source = NSString::from_str(&[prelude.as_str(), &KERNEL_SOURCES.concat()].concat());
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|error| {
                error!(
                    message = "Metal library compilation failed.",
                    error = %error.localizedDescription(),
                );
                MetalError::LibraryCompilation
            })?;
        let allocated_bytes_baseline = allocated_bytes(&device)?;
        Ok(Self {
            device,
            library,
            pipelines: HashMap::new(),
            stream: CommandStream::new(queue),
            allocated_bytes_baseline,
            allocated_bytes_peak: allocated_bytes_baseline,
        })
    }

    fn allocate(
        &mut self,
        nbytes: usize,
    ) -> Result<MetalStorage, MetalError> {
        let max_bytes = self.device.maxBufferLength();
        if nbytes > max_bytes {
            error!(
                message = "Buffer exceeds the maximum Metal buffer length.",
                bytes = nbytes,
                max_bytes,
            );
            return Err(MetalError::BufferTooLarge);
        }
        let buffer = self
            .device
            .newBufferWithLength_options(
                nbytes.max(MAX_ITEMSIZE),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| {
                error!(message = "Metal buffer allocation failed.", bytes = nbytes);
                MetalError::BufferAllocation
            })?;
        self.allocated_bytes_peak = self
            .allocated_bytes_peak
            .max(allocated_bytes(&self.device)?);
        Ok(MetalStorage {
            buffer,
            nbytes,
            queue: self.stream.queue().clone(),
        })
    }

    fn owns(
        &self,
        storage: &MetalStorage,
    ) -> Result<(), MetalError> {
        if std::ptr::eq::<ProtocolObject<dyn MTLCommandQueue>>(
            &*storage.queue,
            &**self.stream.queue(),
        ) {
            Ok(())
        } else {
            error!(message = "Storage was produced by another Metal device instance.");
            Err(MetalError::ForeignStorage)
        }
    }

    fn view<'buffer>(
        &self,
        operand: &Operand<'buffer, &'buffer MetalStorage>,
    ) -> Result<BufferView<'buffer>, MetalError> {
        let storage: &'buffer MetalStorage = operand.storage();
        self.owns(storage)?;
        Ok(BufferView {
            storage,
            layout: operand.layout(),
        })
    }

    fn pipeline(
        &mut self,
        kernel: Kernel,
    ) -> Result<Pipeline, MetalError> {
        if let Some(pipeline) = self.pipelines.get(&kernel) {
            return Ok(pipeline.clone());
        }
        let name = kernel.name();
        let function = self
            .library
            .newFunctionWithName(&NSString::from_str(&name))
            .ok_or_else(|| {
                error!(message = "Kernel is not in the Metal library.", %name);
                MetalError::KernelNotFound
            })?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| {
                error!(
                    message = "Metal pipeline creation failed.",
                    %name,
                    error = %error.localizedDescription(),
                );
                MetalError::PipelineCreation
            })?;
        self.pipelines.insert(kernel, pipeline.clone());
        Ok(pipeline)
    }

    fn dispatch(
        &mut self,
        kernel: Kernel,
        buffers: &[&MetalStorage],
        args: &[KernelArgs<'_>],
        grid: Grid,
    ) -> Result<(), MetalError> {
        if grid.is_empty() {
            return Ok(());
        }
        let pipeline = self.pipeline(kernel)?;
        let threads_max = pipeline.maxTotalThreadsPerThreadgroup();
        let threads_required = grid.threads_required();
        if threads_max < threads_required {
            return Err(threadgroup_too_small(kernel, threads_max, threads_required));
        }
        let group = grid.group(threads_max);
        self.stream.compute(&pipeline, |encoder| {
            // SAFETY: every buffer is borrowed for the whole call and the
            // command buffer retains it. Each one was produced on this
            // device's serial queue, since `view` and `owns` reject foreign
            // storage, so every earlier access to it is ordered before this
            // dispatch. Each caller builds `args` from the views it binds: a
            // view comes from a `Tensor` operand, which only `crate::nn` can
            // build, or from a buffer this device allocated with a contiguous
            // layout of its shape, so every index its layout addresses stays
            // inside its storage, and each output holds one element of the
            // kernel's output type per index the kernel writes. Every `Shape`
            // has at most `i32::MAX` non-zero elements, so addresses and
            // kernel loop counters stay below `u32::MAX` and do not wrap.
            // Each `KernelArgs` points to a `#[repr(C)]` struct of `u32` that
            // matches the struct of the same name in `common.metal`,
            // `reduce.metal`, `matmul.metal` or `index_select.metal`, and Metal
            // copies the bytes before the call returns.
            unsafe {
                buffers.iter().zip(0..).for_each(|(storage, index)| {
                    encoder.setBuffer_offset_atIndex(Some(&storage.buffer), 0, index);
                });
                args.iter()
                    .zip(buffers.len()..)
                    .for_each(|(kernel_args, index)| {
                        let (bytes, length) = kernel_args.bytes();
                        encoder.setBytes_length_atIndex(bytes, length, index);
                    });
            };
            match grid {
                Grid::Threads(count) => encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: count,
                        height: 1,
                        depth: 1,
                    },
                    group,
                ),
                Grid::Groups {
                    grid, ..
                } => encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, group),
            }
        })
    }

    fn scatter_window(
        &mut self,
        kernel: Kernel,
        input: &BufferView<'_>,
        window: &StridedArgs,
        output: &mut MetalStorage,
    ) -> Result<(), MetalError> {
        let input_args = StridedArgs::new(input.layout)?;
        // The output is borrowed mutably, so no other reference to it is live
        // during encoding: either a slice scatter target, which
        // `Tensor::slice_scatter` checked to be contiguous and owned by no
        // other tensor before building its `OperandMut`, or a fresh cat output
        // with a contiguous layout. The shape rule bounds `window` inside that
        // layout, so every written index stays inside the storage and is
        // written by one thread only.
        self.dispatch(
            kernel,
            &[input.storage, &*output],
            &[
                KernelArgs::Strided(&input_args),
                KernelArgs::Strided(window),
            ],
            Grid::Threads(input.layout.shape().numel()),
        )
    }

    fn elementwise(
        &mut self,
        kernel: Kernel,
        inputs: &[&BufferView<'_>],
        output_shape: &Shape,
        dtype: DType,
    ) -> Result<MetalStorage, MetalError> {
        let output = self.allocate(output_shape.nbytes(dtype))?;
        let args = inputs
            .iter()
            .map(|input| StridedArgs::new(input.layout))
            .collect::<Result<Vec<_>, _>>()?;
        let kernel_args: Vec<KernelArgs<'_>> = args.iter().map(KernelArgs::Strided).collect();
        let buffers: Vec<&MetalStorage> = inputs
            .iter()
            .map(|input| input.storage)
            .chain([&output])
            .collect();
        // `Tensor` broadcasts every input to the output shape before calling
        // the device, so the kernel reads each input only for `gid < count`,
        // inside its layout.
        self.dispatch(
            kernel,
            &buffers,
            &kernel_args,
            Grid::Threads(output_shape.numel()),
        )?;
        Ok(output)
    }

    fn matmul_operand(
        &mut self,
        operand: &BufferView<'_>,
        matrix: &MatrixLayout,
        [rows, cols]: [usize; 2],
        dtype: DType,
    ) -> Result<Option<MetalStorage>, MetalError> {
        if is_matmul_ready(matrix, rows, cols) {
            return Ok(None);
        }
        self.elementwise(
            Kernel::new("copy", dtype),
            &[operand],
            operand.layout.shape(),
            dtype,
        )
        .map(Some)
    }

    fn reduce(
        &mut self,
        (op, dim): (ReduceOp, usize),
        input: &BufferView<'_>,
        input_dtype: DType,
        output_shape: &Shape,
        dtype: DType,
    ) -> Result<MetalStorage, MetalError> {
        let args = ReduceArgs::new(input.layout, output_shape, dim)?;
        let output = self.allocate(output_shape.nbytes(dtype))?;
        // One threadgroup per output row.
        self.dispatch(
            Kernel::float(reduce_kernel(op), input_dtype)?,
            &[input.storage, &output],
            &[KernelArgs::Reduce(&args)],
            Grid::Groups {
                grid: MTLSize {
                    width: output_shape.numel(),
                    height: 1,
                    depth: 1,
                },
                group: MTLSize {
                    width: REDUCE_THREADS,
                    height: 1,
                    depth: 1,
                },
            },
        )?;
        Ok(output)
    }

    fn matmul(
        &mut self,
        lhs: BufferView<'_>,
        rhs: BufferView<'_>,
        output_shape: &Shape,
        dtype: DType,
    ) -> Result<MetalStorage, MetalError> {
        let geometry = MatmulGeometry::of(lhs.layout, rhs.layout)
            .ok_or_else(|| invalid_operation(Op::Matmul))?;
        let output = self.allocate(output_shape.nbytes(dtype))?;
        if output_shape.numel() == 0 {
            return Ok(output);
        }
        let lhs_copy = self.matmul_operand(&lhs, &geometry.lhs, [geometry.m, geometry.k], dtype)?;
        let rhs_copy = self.matmul_operand(&rhs, &geometry.rhs, [geometry.k, geometry.n], dtype)?;
        let lhs_layout = Layout::contiguous(*lhs.layout.shape());
        let rhs_layout = Layout::contiguous(*rhs.layout.shape());
        let lhs = lhs_copy.as_ref().map_or(lhs, |storage| BufferView {
            storage,
            layout: &lhs_layout,
        });
        let rhs = rhs_copy.as_ref().map_or(rhs, |storage| BufferView {
            storage,
            layout: &rhs_layout,
        });
        // The geometry is read again from the views bound to the kernel, so a
        // copied operand is addressed through its contiguous layout.
        let geometry = MatmulGeometry::of(lhs.layout, rhs.layout)
            .ok_or_else(|| invalid_operation(Op::Matmul))?;
        let dispatch = MatmulDispatch::new(&geometry)?;
        // The output holds `batch * m * n` elements, and the kernel writes
        // only inside `[m, n]` of each batch.
        self.dispatch(
            Kernel::float("matmul", dtype)?,
            &[lhs.storage, rhs.storage, &output],
            &[KernelArgs::Matmul(&dispatch.args)],
            Grid::Groups {
                grid: dispatch.grid,
                group: MTLSize {
                    width: MATMUL_TILE,
                    height: MATMUL_TILE,
                    depth: 1,
                },
            },
        )?;
        Ok(output)
    }

    fn index_select(
        &mut self,
        table: &BufferView<'_>,
        indices: &BufferView<'_>,
        output_shape: &Shape,
        dtype: DType,
    ) -> Result<MetalStorage, MetalError> {
        let args = IndexSelectArgs::new(table.layout, indices.layout, output_shape)?;
        let output = self.allocate(output_shape.nbytes(dtype))?;
        // The kernel reads a table row only when its index is below `rows`.
        self.dispatch(
            Kernel::float("index_select", dtype)?,
            &[table.storage, indices.storage, &output],
            &[KernelArgs::IndexSelect(&args)],
            Grid::Threads(output_shape.numel()),
        )?;
        Ok(output)
    }

    fn cat(
        &mut self,
        lhs: &BufferView<'_>,
        rhs: &BufferView<'_>,
        dim: usize,
        output_shape: &Shape,
        dtype: DType,
    ) -> Result<MetalStorage, MetalError> {
        let kernel = Kernel::float("slice_scatter", dtype)?;
        let mut output = self.allocate(output_shape.nbytes(dtype))?;
        let layout = Layout::contiguous(*output_shape);
        let rhs_offset = slice_offset(&layout, dim, lhs.layout.shape().dims().get(dim).copied())
            .ok_or_else(|| {
                invalid_operation(Op::Cat {
                    dim,
                })
            })?;
        let lhs_window = StridedArgs::window(&layout, lhs.layout.shape(), 0)?;
        let rhs_window = StridedArgs::window(&layout, rhs.layout.shape(), rhs_offset)?;
        self.scatter_window(kernel, lhs, &lhs_window, &mut output)?;
        self.scatter_window(kernel, rhs, &rhs_window, &mut output)?;
        Ok(output)
    }
}

impl Device for MetalDevice {
    type Error = MetalError;
    type Storage = MetalStorage;

    fn device_name(&self) -> String { self.device.name().to_string() }

    fn max_memory_allocated(&self) -> Option<AllocatedMemory> {
        Some(AllocatedMemory {
            bytes: self
                .allocated_bytes_peak
                .saturating_sub(self.allocated_bytes_baseline),
            source: "metal_current_allocated_size",
        })
    }

    fn storage_from_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<MetalStorage, MetalError> {
        let storage = self.allocate(bytes.len())?;
        // SAFETY: the buffer is freshly allocated in shared storage with at
        // least `storage.nbytes == bytes.len()` bytes, and no GPU
        // work references it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                storage.buffer.contents().cast::<u8>().as_ptr(),
                bytes.len(),
            )
        };
        Ok(storage)
    }

    fn zeros(
        &mut self,
        nbytes: usize,
    ) -> Result<MetalStorage, MetalError> {
        let storage = self.allocate(nbytes)?;
        self.stream.fill_zero(&storage.buffer)?;
        Ok(storage)
    }

    fn storage_to_bytes(
        &mut self,
        input: &Operand<&MetalStorage>,
    ) -> Result<Vec<u8>, MetalError> {
        let dtype = input.dtype();
        let input = self.view(input)?;
        let bytes = input.layout.shape().nbytes(dtype);
        let bytes_offset = input
            .layout
            .storage_offset()
            .saturating_mul(dtype.itemsize());
        let bytes_len = input.storage.nbytes;
        match bytes_offset.checked_add(bytes) {
            Some(end) if end <= bytes_len => {}
            Some(_) | None => {
                error!(
                    message = "Reading bytes exceeds the buffer length.",
                    bytes, bytes_offset, bytes_len,
                );
                return Err(MetalError::ToBytesOutOfBounds);
            }
        }
        self.stream.synchronize()?;
        let mut output = vec![0_u8; bytes];
        // SAFETY: the buffer uses shared storage, `bytes_offset + bytes <=
        // nbytes` was checked above, `output` holds `bytes` bytes,
        // and every command buffer that writes the buffer has completed:
        // `view` checked that the storage was produced on this device's
        // queue, and `synchronize` committed the open command buffer, then
        // waited on every committed command buffer and saw each complete.
        unsafe {
            std::ptr::copy_nonoverlapping(
                input
                    .storage
                    .buffer
                    .contents()
                    .cast::<u8>()
                    .as_ptr()
                    .add(bytes_offset)
                    .cast_const(),
                output.as_mut_ptr(),
                bytes,
            )
        };
        Ok(output)
    }

    fn run(
        &mut self,
        op: Op,
        inputs: &[Operand<&MetalStorage>],
        output: &Shape,
        dtype: DType,
    ) -> Result<MetalStorage, MetalError> {
        match (op, inputs) {
            (Op::Unary(unary), [input]) => self.elementwise(
                Kernel::float(unary_kernel(unary), dtype)?,
                &[&self.view(input)?],
                output,
                dtype,
            ),
            (Op::Binary(binary), [lhs, rhs]) => self.elementwise(
                Kernel::float(binary_kernel(binary), dtype)?,
                &[&self.view(lhs)?, &self.view(rhs)?],
                output,
                dtype,
            ),
            (
                Op::Reduce {
                    op,
                    dim,
                },
                [input],
            ) => self.reduce((op, dim), &self.view(input)?, input.dtype(), output, dtype),
            (Op::Matmul, [lhs, rhs]) => {
                self.matmul(self.view(lhs)?, self.view(rhs)?, output, dtype)
            }
            (Op::Copy, [input]) => self.elementwise(
                Kernel::new("copy", dtype),
                &[&self.view(input)?],
                output,
                dtype,
            ),
            (Op::ToDtype(_), [input]) => self.elementwise(
                to_dtype_kernel(input.dtype(), dtype)?,
                &[&self.view(input)?],
                output,
                dtype,
            ),
            (
                Op::IndexSelect {
                    ..
                },
                [table, indices],
            ) => self.index_select(&self.view(table)?, &self.view(indices)?, output, dtype),
            (
                Op::Cat {
                    dim,
                },
                [lhs, rhs],
            ) => self.cat(&self.view(lhs)?, &self.view(rhs)?, dim, output, dtype),
            _ => Err(invalid_operation(op)),
        }
    }

    fn run_in_place(
        &mut self,
        op: Op,
        mut target: Operand<&mut MetalStorage>,
        inputs: &[Operand<&MetalStorage>],
    ) -> Result<(), MetalError> {
        let (
            Op::SliceScatter {
                dim,
                start,
            },
            [src],
        ) = (op, inputs)
        else {
            return Err(invalid_operation(op));
        };
        self.owns(target.storage())?;
        let layout = target.layout();
        let src = self.view(src)?;
        let offset = slice_offset(layout, dim, Some(start)).ok_or_else(|| invalid_operation(op))?;
        let window = StridedArgs::window(layout, src.layout.shape(), offset)?;
        self.scatter_window(
            Kernel::float("slice_scatter", target.dtype())?,
            &src,
            &window,
            target.storage_mut(),
        )
    }
}

fn threadgroup_too_small(
    kernel: Kernel,
    threads: usize,
    threads_required: usize,
) -> MetalError {
    error!(
        message = "Kernel allows too few threads per threadgroup.",
        name = %kernel.name(),
        threads,
        threads_required,
    );
    MetalError::ThreadgroupTooSmall
}

fn allocated_bytes(device: &MtlDevice) -> Result<u64, MetalError> {
    let allocated_bytes = device.currentAllocatedSize();
    u64::try_from(allocated_bytes).map_err(|_| {
        error!(
            message = "Metal allocated size does not fit in u64.",
            allocated_bytes
        );
        MetalError::AllocatedSizeOverflow
    })
}

#[cfg(test)]
mod tests {
    use std::f32::consts::{
        E,
        PI,
    };

    use crate::metal::MetalDevice;
    use crate::metal::args::{
        MatmulGeometry,
        is_matmul_ready,
    };
    use crate::nn::{
        DType,
        Device,
        Shape,
        Tensor,
        TensorError,
    };

    #[test]
    fn device_name_is_not_empty() {
        let device = MetalDevice::new().expect("a Metal device is available");
        assert_ne!(device.device_name(), "", "the Metal device has a name");
    }

    #[test]
    fn max_memory_allocated_counts_from_bytes_and_never_decreases() {
        let nbytes = 1 << 20;
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let storage = device
            .storage_from_bytes(&vec![0; nbytes])
            .expect("storage_from_bytes succeeds");
        let peak = device
            .max_memory_allocated()
            .expect("Metal tracks peak memory");
        assert_eq!(
            peak.source, "metal_current_allocated_size",
            "Metal reads the device allocated size"
        );
        assert!(
            peak.bytes >= u64::try_from(nbytes).expect("the length fits in u64"),
            "peak {} bytes covers the {nbytes} written bytes",
            peak.bytes
        );
        drop(storage);
        device.zeros(16).expect("zeros succeeds");
        let peak_after_free = device
            .max_memory_allocated()
            .expect("Metal tracks peak memory");
        assert!(
            peak_after_free.bytes >= peak.bytes,
            "peak {} bytes does not fall below {} bytes after a free",
            peak_after_free.bytes,
            peak.bytes
        );
    }

    #[test]
    fn round_trip_returns_the_same_bytes_for_each_dtype() {
        struct DtypeBytesDimsCase {
            dtype: DType,
            bytes: Vec<u8>,
            dims: Vec<usize>,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let cases = [
            DtypeBytesDimsCase {
                dtype: DType::F32,
                bytes: [1.0_f32, -2.5, 1.5e-3, f32::MIN_POSITIVE, -0.0, f32::MAX]
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                dims: vec![2, 3],
            },
            DtypeBytesDimsCase {
                dtype: DType::BF16,
                bytes: [0x3F80_u16, 0xC020, 0x4049, 0x0001, 0x8000, 0x7F7F]
                    .iter()
                    .flat_map(|bits| bits.to_le_bytes())
                    .collect(),
                dims: vec![6],
            },
            DtypeBytesDimsCase {
                dtype: DType::U32,
                bytes: [0_u32, 1, 128_000, u32::MAX]
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                dims: vec![2, 1, 2],
            },
            DtypeBytesDimsCase {
                dtype: DType::F32,
                bytes: vec![],
                dims: vec![0, 3],
            },
        ];
        let read_back: Vec<Vec<u8>> = cases
            .iter()
            .map(|case| {
                Tensor::from_bytes(
                    &mut device,
                    &case.bytes,
                    case.dtype,
                    Shape::try_from(case.dims.as_slice()).expect("the shape is valid"),
                )
                .expect("from_bytes succeeds")
                .to_bytes(&mut device)
                .expect("to_bytes succeeds")
            })
            .collect();
        assert_eq!(
            read_back,
            cases.map(|case| case.bytes),
            "f32, bf16, u32 and empty bytes survive the round trip"
        );
    }

    #[test]
    fn zeros_to_bytes_only_zero_bytes() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        [DType::F32, DType::BF16, DType::U32]
            .into_iter()
            .for_each(|dtype| {
                let read_back = Tensor::zeros(
                    &mut device,
                    dtype,
                    Shape::try_from([2, 3]).expect("the shape is valid"),
                )
                .expect("zeros succeeds")
                .to_bytes(&mut device)
                .expect("to_bytes succeeds");
                assert_eq!(
                    read_back.len(),
                    6 * dtype.itemsize(),
                    "zeros of {dtype:?} has the shape's byte length"
                );
                assert!(
                    read_back.iter().all(|&byte| byte == 0),
                    "zeros of {dtype:?} reads back only zero bytes"
                );
            });
    }

    #[test]
    fn zeros_after_from_bytes_keeps_both_values() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let bytes: Vec<u8> = [7_u32, 8, 9]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let stored = Tensor::from_bytes(
            &mut device,
            &bytes,
            DType::U32,
            Shape::try_from([3]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds");
        let zeros = Tensor::zeros(
            &mut device,
            DType::U32,
            Shape::try_from([3]).expect("the shape is valid"),
        )
        .expect("zeros succeeds");
        assert_eq!(
            zeros.to_bytes(&mut device).expect("to_bytes succeeds"),
            vec![0; 12],
            "zeros reads back zero bytes"
        );
        assert_eq!(
            stored.to_bytes(&mut device).expect("to_bytes succeeds"),
            bytes,
            "the earlier from_bytes is unchanged"
        );
    }

    #[test]
    fn from_bytes_with_wrong_byte_length_is_rejected_before_allocation() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        assert_eq!(
            [
                Tensor::from_bytes(
                    &mut device,
                    &[0; 10],
                    DType::F32,
                    Shape::try_from([2, 2]).expect("the shape is valid"),
                )
                .err(),
                Tensor::from_bytes(
                    &mut device,
                    &[],
                    DType::F32,
                    Shape::try_from([(1 << 31) - 1]).expect("the shape is valid"),
                )
                .err(),
            ],
            [Some(TensorError::ByteLengthMismatch); 2],
            "a byte length that does not match shape × dtype is rejected before allocation"
        );
    }

    #[test]
    fn narrow_on_leading_dim_reads_the_sub_range() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let bytes: Vec<u8> = (0_u32..6).flat_map(u32::to_le_bytes).collect();
        let narrowed = Tensor::from_bytes(
            &mut device,
            &bytes,
            DType::U32,
            Shape::try_from([3, 2]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds")
        .narrow(0, 1, 2)
        .expect("the narrow is valid");
        let expected: Vec<u8> = (2_u32..6).flat_map(u32::to_le_bytes).collect();
        assert_eq!(
            narrowed.to_bytes(&mut device).expect("to_bytes succeeds"),
            expected,
            "the narrowed rows are read back"
        );
    }

    #[test]
    fn permuted_to_bytes_is_rejected() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let bytes: Vec<u8> = (0_u32..6).flat_map(u32::to_le_bytes).collect();
        let permuted = Tensor::from_bytes(
            &mut device,
            &bytes,
            DType::U32,
            Shape::try_from([3, 2]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid");
        assert_eq!(
            permuted.to_bytes(&mut device).err(),
            Some(TensorError::ToBytesNonContiguous),
            "a non-contiguous layout cannot be read as bytes"
        );
    }

    #[test]
    fn copy_of_permuted_tensor_is_contiguous_for_each_float_dtype() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let expected: Vec<f32> = [
            0_u16, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19,
            23,
        ]
        .into_iter()
        .map(f32::from)
        .collect();
        let source = Tensor::from_bytes(
            &mut device,
            &(0_u16..24)
                .flat_map(|value| f32::from(value).to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3, 4]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds");
        [DType::F32, DType::BF16].into_iter().for_each(|dtype| {
            let copied = source
                .to_dtype(&mut device, dtype)
                .expect("to_dtype succeeds")
                .permute(&[2, 0, 1])
                .expect("the permutation is valid")
                .contiguous(&mut device)
                .expect("the copy succeeds");
            assert_eq!(copied.dtype(), dtype, "the copy keeps the dtype");
            assert_eq!(copied.shape().dims(), &[4, 2, 3], "permuted dims");
            assert!(copied.layout().is_contiguous(), "the copy is contiguous");
            let actual: Vec<f32> = copied
                .to_dtype(&mut device, DType::F32)
                .expect("to_dtype succeeds")
                .to_bytes(&mut device)
                .expect("to_bytes succeeds")
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&chunk| f32::from_le_bytes(chunk))
                .collect();
            assert_eq!(actual, expected, "permuted values for {dtype:?}");
        });
    }

    #[test]
    fn copy_of_narrowed_tensor_keeps_the_sub_range() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let copied = Tensor::from_bytes(
            &mut device,
            &(0_u16..24)
                .flat_map(|value| f32::from(value).to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3, 4]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds")
        .narrow(1, 1, 2)
        .expect("the narrow is valid")
        .contiguous(&mut device)
        .expect("the copy succeeds");
        let expected: Vec<f32> = (4_u16..12).chain(16..24).map(f32::from).collect();
        assert_eq!(copied.shape().dims(), &[2, 2, 4], "narrowed dims");
        let actual: Vec<f32> = copied
            .to_bytes(&mut device)
            .expect("to_bytes succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(actual, expected, "narrowed values");
    }

    #[test]
    fn copy_of_broadcast_tensor_repeats_values() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let copied = Tensor::from_bytes(
            &mut device,
            &[1.0_f32, 2.0, 3.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([3, 1]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds")
        .expand(Shape::try_from([2, 3, 4]).expect("the shape is valid"))
        .expect("the broadcast is valid")
        .contiguous(&mut device)
        .expect("the copy succeeds");
        let block: [f32; 12] = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0];
        let expected: Vec<f32> = block.iter().chain(block.iter()).copied().collect();
        let actual: Vec<f32> = copied
            .to_bytes(&mut device)
            .expect("to_bytes succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(actual, expected, "broadcast values");
    }

    #[test]
    fn copy_of_permuted_u32_tensor_is_contiguous() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let bytes: Vec<u8> = (0_u32..6).flat_map(u32::to_le_bytes).collect();
        let copied = Tensor::from_bytes(
            &mut device,
            &bytes,
            DType::U32,
            Shape::try_from([3, 2]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid")
        .contiguous(&mut device)
        .expect("the copy succeeds");
        let expected: Vec<u8> = [0_u32, 2, 4, 1, 3, 5]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        assert_eq!(
            copied.to_bytes(&mut device).expect("to_bytes succeeds"),
            expected,
            "transposed u32 values"
        );
    }

    #[test]
    fn copy_and_add_of_empty_tensors_read_no_bytes() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let empty = Tensor::from_bytes(
            &mut device,
            &[],
            DType::F32,
            Shape::try_from([0, 3]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds");
        let copied = empty
            .permute(&[1, 0])
            .expect("the permutation is valid")
            .contiguous(&mut device)
            .expect("the copy succeeds");
        let sum = empty.add(&mut device, &empty).expect("the add succeeds");
        assert_eq!(
            [
                copied.to_bytes(&mut device).expect("to_bytes succeeds"),
                sum.to_bytes(&mut device).expect("to_bytes succeeds"),
            ],
            [Vec::<u8>::new(), Vec::new()],
            "an empty copy and an empty add read back no bytes"
        );
    }

    #[test]
    fn to_dtype_f32_to_bf16_and_back_rounds_known_values() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let actual: Vec<f32> = Tensor::from_bytes(
            &mut device,
            &[1.0_f32, -2.5, PI]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([3]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds")
        .to_dtype(&mut device, DType::BF16)
        .expect("to_dtype succeeds")
        .to_dtype(&mut device, DType::F32)
        .expect("to_dtype succeeds")
        .to_bytes(&mut device)
        .expect("to_bytes succeeds")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&chunk| f32::from_le_bytes(chunk))
        .collect();
        assert_eq!(actual, vec![1.0, -2.5, 3.140625], "bf16 round trip");
    }

    #[test]
    fn to_dtype_f32_to_bf16_rounds_to_nearest_even() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let actual: Vec<u16> = Tensor::from_bytes(
            &mut device,
            &[0x3F80_8000_u32, 0x3F81_8000, 0x3F80_8001, 0xBF80_8000]
                .into_iter()
                .flat_map(|bits| f32::from_bits(bits).to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([4]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds")
        .to_dtype(&mut device, DType::BF16)
        .expect("to_dtype succeeds")
        .to_bytes(&mut device)
        .expect("to_bytes succeeds")
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&chunk| u16::from_le_bytes(chunk))
        .collect();
        assert_eq!(
            actual,
            vec![0x3F80, 0x3F82, 0x3F81, 0xBF80],
            "ties round to even, others to nearest"
        );
    }

    #[test]
    fn to_dtype_of_permuted_tensor_reads_strided_input() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let converted = Tensor::from_bytes(
            &mut device,
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid")
        .to_dtype(&mut device, DType::BF16)
        .expect("to_dtype succeeds");
        assert_eq!(converted.shape().dims(), &[3, 2], "transposed dims");
        let bits: Vec<u16> = converted
            .to_bytes(&mut device)
            .expect("to_bytes succeeds")
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&chunk| u16::from_le_bytes(chunk))
            .collect();
        assert_eq!(
            bits,
            vec![0x3F80, 0x4080, 0x4000, 0x40A0, 0x4040, 0x40C0],
            "transposed bf16 bits of 1, 4, 2, 5, 3, 6"
        );
    }

    #[test]
    fn unary_ops_match_hand_computed_values_for_each_float_dtype() {
        struct NameOutputExpectedCase {
            name: &'static str,
            output: Result<Tensor<MetalDevice>, TensorError>,
            expected: [f32; 3],
        }
        struct DtypeToleranceCase {
            dtype: DType,
            tolerance: f32,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5,
            },
            DtypeToleranceCase {
                dtype: DType::BF16,
                tolerance: 1e-2,
            },
        ]
        .into_iter()
        .for_each(
            |DtypeToleranceCase {
                 dtype,
                 tolerance,
             }| {
                let [neg_input, exp_input, sqrt_input, reciprocal_input] = [
                    [1.0_f32, -2.5, 0.0],
                    [0.0, 1.0, -1.0],
                    [4.0, 2.25, 0.0],
                    [2.0, -4.0, 0.5],
                ]
                .map(|values| {
                    Tensor::from_bytes(
                        &mut device,
                        &values
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<u8>>(),
                        DType::F32,
                        Shape::try_from([3]).expect("the shape is valid"),
                    )
                    .expect("from_bytes succeeds")
                    .to_dtype(&mut device, dtype)
                    .expect("to_dtype succeeds")
                });
                [
                    NameOutputExpectedCase {
                        name: "neg",
                        output: neg_input.neg(&mut device),
                        expected: [-1.0, 2.5, 0.0],
                    },
                    NameOutputExpectedCase {
                        name: "exp",
                        output: exp_input.exp(&mut device),
                        expected: [1.0, E, 1.0 / E],
                    },
                    NameOutputExpectedCase {
                        name: "sqrt",
                        output: sqrt_input.sqrt(&mut device),
                        expected: [2.0, 1.5, 0.0],
                    },
                    NameOutputExpectedCase {
                        name: "reciprocal",
                        output: reciprocal_input.reciprocal(&mut device),
                        expected: [0.5, -0.25, 2.0],
                    },
                ]
                .into_iter()
                .for_each(
                    |NameOutputExpectedCase {
                         name,
                         output,
                         expected,
                     }| {
                        let actual: Vec<f32> = output
                            .expect("the op succeeds")
                            .to_dtype(&mut device, DType::F32)
                            .expect("to_dtype succeeds")
                            .to_bytes(&mut device)
                            .expect("to_bytes succeeds")
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|&chunk| f32::from_le_bytes(chunk))
                            .collect();
                        assert_eq!(actual.len(), expected.len(), "{name}: element count");
                        actual.iter().zip(expected).for_each(|(&actual, expected)| {
                            assert!(
                                (actual - expected).abs()
                                    <= tolerance * expected.abs().max(f32::MIN_POSITIVE),
                                "{name} for {dtype:?}: {actual} is not within {tolerance} of \
                                 {expected}"
                            );
                        });
                    },
                );
            },
        );
    }

    #[test]
    fn binary_ops_match_hand_computed_values_for_each_float_dtype() {
        struct NameOutputExpectedCase {
            name: &'static str,
            output: Result<Tensor<MetalDevice>, TensorError>,
            expected: [f32; 4],
        }
        struct DtypeToleranceCase {
            dtype: DType,
            tolerance: f32,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5,
            },
            DtypeToleranceCase {
                dtype: DType::BF16,
                tolerance: 1e-2,
            },
        ]
        .into_iter()
        .for_each(
            |DtypeToleranceCase {
                 dtype,
                 tolerance,
             }| {
                let [lhs, rhs] = [[1.0_f32, 2.0, 3.0, 4.0], [2.0, 4.0, 0.5, -1.0]].map(|values| {
                    Tensor::from_bytes(
                        &mut device,
                        &values
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<u8>>(),
                        DType::F32,
                        Shape::try_from([4]).expect("the shape is valid"),
                    )
                    .expect("from_bytes succeeds")
                    .to_dtype(&mut device, dtype)
                    .expect("to_dtype succeeds")
                });
                [
                    NameOutputExpectedCase {
                        name: "add",
                        output: lhs.add(&mut device, &rhs),
                        expected: [3.0, 6.0, 3.5, 3.0],
                    },
                    NameOutputExpectedCase {
                        name: "sub",
                        output: lhs.sub(&mut device, &rhs),
                        expected: [-1.0, -2.0, 2.5, 5.0],
                    },
                    NameOutputExpectedCase {
                        name: "mul",
                        output: lhs.mul(&mut device, &rhs),
                        expected: [2.0, 8.0, 1.5, -4.0],
                    },
                    NameOutputExpectedCase {
                        name: "div",
                        output: lhs.div(&mut device, &rhs),
                        expected: [0.5, 0.5, 6.0, -4.0],
                    },
                ]
                .into_iter()
                .for_each(
                    |NameOutputExpectedCase {
                         name,
                         output,
                         expected,
                     }| {
                        let actual: Vec<f32> = output
                            .expect("the op succeeds")
                            .to_dtype(&mut device, DType::F32)
                            .expect("to_dtype succeeds")
                            .to_bytes(&mut device)
                            .expect("to_bytes succeeds")
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|&chunk| f32::from_le_bytes(chunk))
                            .collect();
                        assert_eq!(actual.len(), expected.len(), "{name}: element count");
                        actual.iter().zip(expected).for_each(|(&actual, expected)| {
                            assert!(
                                (actual - expected).abs()
                                    <= tolerance * expected.abs().max(f32::MIN_POSITIVE),
                                "{name} for {dtype:?}: {actual} is not within {tolerance} of \
                                 {expected}"
                            );
                        });
                    },
                );
            },
        );
    }

    #[test]
    fn add_broadcasts_a_row_over_a_matrix() {
        struct ValuesDimsCase {
            values: Vec<f32>,
            dims: Vec<usize>,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let [matrix, row] = [
            ValuesDimsCase {
                values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                dims: vec![2, 3],
            },
            ValuesDimsCase {
                values: vec![10.0, 20.0, 30.0],
                dims: vec![3],
            },
        ]
        .map(
            |ValuesDimsCase {
                 values,
                 dims,
             }| {
                Tensor::from_bytes(
                    &mut device,
                    &values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                )
                .expect("from_bytes succeeds")
            },
        );
        let sum = matrix.add(&mut device, &row).expect("the add succeeds");
        assert_eq!(sum.shape().dims(), &[2, 3], "broadcast dims");
        let actual: Vec<f32> = sum
            .to_bytes(&mut device)
            .expect("to_bytes succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(
            actual,
            vec![10.0, 21.0, 32.0, 13.0, 24.0, 35.0],
            "row added to each matrix row"
        );
    }

    #[test]
    fn neg_of_permuted_tensor_reads_strided_input() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let negated = Tensor::from_bytes(
            &mut device,
            &[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid")
        .neg(&mut device)
        .expect("the neg succeeds");
        assert_eq!(negated.shape().dims(), &[3, 2], "transposed dims");
        let actual: Vec<f32> = negated
            .to_bytes(&mut device)
            .expect("to_bytes succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(
            actual,
            vec![-0.0, -3.0, -1.0, -4.0, -2.0, -5.0],
            "negated transposed values"
        );
    }

    #[test]
    fn mul_of_permuted_and_contiguous_operands_reads_both_layouts() {
        struct ValuesDimsCase {
            values: Vec<f32>,
            dims: Vec<usize>,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let [transposed, contiguous] = [
            ValuesDimsCase {
                values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                dims: vec![3, 2],
            },
            ValuesDimsCase {
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                dims: vec![2, 3],
            },
        ]
        .map(
            |ValuesDimsCase {
                 values,
                 dims,
             }| {
                Tensor::from_bytes(
                    &mut device,
                    &values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                )
                .expect("from_bytes succeeds")
            },
        );
        let actual: Vec<f32> = transposed
            .permute(&[1, 0])
            .expect("the permutation is valid")
            .mul(&mut device, &contiguous)
            .expect("the mul succeeds")
            .to_bytes(&mut device)
            .expect("to_bytes succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(
            actual,
            vec![0.0, 4.0, 12.0, 4.0, 15.0, 30.0],
            "products of [[0, 2, 4], [1, 3, 5]] and [[1, 2, 3], [4, 5, 6]]"
        );
    }

    #[test]
    fn long_chain_of_adds_spans_several_command_buffers() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let ones = Tensor::from_bytes(
            &mut device,
            &[1.0_f32; 4]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([4]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds");
        let start = Tensor::zeros(
            &mut device,
            DType::F32,
            Shape::try_from([4]).expect("the shape is valid"),
        )
        .expect("zeros succeeds");
        let actual: Vec<f32> = (0..300)
            .try_fold(start, |sum, _| sum.add(&mut device, &ones))
            .expect("every add succeeds")
            .to_bytes(&mut device)
            .expect("to_bytes succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(actual, vec![300.0; 4], "three hundred additions of one");
    }

    #[test]
    fn zeros_interleaved_with_adds_keep_submission_order() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let [ones, start] = [0_u8; 2].map(|_| {
            Tensor::from_bytes(
                &mut device,
                &[1.0_f32; 3]
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<u8>>(),
                DType::F32,
                Shape::try_from([3]).expect("the shape is valid"),
            )
            .expect("from_bytes succeeds")
        });
        let actual: Vec<f32> = (0..100)
            .try_fold(start, |sum, _| {
                let zeros = Tensor::zeros(
                    &mut device,
                    DType::F32,
                    Shape::try_from([3]).expect("the shape is valid"),
                )?;
                sum.add(&mut device, &zeros)?.add(&mut device, &ones)
            })
            .expect("every zeros and add succeeds")
            .to_bytes(&mut device)
            .expect("to_bytes succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(
            actual,
            vec![101.0; 3],
            "one plus one hundred additions of zero then one"
        );
    }

    #[test]
    fn sum_and_amax_over_each_dim_match_hand_computed_values_for_each_float_dtype() {
        struct NameReducedDimsExpectedCase {
            name: &'static str,
            reduced: Result<Tensor<MetalDevice>, TensorError>,
            dims: &'static [usize],
            expected: &'static [f32],
        }
        struct DtypeToleranceCase {
            dtype: DType,
            tolerance: f32,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5,
            },
            DtypeToleranceCase {
                dtype: DType::BF16,
                tolerance: 1e-2,
            },
        ]
        .into_iter()
        .for_each(
            |DtypeToleranceCase {
                 dtype,
                 tolerance,
             }| {
                let input = Tensor::from_bytes(
                    &mut device,
                    &[1.0_f32, 5.0, 3.0, 4.0, 2.0, 6.0]
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from([2, 3]).expect("the shape is valid"),
                )
                .expect("from_bytes succeeds")
                .to_dtype(&mut device, dtype)
                .expect("to_dtype succeeds");
                [
                    NameReducedDimsExpectedCase {
                        name: "sum dim 0",
                        reduced: input.sum_keepdim(&mut device, 0),
                        dims: &[1, 3],
                        expected: &[5.0, 7.0, 9.0],
                    },
                    NameReducedDimsExpectedCase {
                        name: "sum dim 1",
                        reduced: input.sum_keepdim(&mut device, 1),
                        dims: &[2, 1],
                        expected: &[9.0, 12.0],
                    },
                    NameReducedDimsExpectedCase {
                        name: "max dim 0",
                        reduced: input.amax_keepdim(&mut device, 0),
                        dims: &[1, 3],
                        expected: &[4.0, 5.0, 6.0],
                    },
                    NameReducedDimsExpectedCase {
                        name: "max dim 1",
                        reduced: input.amax_keepdim(&mut device, 1),
                        dims: &[2, 1],
                        expected: &[5.0, 6.0],
                    },
                ]
                .into_iter()
                .for_each(
                    |NameReducedDimsExpectedCase {
                         name,
                         reduced,
                         dims,
                         expected,
                     }| {
                        let reduced = reduced.expect("the reduction succeeds");
                        assert_eq!(reduced.dtype(), dtype, "{name} keeps the dtype");
                        assert_eq!(reduced.shape().dims(), dims, "{name} keeps the dim");
                        let actual: Vec<f32> = reduced
                            .to_dtype(&mut device, DType::F32)
                            .expect("to_dtype succeeds")
                            .to_bytes(&mut device)
                            .expect("to_bytes succeeds")
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|&chunk| f32::from_le_bytes(chunk))
                            .collect();
                        assert_eq!(actual.len(), expected.len(), "{name}: element count");
                        actual
                            .iter()
                            .zip(expected)
                            .for_each(|(&actual, &expected)| {
                                assert!(
                                    (actual - expected).abs()
                                        <= tolerance * expected.abs().max(f32::MIN_POSITIVE),
                                    "{name} for {dtype:?}: {actual} is not within {tolerance} of \
                                     {expected}"
                                );
                            });
                    },
                );
            },
        );
    }

    #[test]
    fn argmax_over_each_dim_matches_hand_computed_indices_for_each_float_dtype() {
        struct DimDimsExpectedCase {
            dim: usize,
            dims: &'static [usize],
            expected: &'static [u32],
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        [DType::F32, DType::BF16].into_iter().for_each(|dtype| {
            [
                DimDimsExpectedCase {
                    dim: 0,
                    dims: &[1, 3],
                    expected: &[1, 0, 1],
                },
                DimDimsExpectedCase {
                    dim: 1,
                    dims: &[2, 1],
                    expected: &[1, 2],
                },
            ]
            .into_iter()
            .for_each(
                |DimDimsExpectedCase {
                     dim,
                     dims,
                     expected,
                 }| {
                    let indices = Tensor::from_bytes(
                        &mut device,
                        &[1.0_f32, 5.0, 3.0, 4.0, 2.0, 6.0]
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<u8>>(),
                        DType::F32,
                        Shape::try_from([2, 3]).expect("the shape is valid"),
                    )
                    .expect("from_bytes succeeds")
                    .to_dtype(&mut device, dtype)
                    .expect("to_dtype succeeds")
                    .argmax_keepdim(&mut device, dim)
                    .expect("the argmax succeeds");
                    assert_eq!(indices.dtype(), DType::U32, "argmax outputs u32");
                    assert_eq!(indices.shape().dims(), dims, "argmax keeps the dim");
                    let actual: Vec<u32> = indices
                        .to_bytes(&mut device)
                        .expect("to_bytes succeeds")
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|&chunk| u32::from_le_bytes(chunk))
                        .collect();
                    assert_eq!(actual, expected, "argmax over dim {dim} for {dtype:?}");
                },
            );
        });
    }

    #[test]
    fn sum_of_4096_bf16_ones_equals_4096() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let actual: Vec<f32> = Tensor::from_bytes(
            &mut device,
            &[1.0_f32; 4096]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([4096]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds")
        .to_dtype(&mut device, DType::BF16)
        .expect("to_dtype succeeds")
        .sum_keepdim(&mut device, 0)
        .expect("the sum succeeds")
        .to_dtype(&mut device, DType::F32)
        .expect("to_dtype succeeds")
        .to_bytes(&mut device)
        .expect("to_bytes succeeds")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&chunk| f32::from_le_bytes(chunk))
        .collect();
        assert_eq!(actual, vec![4096.0], "bf16 sum of ones");
    }

    #[test]
    fn argmax_returns_the_first_index_on_ties() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let actual: Vec<u32> = Tensor::from_bytes(
            &mut device,
            &[3.0_f32, 7.0, 7.0, 1.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([4]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds")
        .argmax_keepdim(&mut device, 0)
        .expect("the argmax succeeds")
        .to_bytes(&mut device)
        .expect("to_bytes succeeds")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&chunk| u32::from_le_bytes(chunk))
        .collect();
        assert_eq!(actual, vec![1], "first of two maxima");
    }

    #[test]
    fn argmax_returns_the_first_index_on_ties_across_threads() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let actual: Vec<u32> = Tensor::from_bytes(
            &mut device,
            &(0_u16..600)
                .flat_map(|index| {
                    if index == 10 || index == 300 {
                        9.0_f32.to_le_bytes()
                    } else {
                        f32::from(index % 7).to_le_bytes()
                    }
                })
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([600]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds")
        .argmax_keepdim(&mut device, 0)
        .expect("the argmax succeeds")
        .to_bytes(&mut device)
        .expect("to_bytes succeeds")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&chunk| u32::from_le_bytes(chunk))
        .collect();
        assert_eq!(
            actual,
            vec![10],
            "maxima at 10 and 300 read by different threads"
        );
    }

    #[test]
    fn sum_of_permuted_tensor_reads_strided_input() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let sum = Tensor::from_bytes(
            &mut device,
            &[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid")
        .sum_keepdim(&mut device, 1)
        .expect("the sum succeeds");
        assert_eq!(sum.shape().dims(), &[3, 1], "transposed rows");
        let actual: Vec<f32> = sum
            .to_bytes(&mut device)
            .expect("to_bytes succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(
            actual,
            vec![3.0, 5.0, 7.0],
            "row sums of [[0, 3], [1, 4], [2, 5]]"
        );
    }

    #[test]
    fn sum_over_empty_dim_reads_zeros() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let sum = Tensor::from_bytes(
            &mut device,
            &[],
            DType::F32,
            Shape::try_from([2, 0]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds")
        .sum_keepdim(&mut device, 1)
        .expect("the sum succeeds");
        assert_eq!(sum.shape().dims(), &[2, 1], "the dim is kept");
        let actual: Vec<f32> = sum
            .to_bytes(&mut device)
            .expect("to_bytes succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(actual, vec![0.0, 0.0], "an empty sum is zero");
    }

    #[test]
    fn matmul_matches_hand_computed_values_for_each_float_dtype() {
        struct ValuesDimsCase {
            values: Vec<f32>,
            dims: Vec<usize>,
        }
        struct DtypeToleranceCase {
            dtype: DType,
            tolerance: f32,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5,
            },
            DtypeToleranceCase {
                dtype: DType::BF16,
                tolerance: 1e-2,
            },
        ]
        .into_iter()
        .for_each(
            |DtypeToleranceCase {
                 dtype,
                 tolerance,
             }| {
                let [lhs, rhs] = [
                    ValuesDimsCase {
                        values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        dims: vec![2, 3],
                    },
                    ValuesDimsCase {
                        values: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                        dims: vec![3, 2],
                    },
                ]
                .map(
                    |ValuesDimsCase {
                         values,
                         dims,
                     }| {
                        Tensor::from_bytes(
                            &mut device,
                            &values
                                .iter()
                                .flat_map(|value| value.to_le_bytes())
                                .collect::<Vec<u8>>(),
                            DType::F32,
                            Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                        )
                        .expect("from_bytes succeeds")
                        .to_dtype(&mut device, dtype)
                        .expect("to_dtype succeeds")
                    },
                );
                let actual: Vec<f32> = lhs
                    .matmul(&mut device, &rhs)
                    .expect("the matmul succeeds")
                    .to_dtype(&mut device, DType::F32)
                    .expect("to_dtype succeeds")
                    .to_bytes(&mut device)
                    .expect("to_bytes succeeds")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&chunk| f32::from_le_bytes(chunk))
                    .collect();
                let expected = [58.0_f32, 64.0, 139.0, 154.0];
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "[2, 3] x [3, 2]: element count"
                );
                actual.iter().zip(expected).for_each(|(&actual, expected)| {
                    assert!(
                        (actual - expected).abs() <= tolerance * expected.abs(),
                        "[2, 3] x [3, 2] for {dtype:?}: {actual} is not within {tolerance} of \
                         {expected}"
                    );
                });
            },
        );
    }

    #[test]
    fn matmul_reads_a_transposed_left_or_right_operand_without_copy() {
        struct ValuesDimsCase {
            values: [f32; 6],
            dims: [usize; 2],
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let [x, weight, lhs, rhs] = [
            ValuesDimsCase {
                values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                dims: [2, 3],
            },
            ValuesDimsCase {
                values: [1.0, 0.0, -1.0, 2.0, 1.0, 0.0],
                dims: [2, 3],
            },
            ValuesDimsCase {
                values: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
                dims: [3, 2],
            },
            ValuesDimsCase {
                values: [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                dims: [3, 2],
            },
        ]
        .map(
            |ValuesDimsCase {
                 values,
                 dims,
             }| {
                Tensor::from_bytes(
                    &mut device,
                    &values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                )
                .expect("from_bytes succeeds")
            },
        );
        let [weight_transposed, lhs_transposed] = [weight, lhs]
            .map(|operand| operand.permute(&[1, 0]).expect("the permutation is valid"));
        assert_eq!(
            [
                MatmulGeometry::of(x.layout(), weight_transposed.layout())
                    .map(|geometry| is_matmul_ready(&geometry.rhs, geometry.k, geometry.n)),
                MatmulGeometry::of(lhs_transposed.layout(), rhs.layout())
                    .map(|geometry| is_matmul_ready(&geometry.lhs, geometry.m, geometry.k)),
            ],
            [Some(true); 2],
            "a transposed rhs or lhs is read in place"
        );
        let actual = [
            x.matmul(&mut device, &weight_transposed),
            lhs_transposed.matmul(&mut device, &rhs),
        ]
        .map(|product| {
            product
                .expect("the matmul succeeds")
                .to_bytes(&mut device)
                .expect("to_bytes succeeds")
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&chunk| f32::from_le_bytes(chunk))
                .collect::<Vec<f32>>()
        });
        assert_eq!(
            actual,
            [vec![-2.0, 4.0, -2.0, 13.0], vec![58.0, 64.0, 139.0, 154.0]],
            "x . W^T and L^T x R"
        );
    }

    #[test]
    fn batched_matmul_matches_sum_of_products() {
        struct ValuesDimsCase<'values> {
            values: &'values [f32],
            dims: [usize; 3],
        }
        struct DtypeToleranceCase {
            dtype: DType,
            tolerance: f32,
        }
        struct CountSeedCase {
            count: u16,
            seed: u16,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let [lhs_values, rhs_values] = [
            CountSeedCase {
                count: 2 * 3 * 40,
                seed: 7,
            },
            CountSeedCase {
                count: 2 * 40 * 5,
                seed: 5,
            },
        ]
        .map(
            |CountSeedCase {
                 count,
                 seed,
             }| {
                (0..count)
                    .map(|index: u16| {
                        let step = index.wrapping_mul(seed).wrapping_add(3) % 17;
                        (f32::from(step) - 8.0) / 8.0
                    })
                    .collect::<Vec<f32>>()
            },
        );
        let expected: Vec<f32> = lhs_values
            .chunks(3 * 40)
            .zip(rhs_values.chunks(40 * 5))
            .flat_map(|(lhs_matrix, rhs_matrix)| {
                lhs_matrix.chunks(40).flat_map(move |row| {
                    (0..5).map(move |col| {
                        row.iter()
                            .zip(rhs_matrix.iter().skip(col).step_by(5))
                            .map(|(lhs_value, rhs_value)| lhs_value * rhs_value)
                            .sum::<f32>()
                    })
                })
            })
            .collect();
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5,
            },
            DtypeToleranceCase {
                dtype: DType::BF16,
                tolerance: 1e-2,
            },
        ]
        .into_iter()
        .for_each(
            |DtypeToleranceCase {
                 dtype,
                 tolerance,
             }| {
                let [lhs, rhs] = [
                    ValuesDimsCase {
                        values: &lhs_values,
                        dims: [2, 3, 40],
                    },
                    ValuesDimsCase {
                        values: &rhs_values,
                        dims: [2, 40, 5],
                    },
                ]
                .map(
                    |ValuesDimsCase {
                         values,
                         dims,
                     }| {
                        Tensor::from_bytes(
                            &mut device,
                            &values
                                .iter()
                                .flat_map(|value| value.to_le_bytes())
                                .collect::<Vec<u8>>(),
                            DType::F32,
                            Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                        )
                        .expect("from_bytes succeeds")
                        .to_dtype(&mut device, dtype)
                        .expect("to_dtype succeeds")
                    },
                );
                let actual: Vec<f32> = lhs
                    .matmul(&mut device, &rhs)
                    .expect("the matmul succeeds")
                    .to_dtype(&mut device, DType::F32)
                    .expect("to_dtype succeeds")
                    .to_bytes(&mut device)
                    .expect("to_bytes succeeds")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&chunk| f32::from_le_bytes(chunk))
                    .collect();
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "[2, 3, 40] x [2, 40, 5]: element count"
                );
                actual
                    .iter()
                    .zip(&expected)
                    .for_each(|(&actual, &expected)| {
                        assert!(
                            (actual - expected).abs()
                                <= tolerance * expected.abs().max(f32::MIN_POSITIVE),
                            "[2, 3, 40] x [2, 40, 5] for {dtype:?}: {actual} is not within \
                             {tolerance} of {expected}"
                        );
                    });
            },
        );
    }

    #[test]
    fn matmul_spans_several_tiles() {
        struct ValuesDimsCase<'values> {
            values: &'values [f32],
            dims: [usize; 2],
        }
        struct CountSeedCase {
            count: u16,
            seed: u16,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let [lhs_values, rhs_values] = [
            CountSeedCase {
                count: 37 * 20,
                seed: 3,
            },
            CountSeedCase {
                count: 20 * 33,
                seed: 11,
            },
        ]
        .map(
            |CountSeedCase {
                 count,
                 seed,
             }| {
                (0..count)
                    .map(|index: u16| {
                        let step = index.wrapping_mul(seed).wrapping_add(3) % 17;
                        (f32::from(step) - 8.0) / 8.0
                    })
                    .collect::<Vec<f32>>()
            },
        );
        let expected: Vec<f32> = lhs_values
            .chunks(20)
            .flat_map(|row| {
                let rhs_values = &rhs_values;
                (0..33).map(move |col| {
                    row.iter()
                        .zip(rhs_values.iter().skip(col).step_by(33))
                        .map(|(lhs_value, rhs_value)| lhs_value * rhs_value)
                        .sum::<f32>()
                })
            })
            .collect();
        let [lhs, rhs] = [
            ValuesDimsCase {
                values: &lhs_values,
                dims: [37, 20],
            },
            ValuesDimsCase {
                values: &rhs_values,
                dims: [20, 33],
            },
        ]
        .map(
            |ValuesDimsCase {
                 values,
                 dims,
             }| {
                Tensor::from_bytes(
                    &mut device,
                    &values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                )
                .expect("from_bytes succeeds")
            },
        );
        let actual: Vec<f32> = lhs
            .matmul(&mut device, &rhs)
            .expect("the matmul succeeds")
            .to_bytes(&mut device)
            .expect("to_bytes succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(
            actual.len(),
            expected.len(),
            "[37, 20] x [20, 33]: element count"
        );
        actual
            .iter()
            .zip(&expected)
            .for_each(|(&actual, &expected)| {
                assert!(
                    (actual - expected).abs() <= 1e-5 * expected.abs().max(f32::MIN_POSITIVE),
                    "[37, 20] x [20, 33]: {actual} is not within 1e-5 of {expected}"
                );
            });
    }

    #[test]
    fn matmul_copies_operand_without_unit_stride() {
        struct DtypeToleranceCase {
            dtype: DType,
            tolerance: f32,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let rhs_contiguous: Vec<f32> = [
            0_u16, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19,
            23,
        ]
        .into_iter()
        .map(f32::from)
        .collect();
        let lhs_values: Vec<f32> = (0_u16..16).map(f32::from).collect();
        let expected: Vec<f32> = lhs_values
            .chunks(2 * 2)
            .zip(rhs_contiguous.chunks(2 * 3))
            .flat_map(|(lhs_matrix, rhs_matrix)| {
                lhs_matrix.chunks(2).flat_map(move |row| {
                    (0..3).map(move |col| {
                        row.iter()
                            .zip(rhs_matrix.iter().skip(col).step_by(3))
                            .map(|(lhs_value, rhs_value)| lhs_value * rhs_value)
                            .sum::<f32>()
                    })
                })
            })
            .collect();
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5,
            },
            DtypeToleranceCase {
                dtype: DType::BF16,
                tolerance: 1e-2,
            },
        ]
        .into_iter()
        .for_each(
            |DtypeToleranceCase {
                 dtype,
                 tolerance,
             }| {
                let rhs = Tensor::from_bytes(
                    &mut device,
                    &(0_u16..24)
                        .flat_map(|value| f32::from(value).to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from([2, 3, 4]).expect("the shape is valid"),
                )
                .expect("from_bytes succeeds")
                .to_dtype(&mut device, dtype)
                .expect("to_dtype succeeds")
                .permute(&[2, 0, 1])
                .expect("the permutation is valid");
                let lhs = Tensor::from_bytes(
                    &mut device,
                    &lhs_values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from([4, 2, 2]).expect("the shape is valid"),
                )
                .expect("from_bytes succeeds")
                .to_dtype(&mut device, dtype)
                .expect("to_dtype succeeds");
                assert!(
                    MatmulGeometry::of(lhs.layout(), rhs.layout()).is_some_and(|geometry| {
                        !is_matmul_ready(&geometry.rhs, geometry.k, geometry.n)
                    }),
                    "strides (12, 4) on the last two dims need a copy"
                );
                let actual: Vec<f32> = lhs
                    .matmul(&mut device, &rhs)
                    .expect("the matmul succeeds")
                    .to_dtype(&mut device, DType::F32)
                    .expect("to_dtype succeeds")
                    .to_bytes(&mut device)
                    .expect("to_bytes succeeds")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&chunk| f32::from_le_bytes(chunk))
                    .collect();
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "[4, 2, 2] x permuted [4, 2, 3]: element count"
                );
                actual
                    .iter()
                    .zip(&expected)
                    .for_each(|(&actual, &expected)| {
                        assert!(
                            (actual - expected).abs()
                                <= tolerance * expected.abs().max(f32::MIN_POSITIVE),
                            "[4, 2, 2] x permuted [4, 2, 3] for {dtype:?}: {actual} is not within \
                             {tolerance} of {expected}"
                        );
                    });
            },
        );
    }

    #[test]
    fn matmul_reads_permuted_batch_dims_in_place() {
        struct CountDimsCase {
            count: u16,
            dims: [usize; 4],
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let [lhs, rhs] = [
            CountDimsCase {
                count: 48,
                dims: [2, 3, 2, 4],
            },
            CountDimsCase {
                count: 24,
                dims: [3, 2, 4, 1],
            },
        ]
        .map(
            |CountDimsCase {
                 count,
                 dims,
             }| {
                Tensor::from_bytes(
                    &mut device,
                    &(0..count)
                        .flat_map(|value| f32::from(value).to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                )
                .expect("from_bytes succeeds")
            },
        );
        let lhs = lhs
            .permute(&[1, 0, 2, 3])
            .expect("the permutation is valid");
        let [expected, actual] = [lhs.contiguous(&mut device).expect("the copy succeeds"), lhs]
            .map(|lhs| {
                lhs.matmul(&mut device, &rhs)
                    .expect("the matmul succeeds")
                    .to_bytes(&mut device)
                    .expect("to_bytes succeeds")
            });
        assert_eq!(
            actual, expected,
            "permuted batch strides match the contiguous copy"
        );
    }

    #[test]
    fn matmul_with_empty_output_reads_nothing() {
        struct ValuesDimsCase {
            values: Vec<f32>,
            dims: Vec<usize>,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let [lhs, rhs] = [
            ValuesDimsCase {
                values: vec![],
                dims: vec![0, 3],
            },
            ValuesDimsCase {
                values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                dims: vec![3, 2],
            },
        ]
        .map(
            |ValuesDimsCase {
                 values,
                 dims,
             }| {
                Tensor::from_bytes(
                    &mut device,
                    &values
                        .iter()
                        .flat_map(|value: &f32| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                )
                .expect("from_bytes succeeds")
            },
        );
        let output = lhs.matmul(&mut device, &rhs).expect("the matmul succeeds");
        assert_eq!(output.shape().dims(), &[0, 2], "output shape is [m, n]");
        assert!(
            output
                .to_bytes(&mut device)
                .expect("to_bytes succeeds")
                .is_empty(),
            "an empty matmul reads back no values"
        );
    }

    #[test]
    fn index_select_selects_rows_for_each_float_dtype() {
        struct DtypeToleranceCase {
            dtype: DType,
            tolerance: f32,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let indices = Tensor::from_bytes(
            &mut device,
            &[2_u32, 0]
                .iter()
                .flat_map(|index| index.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::U32,
            Shape::try_from([2]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds");
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5,
            },
            DtypeToleranceCase {
                dtype: DType::BF16,
                tolerance: 1e-2,
            },
        ]
        .into_iter()
        .for_each(
            |DtypeToleranceCase {
                 dtype,
                 tolerance,
             }| {
                let rows = Tensor::from_bytes(
                    &mut device,
                    &(0_u16..12)
                        .flat_map(|value| f32::from(value).to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from([3, 4]).expect("the shape is valid"),
                )
                .expect("from_bytes succeeds")
                .to_dtype(&mut device, dtype)
                .expect("to_dtype succeeds")
                .index_select(&mut device, 0, &indices)
                .expect("index_select succeeds");
                assert_eq!(
                    rows.shape().dims(),
                    &[2, 4],
                    "output shape is [indices, cols]"
                );
                let actual: Vec<f32> = rows
                    .to_dtype(&mut device, DType::F32)
                    .expect("to_dtype succeeds")
                    .to_bytes(&mut device)
                    .expect("to_bytes succeeds")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&chunk| f32::from_le_bytes(chunk))
                    .collect();
                let expected = [8.0_f32, 9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 3.0];
                assert_eq!(actual.len(), expected.len(), "index_select: element count");
                actual.iter().zip(expected).for_each(|(&actual, expected)| {
                    assert!(
                        (actual - expected).abs()
                            <= tolerance * expected.abs().max(f32::MIN_POSITIVE),
                        "rows [2, 0] of a [3, 4] table for {dtype:?}: {actual} is not within \
                         {tolerance} of {expected}"
                    );
                });
            },
        );
    }

    #[test]
    fn index_select_with_out_of_range_index_writes_a_zero_row() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let table = Tensor::from_bytes(
            &mut device,
            &[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([3, 2]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds");
        let indices = Tensor::from_bytes(
            &mut device,
            &[1_u32, 3]
                .iter()
                .flat_map(|index| index.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::U32,
            Shape::try_from([2]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds");
        let actual: Vec<f32> = table
            .index_select(&mut device, 0, &indices)
            .expect("index_select succeeds")
            .to_bytes(&mut device)
            .expect("to_bytes succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(
            actual,
            [2.0, 3.0, 0.0, 0.0],
            "an index past the table gives zeros"
        );
    }

    #[test]
    fn index_select_of_permuted_table_reads_strided_rows() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let table = Tensor::from_bytes(
            &mut device,
            &[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid");
        let indices = Tensor::from_bytes(
            &mut device,
            &[2_u32, 0]
                .iter()
                .flat_map(|index| index.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::U32,
            Shape::try_from([2]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds");
        let actual: Vec<f32> = table
            .index_select(&mut device, 0, &indices)
            .expect("index_select succeeds")
            .to_bytes(&mut device)
            .expect("to_bytes succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(
            actual,
            [2.0, 5.0, 0.0, 3.0],
            "rows [2, 0] of the transposed table"
        );
    }

    #[test]
    fn cat_on_first_and_last_dim_for_each_float_dtype() {
        struct NameOutputDimsExpectedCase {
            name: &'static str,
            output: Result<Tensor<MetalDevice>, TensorError>,
            dims: [usize; 2],
            expected: Vec<f32>,
        }
        struct ValuesDimsCase {
            values: Vec<f32>,
            dims: Vec<usize>,
        }
        struct DtypeToleranceCase {
            dtype: DType,
            tolerance: f32,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5,
            },
            DtypeToleranceCase {
                dtype: DType::BF16,
                tolerance: 1e-2,
            },
        ]
        .into_iter()
        .for_each(
            |DtypeToleranceCase {
                 dtype,
                 tolerance,
             }| {
                let [top, bottom, left, right] = [
                    ValuesDimsCase {
                        values: vec![0.0, 1.0, 2.0],
                        dims: vec![1, 3],
                    },
                    ValuesDimsCase {
                        values: vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                        dims: vec![2, 3],
                    },
                    ValuesDimsCase {
                        values: vec![10.0, 20.0],
                        dims: vec![2, 1],
                    },
                    ValuesDimsCase {
                        values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                        dims: vec![2, 3],
                    },
                ]
                .map(
                    |ValuesDimsCase {
                         values,
                         dims,
                     }| {
                        Tensor::from_bytes(
                            &mut device,
                            &values
                                .iter()
                                .flat_map(|value| value.to_le_bytes())
                                .collect::<Vec<u8>>(),
                            DType::F32,
                            Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                        )
                        .expect("from_bytes succeeds")
                        .to_dtype(&mut device, dtype)
                        .expect("to_dtype succeeds")
                    },
                );
                [
                    NameOutputDimsExpectedCase {
                        name: "cat on dim 0",
                        output: Tensor::cat(&mut device, [&top, &bottom], 0),
                        dims: [3, 3],
                        expected: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    },
                    NameOutputDimsExpectedCase {
                        name: "cat on the last dim",
                        output: Tensor::cat(&mut device, [&left, &right], 1),
                        dims: [2, 4],
                        expected: vec![10.0, 0.0, 1.0, 2.0, 20.0, 3.0, 4.0, 5.0],
                    },
                ]
                .into_iter()
                .for_each(
                    |NameOutputDimsExpectedCase {
                         name,
                         output,
                         dims,
                         expected,
                     }| {
                        let output = output.expect("cat succeeds");
                        assert_eq!(output.shape().dims(), &dims, "{name}: output dims");
                        let actual: Vec<f32> = output
                            .to_dtype(&mut device, DType::F32)
                            .expect("to_dtype succeeds")
                            .to_bytes(&mut device)
                            .expect("to_bytes succeeds")
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|&chunk| f32::from_le_bytes(chunk))
                            .collect();
                        assert_eq!(actual.len(), expected.len(), "{name}: element count");
                        actual
                            .iter()
                            .zip(&expected)
                            .for_each(|(&actual, &expected)| {
                                assert!(
                                    (actual - expected).abs()
                                        <= tolerance * expected.abs().max(f32::MIN_POSITIVE),
                                    "{name} for {dtype:?}: {actual} is not within {tolerance} of \
                                     {expected}"
                                );
                            });
                    },
                );
            },
        );
    }

    #[test]
    fn cat_of_permuted_input_reads_strided_values() {
        struct ValuesDimsCase {
            values: Vec<f32>,
            dims: Vec<usize>,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let [lhs, rhs] = [
            ValuesDimsCase {
                values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                dims: vec![2, 3],
            },
            ValuesDimsCase {
                values: vec![10.0, 11.0],
                dims: vec![1, 2],
            },
        ]
        .map(
            |ValuesDimsCase {
                 values,
                 dims,
             }| {
                Tensor::from_bytes(
                    &mut device,
                    &values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                )
                .expect("from_bytes succeeds")
            },
        );
        let permuted = lhs.permute(&[1, 0]).expect("the permutation is valid");
        let actual: Vec<f32> = Tensor::cat(&mut device, [&permuted, &rhs], 0)
            .expect("cat succeeds")
            .to_bytes(&mut device)
            .expect("to_bytes succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(
            actual,
            [0.0, 3.0, 1.0, 4.0, 2.0, 5.0, 10.0, 11.0],
            "the permuted lhs is read in place"
        );
    }

    #[test]
    fn cat_with_empty_input_keeps_the_other_input() {
        struct ValuesDimsCase {
            values: Vec<f32>,
            dims: Vec<usize>,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let [empty, rows] = [
            ValuesDimsCase {
                values: vec![],
                dims: vec![0, 3],
            },
            ValuesDimsCase {
                values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                dims: vec![2, 3],
            },
        ]
        .map(
            |ValuesDimsCase {
                 values,
                 dims,
             }| {
                Tensor::from_bytes(
                    &mut device,
                    &values
                        .iter()
                        .flat_map(|value: &f32| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                )
                .expect("from_bytes succeeds")
            },
        );
        let output = Tensor::cat(&mut device, [&empty, &rows], 0).expect("cat succeeds");
        assert_eq!(
            output.shape().dims(),
            &[2, 3],
            "the empty input adds no rows"
        );
        let actual: Vec<f32> = output
            .to_bytes(&mut device)
            .expect("to_bytes succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(
            actual,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "the output equals the non-empty input"
        );
    }

    #[test]
    fn slice_scatter_writes_one_position_and_keeps_the_others_for_each_float_dtype() {
        struct DtypeToleranceCase {
            dtype: DType,
            tolerance: f32,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let heads = 2_usize;
        let positions = 4_usize;
        let head_dim = 3_usize;
        let position = 2_usize;
        let update_values: Vec<f32> = (0_u16..6).map(|index| f32::from(index + 1)).collect();
        let expected: Vec<f32> = (0..heads)
            .flat_map(|head| {
                let update_values = &update_values;
                (0..positions).flat_map(move |slot| {
                    (0..head_dim).map(move |dim| {
                        if slot == position {
                            update_values
                                .get(head * head_dim + dim)
                                .copied()
                                .expect("the update holds every head and dim")
                        } else {
                            0.0
                        }
                    })
                })
            })
            .collect();
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5,
            },
            DtypeToleranceCase {
                dtype: DType::BF16,
                tolerance: 1e-2,
            },
        ]
        .into_iter()
        .for_each(
            |DtypeToleranceCase {
                 dtype,
                 tolerance,
             }| {
                let mut cache = Tensor::zeros(
                    &mut device,
                    dtype,
                    Shape::try_from([heads, positions, head_dim]).expect("the shape is valid"),
                )
                .expect("the zeros succeed");
                let update = Tensor::from_bytes(
                    &mut device,
                    &update_values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from([heads, 1, head_dim]).expect("the shape is valid"),
                )
                .expect("from_bytes succeeds")
                .to_dtype(&mut device, dtype)
                .expect("to_dtype succeeds");
                cache
                    .slice_scatter(&mut device, &update, 1, position)
                    .expect("the slice scatter succeeds");
                let actual: Vec<f32> = cache
                    .to_dtype(&mut device, DType::F32)
                    .expect("to_dtype succeeds")
                    .to_bytes(&mut device)
                    .expect("to_bytes succeeds")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&chunk| f32::from_le_bytes(chunk))
                    .collect();
                assert_eq!(actual.len(), expected.len(), "slice scatter: element count");
                actual
                    .iter()
                    .zip(&expected)
                    .for_each(|(&actual, &expected)| {
                        assert!(
                            (actual - expected).abs()
                                <= tolerance * expected.abs().max(f32::MIN_POSITIVE),
                            "[2, 1, 3] into [2, 4, 3] at 2 for {dtype:?}: {actual} is not within \
                             {tolerance} of {expected}"
                        );
                    });
            },
        );
    }

    #[test]
    fn slice_scatter_of_permuted_update_reads_strided_values() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let mut target = Tensor::zeros(
            &mut device,
            DType::F32,
            Shape::try_from([2, 4]).expect("the shape is valid"),
        )
        .expect("the zeros succeed");
        let update = Tensor::from_bytes(
            &mut device,
            &[0.0_f32, 1.0, 2.0, 3.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 2]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid");
        target
            .slice_scatter(&mut device, &update, 1, 1)
            .expect("the slice scatter succeeds");
        let actual: Vec<f32> = target
            .to_bytes(&mut device)
            .expect("to_bytes succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(
            actual,
            [0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0],
            "the transposed update lands in columns 1 and 2"
        );
    }

    #[test]
    fn slice_scatter_of_broadcast_target_is_rejected() {
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let mut target = Tensor::zeros(
            &mut device,
            DType::F32,
            Shape::try_from([1, 4]).expect("the shape is valid"),
        )
        .expect("the zeros succeed")
        .expand(Shape::try_from([3, 4]).expect("the shape is valid"))
        .expect("the broadcast is valid");
        let update = Tensor::from_bytes(
            &mut device,
            &[0.0_f32, 1.0, 2.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([3, 1]).expect("the shape is valid"),
        )
        .expect("from_bytes succeeds");
        assert_eq!(
            target.slice_scatter(&mut device, &update, 1, 0),
            Err(TensorError::SliceScatterNonContiguous),
            "a broadcast target would be written by several threads"
        );
    }

    #[test]
    fn slice_scatter_while_another_view_of_the_target_is_alive_is_rejected() {
        struct ValuesDimsCase {
            values: Vec<f32>,
            dims: Vec<usize>,
        }
        let mut device = MetalDevice::new().expect("a Metal device is available");
        let [mut target, update] = [
            ValuesDimsCase {
                values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                dims: vec![2, 4],
            },
            ValuesDimsCase {
                values: vec![100.0, 101.0],
                dims: vec![2, 1],
            },
        ]
        .map(
            |ValuesDimsCase {
                 values,
                 dims,
             }| {
                Tensor::from_bytes(
                    &mut device,
                    &values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                )
                .expect("from_bytes succeeds")
            },
        );
        let view = target.narrow(0, 0, 1).expect("the narrow is valid");
        assert_eq!(
            target.slice_scatter(&mut device, &update, 1, 3),
            Err(TensorError::SharedStorage),
            "a live view would observe the write"
        );
        assert_eq!(view.shape().dims(), &[1, 4], "the view is still alive");
    }

    #[test]
    fn storage_from_another_device_cannot_be_read() {
        let mut producer = MetalDevice::new().expect("a Metal device is available");
        let mut consumer = MetalDevice::new().expect("a Metal device is available");
        let shape = Shape::try_from([2]).expect("the shape is valid");
        let tensor = Tensor::from_bytes(&mut producer, &[0_u8; 8], DType::F32, shape)
            .expect("from_bytes succeeds");
        assert_eq!(
            [
                tensor.to_bytes(&mut consumer).err(),
                tensor.neg(&mut consumer).err()
            ],
            [Some(TensorError::Device); 2],
            "a storage from another device cannot be read as bytes or by a kernel"
        );
    }

    #[test]
    fn slice_scatter_of_target_from_another_device_is_rejected() {
        let mut producer = MetalDevice::new().expect("a Metal device is available");
        let mut consumer = MetalDevice::new().expect("a Metal device is available");
        let mut target = Tensor::zeros(
            &mut producer,
            DType::F32,
            Shape::try_from([2, 2]).expect("the shape is valid"),
        )
        .expect("the zeros succeed");
        let update = Tensor::zeros(
            &mut consumer,
            DType::F32,
            Shape::try_from([2, 1]).expect("the shape is valid"),
        )
        .expect("the zeros succeed");
        assert_eq!(
            target.slice_scatter(&mut consumer, &update, 1, 0),
            Err(TensorError::Device),
            "a target from another device cannot be written"
        );
    }
}
