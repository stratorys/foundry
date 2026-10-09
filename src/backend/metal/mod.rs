mod error;
mod stream;

use std::collections::HashMap;
use std::ptr::NonNull;

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

pub use self::error::MetalError;
use self::stream::CommandStream;
use crate::core::primitive::{
    BinaryOp,
    ConcatSpec,
    GatherSpec,
    MatmulSpec,
    MatrixLayout,
    ReduceOp,
    ReduceSpec,
    SliceUpdateSpec,
    UnaryOp,
};
use crate::core::{
    Backend,
    DType,
    FloatDType,
    Layout,
    Operand,
    OperandMut,
    RANK_MAX,
    Shape,
};

const KERNEL_SOURCES: &[&str] = &[
    include_str!("kernels/common.metal"),
    include_str!("kernels/copy.metal"),
    include_str!("kernels/unary.metal"),
    include_str!("kernels/binary.metal"),
    include_str!("kernels/reduce.metal"),
    include_str!("kernels/matmul.metal"),
    include_str!("kernels/gather.metal"),
    include_str!("kernels/slice_update.metal"),
];

const REDUCE_THREADS: usize = 256;

const MATMUL_TILE: usize = 16;

const MATMUL_THREADS: usize = MATMUL_TILE * MATMUL_TILE;

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type ComputeEncoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

pub struct MetalStorage {
    buffer: Buffer,
    byte_len: usize,
    queue: Queue,
}

pub struct MetalBackend {
    device: Device,
    library: Library,
    pipelines: HashMap<&'static str, Pipeline>,
    stream: CommandStream,
}

struct BufferView<'buffer> {
    storage: &'buffer MetalStorage,
    layout: &'buffer Layout,
}

#[repr(C)]
struct StridedArgs {
    count: u32,
    rank: u32,
    dims: [u32; RANK_MAX],
    strides: [u32; RANK_MAX],
    offset: u32,
}

#[repr(C)]
struct ReduceArgs {
    rows: StridedArgs,
    axis_len: u32,
    axis_stride: u32,
}

#[repr(C)]
struct MatmulOperand {
    offset: u32,
    row_stride: u32,
    col_stride: u32,
    batch_strides: [u32; RANK_MAX],
}

#[repr(C)]
struct MatmulArgs {
    m: u32,
    n: u32,
    k: u32,
    batch_rank: u32,
    batch_dims: [u32; RANK_MAX],
    lhs: MatmulOperand,
    rhs: MatmulOperand,
}

struct MatmulDispatch {
    args: MatmulArgs,
    grid: MTLSize,
}

struct MatmulInput<'buffer> {
    view: BufferView<'buffer>,
    layout: MatrixLayout,
    copy: Option<MetalStorage>,
}

#[repr(C)]
struct GatherArgs {
    count: u32,
    rows: u32,
    cols: u32,
    table_offset: u32,
    table_row_stride: u32,
    table_col_stride: u32,
    indices: StridedArgs,
}

impl StridedArgs {
    fn new(input: &BufferView<'_>) -> Result<Self, MetalError> {
        Self::with_shape(input.layout, input.layout.shape())
    }

    fn with_shape(
        layout: &Layout,
        shape: &Shape,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            count: index_u32(shape.element_count())?,
            rank: index_u32(shape.rank())?,
            dims: padded_u32(shape.dims())?,
            strides: padded_u32(layout.strides())?,
            offset: index_u32(layout.offset())?,
        })
    }

    fn window(
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

impl GatherArgs {
    fn new(
        table: &BufferView<'_>,
        indices: &BufferView<'_>,
        spec: &GatherSpec,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            count: index_u32(spec.output().element_count())?,
            rows: index_u32(spec.rows())?,
            cols: index_u32(spec.cols())?,
            table_offset: index_u32(table.layout.offset())?,
            table_row_stride: index_u32(spec.row_stride())?,
            table_col_stride: index_u32(spec.col_stride())?,
            indices: StridedArgs::new(indices)?,
        })
    }
}

impl ReduceArgs {
    fn new(
        input: &BufferView<'_>,
        spec: &ReduceSpec,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            rows: StridedArgs::with_shape(input.layout, spec.output())?,
            axis_len: index_u32(spec.axis_len())?,
            axis_stride: index_u32(spec.axis_stride())?,
        })
    }
}

impl MatmulOperand {
    fn new(matrix: &MatrixLayout) -> Result<Self, MetalError> {
        Ok(Self {
            offset: index_u32(matrix.offset())?,
            row_stride: index_u32(matrix.row_stride())?,
            col_stride: index_u32(matrix.col_stride())?,
            batch_strides: padded_u32(matrix.batch_strides())?,
        })
    }
}

impl MatmulDispatch {
    fn new(
        spec: &MatmulSpec,
        lhs: &MatrixLayout,
        rhs: &MatrixLayout,
    ) -> Result<Self, MetalError> {
        let batch_dims = spec.batch().dims();
        let args = MatmulArgs {
            m: index_u32(spec.m())?,
            n: index_u32(spec.n())?,
            k: index_u32(spec.k())?,
            batch_rank: index_u32(batch_dims.len())?,
            batch_dims: padded_u32(batch_dims)?,
            lhs: MatmulOperand::new(lhs)?,
            rhs: MatmulOperand::new(rhs)?,
        };
        let grid = MTLSize {
            width: spec.n().div_ceil(MATMUL_TILE),
            height: spec.m().div_ceil(MATMUL_TILE),
            depth: spec.batch().element_count(),
        };
        Ok(Self {
            args,
            grid,
        })
    }
}

impl MetalBackend {
    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceUnavailable)?;
        let queue = device.newCommandQueue().ok_or(MetalError::QueueCreation)?;
        let source = NSString::from_str(&KERNEL_SOURCES.concat());
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|error| {
                error!(
                    message = "Metal library compilation failed.",
                    error = %error.localizedDescription(),
                );
                MetalError::LibraryCompilation
            })?;
        Ok(Self {
            device,
            library,
            pipelines: HashMap::new(),
            stream: CommandStream::new(queue),
        })
    }

    fn allocate(
        &self,
        byte_len: usize,
    ) -> Result<MetalStorage, MetalError> {
        let bytes_max = self.device.maxBufferLength();
        if byte_len > bytes_max {
            error!(
                message = "Buffer exceeds the maximum Metal buffer length.",
                bytes = byte_len,
                bytes_max,
            );
            return Err(MetalError::BufferTooLarge);
        }
        let buffer = self
            .device
            .newBufferWithLength_options(byte_len.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                error!(
                    message = "Metal buffer allocation failed.",
                    bytes = byte_len
                );
                MetalError::BufferAllocation
            })?;
        Ok(MetalStorage {
            buffer,
            byte_len,
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
            error!(message = "Storage was produced by another Metal backend instance.");
            Err(MetalError::ForeignStorage)
        }
    }

    fn view<'buffer, D: Copy>(
        &self,
        operand: &Operand<'buffer, MetalStorage, D>,
    ) -> Result<BufferView<'buffer>, MetalError> {
        self.owns(operand.storage())?;
        Ok(BufferView {
            storage: operand.storage(),
            layout: operand.layout(),
        })
    }

    fn pipeline(
        &mut self,
        name: &'static str,
    ) -> Result<Pipeline, MetalError> {
        if let Some(pipeline) = self.pipelines.get(name) {
            return Ok(pipeline.clone());
        }
        let function = self
            .library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| {
                error!(message = "Kernel is not in the Metal library.", name);
                MetalError::KernelNotFound
            })?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| {
                error!(
                    message = "Metal pipeline creation failed.",
                    name,
                    error = %error.localizedDescription(),
                );
                MetalError::PipelineCreation
            })?;
        self.pipelines.insert(name, pipeline.clone());
        Ok(pipeline)
    }

    fn encode(
        &mut self,
        pipeline: &Pipeline,
        bind: impl FnOnce(&ComputeEncoder),
    ) -> Result<(), MetalError> {
        self.stream.compute(pipeline, bind)
    }

    fn dispatch_strided(
        &mut self,
        kernel: &'static str,
        inputs: &[&BufferView<'_>],
        output: &MetalStorage,
    ) -> Result<(), MetalError> {
        let count = inputs
            .first()
            .map_or(0, |input| input.layout.shape().element_count());
        if count == 0 {
            return Ok(());
        }
        let args = inputs
            .iter()
            .map(|input| StridedArgs::new(input))
            .collect::<Result<Vec<_>, _>>()?;
        let output_index = inputs.len();
        let pipeline = self.pipeline(kernel)?;
        let width = pipeline.maxTotalThreadsPerThreadgroup().min(count);
        self.encode(&pipeline, |encoder| {
            inputs.iter().enumerate().for_each(|(index, input)| {
                // SAFETY: the input buffer is alive for the whole call and the
                // command buffer retains it. The view comes from a `Tensor`
                // operand, which only `crate::core` can build, or from a
                // buffer this backend allocated with a contiguous layout of
                // its shape, so every index its layout addresses stays inside
                // its storage. Every `Shape` has at most `i32::MAX` non-zero
                // elements, so the storage size and every address its layout
                // produces stay below `u32::MAX` and do not wrap. Every
                // input has the dispatch shape: `count` is the first input's
                // element count, and `Tensor` broadcasts both binary operands
                // to the same shape before calling the backend, so the kernel
                // reads each input, `rhs` included, only for `gid < count`
                // inside its layout. The storage was produced on this serial
                // queue, so every earlier write to it is ordered before this
                // dispatch.
                unsafe { encoder.setBuffer_offset_atIndex(Some(&input.storage.buffer), 0, index) };
            });
            // SAFETY: the output buffer holds `count` elements of the kernel's
            // output type, one per dispatched thread.
            unsafe { encoder.setBuffer_offset_atIndex(Some(&output.buffer), 0, output_index) };
            args.iter()
                .zip(output_index.saturating_add(1)..)
                .for_each(|(input_args, index)| {
                    // SAFETY: `input_args` is a `#[repr(C)]` struct of `u32`
                    // that matches the layout of
                    // `StridedArgs` in `common.metal`; Metal
                    // copies the bytes before the call returns.
                    unsafe {
                        encoder.setBytes_length_atIndex(
                            NonNull::from(input_args).cast(),
                            size_of::<StridedArgs>(),
                            index,
                        )
                    };
                });
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: count,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width,
                    height: 1,
                    depth: 1,
                },
            );
        })
    }

    fn dispatch_reduce(
        &mut self,
        kernel: &'static str,
        input: &BufferView<'_>,
        args: &ReduceArgs,
        rows: usize,
        output: &MetalStorage,
    ) -> Result<(), MetalError> {
        if rows == 0 {
            return Ok(());
        }
        let pipeline = self.pipeline(kernel)?;
        let threads = pipeline.maxTotalThreadsPerThreadgroup();
        if threads < REDUCE_THREADS {
            return Err(threadgroup_too_small(kernel, threads, REDUCE_THREADS));
        }
        self.encode(&pipeline, |encoder| {
            // SAFETY: the input buffer is alive for the whole call and the
            // command buffer retains it. The view comes from a `Tensor`
            // operand, which only `crate::core` can build, so every index its
            // layout addresses stays inside its storage. Every `Shape` has at
            // most `i32::MAX` non-zero elements, so addresses and the loop
            // counter `k += 256` stay below `u32::MAX` and do not wrap. The
            // storage was produced on this serial queue, so every earlier
            // write to it is ordered before this dispatch.
            unsafe { encoder.setBuffer_offset_atIndex(Some(&input.storage.buffer), 0, 0) };
            // SAFETY: the output buffer holds one element of the kernel's
            // output type per dispatched threadgroup.
            unsafe { encoder.setBuffer_offset_atIndex(Some(&output.buffer), 0, 1) };
            // SAFETY: `args` is a `#[repr(C)]` struct of `u32` that matches the
            // layout of `ReduceArgs` in `reduce.metal`; Metal copies the bytes
            // before the call returns.
            unsafe {
                encoder.setBytes_length_atIndex(
                    NonNull::from(args).cast(),
                    size_of::<ReduceArgs>(),
                    2,
                )
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: rows,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: REDUCE_THREADS,
                    height: 1,
                    depth: 1,
                },
            );
        })
    }

    fn dispatch_matmul(
        &mut self,
        kernel: &'static str,
        lhs: &BufferView<'_>,
        rhs: &BufferView<'_>,
        args: &MatmulArgs,
        grid: MTLSize,
        output: &MetalStorage,
    ) -> Result<(), MetalError> {
        let pipeline = self.pipeline(kernel)?;
        let threads = pipeline.maxTotalThreadsPerThreadgroup();
        if threads < MATMUL_THREADS {
            return Err(threadgroup_too_small(kernel, threads, MATMUL_THREADS));
        }
        self.encode(&pipeline, |encoder| {
            // SAFETY: both input buffers are alive for the whole call and the
            // command buffer retains them. Each view comes from a `Tensor`
            // operand, which only `crate::core` can build, or from a copy this
            // backend allocated with a contiguous layout of its shape, so every
            // index its layout addresses stays inside its storage. Every
            // `Shape` has at most `i32::MAX` non-zero elements, so addresses
            // and the loop counter `k_start += 16` stay below `u32::MAX` and do
            // not wrap. Both storages were produced on this serial queue, so
            // every earlier write to them is ordered before this dispatch.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&lhs.storage.buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&rhs.storage.buffer), 0, 1);
            };
            // SAFETY: the output buffer holds `batch * m * n` elements of the
            // kernel's type, and the kernel writes only inside `[m, n]` of each
            // batch.
            unsafe { encoder.setBuffer_offset_atIndex(Some(&output.buffer), 0, 2) };
            // SAFETY: `args` is a `#[repr(C)]` struct of `u32` that matches the
            // layout of `MatmulArgs` in `matmul.metal`; Metal copies the bytes
            // before the call returns.
            unsafe {
                encoder.setBytes_length_atIndex(
                    NonNull::from(args).cast(),
                    size_of::<MatmulArgs>(),
                    3,
                )
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                grid,
                MTLSize {
                    width: MATMUL_TILE,
                    height: MATMUL_TILE,
                    depth: 1,
                },
            );
        })
    }

    fn dispatch_gather(
        &mut self,
        kernel: &'static str,
        table: &BufferView<'_>,
        indices: &BufferView<'_>,
        args: &GatherArgs,
        count: usize,
        output: &MetalStorage,
    ) -> Result<(), MetalError> {
        if count == 0 {
            return Ok(());
        }
        let pipeline = self.pipeline(kernel)?;
        let width = pipeline.maxTotalThreadsPerThreadgroup().min(count);
        self.encode(&pipeline, |encoder| {
            // SAFETY: both input buffers are alive for the whole call and the
            // command buffer retains them. Both views come from `Tensor`
            // operands, which only `crate::core` can build, so every index
            // their layouts address stays inside their storage. Every `Shape`
            // has at most `i32::MAX` non-zero elements, so addresses stay below
            // `u32::MAX` and do not wrap, and the kernel reads a table row only
            // when its index is below `rows`. Both storages were produced on
            // this serial queue, so every earlier write to them is ordered
            // before this dispatch.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&table.storage.buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&indices.storage.buffer), 0, 1);
            };
            // SAFETY: the output buffer holds `count` elements of the kernel's
            // type, one per dispatched thread.
            unsafe { encoder.setBuffer_offset_atIndex(Some(&output.buffer), 0, 2) };
            // SAFETY: `args` is a `#[repr(C)]` struct of `u32` that matches the
            // layout of `GatherArgs` in `gather.metal`; Metal copies the bytes
            // before the call returns.
            unsafe {
                encoder.setBytes_length_atIndex(
                    NonNull::from(args).cast(),
                    size_of::<GatherArgs>(),
                    3,
                )
            };
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: count,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width,
                    height: 1,
                    depth: 1,
                },
            );
        })
    }

    fn dispatch_write(
        &mut self,
        kernel: &'static str,
        input: &BufferView<'_>,
        window: &StridedArgs,
        output: &mut MetalStorage,
    ) -> Result<(), MetalError> {
        let count = input.layout.shape().element_count();
        if count == 0 {
            return Ok(());
        }
        let input_args = StridedArgs::new(input)?;
        let pipeline = self.pipeline(kernel)?;
        let width = pipeline.maxTotalThreadsPerThreadgroup().min(count);
        self.encode(&pipeline, |encoder| {
            // SAFETY: the input buffer is alive for the whole call and the
            // command buffer retains it. The view comes from a `Tensor`
            // operand, which only `crate::core` can build, so every index its
            // layout addresses stays inside its storage. Every `Shape` has at
            // most `i32::MAX` non-zero elements, so addresses stay below
            // `u32::MAX` and do not wrap. The storage was produced on this
            // serial queue, so every earlier write to it is ordered before
            // this dispatch.
            unsafe { encoder.setBuffer_offset_atIndex(Some(&input.storage.buffer), 0, 0) };
            // SAFETY: the output buffer is alive for the whole call and the
            // command buffer retains it. It was produced on this serial queue,
            // so this write is ordered after every earlier access to it. It is
            // borrowed mutably, so no other reference to it is live
            // during encoding: either a slice update target, which
            // `Tensor::slice_update` checked to be contiguous
            // and owned by no other tensor before building its `OperandMut`,
            // or a fresh concat output with a contiguous layout. The shape rule
            // bounds `window` inside that layout, so every written
            // index stays inside the storage and is written by one
            // thread only.
            unsafe { encoder.setBuffer_offset_atIndex(Some(&output.buffer), 0, 1) };
            // SAFETY: `input_args` and `window` are `#[repr(C)]` structs of
            // `u32` that match the layout of `StridedArgs` in `common.metal`;
            // Metal copies the bytes before the call returns.
            unsafe {
                encoder.setBytes_length_atIndex(
                    NonNull::from(&input_args).cast(),
                    size_of::<StridedArgs>(),
                    2,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::from(window).cast(),
                    size_of::<StridedArgs>(),
                    3,
                );
            };
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: count,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width,
                    height: 1,
                    depth: 1,
                },
            );
        })
    }

    fn copy_view(
        &mut self,
        input: &BufferView<'_>,
        dtype: DType,
    ) -> Result<MetalStorage, MetalError> {
        let output = self.allocate(input.layout.shape().byte_len(dtype))?;
        self.dispatch_strided(copy_kernel(dtype), &[input], &output)?;
        Ok(output)
    }

    fn matmul_operand<'buffer>(
        &mut self,
        operand: BufferView<'buffer>,
        matrix: &MatrixLayout,
        spec: &MatmulSpec,
        rows: usize,
        cols: usize,
    ) -> Result<MatmulInput<'buffer>, MetalError> {
        if is_matmul_ready(matrix, rows, cols) {
            return Ok(MatmulInput {
                view: operand,
                layout: *matrix,
                copy: None,
            });
        }
        let storage = self.copy_view(&operand, spec.dtype().into())?;
        Ok(MatmulInput {
            view: operand,
            layout: MatrixLayout::contiguous(spec.batch(), rows, cols),
            copy: Some(storage),
        })
    }
}

impl Backend for MetalBackend {
    type Error = MetalError;
    type Storage = MetalStorage;

    fn upload(
        &mut self,
        bytes: &[u8],
    ) -> Result<MetalStorage, MetalError> {
        let storage = self.allocate(bytes.len())?;
        // SAFETY: the buffer is freshly allocated in shared storage with at
        // least `storage.byte_len == bytes.len()` bytes, and no GPU
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
        byte_len: usize,
    ) -> Result<MetalStorage, MetalError> {
        let storage = self.allocate(byte_len)?;
        self.stream.fill_zero(&storage.buffer)?;
        Ok(storage)
    }

    fn download(
        &mut self,
        input: Operand<'_, MetalStorage>,
    ) -> Result<Vec<u8>, MetalError> {
        let dtype = input.dtype();
        let input = self.view(&input)?;
        let bytes = input.layout.shape().byte_len(dtype);
        let bytes_offset = input.layout.offset().saturating_mul(dtype.size_bytes());
        let bytes_len = input.storage.byte_len;
        match bytes_offset.checked_add(bytes) {
            Some(end) if end <= bytes_len => {}
            Some(_) | None => {
                error!(
                    message = "Download exceeds the buffer length.",
                    bytes, bytes_offset, bytes_len,
                );
                return Err(MetalError::DownloadOutOfBounds);
            }
        }
        self.stream.synchronize()?;
        let mut output = vec![0_u8; bytes];
        // SAFETY: the buffer uses shared storage, `bytes_offset + bytes <=
        // byte_len` was checked above, `output` holds `bytes` bytes,
        // and every command buffer that writes the buffer has completed:
        // `view` checked that the storage was produced on this backend's
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

    fn unary(
        &mut self,
        op: UnaryOp,
        input: Operand<'_, MetalStorage, FloatDType>,
    ) -> Result<MetalStorage, MetalError> {
        let dtype = input.dtype();
        let input = self.view(&input)?;
        let output = self.allocate(input.layout.shape().byte_len(dtype.into()))?;
        self.dispatch_strided(unary_kernel(op, dtype), &[&input], &output)?;
        Ok(output)
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        lhs: Operand<'_, MetalStorage, FloatDType>,
        rhs: Operand<'_, MetalStorage, FloatDType>,
    ) -> Result<MetalStorage, MetalError> {
        let dtype = lhs.dtype();
        let lhs = self.view(&lhs)?;
        let rhs = self.view(&rhs)?;
        let output = self.allocate(lhs.layout.shape().byte_len(dtype.into()))?;
        self.dispatch_strided(binary_kernel(op, dtype), &[&lhs, &rhs], &output)?;
        Ok(output)
    }

    fn reduce(
        &mut self,
        input: Operand<'_, MetalStorage, FloatDType>,
        spec: &ReduceSpec,
    ) -> Result<MetalStorage, MetalError> {
        let input = self.view(&input)?;
        let args = ReduceArgs::new(&input, spec)?;
        let output = self.allocate(spec.output().byte_len(spec.dtype_output()))?;
        self.dispatch_reduce(
            reduce_kernel(spec.op(), spec.dtype()),
            &input,
            &args,
            spec.output().element_count(),
            &output,
        )?;
        Ok(output)
    }

    fn matmul(
        &mut self,
        lhs: Operand<'_, MetalStorage, FloatDType>,
        rhs: Operand<'_, MetalStorage, FloatDType>,
        spec: &MatmulSpec,
    ) -> Result<MetalStorage, MetalError> {
        let lhs = self.view(&lhs)?;
        let rhs = self.view(&rhs)?;
        let output = self.allocate(spec.output().byte_len(spec.dtype().into()))?;
        if spec.output().element_count() == 0 {
            return Ok(output);
        }
        let lhs_input = self.matmul_operand(lhs, spec.lhs(), spec, spec.m(), spec.k())?;
        let rhs_input = self.matmul_operand(rhs, spec.rhs(), spec, spec.k(), spec.n())?;
        let lhs_layout = Layout::contiguous(*lhs_input.view.layout.shape());
        let rhs_layout = Layout::contiguous(*rhs_input.view.layout.shape());
        let dispatch = MatmulDispatch::new(spec, &lhs_input.layout, &rhs_input.layout)?;
        let lhs = copied_view(lhs_input.view, lhs_input.copy.as_ref(), &lhs_layout);
        let rhs = copied_view(rhs_input.view, rhs_input.copy.as_ref(), &rhs_layout);
        self.dispatch_matmul(
            matmul_kernel(spec.dtype()),
            &lhs,
            &rhs,
            &dispatch.args,
            dispatch.grid,
            &output,
        )?;
        Ok(output)
    }

    fn copy(
        &mut self,
        input: Operand<'_, MetalStorage>,
    ) -> Result<MetalStorage, MetalError> {
        let dtype = input.dtype();
        self.copy_view(&self.view(&input)?, dtype)
    }

    fn cast(
        &mut self,
        input: Operand<'_, MetalStorage, FloatDType>,
        dtype: FloatDType,
    ) -> Result<MetalStorage, MetalError> {
        let dtype_from = input.dtype();
        let input = self.view(&input)?;
        let output = self.allocate(input.layout.shape().byte_len(dtype.into()))?;
        self.dispatch_strided(cast_kernel(dtype_from, dtype), &[&input], &output)?;
        Ok(output)
    }

    fn gather(
        &mut self,
        table: Operand<'_, MetalStorage, FloatDType>,
        indices: Operand<'_, MetalStorage>,
        spec: &GatherSpec,
    ) -> Result<MetalStorage, MetalError> {
        let table = self.view(&table)?;
        let indices = self.view(&indices)?;
        let args = GatherArgs::new(&table, &indices, spec)?;
        let output = self.allocate(spec.output().byte_len(spec.dtype().into()))?;
        self.dispatch_gather(
            gather_kernel(spec.dtype()),
            &table,
            &indices,
            &args,
            spec.output().element_count(),
            &output,
        )?;
        Ok(output)
    }

    fn concat(
        &mut self,
        lhs: Operand<'_, MetalStorage, FloatDType>,
        rhs: Operand<'_, MetalStorage, FloatDType>,
        spec: &ConcatSpec,
    ) -> Result<MetalStorage, MetalError> {
        let lhs = self.view(&lhs)?;
        let rhs = self.view(&rhs)?;
        let kernel = slice_update_kernel(spec.dtype());
        let mut output = self.allocate(spec.output().byte_len(spec.dtype().into()))?;
        let layout = Layout::contiguous(*spec.output());
        let lhs_window = StridedArgs::window(&layout, lhs.layout.shape(), 0)?;
        let rhs_window = StridedArgs::window(&layout, rhs.layout.shape(), spec.rhs_offset())?;
        self.dispatch_write(kernel, &lhs, &lhs_window, &mut output)?;
        self.dispatch_write(kernel, &rhs, &rhs_window, &mut output)?;
        Ok(output)
    }

    fn slice_update(
        &mut self,
        mut target: OperandMut<'_, MetalStorage, FloatDType>,
        update: Operand<'_, MetalStorage, FloatDType>,
        spec: &SliceUpdateSpec,
    ) -> Result<(), MetalError> {
        self.owns(target.storage_mut())?;
        let layout = target.layout();
        let update = self.view(&update)?;
        let window = StridedArgs::window(layout, update.layout.shape(), spec.offset())?;
        self.dispatch_write(
            slice_update_kernel(spec.dtype()),
            &update,
            &window,
            target.storage_mut(),
        )
    }
}

fn threadgroup_too_small(
    name: &'static str,
    threads: usize,
    threads_required: usize,
) -> MetalError {
    error!(
        message = "Kernel allows too few threads per threadgroup.",
        name, threads, threads_required,
    );
    MetalError::ThreadgroupTooSmall
}

fn copy_kernel(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "copy_f32",
        DType::F16 => "copy_f16",
        DType::BF16 => "copy_bf16",
        DType::U32 => "copy_u32",
    }
}

fn cast_kernel(
    dtype_from: FloatDType,
    dtype_to: FloatDType,
) -> &'static str {
    match (dtype_from, dtype_to) {
        (FloatDType::F32, FloatDType::F32) => "cast_f32_f32",
        (FloatDType::F32, FloatDType::F16) => "cast_f32_f16",
        (FloatDType::F32, FloatDType::BF16) => "cast_f32_bf16",
        (FloatDType::F16, FloatDType::F32) => "cast_f16_f32",
        (FloatDType::F16, FloatDType::F16) => "cast_f16_f16",
        (FloatDType::F16, FloatDType::BF16) => "cast_f16_bf16",
        (FloatDType::BF16, FloatDType::F32) => "cast_bf16_f32",
        (FloatDType::BF16, FloatDType::F16) => "cast_bf16_f16",
        (FloatDType::BF16, FloatDType::BF16) => "cast_bf16_bf16",
    }
}

fn unary_kernel(
    op: UnaryOp,
    dtype: FloatDType,
) -> &'static str {
    match (op, dtype) {
        (UnaryOp::Neg, FloatDType::F32) => "unary_neg_f32",
        (UnaryOp::Neg, FloatDType::F16) => "unary_neg_f16",
        (UnaryOp::Neg, FloatDType::BF16) => "unary_neg_bf16",
        (UnaryOp::Exp, FloatDType::F32) => "unary_exp_f32",
        (UnaryOp::Exp, FloatDType::F16) => "unary_exp_f16",
        (UnaryOp::Exp, FloatDType::BF16) => "unary_exp_bf16",
        (UnaryOp::Sqrt, FloatDType::F32) => "unary_sqrt_f32",
        (UnaryOp::Sqrt, FloatDType::F16) => "unary_sqrt_f16",
        (UnaryOp::Sqrt, FloatDType::BF16) => "unary_sqrt_bf16",
        (UnaryOp::Recip, FloatDType::F32) => "unary_recip_f32",
        (UnaryOp::Recip, FloatDType::F16) => "unary_recip_f16",
        (UnaryOp::Recip, FloatDType::BF16) => "unary_recip_bf16",
    }
}

fn binary_kernel(
    op: BinaryOp,
    dtype: FloatDType,
) -> &'static str {
    match (op, dtype) {
        (BinaryOp::Add, FloatDType::F32) => "binary_add_f32",
        (BinaryOp::Add, FloatDType::F16) => "binary_add_f16",
        (BinaryOp::Add, FloatDType::BF16) => "binary_add_bf16",
        (BinaryOp::Sub, FloatDType::F32) => "binary_sub_f32",
        (BinaryOp::Sub, FloatDType::F16) => "binary_sub_f16",
        (BinaryOp::Sub, FloatDType::BF16) => "binary_sub_bf16",
        (BinaryOp::Mul, FloatDType::F32) => "binary_mul_f32",
        (BinaryOp::Mul, FloatDType::F16) => "binary_mul_f16",
        (BinaryOp::Mul, FloatDType::BF16) => "binary_mul_bf16",
        (BinaryOp::Div, FloatDType::F32) => "binary_div_f32",
        (BinaryOp::Div, FloatDType::F16) => "binary_div_f16",
        (BinaryOp::Div, FloatDType::BF16) => "binary_div_bf16",
    }
}

fn reduce_kernel(
    op: ReduceOp,
    dtype: FloatDType,
) -> &'static str {
    match (op, dtype) {
        (ReduceOp::Sum, FloatDType::F32) => "reduce_sum_f32",
        (ReduceOp::Sum, FloatDType::F16) => "reduce_sum_f16",
        (ReduceOp::Sum, FloatDType::BF16) => "reduce_sum_bf16",
        (ReduceOp::Max, FloatDType::F32) => "reduce_max_f32",
        (ReduceOp::Max, FloatDType::F16) => "reduce_max_f16",
        (ReduceOp::Max, FloatDType::BF16) => "reduce_max_bf16",
        (ReduceOp::Argmax, FloatDType::F32) => "reduce_argmax_f32",
        (ReduceOp::Argmax, FloatDType::F16) => "reduce_argmax_f16",
        (ReduceOp::Argmax, FloatDType::BF16) => "reduce_argmax_bf16",
    }
}

fn matmul_kernel(dtype: FloatDType) -> &'static str {
    match dtype {
        FloatDType::F32 => "matmul_f32",
        FloatDType::F16 => "matmul_f16",
        FloatDType::BF16 => "matmul_bf16",
    }
}

fn gather_kernel(dtype: FloatDType) -> &'static str {
    match dtype {
        FloatDType::F32 => "gather_f32",
        FloatDType::F16 => "gather_f16",
        FloatDType::BF16 => "gather_bf16",
    }
}

fn slice_update_kernel(dtype: FloatDType) -> &'static str {
    match dtype {
        FloatDType::F32 => "slice_update_f32",
        FloatDType::F16 => "slice_update_f16",
        FloatDType::BF16 => "slice_update_bf16",
    }
}

fn is_matmul_ready(
    matrix: &MatrixLayout,
    rows: usize,
    cols: usize,
) -> bool {
    let is_row_major = matrix.col_stride() == 1 || cols == 1;
    let is_transposed = matrix.row_stride() == 1 || rows == 1;
    is_row_major || is_transposed
}

fn copied_view<'buffer>(
    operand: BufferView<'buffer>,
    copy: Option<&'buffer MetalStorage>,
    layout: &'buffer Layout,
) -> BufferView<'buffer> {
    match copy {
        Some(storage) => BufferView {
            storage,
            layout,
        },
        None => operand,
    }
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

fn padded_u32(values: &[usize]) -> Result<[u32; RANK_MAX], MetalError> {
    values
        .iter()
        .zip(0..RANK_MAX)
        .try_fold([0_u32; RANK_MAX], |mut padded, (&value, axis)| {
            if let Some(slot) = padded.get_mut(axis) {
                *slot = index_u32(value)?;
            }
            Ok(padded)
        })
}

#[cfg(test)]
mod tests {
    use std::f32::consts::{
        E,
        PI,
    };

    use crate::backend::metal::{
        MetalBackend,
        MetalError,
        is_matmul_ready,
    };
    use crate::core::primitive::matmul_rule;
    use crate::core::{
        CoreError,
        DType,
        Shape,
        Tensor,
        TensorError,
    };

    #[test]
    fn f32_round_trip_returns_the_same_bytes() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let bytes: Vec<u8> = [1.0_f32, -2.5, 1.5e-3, f32::MIN_POSITIVE, -0.0, f32::MAX]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let downloaded = Tensor::upload(
            &mut backend,
            &bytes,
            DType::F32,
            Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .download(&mut backend)
        .expect("the download succeeds");
        assert_eq!(downloaded, bytes, "f32 bytes survive the round trip");
    }

    #[test]
    fn f16_round_trip_returns_the_same_bytes() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let bytes: Vec<u8> = [0x3C00_u16, 0xC100, 0x4248, 0x0001, 0x8000, 0x7BFF]
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect();
        let downloaded = Tensor::upload(
            &mut backend,
            &bytes,
            DType::F16,
            Shape::try_from([3, 2].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .download(&mut backend)
        .expect("the download succeeds");
        assert_eq!(downloaded, bytes, "f16 bytes survive the round trip");
    }

    #[test]
    fn bf16_round_trip_returns_the_same_bytes() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let bytes: Vec<u8> = [0x3F80_u16, 0xC020, 0x4049, 0x0001, 0x8000, 0x7F7F]
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect();
        let downloaded = Tensor::upload(
            &mut backend,
            &bytes,
            DType::BF16,
            Shape::try_from([6].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .download(&mut backend)
        .expect("the download succeeds");
        assert_eq!(downloaded, bytes, "bf16 bytes survive the round trip");
    }

    #[test]
    fn u32_round_trip_returns_the_same_bytes() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let bytes: Vec<u8> = [0_u32, 1, 128_000, u32::MAX]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let downloaded = Tensor::upload(
            &mut backend,
            &bytes,
            DType::U32,
            Shape::try_from([2, 1, 2].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .download(&mut backend)
        .expect("the download succeeds");
        assert_eq!(downloaded, bytes, "u32 bytes survive the round trip");
    }

    #[test]
    fn empty_round_trip_returns_no_bytes() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let downloaded = Tensor::upload(
            &mut backend,
            &[],
            DType::F32,
            Shape::try_from([0, 3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .download(&mut backend)
        .expect("the download succeeds");
        assert!(downloaded.is_empty(), "an empty tensor downloads no bytes");
    }

    #[test]
    fn zeros_download_only_zero_bytes() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        [DType::F32, DType::F16, DType::BF16, DType::U32]
            .into_iter()
            .for_each(|dtype| {
                let downloaded = Tensor::zeros(
                    &mut backend,
                    dtype,
                    Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
                )
                .expect("zeros succeeds")
                .download(&mut backend)
                .expect("the download succeeds");
                assert_eq!(
                    downloaded.len(),
                    6 * dtype.size_bytes(),
                    "zeros of {dtype:?} has the shape's byte length"
                );
                assert!(
                    downloaded.iter().all(|&byte| byte == 0),
                    "zeros of {dtype:?} downloads only zero bytes"
                );
            });
    }

    #[test]
    fn zeros_after_upload_keeps_both_values() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let bytes: Vec<u8> = [7_u32, 8, 9]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let uploaded = Tensor::upload(
            &mut backend,
            &bytes,
            DType::U32,
            Shape::try_from([3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds");
        let zeros = Tensor::zeros(
            &mut backend,
            DType::U32,
            Shape::try_from([3].as_slice()).expect("the shape is valid"),
        )
        .expect("zeros succeeds");
        assert_eq!(
            zeros.download(&mut backend).expect("the download succeeds"),
            vec![0; 12],
            "zeros downloads zero bytes"
        );
        assert_eq!(
            uploaded
                .download(&mut backend)
                .expect("the download succeeds"),
            bytes,
            "the earlier upload is unchanged"
        );
    }

    #[test]
    fn upload_with_wrong_byte_length_fails() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let result = Tensor::upload(
            &mut backend,
            &[0; 10],
            DType::F32,
            Shape::try_from([2, 2].as_slice()).expect("the shape is valid"),
        );
        assert!(
            matches!(
                result.err(),
                Some(TensorError::Validation(CoreError::ByteLengthMismatch))
            ),
            "a byte length that does not match shape × dtype is rejected"
        );
    }

    #[test]
    fn upload_of_empty_bytes_for_oversized_shape_reports_byte_length_mismatch() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let result = Tensor::upload(
            &mut backend,
            &[],
            DType::F32,
            Shape::try_from([(1 << 31) - 1].as_slice()).expect("the shape is valid"),
        );
        assert!(
            matches!(
                result.err(),
                Some(TensorError::Validation(CoreError::ByteLengthMismatch))
            ),
            "the byte length is checked before allocation"
        );
    }

    #[test]
    fn narrow_on_leading_axis_downloads_the_sub_range() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let bytes: Vec<u8> = (0_u32..6).flat_map(u32::to_le_bytes).collect();
        let narrowed = Tensor::upload(
            &mut backend,
            &bytes,
            DType::U32,
            Shape::try_from([3, 2].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .narrow(0, 1, 2)
        .expect("the narrow is valid");
        let expected: Vec<u8> = (2_u32..6).flat_map(u32::to_le_bytes).collect();
        assert_eq!(
            narrowed
                .download(&mut backend)
                .expect("the download succeeds"),
            expected,
            "the narrowed rows are downloaded"
        );
    }

    #[test]
    fn permuted_download_is_rejected() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let bytes: Vec<u8> = (0_u32..6).flat_map(u32::to_le_bytes).collect();
        let permuted = Tensor::upload(
            &mut backend,
            &bytes,
            DType::U32,
            Shape::try_from([3, 2].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid");
        assert!(
            matches!(
                permuted.download(&mut backend).err(),
                Some(TensorError::Validation(CoreError::DownloadNonContiguous))
            ),
            "a non-contiguous layout cannot be downloaded"
        );
    }

    #[test]
    fn copy_of_permuted_tensor_is_contiguous_for_each_float_dtype() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let expected: Vec<f32> = [
            0_u16, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19,
            23,
        ]
        .into_iter()
        .map(f32::from)
        .collect();
        let source = Tensor::upload(
            &mut backend,
            &(0_u16..24)
                .flat_map(|value| f32::from(value).to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3, 4].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds");
        [DType::F32, DType::F16, DType::BF16]
            .into_iter()
            .for_each(|dtype| {
                let copied = source
                    .cast(&mut backend, dtype)
                    .expect("the cast succeeds")
                    .permute(&[2, 0, 1])
                    .expect("the permutation is valid")
                    .contiguous(&mut backend)
                    .expect("the copy succeeds");
                assert_eq!(copied.dtype(), dtype, "the copy keeps the dtype");
                assert_eq!(copied.shape().dims(), &[4, 2, 3], "permuted dims");
                assert!(copied.layout().is_contiguous(), "the copy is contiguous");
                let actual: Vec<f32> = copied
                    .cast(&mut backend, DType::F32)
                    .expect("the cast succeeds")
                    .download(&mut backend)
                    .expect("the download succeeds")
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
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let copied = Tensor::upload(
            &mut backend,
            &(0_u16..24)
                .flat_map(|value| f32::from(value).to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3, 4].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .narrow(1, 1, 2)
        .expect("the narrow is valid")
        .contiguous(&mut backend)
        .expect("the copy succeeds");
        let expected: Vec<f32> = (4_u16..12).chain(16..24).map(f32::from).collect();
        assert_eq!(copied.shape().dims(), &[2, 2, 4], "narrowed dims");
        let actual: Vec<f32> = copied
            .download(&mut backend)
            .expect("the download succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(actual, expected, "narrowed values");
    }

    #[test]
    fn copy_of_broadcast_tensor_repeats_values() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let copied = Tensor::upload(
            &mut backend,
            &[1.0_f32, 2.0, 3.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([3, 1].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .broadcast_as(Shape::try_from([2, 3, 4].as_slice()).expect("the shape is valid"))
        .expect("the broadcast is valid")
        .contiguous(&mut backend)
        .expect("the copy succeeds");
        let block: [f32; 12] = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0];
        let expected: Vec<f32> = block.iter().chain(block.iter()).copied().collect();
        let actual: Vec<f32> = copied
            .download(&mut backend)
            .expect("the download succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(actual, expected, "broadcast values");
    }

    #[test]
    fn copy_of_permuted_u32_tensor_is_contiguous() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let bytes: Vec<u8> = (0_u32..6).flat_map(u32::to_le_bytes).collect();
        let copied = Tensor::upload(
            &mut backend,
            &bytes,
            DType::U32,
            Shape::try_from([3, 2].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid")
        .contiguous(&mut backend)
        .expect("the copy succeeds");
        let expected: Vec<u8> = [0_u32, 2, 4, 1, 3, 5]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        assert_eq!(
            copied
                .download(&mut backend)
                .expect("the download succeeds"),
            expected,
            "transposed u32 values"
        );
    }

    #[test]
    fn copy_of_empty_tensor_downloads_no_bytes() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let copied = Tensor::upload(
            &mut backend,
            &[],
            DType::F32,
            Shape::try_from([0, 3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid")
        .contiguous(&mut backend)
        .expect("the copy succeeds");
        assert!(
            copied
                .download(&mut backend)
                .expect("the download succeeds")
                .is_empty(),
            "an empty copy downloads no bytes"
        );
    }

    #[test]
    fn cast_f32_to_bf16_and_back_rounds_known_values() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let actual: Vec<f32> = Tensor::upload(
            &mut backend,
            &[1.0_f32, -2.5, PI]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .cast(&mut backend, DType::BF16)
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
        assert_eq!(actual, vec![1.0, -2.5, 3.140625], "bf16 round trip");
    }

    #[test]
    fn cast_f32_to_bf16_rounds_to_nearest_even() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let actual: Vec<u16> = Tensor::upload(
            &mut backend,
            &[0x3F80_8000_u32, 0x3F81_8000, 0x3F80_8001, 0xBF80_8000]
                .into_iter()
                .flat_map(|bits| f32::from_bits(bits).to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([4].as_slice()).expect("the shape is valid"),
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
            actual,
            vec![0x3F80, 0x3F82, 0x3F81, 0xBF80],
            "ties round to even, others to nearest"
        );
    }

    #[test]
    fn cast_f32_to_f16_and_back_rounds_known_values() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let cast = Tensor::upload(
            &mut backend,
            &[1.0_f32, -2.5, 0.5, PI]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([4].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .cast(&mut backend, DType::F16)
        .expect("the cast succeeds");
        let bits: Vec<u16> = cast
            .download(&mut backend)
            .expect("the download succeeds")
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&chunk| u16::from_le_bytes(chunk))
            .collect();
        assert_eq!(bits, vec![0x3C00, 0xC100, 0x3800, 0x4248], "f16 bits");
        let round_tripped: Vec<f32> = cast
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
            round_tripped,
            vec![1.0, -2.5, 0.5, 3.140625],
            "f16 round trip"
        );
    }

    #[test]
    fn cast_of_permuted_tensor_reads_strided_input() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let cast = Tensor::upload(
            &mut backend,
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid")
        .cast(&mut backend, DType::BF16)
        .expect("the cast succeeds");
        assert_eq!(cast.shape().dims(), &[3, 2], "transposed dims");
        let bits: Vec<u16> = cast
            .download(&mut backend)
            .expect("the download succeeds")
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
        struct NameOutputExpectedCase<T0, T1, T2> {
            name: T0,
            output: T1,
            expected: T2,
        }
        struct DtypeToleranceCase<T0, T1> {
            dtype: T0,
            tolerance: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5_f32,
            },
            DtypeToleranceCase {
                dtype: DType::F16,
                tolerance: 1e-2,
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
                let [neg_input, exp_input, sqrt_input, recip_input] = [
                    [1.0_f32, -2.5, 0.0],
                    [0.0, 1.0, -1.0],
                    [4.0, 2.25, 0.0],
                    [2.0, -4.0, 0.5],
                ]
                .map(|values| {
                    Tensor::upload(
                        &mut backend,
                        &values
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<u8>>(),
                        DType::F32,
                        Shape::try_from([3].as_slice()).expect("the shape is valid"),
                    )
                    .expect("the upload succeeds")
                    .cast(&mut backend, dtype)
                    .expect("the cast succeeds")
                });
                [
                    NameOutputExpectedCase {
                        name: "neg",
                        output: neg_input.neg(&mut backend),
                        expected: [-1.0_f32, 2.5, 0.0],
                    },
                    NameOutputExpectedCase {
                        name: "exp",
                        output: exp_input.exp(&mut backend),
                        expected: [1.0, E, 1.0 / E],
                    },
                    NameOutputExpectedCase {
                        name: "sqrt",
                        output: sqrt_input.sqrt(&mut backend),
                        expected: [2.0, 1.5, 0.0],
                    },
                    NameOutputExpectedCase {
                        name: "recip",
                        output: recip_input.recip(&mut backend),
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
                            .cast(&mut backend, DType::F32)
                            .expect("the cast succeeds")
                            .download(&mut backend)
                            .expect("the download succeeds")
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
        struct NameOutputExpectedCase<T0, T1, T2> {
            name: T0,
            output: T1,
            expected: T2,
        }
        struct DtypeToleranceCase<T0, T1> {
            dtype: T0,
            tolerance: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5_f32,
            },
            DtypeToleranceCase {
                dtype: DType::F16,
                tolerance: 1e-2,
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
                    Tensor::upload(
                        &mut backend,
                        &values
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<u8>>(),
                        DType::F32,
                        Shape::try_from([4].as_slice()).expect("the shape is valid"),
                    )
                    .expect("the upload succeeds")
                    .cast(&mut backend, dtype)
                    .expect("the cast succeeds")
                });
                [
                    NameOutputExpectedCase {
                        name: "add",
                        output: lhs.add(&mut backend, &rhs),
                        expected: [3.0_f32, 6.0, 3.5, 3.0],
                    },
                    NameOutputExpectedCase {
                        name: "sub",
                        output: lhs.sub(&mut backend, &rhs),
                        expected: [-1.0, -2.0, 2.5, 5.0],
                    },
                    NameOutputExpectedCase {
                        name: "mul",
                        output: lhs.mul(&mut backend, &rhs),
                        expected: [2.0, 8.0, 1.5, -4.0],
                    },
                    NameOutputExpectedCase {
                        name: "div",
                        output: lhs.div(&mut backend, &rhs),
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
                            .cast(&mut backend, DType::F32)
                            .expect("the cast succeeds")
                            .download(&mut backend)
                            .expect("the download succeeds")
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
        struct ValuesDimsCase<T0, T1> {
            values: T0,
            dims: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let [matrix, row] = [
            ValuesDimsCase {
                values: vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0],
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
            },
        );
        let sum = matrix.add(&mut backend, &row).expect("the add succeeds");
        assert_eq!(sum.shape().dims(), &[2, 3], "broadcast dims");
        let actual: Vec<f32> = sum
            .download(&mut backend)
            .expect("the download succeeds")
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
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let negated = Tensor::upload(
            &mut backend,
            &[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid")
        .neg(&mut backend)
        .expect("the neg succeeds");
        assert_eq!(negated.shape().dims(), &[3, 2], "transposed dims");
        let actual: Vec<f32> = negated
            .download(&mut backend)
            .expect("the download succeeds")
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
        struct ValuesDimsCase<T0, T1> {
            values: T0,
            dims: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let [transposed, contiguous] = [
            ValuesDimsCase {
                values: vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0],
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
            },
        );
        let actual: Vec<f32> = transposed
            .permute(&[1, 0])
            .expect("the permutation is valid")
            .mul(&mut backend, &contiguous)
            .expect("the mul succeeds")
            .download(&mut backend)
            .expect("the download succeeds")
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
    fn binary_on_u32_is_rejected() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let bytes: Vec<u8> = (0_u32..3).flat_map(u32::to_le_bytes).collect();
        let indices = Tensor::upload(
            &mut backend,
            &bytes,
            DType::U32,
            Shape::try_from([3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds");
        assert!(
            matches!(
                indices.add(&mut backend, &indices).err(),
                Some(TensorError::Validation(CoreError::DTypeNotFloat))
            ),
            "binary ops reject u32 on Metal"
        );
    }

    #[test]
    fn add_of_empty_tensors_downloads_no_bytes() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let empty = Tensor::upload(
            &mut backend,
            &[],
            DType::F32,
            Shape::try_from([0, 3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds");
        let sum = empty.add(&mut backend, &empty).expect("the add succeeds");
        assert!(
            sum.download(&mut backend)
                .expect("the download succeeds")
                .is_empty(),
            "an empty add downloads no bytes"
        );
    }

    #[test]
    fn long_chain_of_adds_spans_several_command_buffers() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let ones = Tensor::upload(
            &mut backend,
            &[1.0_f32; 4]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([4].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds");
        let start = Tensor::zeros(
            &mut backend,
            DType::F32,
            Shape::try_from([4].as_slice()).expect("the shape is valid"),
        )
        .expect("zeros succeeds");
        let actual: Vec<f32> = (0..300)
            .try_fold(start, |sum, _| sum.add(&mut backend, &ones))
            .expect("every add succeeds")
            .download(&mut backend)
            .expect("the download succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(actual, vec![300.0; 4], "three hundred additions of one");
    }

    #[test]
    fn zeros_interleaved_with_adds_keep_submission_order() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let [ones, start] = [0_u8; 2].map(|_| {
            Tensor::upload(
                &mut backend,
                &[1.0_f32; 3]
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<u8>>(),
                DType::F32,
                Shape::try_from([3].as_slice()).expect("the shape is valid"),
            )
            .expect("the upload succeeds")
        });
        let actual: Vec<f32> = (0..100)
            .try_fold(start, |sum, _| {
                let zeros = Tensor::zeros(
                    &mut backend,
                    DType::F32,
                    Shape::try_from([3].as_slice()).expect("the shape is valid"),
                )?;
                sum.add(&mut backend, &zeros)?.add(&mut backend, &ones)
            })
            .expect("every zeros and add succeeds")
            .download(&mut backend)
            .expect("the download succeeds")
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
    fn sum_and_max_over_each_axis_match_hand_computed_values_for_each_float_dtype() {
        struct NameReducedDimsExpectedCase<T0, T1, T2, T3> {
            name: T0,
            reduced: T1,
            dims: T2,
            expected: T3,
        }
        struct DtypeToleranceCase<T0, T1> {
            dtype: T0,
            tolerance: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5_f32,
            },
            DtypeToleranceCase {
                dtype: DType::F16,
                tolerance: 1e-2,
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
                let input = Tensor::upload(
                    &mut backend,
                    &[1.0_f32, 5.0, 3.0, 4.0, 2.0, 6.0]
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
                )
                .expect("the upload succeeds")
                .cast(&mut backend, dtype)
                .expect("the cast succeeds");
                [
                    NameReducedDimsExpectedCase {
                        name: "sum axis 0",
                        reduced: input.sum(&mut backend, 0),
                        dims: &[1, 3][..],
                        expected: &[5.0_f32, 7.0, 9.0][..],
                    },
                    NameReducedDimsExpectedCase {
                        name: "sum axis 1",
                        reduced: input.sum(&mut backend, 1),
                        dims: &[2, 1],
                        expected: &[9.0, 12.0],
                    },
                    NameReducedDimsExpectedCase {
                        name: "max axis 0",
                        reduced: input.max(&mut backend, 0),
                        dims: &[1, 3],
                        expected: &[4.0, 5.0, 6.0],
                    },
                    NameReducedDimsExpectedCase {
                        name: "max axis 1",
                        reduced: input.max(&mut backend, 1),
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
                        assert_eq!(reduced.shape().dims(), dims, "{name} keeps the axis");
                        let actual: Vec<f32> = reduced
                            .cast(&mut backend, DType::F32)
                            .expect("the cast succeeds")
                            .download(&mut backend)
                            .expect("the download succeeds")
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
    fn argmax_over_each_axis_matches_hand_computed_indices_for_each_float_dtype() {
        struct AxisDimsExpectedCase<T0, T1, T2> {
            axis: T0,
            dims: T1,
            expected: T2,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        [DType::F32, DType::F16, DType::BF16]
            .into_iter()
            .for_each(|dtype| {
                [
                    AxisDimsExpectedCase {
                        axis: 0,
                        dims: &[1, 3][..],
                        expected: &[1_u32, 0, 1][..],
                    },
                    AxisDimsExpectedCase {
                        axis: 1,
                        dims: &[2, 1],
                        expected: &[1, 2],
                    },
                ]
                .into_iter()
                .for_each(
                    |AxisDimsExpectedCase {
                         axis,
                         dims,
                         expected,
                     }| {
                        let indices = Tensor::upload(
                            &mut backend,
                            &[1.0_f32, 5.0, 3.0, 4.0, 2.0, 6.0]
                                .iter()
                                .flat_map(|value| value.to_le_bytes())
                                .collect::<Vec<u8>>(),
                            DType::F32,
                            Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
                        )
                        .expect("the upload succeeds")
                        .cast(&mut backend, dtype)
                        .expect("the cast succeeds")
                        .argmax(&mut backend, axis)
                        .expect("the argmax succeeds");
                        assert_eq!(indices.dtype(), DType::U32, "argmax outputs u32");
                        assert_eq!(indices.shape().dims(), dims, "argmax keeps the axis");
                        let actual: Vec<u32> = indices
                            .download(&mut backend)
                            .expect("the download succeeds")
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|&chunk| u32::from_le_bytes(chunk))
                            .collect();
                        assert_eq!(actual, expected, "argmax over axis {axis} for {dtype:?}");
                    },
                );
            });
    }

    #[test]
    fn sum_of_4096_bf16_ones_equals_4096() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let actual: Vec<f32> = Tensor::upload(
            &mut backend,
            &[1.0_f32; 4096]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([4096].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .cast(&mut backend, DType::BF16)
        .expect("the cast succeeds")
        .sum(&mut backend, 0)
        .expect("the sum succeeds")
        .cast(&mut backend, DType::F32)
        .expect("the cast succeeds")
        .download(&mut backend)
        .expect("the download succeeds")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&chunk| f32::from_le_bytes(chunk))
        .collect();
        assert_eq!(actual, vec![4096.0], "bf16 sum of ones");
    }

    #[test]
    fn argmax_returns_the_first_index_on_ties() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let actual: Vec<u32> = Tensor::upload(
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
        .expect("the download succeeds")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&chunk| u32::from_le_bytes(chunk))
        .collect();
        assert_eq!(actual, vec![1], "first of two maxima");
    }

    #[test]
    fn argmax_returns_the_first_index_on_ties_across_threads() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let actual: Vec<u32> = Tensor::upload(
            &mut backend,
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
            Shape::try_from([600].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .argmax(&mut backend, 0)
        .expect("the argmax succeeds")
        .download(&mut backend)
        .expect("the download succeeds")
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
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let sum = Tensor::upload(
            &mut backend,
            &[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid")
        .sum(&mut backend, 1)
        .expect("the sum succeeds");
        assert_eq!(sum.shape().dims(), &[3, 1], "transposed rows");
        let actual: Vec<f32> = sum
            .download(&mut backend)
            .expect("the download succeeds")
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
    fn sum_over_empty_axis_downloads_zeros() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let sum = Tensor::upload(
            &mut backend,
            &[],
            DType::F32,
            Shape::try_from([2, 0].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .sum(&mut backend, 1)
        .expect("the sum succeeds");
        assert_eq!(sum.shape().dims(), &[2, 1], "the axis is kept");
        let actual: Vec<f32> = sum
            .download(&mut backend)
            .expect("the download succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        assert_eq!(actual, vec![0.0, 0.0], "an empty sum is zero");
    }

    #[test]
    fn reduce_on_u32_is_rejected() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let bytes: Vec<u8> = (0_u32..3).flat_map(u32::to_le_bytes).collect();
        let indices = Tensor::upload(
            &mut backend,
            &bytes,
            DType::U32,
            Shape::try_from([3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds");
        assert!(
            matches!(
                indices.sum(&mut backend, 0).err(),
                Some(TensorError::Validation(CoreError::DTypeNotFloat))
            ),
            "reductions reject u32 on Metal"
        );
    }

    #[test]
    fn matmul_matches_hand_computed_values_for_each_float_dtype() {
        struct ValuesDimsCase<T0, T1> {
            values: T0,
            dims: T1,
        }
        struct DtypeToleranceCase<T0, T1> {
            dtype: T0,
            tolerance: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5_f32,
            },
            DtypeToleranceCase {
                dtype: DType::F16,
                tolerance: 1e-2,
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
                        values: vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
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
                    },
                );
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
    fn matmul_reads_transposed_right_operand_without_copy() {
        struct DtypeToleranceCase<T0, T1> {
            dtype: T0,
            tolerance: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5_f32,
            },
            DtypeToleranceCase {
                dtype: DType::F16,
                tolerance: 1e-2,
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
                let [x, weight] = [
                    [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [1.0, 0.0, -1.0, 2.0, 1.0, 0.0],
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
                    .cast(&mut backend, dtype)
                    .expect("the cast succeeds")
                });
                let weight_transposed = weight.permute(&[1, 0]).expect("the permutation is valid");
                assert!(
                    !weight_transposed.layout().is_contiguous()
                        && matmul_rule(dtype, x.layout(), dtype, weight_transposed.layout())
                            .is_ok_and(|spec| is_matmul_ready(spec.rhs(), spec.k(), spec.n())),
                    "a transposed weight is read in place"
                );
                let actual: Vec<f32> = x
                    .matmul(&mut backend, &weight_transposed)
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
                let expected = [-2.0_f32, 4.0, -2.0, 13.0];
                assert_eq!(actual.len(), expected.len(), "x . W^T: element count");
                actual.iter().zip(expected).for_each(|(&actual, expected)| {
                    assert!(
                        (actual - expected).abs() <= tolerance * expected.abs(),
                        "x . W^T for {dtype:?}: {actual} is not within {tolerance} of {expected}"
                    );
                });
            },
        );
    }

    #[test]
    fn matmul_reads_transposed_left_operand_without_copy() {
        struct DtypeToleranceCase<T0, T1> {
            dtype: T0,
            tolerance: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5_f32,
            },
            DtypeToleranceCase {
                dtype: DType::F16,
                tolerance: 1e-2,
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
                    [1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0],
                    [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                ]
                .map(|values| {
                    Tensor::upload(
                        &mut backend,
                        &values
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<u8>>(),
                        DType::F32,
                        Shape::try_from([3, 2].as_slice()).expect("the shape is valid"),
                    )
                    .expect("the upload succeeds")
                    .cast(&mut backend, dtype)
                    .expect("the cast succeeds")
                });
                let lhs_transposed = lhs.permute(&[1, 0]).expect("the permutation is valid");
                assert!(
                    matmul_rule(dtype, lhs_transposed.layout(), dtype, rhs.layout())
                        .is_ok_and(|spec| is_matmul_ready(spec.lhs(), spec.m(), spec.k())),
                    "a transposed lhs is read in place"
                );
                let actual: Vec<f32> = lhs_transposed
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
                let expected = [58.0_f32, 64.0, 139.0, 154.0];
                assert_eq!(actual.len(), expected.len(), "L^T x R: element count");
                actual.iter().zip(expected).for_each(|(&actual, expected)| {
                    assert!(
                        (actual - expected).abs() <= tolerance * expected.abs(),
                        "L^T x R for {dtype:?}: {actual} is not within {tolerance} of {expected}"
                    );
                });
            },
        );
    }

    #[test]
    fn batched_matmul_matches_sum_of_products() {
        struct ValuesDimsCase<T0, T1> {
            values: T0,
            dims: T1,
        }
        struct DtypeToleranceCase<T0, T1> {
            dtype: T0,
            tolerance: T1,
        }
        struct CountSeedCase<T0, T1> {
            count: T0,
            seed: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let [lhs_values, rhs_values] = [
            CountSeedCase {
                count: 2 * 3 * 40,
                seed: 7_u16,
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
                tolerance: 1e-5_f32,
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
                    },
                );
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
        struct ValuesDimsCase<T0, T1> {
            values: T0,
            dims: T1,
        }
        struct CountSeedCase<T0, T1> {
            count: T0,
            seed: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let [lhs_values, rhs_values] = [
            CountSeedCase {
                count: 37 * 20,
                seed: 3_u16,
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
            },
        );
        let actual: Vec<f32> = lhs
            .matmul(&mut backend, &rhs)
            .expect("the matmul succeeds")
            .download(&mut backend)
            .expect("the download succeeds")
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
        struct DtypeToleranceCase<T0, T1> {
            dtype: T0,
            tolerance: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
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
                tolerance: 1e-5_f32,
            },
            DtypeToleranceCase {
                dtype: DType::F16,
                tolerance: 1e-2,
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
                let rhs = Tensor::upload(
                    &mut backend,
                    &(0_u16..24)
                        .flat_map(|value| f32::from(value).to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from([2, 3, 4].as_slice()).expect("the shape is valid"),
                )
                .expect("the upload succeeds")
                .cast(&mut backend, dtype)
                .expect("the cast succeeds")
                .permute(&[2, 0, 1])
                .expect("the permutation is valid");
                let lhs = Tensor::upload(
                    &mut backend,
                    &lhs_values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from([4, 2, 2].as_slice()).expect("the shape is valid"),
                )
                .expect("the upload succeeds")
                .cast(&mut backend, dtype)
                .expect("the cast succeeds");
                assert!(
                    matmul_rule(dtype, lhs.layout(), dtype, rhs.layout())
                        .is_ok_and(|spec| !is_matmul_ready(spec.rhs(), spec.k(), spec.n())),
                    "strides (12, 4) on the last two axes need a copy"
                );
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
    fn matmul_reads_permuted_batch_axes_in_place() {
        struct CountDimsCase<T0, T1> {
            count: T0,
            dims: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let [lhs, rhs] = [
            CountDimsCase {
                count: 48_u16,
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
                Tensor::upload(
                    &mut backend,
                    &(0..count)
                        .flat_map(|value| f32::from(value).to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                )
                .expect("the upload succeeds")
            },
        );
        let lhs = lhs
            .permute(&[1, 0, 2, 3])
            .expect("the permutation is valid");
        let [expected, actual] = [
            lhs.contiguous(&mut backend).expect("the copy succeeds"),
            lhs,
        ]
        .map(|lhs| {
            lhs.matmul(&mut backend, &rhs)
                .expect("the matmul succeeds")
                .download(&mut backend)
                .expect("the download succeeds")
        });
        assert_eq!(
            actual, expected,
            "permuted batch strides match the contiguous copy"
        );
    }

    #[test]
    fn matmul_with_empty_output_downloads_nothing() {
        struct ValuesDimsCase<T0, T1> {
            values: T0,
            dims: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let [lhs, rhs] = [
            ValuesDimsCase {
                values: vec![],
                dims: vec![0, 3],
            },
            ValuesDimsCase {
                values: vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0],
                dims: vec![3, 2],
            },
        ]
        .map(
            |ValuesDimsCase {
                 values,
                 dims,
             }| {
                Tensor::upload(
                    &mut backend,
                    &values
                        .iter()
                        .flat_map(|value: &f32| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                )
                .expect("the upload succeeds")
            },
        );
        let output = lhs.matmul(&mut backend, &rhs).expect("the matmul succeeds");
        assert_eq!(output.shape().dims(), &[0, 2], "output shape is [m, n]");
        assert!(
            output
                .download(&mut backend)
                .expect("the download succeeds")
                .is_empty(),
            "an empty matmul downloads no values"
        );
    }

    #[test]
    fn gather_selects_rows_for_each_float_dtype() {
        struct DtypeToleranceCase<T0, T1> {
            dtype: T0,
            tolerance: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let indices = Tensor::upload(
            &mut backend,
            &[2_u32, 0]
                .iter()
                .flat_map(|index| index.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::U32,
            Shape::try_from([2].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds");
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5_f32,
            },
            DtypeToleranceCase {
                dtype: DType::F16,
                tolerance: 1e-2,
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
                let rows = Tensor::upload(
                    &mut backend,
                    &(0_u16..12)
                        .flat_map(|value| f32::from(value).to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from([3, 4].as_slice()).expect("the shape is valid"),
                )
                .expect("the upload succeeds")
                .cast(&mut backend, dtype)
                .expect("the cast succeeds")
                .gather(&mut backend, &indices)
                .expect("the gather succeeds");
                assert_eq!(
                    rows.shape().dims(),
                    &[2, 4],
                    "output shape is [indices, cols]"
                );
                let actual: Vec<f32> = rows
                    .cast(&mut backend, DType::F32)
                    .expect("the cast succeeds")
                    .download(&mut backend)
                    .expect("the download succeeds")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&chunk| f32::from_le_bytes(chunk))
                    .collect();
                let expected = [8.0_f32, 9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 3.0];
                assert_eq!(actual.len(), expected.len(), "gather: element count");
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
    fn gather_with_out_of_range_index_writes_a_zero_row() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let table = Tensor::upload(
            &mut backend,
            &[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([3, 2].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds");
        let indices = Tensor::upload(
            &mut backend,
            &[1_u32, 3]
                .iter()
                .flat_map(|index| index.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::U32,
            Shape::try_from([2].as_slice()).expect("the shape is valid"),
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
            [2.0, 3.0, 0.0, 0.0],
            "an index past the table gives zeros"
        );
    }

    #[test]
    fn gather_of_permuted_table_reads_strided_rows() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let table = Tensor::upload(
            &mut backend,
            &[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid");
        let indices = Tensor::upload(
            &mut backend,
            &[2_u32, 0]
                .iter()
                .flat_map(|index| index.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::U32,
            Shape::try_from([2].as_slice()).expect("the shape is valid"),
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
            [2.0, 5.0, 0.0, 3.0],
            "rows [2, 0] of the transposed table"
        );
    }

    #[test]
    fn concat_on_first_and_last_axis_for_each_float_dtype() {
        struct NameOutputDimsExpectedCase<T0, T1, T2, T3> {
            name: T0,
            output: T1,
            dims: T2,
            expected: T3,
        }
        struct ValuesDimsCase<T0, T1> {
            values: T0,
            dims: T1,
        }
        struct DtypeToleranceCase<T0, T1> {
            dtype: T0,
            tolerance: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        [
            DtypeToleranceCase {
                dtype: DType::F32,
                tolerance: 1e-5_f32,
            },
            DtypeToleranceCase {
                dtype: DType::F16,
                tolerance: 1e-2,
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
                        values: vec![0.0_f32, 1.0, 2.0],
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
                    },
                );
                [
                    NameOutputDimsExpectedCase {
                        name: "concat on axis 0",
                        output: top.concat(&mut backend, &bottom, 0),
                        dims: [3, 3],
                        expected: vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    },
                    NameOutputDimsExpectedCase {
                        name: "concat on the last axis",
                        output: left.concat(&mut backend, &right, 1),
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
                        let output = output.expect("the concat succeeds");
                        assert_eq!(output.shape().dims(), &dims, "{name}: output dims");
                        let actual: Vec<f32> = output
                            .cast(&mut backend, DType::F32)
                            .expect("the cast succeeds")
                            .download(&mut backend)
                            .expect("the download succeeds")
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
    fn concat_of_permuted_input_reads_strided_values() {
        struct ValuesDimsCase<T0, T1> {
            values: T0,
            dims: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let [lhs, rhs] = [
            ValuesDimsCase {
                values: vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0],
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
            },
        );
        let actual: Vec<f32> = lhs
            .permute(&[1, 0])
            .expect("the permutation is valid")
            .concat(&mut backend, &rhs, 0)
            .expect("the concat succeeds")
            .download(&mut backend)
            .expect("the download succeeds")
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
    fn concat_with_empty_input_keeps_the_other_input() {
        struct ValuesDimsCase<T0, T1> {
            values: T0,
            dims: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let [empty, rows] = [
            ValuesDimsCase {
                values: vec![],
                dims: vec![0, 3],
            },
            ValuesDimsCase {
                values: vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0],
                dims: vec![2, 3],
            },
        ]
        .map(
            |ValuesDimsCase {
                 values,
                 dims,
             }| {
                Tensor::upload(
                    &mut backend,
                    &values
                        .iter()
                        .flat_map(|value: &f32| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                )
                .expect("the upload succeeds")
            },
        );
        let output = empty
            .concat(&mut backend, &rows, 0)
            .expect("the concat succeeds");
        assert_eq!(
            output.shape().dims(),
            &[2, 3],
            "the empty input adds no rows"
        );
        let actual: Vec<f32> = output
            .download(&mut backend)
            .expect("the download succeeds")
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
    fn slice_update_writes_one_position_and_keeps_the_others_for_each_float_dtype() {
        struct DtypeToleranceCase<T0, T1> {
            dtype: T0,
            tolerance: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
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
                tolerance: 1e-5_f32,
            },
            DtypeToleranceCase {
                dtype: DType::F16,
                tolerance: 1e-2,
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
                    &mut backend,
                    dtype,
                    Shape::try_from([heads, positions, head_dim].as_slice())
                        .expect("the shape is valid"),
                )
                .expect("the zeros succeed");
                let update = Tensor::upload(
                    &mut backend,
                    &update_values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::F32,
                    Shape::try_from([heads, 1, head_dim].as_slice()).expect("the shape is valid"),
                )
                .expect("the upload succeeds")
                .cast(&mut backend, dtype)
                .expect("the cast succeeds");
                cache
                    .slice_update(&mut backend, &update, 1, position)
                    .expect("the slice update succeeds");
                let actual: Vec<f32> = cache
                    .cast(&mut backend, DType::F32)
                    .expect("the cast succeeds")
                    .download(&mut backend)
                    .expect("the download succeeds")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&chunk| f32::from_le_bytes(chunk))
                    .collect();
                assert_eq!(actual.len(), expected.len(), "slice update: element count");
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
    fn slice_update_of_permuted_update_reads_strided_values() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let mut target = Tensor::zeros(
            &mut backend,
            DType::F32,
            Shape::try_from([2, 4].as_slice()).expect("the shape is valid"),
        )
        .expect("the zeros succeed");
        let update = Tensor::upload(
            &mut backend,
            &[0.0_f32, 1.0, 2.0, 3.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 2].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid");
        target
            .slice_update(&mut backend, &update, 1, 1)
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
            [0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0],
            "the transposed update lands in columns 1 and 2"
        );
    }

    #[test]
    fn slice_update_of_broadcast_target_is_rejected() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let mut target = Tensor::zeros(
            &mut backend,
            DType::F32,
            Shape::try_from([1, 4].as_slice()).expect("the shape is valid"),
        )
        .expect("the zeros succeed")
        .broadcast_as(Shape::try_from([3, 4].as_slice()).expect("the shape is valid"))
        .expect("the broadcast is valid");
        let update = Tensor::upload(
            &mut backend,
            &[0.0_f32, 1.0, 2.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([3, 1].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds");
        assert!(
            matches!(
                target.slice_update(&mut backend, &update, 1, 0),
                Err(TensorError::Validation(CoreError::SliceUpdateNonContiguous))
            ),
            "a broadcast target would be written by several threads"
        );
    }

    #[test]
    fn slice_update_with_a_view_of_the_target_as_update_is_rejected() {
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let mut target = Tensor::upload(
            &mut backend,
            &(0_u16..8)
                .flat_map(|value| f32::from(value).to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 4].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds");
        let update = target.narrow(1, 0, 1).expect("the narrow is valid");
        let result = target.slice_update(&mut backend, &update, 1, 3);
        assert!(
            matches!(
                result.err(),
                Some(TensorError::Validation(CoreError::SharedStorage))
            ),
            "an update aliasing the target is rejected"
        );
    }

    #[test]
    fn slice_update_while_another_view_of_the_target_is_alive_is_rejected() {
        struct ValuesDimsCase<T0, T1> {
            values: T0,
            dims: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let [mut target, update] = [
            ValuesDimsCase {
                values: vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
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
            },
        );
        let view = target.narrow(0, 0, 1).expect("the narrow is valid");
        let result = target.slice_update(&mut backend, &update, 1, 3);
        assert!(
            matches!(
                result.err(),
                Some(TensorError::Validation(CoreError::SharedStorage))
            ),
            "a live view would observe the write"
        );
        assert_eq!(view.shape().dims(), &[1, 4], "the view is still alive");
    }

    #[test]
    fn slice_update_without_live_views_writes_the_update() {
        struct ValuesDimsCase<T0, T1> {
            values: T0,
            dims: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let [mut target, update] = [
            ValuesDimsCase {
                values: vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
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
            },
        );
        target
            .slice_update(&mut backend, &update, 1, 3)
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
            [0.0, 1.0, 2.0, 100.0, 4.0, 5.0, 6.0, 101.0],
            "the update lands in the last column"
        );
    }

    #[test]
    fn download_of_storage_from_another_backend_is_rejected() {
        let mut producer = MetalBackend::new().expect("a Metal device is available");
        let mut consumer = MetalBackend::new().expect("a Metal device is available");
        let shape = Shape::try_from(&[2_usize][..]).expect("the shape is valid");
        let tensor = Tensor::upload(&mut producer, &[0_u8; 8], DType::F32, shape)
            .expect("the upload succeeds");
        assert!(
            matches!(
                tensor.download(&mut consumer).err(),
                Some(TensorError::Backend {
                    operation: "download",
                    source: MetalError::ForeignStorage
                })
            ),
            "a storage from another backend cannot be downloaded"
        );
    }

    #[test]
    fn unary_of_storage_from_another_backend_is_rejected() {
        let mut producer = MetalBackend::new().expect("a Metal device is available");
        let mut consumer = MetalBackend::new().expect("a Metal device is available");
        let shape = Shape::try_from(&[2_usize][..]).expect("the shape is valid");
        let tensor = Tensor::upload(&mut producer, &[0_u8; 8], DType::F32, shape)
            .expect("the upload succeeds");
        assert!(
            matches!(
                tensor.neg(&mut consumer).err(),
                Some(TensorError::Backend {
                    operation: "unary",
                    source: MetalError::ForeignStorage
                })
            ),
            "a storage from another backend cannot be read by a kernel"
        );
    }

    #[test]
    fn slice_update_of_target_from_another_backend_is_rejected() {
        let mut producer = MetalBackend::new().expect("a Metal device is available");
        let mut consumer = MetalBackend::new().expect("a Metal device is available");
        let mut target = Tensor::zeros(
            &mut producer,
            DType::F32,
            Shape::try_from(&[2_usize, 2][..]).expect("the shape is valid"),
        )
        .expect("the zeros succeed");
        let update = Tensor::zeros(
            &mut consumer,
            DType::F32,
            Shape::try_from(&[2_usize, 1][..]).expect("the shape is valid"),
        )
        .expect("the zeros succeed");
        assert!(
            matches!(
                target.slice_update(&mut consumer, &update, 1, 0),
                Err(TensorError::Backend {
                    operation: "slice_update",
                    source: MetalError::ForeignStorage
                })
            ),
            "a target from another backend cannot be written"
        );
    }

    #[test]
    fn gather_and_concat_on_u32_are_rejected() {
        struct ValuesDimsCase<T0, T1> {
            values: T0,
            dims: T1,
        }
        let mut backend = MetalBackend::new().expect("a Metal device is available");
        let [values, indices] = [
            ValuesDimsCase {
                values: vec![0_u32, 1, 2, 3],
                dims: vec![2, 2],
            },
            ValuesDimsCase {
                values: vec![0],
                dims: vec![1],
            },
        ]
        .map(
            |ValuesDimsCase {
                 values,
                 dims,
             }| {
                Tensor::upload(
                    &mut backend,
                    &values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    DType::U32,
                    Shape::try_from(dims.as_slice()).expect("the shape is valid"),
                )
                .expect("the upload succeeds")
            },
        );
        assert!(
            matches!(
                values.gather(&mut backend, &indices).err(),
                Some(TensorError::Validation(CoreError::DTypeNotFloat))
            ),
            "gather rejects a u32 table on Metal"
        );
        assert!(
            matches!(
                values.concat(&mut backend, &values, 0).err(),
                Some(TensorError::Validation(CoreError::DTypeNotFloat))
            ),
            "concat rejects u32 on Metal"
        );
    }
}
