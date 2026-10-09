mod error;

use std::collections::HashMap;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{
    NSRange,
    NSString,
};
use objc2_metal::{
    MTLBlitCommandEncoder,
    MTLBuffer,
    MTLCommandBuffer,
    MTLCommandBufferStatus,
    MTLCommandEncoder,
    MTLCommandQueue,
    MTLComputeCommandEncoder,
    MTLComputePipelineState,
    MTLCreateSystemDefaultDevice,
    MTLDevice,
    MTLLibrary,
    MTLResourceOptions,
    MTLSize,
};

pub use self::error::MetalError;
use crate::core::primitive::{
    BinaryOp,
    ReduceOp,
    UnaryOp,
};
use crate::core::{
    Backend,
    DType,
    Operand,
    RANK_MAX,
    Shape,
};

const KERNEL_SOURCES: &[&str] = &[
    include_str!("kernels/common.metal"),
    include_str!("kernels/copy.metal"),
    include_str!("kernels/unary.metal"),
    include_str!("kernels/binary.metal"),
];

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub struct MetalStorage {
    buffer: Buffer,
    byte_len: usize,
}

pub struct MetalBackend {
    device: Device,
    queue: Queue,
    library: Library,
    pipelines: HashMap<&'static str, Pipeline>,
    pending: Option<CommandBuffer>,
}

#[repr(C)]
struct StridedArgs {
    count: u32,
    rank: u32,
    dims: [u32; RANK_MAX],
    strides: [u32; RANK_MAX],
    offset: u32,
}

impl StridedArgs {
    fn new(input: &Operand<'_, MetalStorage>) -> Result<Self, MetalError> {
        let elements_buffer = input
            .storage
            .byte_len
            .checked_div(input.dtype.size_bytes())
            .unwrap_or(0);
        index_u32(elements_buffer)?;
        let shape = input.layout.shape();
        Ok(Self {
            count: index_u32(shape.element_count())?,
            rank: index_u32(shape.rank())?,
            dims: padded_u32(shape.dims())?,
            strides: padded_u32(input.layout.strides())?,
            offset: index_u32(input.layout.offset())?,
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
            .map_err(|error| MetalError::LibraryCompilation {
                message: error.localizedDescription().to_string(),
            })?;
        Ok(Self {
            device,
            queue,
            library,
            pipelines: HashMap::new(),
            pending: None,
        })
    }

    fn allocate(
        &self,
        dtype: DType,
        shape: &Shape,
    ) -> Result<MetalStorage, MetalError> {
        let byte_len = byte_len(dtype, shape)?;
        let bytes_max = self.device.maxBufferLength();
        if byte_len > bytes_max {
            return Err(MetalError::BufferTooLarge {
                bytes: byte_len,
                bytes_max,
            });
        }
        let buffer = self
            .device
            .newBufferWithLength_options(byte_len.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAllocation {
                bytes: byte_len,
            })?;
        Ok(MetalStorage {
            buffer,
            byte_len,
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
            .ok_or(MetalError::KernelNotFound {
                name,
            })?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| MetalError::PipelineCreation {
                name,
                message: error.localizedDescription().to_string(),
            })?;
        self.pipelines.insert(name, pipeline.clone());
        Ok(pipeline)
    }

    fn dispatch_strided(
        &mut self,
        kernel: &'static str,
        inputs: &[&Operand<'_, MetalStorage>],
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
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreation)?;
        encoder.setComputePipelineState(&pipeline);
        inputs.iter().enumerate().for_each(|(index, input)| {
            // SAFETY: the input buffer is alive for the whole call and the
            // command buffer retains it; `StridedArgs::new` checked that every
            // index the kernel reads fits in the buffer's element count.
            unsafe { encoder.setBuffer_offset_atIndex(Some(&input.storage.buffer), 0, index) };
        });
        // SAFETY: the output buffer holds `count` elements of the kernel's
        // output type, one per dispatched thread.
        unsafe { encoder.setBuffer_offset_atIndex(Some(&output.buffer), 0, output_index) };
        args.iter()
            .zip(output_index.saturating_add(1)..)
            .for_each(|(input_args, index)| {
                // SAFETY: `input_args` is a `#[repr(C)]` struct of `u32` that
                // matches the layout of `StridedArgs` in `common.metal`; Metal
                // copies the bytes before the call returns.
                unsafe {
                    encoder.setBytes_length_atIndex(
                        NonNull::from(input_args).cast(),
                        size_of::<StridedArgs>(),
                        index,
                    )
                };
            });
        let width = pipeline.maxTotalThreadsPerThreadgroup().min(count);
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
        encoder.endEncoding();
        command_buffer.commit();
        self.pending = Some(command_buffer);
        Ok(())
    }

    fn wait_pending(&mut self) -> Result<(), MetalError> {
        self.pending
            .take()
            .map_or(Ok(()), |command_buffer| wait(&command_buffer))
    }
}

impl Backend for MetalBackend {
    type Error = MetalError;
    type Storage = MetalStorage;

    fn upload(
        &mut self,
        bytes: &[u8],
        dtype: DType,
        shape: &Shape,
    ) -> Result<MetalStorage, MetalError> {
        let storage = self.allocate(dtype, shape)?;
        if bytes.len() != storage.byte_len {
            return Err(MetalError::ByteLengthMismatch {
                bytes: bytes.len(),
                bytes_expected: storage.byte_len,
            });
        }
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
        dtype: DType,
        shape: &Shape,
    ) -> Result<MetalStorage, MetalError> {
        let storage = self.allocate(dtype, shape)?;
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;
        let encoder = command_buffer
            .blitCommandEncoder()
            .ok_or(MetalError::BlitEncoderCreation)?;
        encoder.fillBuffer_range_value(
            &storage.buffer,
            NSRange::new(0, storage.buffer.length()),
            0,
        );
        encoder.endEncoding();
        command_buffer.commit();
        self.pending = Some(command_buffer);
        Ok(storage)
    }

    fn download(
        &mut self,
        input: Operand<'_, MetalStorage>,
    ) -> Result<Vec<u8>, MetalError> {
        if !input.layout.is_contiguous() {
            return Err(MetalError::DownloadNonContiguous);
        }
        let shape = input.layout.shape();
        let bytes = byte_len(input.dtype, shape)?;
        let bytes_offset = input
            .layout
            .offset()
            .checked_mul(input.dtype.size_bytes())
            .ok_or_else(|| MetalError::ByteCountOverflow {
                dims: shape.dims().to_vec(),
            })?;
        let bytes_len = input.storage.byte_len;
        let out_of_bounds = MetalError::DownloadOutOfBounds {
            bytes,
            bytes_offset,
            bytes_len,
        };
        match bytes_offset.checked_add(bytes) {
            Some(end) if end <= bytes_len => {}
            Some(_) | None => return Err(out_of_bounds),
        }
        self.wait_pending()?;
        let mut output = vec![0_u8; bytes];
        // SAFETY: the buffer uses shared storage, `bytes_offset + bytes <=
        // byte_len` was checked above, `output` holds `bytes` bytes,
        // and every command buffer that writes the buffer has completed
        // because the queue is serial and the last committed one was
        // waited on.
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
        input: Operand<'_, MetalStorage>,
    ) -> Result<MetalStorage, MetalError> {
        let kernel = unary_kernel(op, input.dtype)?;
        let output = self.allocate(input.dtype, input.layout.shape())?;
        self.dispatch_strided(kernel, &[&input], &output)?;
        Ok(output)
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        lhs: Operand<'_, MetalStorage>,
        rhs: Operand<'_, MetalStorage>,
    ) -> Result<MetalStorage, MetalError> {
        let kernel = binary_kernel(op, lhs.dtype)?;
        let output = self.allocate(lhs.dtype, lhs.layout.shape())?;
        self.dispatch_strided(kernel, &[&lhs, &rhs], &output)?;
        Ok(output)
    }

    fn reduce(
        &mut self,
        _op: ReduceOp,
        _input: Operand<'_, MetalStorage>,
        _axis: usize,
    ) -> Result<MetalStorage, MetalError> {
        Err(MetalError::NotImplemented {
            primitive: "reduce",
        })
    }

    fn matmul(
        &mut self,
        _lhs: Operand<'_, MetalStorage>,
        _rhs: Operand<'_, MetalStorage>,
    ) -> Result<MetalStorage, MetalError> {
        Err(MetalError::NotImplemented {
            primitive: "matmul",
        })
    }

    fn copy(
        &mut self,
        input: Operand<'_, MetalStorage>,
    ) -> Result<MetalStorage, MetalError> {
        let output = self.allocate(input.dtype, input.layout.shape())?;
        self.dispatch_strided(copy_kernel(input.dtype), &[&input], &output)?;
        Ok(output)
    }

    fn cast(
        &mut self,
        input: Operand<'_, MetalStorage>,
        dtype: DType,
    ) -> Result<MetalStorage, MetalError> {
        let kernel = cast_kernel(input.dtype, dtype)?;
        let output = self.allocate(dtype, input.layout.shape())?;
        self.dispatch_strided(kernel, &[&input], &output)?;
        Ok(output)
    }

    fn gather(
        &mut self,
        _table: Operand<'_, MetalStorage>,
        _indices: Operand<'_, MetalStorage>,
    ) -> Result<MetalStorage, MetalError> {
        Err(MetalError::NotImplemented {
            primitive: "gather",
        })
    }

    fn concat(
        &mut self,
        _lhs: Operand<'_, MetalStorage>,
        _rhs: Operand<'_, MetalStorage>,
        _axis: usize,
    ) -> Result<MetalStorage, MetalError> {
        Err(MetalError::NotImplemented {
            primitive: "concat",
        })
    }

    fn slice_update(
        &mut self,
        _target: Operand<'_, MetalStorage>,
        _update: Operand<'_, MetalStorage>,
        _axis: usize,
        _start: usize,
    ) -> Result<(), MetalError> {
        Err(MetalError::NotImplemented {
            primitive: "slice_update",
        })
    }
}

fn byte_len(
    dtype: DType,
    shape: &Shape,
) -> Result<usize, MetalError> {
    shape
        .element_count()
        .checked_mul(dtype.size_bytes())
        .ok_or_else(|| MetalError::ByteCountOverflow {
            dims: shape.dims().to_vec(),
        })
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
    dtype_from: DType,
    dtype_to: DType,
) -> Result<&'static str, MetalError> {
    let unsupported = |dtype| MetalError::UnsupportedDType {
        primitive: "cast",
        dtype,
    };
    match (dtype_from, dtype_to) {
        (DType::F32, DType::F32) => Ok("cast_f32_f32"),
        (DType::F32, DType::F16) => Ok("cast_f32_f16"),
        (DType::F32, DType::BF16) => Ok("cast_f32_bf16"),
        (DType::F16, DType::F32) => Ok("cast_f16_f32"),
        (DType::F16, DType::F16) => Ok("cast_f16_f16"),
        (DType::F16, DType::BF16) => Ok("cast_f16_bf16"),
        (DType::BF16, DType::F32) => Ok("cast_bf16_f32"),
        (DType::BF16, DType::F16) => Ok("cast_bf16_f16"),
        (DType::BF16, DType::BF16) => Ok("cast_bf16_bf16"),
        (DType::U32, _) => Err(unsupported(DType::U32)),
        (_, DType::U32) => Err(unsupported(DType::U32)),
    }
}

fn unary_kernel(
    op: UnaryOp,
    dtype: DType,
) -> Result<&'static str, MetalError> {
    match (op, dtype) {
        (UnaryOp::Neg, DType::F32) => Ok("unary_neg_f32"),
        (UnaryOp::Neg, DType::F16) => Ok("unary_neg_f16"),
        (UnaryOp::Neg, DType::BF16) => Ok("unary_neg_bf16"),
        (UnaryOp::Exp, DType::F32) => Ok("unary_exp_f32"),
        (UnaryOp::Exp, DType::F16) => Ok("unary_exp_f16"),
        (UnaryOp::Exp, DType::BF16) => Ok("unary_exp_bf16"),
        (UnaryOp::Sqrt, DType::F32) => Ok("unary_sqrt_f32"),
        (UnaryOp::Sqrt, DType::F16) => Ok("unary_sqrt_f16"),
        (UnaryOp::Sqrt, DType::BF16) => Ok("unary_sqrt_bf16"),
        (UnaryOp::Recip, DType::F32) => Ok("unary_recip_f32"),
        (UnaryOp::Recip, DType::F16) => Ok("unary_recip_f16"),
        (UnaryOp::Recip, DType::BF16) => Ok("unary_recip_bf16"),
        (_, DType::U32) => Err(MetalError::UnsupportedDType {
            primitive: "unary",
            dtype,
        }),
    }
}

fn binary_kernel(
    op: BinaryOp,
    dtype: DType,
) -> Result<&'static str, MetalError> {
    match (op, dtype) {
        (BinaryOp::Add, DType::F32) => Ok("binary_add_f32"),
        (BinaryOp::Add, DType::F16) => Ok("binary_add_f16"),
        (BinaryOp::Add, DType::BF16) => Ok("binary_add_bf16"),
        (BinaryOp::Sub, DType::F32) => Ok("binary_sub_f32"),
        (BinaryOp::Sub, DType::F16) => Ok("binary_sub_f16"),
        (BinaryOp::Sub, DType::BF16) => Ok("binary_sub_bf16"),
        (BinaryOp::Mul, DType::F32) => Ok("binary_mul_f32"),
        (BinaryOp::Mul, DType::F16) => Ok("binary_mul_f16"),
        (BinaryOp::Mul, DType::BF16) => Ok("binary_mul_bf16"),
        (BinaryOp::Div, DType::F32) => Ok("binary_div_f32"),
        (BinaryOp::Div, DType::F16) => Ok("binary_div_f16"),
        (BinaryOp::Div, DType::BF16) => Ok("binary_div_bf16"),
        (_, DType::U32) => Err(MetalError::UnsupportedDType {
            primitive: "binary",
            dtype,
        }),
    }
}

fn index_u32(value: usize) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| MetalError::IndexTooLarge {
        value,
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

fn wait(command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
    command_buffer.waitUntilCompleted();
    match command_buffer.status() {
        MTLCommandBufferStatus::Completed => Ok(()),
        status => Err(MetalError::CommandBufferFailed {
            message: command_buffer.error().map_or_else(
                || format!("status {status:?}"),
                |error| error.localizedDescription().to_string(),
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use crate::backend::metal::{
        MetalBackend,
        MetalError,
    };
    use crate::core::{
        DType,
        Shape,
        Tensor,
    };

    fn backend() -> MetalBackend { MetalBackend::new().expect("a Metal device is available") }

    fn shape(dims: &[usize]) -> Shape { Shape::try_from(dims).expect("the shape is valid") }

    fn round_trip(
        backend: &mut MetalBackend,
        bytes: &[u8],
        dtype: DType,
        dims: &[usize],
    ) -> Vec<u8> {
        Tensor::upload(backend, bytes, dtype, shape(dims))
            .expect("the upload succeeds")
            .download(backend)
            .expect("the download succeeds")
    }

    #[test]
    fn f32_round_trip_returns_the_same_bytes() {
        let bytes: Vec<u8> = [1.0_f32, -2.5, 1.5e-3, f32::MIN_POSITIVE, -0.0, f32::MAX]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let downloaded = round_trip(&mut backend(), &bytes, DType::F32, &[2, 3]);
        assert_eq!(downloaded, bytes, "f32 bytes survive the round trip");
    }

    #[test]
    fn f16_round_trip_returns_the_same_bytes() {
        let bytes: Vec<u8> = [0x3C00_u16, 0xC100, 0x4248, 0x0001, 0x8000, 0x7BFF]
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect();
        let downloaded = round_trip(&mut backend(), &bytes, DType::F16, &[3, 2]);
        assert_eq!(downloaded, bytes, "f16 bytes survive the round trip");
    }

    #[test]
    fn bf16_round_trip_returns_the_same_bytes() {
        let bytes: Vec<u8> = [0x3F80_u16, 0xC020, 0x4049, 0x0001, 0x8000, 0x7F7F]
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect();
        let downloaded = round_trip(&mut backend(), &bytes, DType::BF16, &[6]);
        assert_eq!(downloaded, bytes, "bf16 bytes survive the round trip");
    }

    #[test]
    fn u32_round_trip_returns_the_same_bytes() {
        let bytes: Vec<u8> = [0_u32, 1, 128_000, u32::MAX]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let downloaded = round_trip(&mut backend(), &bytes, DType::U32, &[2, 1, 2]);
        assert_eq!(downloaded, bytes, "u32 bytes survive the round trip");
    }

    #[test]
    fn empty_round_trip_returns_no_bytes() {
        let downloaded = round_trip(&mut backend(), &[], DType::F32, &[0, 3]);
        assert!(downloaded.is_empty(), "an empty tensor downloads no bytes");
    }

    #[test]
    fn zeros_download_only_zero_bytes() {
        let mut backend = backend();
        [DType::F32, DType::F16, DType::BF16, DType::U32]
            .into_iter()
            .for_each(|dtype| {
                let downloaded = Tensor::zeros(&mut backend, dtype, shape(&[2, 3]))
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
        let mut backend = backend();
        let bytes: Vec<u8> = [7_u32, 8, 9]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let uploaded = Tensor::upload(&mut backend, &bytes, DType::U32, shape(&[3]))
            .expect("the upload succeeds");
        let zeros = Tensor::zeros(&mut backend, DType::U32, shape(&[3])).expect("zeros succeeds");
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
        let result = Tensor::upload(&mut backend(), &[0; 10], DType::F32, shape(&[2, 2]));
        assert!(
            matches!(
                result,
                Err(MetalError::ByteLengthMismatch {
                    bytes: 10,
                    bytes_expected: 16
                })
            ),
            "a byte length that does not match shape × dtype is rejected"
        );
    }

    #[test]
    fn narrow_on_leading_axis_downloads_the_sub_range() {
        let mut backend = backend();
        let bytes: Vec<u8> = (0_u32..6).flat_map(u32::to_le_bytes).collect();
        let narrowed = Tensor::upload(&mut backend, &bytes, DType::U32, shape(&[3, 2]))
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
        let mut backend = backend();
        let bytes: Vec<u8> = (0_u32..6).flat_map(u32::to_le_bytes).collect();
        let permuted = Tensor::upload(&mut backend, &bytes, DType::U32, shape(&[3, 2]))
            .expect("the upload succeeds")
            .permute(&[1, 0])
            .expect("the permutation is valid");
        assert!(
            matches!(
                permuted.download(&mut backend),
                Err(MetalError::DownloadNonContiguous)
            ),
            "a non-contiguous layout cannot be downloaded"
        );
    }

    fn upload_f32(
        backend: &mut MetalBackend,
        values: &[f32],
        dims: &[usize],
    ) -> Tensor<MetalBackend> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        Tensor::upload(backend, &bytes, DType::F32, shape(dims)).expect("the upload succeeds")
    }

    fn download_f32(
        backend: &mut MetalBackend,
        tensor: &Tensor<MetalBackend>,
    ) -> Vec<f32> {
        tensor
            .download(backend)
            .expect("the download succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect()
    }

    fn download_u16(
        backend: &mut MetalBackend,
        tensor: &Tensor<MetalBackend>,
    ) -> Vec<u16> {
        tensor
            .download(backend)
            .expect("the download succeeds")
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&chunk| u16::from_le_bytes(chunk))
            .collect()
    }

    fn iota(count: u16) -> Vec<f32> { (0..count).map(f32::from).collect() }

    #[test]
    fn copy_of_permuted_tensor_is_contiguous_for_each_float_dtype() {
        let mut backend = backend();
        let expected: Vec<f32> = [
            0_u16, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19,
            23,
        ]
        .into_iter()
        .map(f32::from)
        .collect();
        let source = upload_f32(&mut backend, &iota(24), &[2, 3, 4]);
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
                let values = copied
                    .cast(&mut backend, DType::F32)
                    .expect("the cast succeeds");
                assert_eq!(
                    download_f32(&mut backend, &values),
                    expected,
                    "permuted values for {dtype:?}"
                );
            });
    }

    #[test]
    fn copy_of_narrowed_tensor_keeps_the_sub_range() {
        let mut backend = backend();
        let copied = upload_f32(&mut backend, &iota(24), &[2, 3, 4])
            .narrow(1, 1, 2)
            .expect("the narrow is valid")
            .contiguous(&mut backend)
            .expect("the copy succeeds");
        let expected: Vec<f32> = (4_u16..12).chain(16..24).map(f32::from).collect();
        assert_eq!(copied.shape().dims(), &[2, 2, 4], "narrowed dims");
        assert_eq!(
            download_f32(&mut backend, &copied),
            expected,
            "narrowed values"
        );
    }

    #[test]
    fn copy_of_broadcast_tensor_repeats_values() {
        let mut backend = backend();
        let copied = upload_f32(&mut backend, &[1.0, 2.0, 3.0], &[3, 1])
            .broadcast_as(shape(&[2, 3, 4]))
            .expect("the broadcast is valid")
            .contiguous(&mut backend)
            .expect("the copy succeeds");
        let block: [f32; 12] = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0];
        let expected: Vec<f32> = block.iter().chain(block.iter()).copied().collect();
        assert_eq!(
            download_f32(&mut backend, &copied),
            expected,
            "broadcast values"
        );
    }

    #[test]
    fn copy_of_permuted_u32_tensor_is_contiguous() {
        let mut backend = backend();
        let bytes: Vec<u8> = (0_u32..6).flat_map(u32::to_le_bytes).collect();
        let copied = Tensor::upload(&mut backend, &bytes, DType::U32, shape(&[3, 2]))
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
        let mut backend = backend();
        let copied = upload_f32(&mut backend, &[], &[0, 3])
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
        let mut backend = backend();
        let round_tripped = upload_f32(&mut backend, &[1.0, -2.5, PI], &[3])
            .cast(&mut backend, DType::BF16)
            .expect("the cast succeeds")
            .cast(&mut backend, DType::F32)
            .expect("the cast succeeds");
        assert_eq!(
            download_f32(&mut backend, &round_tripped),
            vec![1.0, -2.5, 3.140625],
            "bf16 round trip"
        );
    }

    #[test]
    fn cast_f32_to_bf16_rounds_to_nearest_even() {
        let mut backend = backend();
        let values: Vec<f32> = [0x3F80_8000_u32, 0x3F81_8000, 0x3F80_8001, 0xBF80_8000]
            .into_iter()
            .map(f32::from_bits)
            .collect();
        let cast = upload_f32(&mut backend, &values, &[4])
            .cast(&mut backend, DType::BF16)
            .expect("the cast succeeds");
        assert_eq!(
            download_u16(&mut backend, &cast),
            vec![0x3F80, 0x3F82, 0x3F81, 0xBF80],
            "ties round to even, others to nearest"
        );
    }

    #[test]
    fn cast_f32_to_f16_and_back_rounds_known_values() {
        let mut backend = backend();
        let cast = upload_f32(&mut backend, &[1.0, -2.5, 0.5, PI], &[4])
            .cast(&mut backend, DType::F16)
            .expect("the cast succeeds");
        assert_eq!(
            download_u16(&mut backend, &cast),
            vec![0x3C00, 0xC100, 0x3800, 0x4248],
            "f16 bits"
        );
        let round_tripped = cast
            .cast(&mut backend, DType::F32)
            .expect("the cast succeeds");
        assert_eq!(
            download_f32(&mut backend, &round_tripped),
            vec![1.0, -2.5, 0.5, 3.140625],
            "f16 round trip"
        );
    }

    #[test]
    fn cast_of_permuted_tensor_reads_strided_input() {
        let mut backend = backend();
        let cast = upload_f32(&mut backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .permute(&[1, 0])
            .expect("the permutation is valid")
            .cast(&mut backend, DType::BF16)
            .expect("the cast succeeds");
        assert_eq!(cast.shape().dims(), &[3, 2], "transposed dims");
        assert_eq!(
            download_u16(&mut backend, &cast),
            vec![0x3F80, 0x4080, 0x4000, 0x40A0, 0x4040, 0x40C0],
            "transposed bf16 bits of 1, 4, 2, 5, 3, 6"
        );
    }

    type UnaryFn =
        fn(&Tensor<MetalBackend>, &mut MetalBackend) -> Result<Tensor<MetalBackend>, MetalError>;

    type BinaryFn = fn(
        &Tensor<MetalBackend>,
        &mut MetalBackend,
        &Tensor<MetalBackend>,
    ) -> Result<Tensor<MetalBackend>, MetalError>;

    const FLOAT_DTYPES: [DType; 3] = [DType::F32, DType::F16, DType::BF16];

    fn tolerance_relative(dtype: DType) -> f32 {
        match dtype {
            DType::F32 => 1e-5,
            DType::F16 | DType::BF16 | DType::U32 => 1e-2,
        }
    }

    fn assert_close(
        actual: &[f32],
        expected: &[f32],
        dtype: DType,
        context: &str,
    ) {
        assert_eq!(actual.len(), expected.len(), "{context}: element count");
        let tolerance = tolerance_relative(dtype);
        actual
            .iter()
            .zip(expected)
            .for_each(|(&actual, &expected)| {
                assert!(
                    (actual - expected).abs() <= tolerance * expected.abs().max(f32::MIN_POSITIVE),
                    "{context} for {dtype:?}: {actual} is not within {tolerance} of {expected}"
                );
            });
    }

    fn unary_in(
        backend: &mut MetalBackend,
        op: UnaryFn,
        values: &[f32],
        dtype: DType,
    ) -> Vec<f32> {
        let input = upload_f32(backend, values, &[values.len()])
            .cast(backend, dtype)
            .expect("the cast succeeds");
        let output = op(&input, backend)
            .expect("the op succeeds")
            .cast(backend, DType::F32)
            .expect("the cast succeeds");
        download_f32(backend, &output)
    }

    #[test]
    fn unary_ops_match_hand_computed_values_for_each_float_dtype() {
        let mut backend = backend();
        let cases: [(&str, UnaryFn, [f32; 3], [f32; 3]); 4] = [
            ("neg", Tensor::neg, [1.0, -2.5, 0.0], [-1.0, 2.5, 0.0]),
            (
                "exp",
                Tensor::exp,
                [0.0, 1.0, -1.0],
                [1.0, std::f32::consts::E, 1.0 / std::f32::consts::E],
            ),
            ("sqrt", Tensor::sqrt, [4.0, 2.25, 0.0], [2.0, 1.5, 0.0]),
            ("recip", Tensor::recip, [2.0, -4.0, 0.5], [0.5, -0.25, 2.0]),
        ];
        cases.into_iter().for_each(|(name, op, values, expected)| {
            FLOAT_DTYPES.into_iter().for_each(|dtype| {
                let actual = unary_in(&mut backend, op, &values, dtype);
                assert_close(&actual, &expected, dtype, name);
            });
        });
    }

    #[test]
    fn binary_ops_match_hand_computed_values_for_each_float_dtype() {
        let mut backend = backend();
        let lhs_values = [1.0_f32, 2.0, 3.0, 4.0];
        let rhs_values = [2.0_f32, 4.0, 0.5, -1.0];
        let cases: [(&str, BinaryFn, [f32; 4]); 4] = [
            ("add", Tensor::add, [3.0, 6.0, 3.5, 3.0]),
            ("sub", Tensor::sub, [-1.0, -2.0, 2.5, 5.0]),
            ("mul", Tensor::mul, [2.0, 8.0, 1.5, -4.0]),
            ("div", Tensor::div, [0.5, 0.5, 6.0, -4.0]),
        ];
        cases.into_iter().for_each(|(name, op, expected)| {
            FLOAT_DTYPES.into_iter().for_each(|dtype| {
                let lhs = upload_f32(&mut backend, &lhs_values, &[4])
                    .cast(&mut backend, dtype)
                    .expect("the cast succeeds");
                let rhs = upload_f32(&mut backend, &rhs_values, &[4])
                    .cast(&mut backend, dtype)
                    .expect("the cast succeeds");
                let output = op(&lhs, &mut backend, &rhs)
                    .expect("the op succeeds")
                    .cast(&mut backend, DType::F32)
                    .expect("the cast succeeds");
                assert_close(&download_f32(&mut backend, &output), &expected, dtype, name);
            });
        });
    }

    #[test]
    fn add_broadcasts_a_row_over_a_matrix() {
        let mut backend = backend();
        let matrix = upload_f32(&mut backend, &iota(6), &[2, 3]);
        let row = upload_f32(&mut backend, &[10.0, 20.0, 30.0], &[3]);
        let sum = matrix.add(&mut backend, &row).expect("the add succeeds");
        assert_eq!(sum.shape().dims(), &[2, 3], "broadcast dims");
        assert_eq!(
            download_f32(&mut backend, &sum),
            vec![10.0, 21.0, 32.0, 13.0, 24.0, 35.0],
            "row added to each matrix row"
        );
    }

    #[test]
    fn neg_of_permuted_tensor_reads_strided_input() {
        let mut backend = backend();
        let negated = upload_f32(&mut backend, &iota(6), &[2, 3])
            .permute(&[1, 0])
            .expect("the permutation is valid")
            .neg(&mut backend)
            .expect("the neg succeeds");
        assert_eq!(negated.shape().dims(), &[3, 2], "transposed dims");
        assert_eq!(
            download_f32(&mut backend, &negated),
            vec![-0.0, -3.0, -1.0, -4.0, -2.0, -5.0],
            "negated transposed values"
        );
    }

    #[test]
    fn mul_of_permuted_and_contiguous_operands_reads_both_layouts() {
        let mut backend = backend();
        let transposed = upload_f32(&mut backend, &iota(6), &[3, 2])
            .permute(&[1, 0])
            .expect("the permutation is valid");
        let contiguous = upload_f32(&mut backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let product = transposed
            .mul(&mut backend, &contiguous)
            .expect("the mul succeeds");
        assert_eq!(
            download_f32(&mut backend, &product),
            vec![0.0, 4.0, 12.0, 4.0, 15.0, 30.0],
            "products of [[0, 2, 4], [1, 3, 5]] and [[1, 2, 3], [4, 5, 6]]"
        );
    }

    #[test]
    fn binary_on_u32_is_rejected() {
        let mut backend = backend();
        let bytes: Vec<u8> = (0_u32..3).flat_map(u32::to_le_bytes).collect();
        let indices = Tensor::upload(&mut backend, &bytes, DType::U32, shape(&[3]))
            .expect("the upload succeeds");
        assert!(
            matches!(
                indices.add(&mut backend, &indices),
                Err(MetalError::UnsupportedDType {
                    primitive: "binary",
                    dtype: DType::U32
                })
            ),
            "binary ops reject u32 on Metal"
        );
    }

    #[test]
    fn add_of_empty_tensors_downloads_no_bytes() {
        let mut backend = backend();
        let empty = upload_f32(&mut backend, &[], &[0, 3]);
        let sum = empty.add(&mut backend, &empty).expect("the add succeeds");
        assert!(
            sum.download(&mut backend)
                .expect("the download succeeds")
                .is_empty(),
            "an empty add downloads no bytes"
        );
    }
}
