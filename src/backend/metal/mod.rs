mod error;
#[cfg(test)]
mod tests;

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
    MTLCreateSystemDefaultDevice,
    MTLDevice,
    MTLLibrary,
    MTLResourceOptions,
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
    Shape,
};

const KERNEL_SOURCES: &[&str] = &[include_str!("kernels/common.metal")];

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;

pub struct MetalStorage {
    buffer: Buffer,
    byte_len: usize,
}

pub struct MetalBackend {
    device: Device,
    queue: Queue,
    #[expect(
        dead_code,
        reason = "Pipelines are created from the library from step 5 on."
    )]
    library: Library,
    pending: Option<CommandBuffer>,
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
        _op: UnaryOp,
        _input: Operand<'_, MetalStorage>,
    ) -> Result<MetalStorage, MetalError> {
        Err(MetalError::NotImplemented {
            primitive: "unary",
        })
    }

    fn binary(
        &mut self,
        _op: BinaryOp,
        _lhs: Operand<'_, MetalStorage>,
        _rhs: Operand<'_, MetalStorage>,
    ) -> Result<MetalStorage, MetalError> {
        Err(MetalError::NotImplemented {
            primitive: "binary",
        })
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
        _input: Operand<'_, MetalStorage>,
    ) -> Result<MetalStorage, MetalError> {
        Err(MetalError::NotImplemented {
            primitive: "copy",
        })
    }

    fn cast(
        &mut self,
        _input: Operand<'_, MetalStorage>,
        _dtype: DType,
    ) -> Result<MetalStorage, MetalError> {
        Err(MetalError::NotImplemented {
            primitive: "cast",
        })
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
