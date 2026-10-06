use half::f16;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAllocation,
    MTLBuffer,
    MTLCommandBuffer,
    MTLCommandBufferStatus,
    MTLCommandQueue,
    MTLCreateSystemDefaultDevice,
    MTLDevice,
    MTLGPUFamily,
    MTLResourceOptions,
};

use crate::metal::error::GpuError;
use crate::metal::kernels::Kernels;

pub(crate) type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub(crate) type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub(crate) type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub(crate) type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;

pub(crate) trait Element: Copy + Default {}

impl Element for u32 {}
impl Element for f32 {}
impl Element for f16 {}

pub(crate) struct Context {
    pub(crate) device: Device,
    pub(crate) queue: Queue,
    pub(crate) kernels: Kernels,
}

impl Context {
    pub(crate) fn new() -> Result<Self, GpuError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(GpuError::DeviceUnavailable)?;
        if !device.supportsFamily(MTLGPUFamily::Apple7) {
            return Err(GpuError::UnsupportedDevice {
                name: device.name().to_string(),
                requirement: "the Apple7 GPU family (simdgroup matrices)",
            });
        }
        let queue = device.newCommandQueue().ok_or(GpuError::QueueCreation)?;
        let kernels = Kernels::compile(&device)?;
        Ok(Self {
            device,
            queue,
            kernels,
        })
    }

    pub(crate) fn device_name(&self) -> String { self.device.name().to_string() }

    pub(crate) fn allocated_bytes(&self) -> usize { self.device.currentAllocatedSize() }

    pub(crate) fn shared(
        &self,
        bytes: usize,
    ) -> Result<Buffer, GpuError> {
        let limit = self.device.maxBufferLength();
        let length = bytes.max(1);
        if length > limit {
            return Err(GpuError::AllocationTooLarge {
                bytes,
                limit,
            });
        }
        self.device
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .ok_or(GpuError::AllocationFailed {
                bytes,
            })
    }

    pub(crate) fn shared_elements<T: Element>(
        &self,
        count: usize,
    ) -> Result<Buffer, GpuError> {
        let bytes = count
            .checked_mul(size_of::<T>())
            .ok_or(GpuError::Overflow {
                what: "buffer size",
            })?;
        self.shared(bytes)
    }

    pub(crate) fn shared_from<T: Element>(
        &self,
        values: &[T],
    ) -> Result<Buffer, GpuError> {
        let buffer = self.shared_elements::<T>(values.len())?;
        write(&buffer, 0, values)?;
        Ok(buffer)
    }

    pub(crate) fn command_buffer(&self) -> Result<CommandBuffer, GpuError> {
        self.queue
            .commandBuffer()
            .ok_or(GpuError::CommandBufferCreation)
    }

    pub(crate) fn synchronize(&self) -> Result<(), GpuError> {
        let command_buffer = self.command_buffer()?;
        command_buffer.commit();
        wait(&command_buffer)
    }
}

pub(crate) fn allocated_size(buffer: &ProtocolObject<dyn MTLBuffer>) -> usize {
    MTLAllocation::allocatedSize(buffer)
}

fn element_range<T: Element>(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    offset: usize,
    count: usize,
) -> Result<usize, GpuError> {
    let bounds = GpuError::OutOfBounds {
        what: "buffer access",
    };
    let end = offset
        .checked_add(count)
        .and_then(|end| end.checked_mul(size_of::<T>()))
        .ok_or_else(|| bounds.clone())?;
    if end <= buffer.length() {
        offset.checked_mul(size_of::<T>()).ok_or(bounds)
    } else {
        Err(bounds)
    }
}

pub(crate) fn write<T: Element>(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    offset: usize,
    values: &[T],
) -> Result<(), GpuError> {
    let start = element_range::<T>(buffer, offset, values.len())?;
    let base = buffer.contents().cast::<u8>().as_ptr();
    // SAFETY: the buffer uses shared storage, `element_range` proved that
    // `values.len()` elements starting at `start` bytes fit in it, and no GPU
    // work referencing this range is in flight when the host writes inputs.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            base.add(start),
            std::mem::size_of_val(values),
        );
    };
    Ok(())
}

pub(crate) fn read<T: Element>(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    offset: usize,
    count: usize,
) -> Result<Vec<T>, GpuError> {
    let start = element_range::<T>(buffer, offset, count)?;
    let mut values = vec![T::default(); count];
    let base = buffer.contents().cast::<u8>().as_ptr();
    // SAFETY: the buffer uses shared storage, `element_range` proved the
    // range is inside it, the destination vector holds `count` elements, and
    // callers read only after the command buffers writing it completed.
    unsafe {
        std::ptr::copy_nonoverlapping(
            base.add(start).cast_const(),
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values.as_slice()),
        );
    };
    Ok(values)
}

pub(crate) fn wait(command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), GpuError> {
    command_buffer.waitUntilCompleted();
    let status = command_buffer.status();
    if status == MTLCommandBufferStatus::Completed {
        Ok(())
    } else if status == MTLCommandBufferStatus::Error {
        Err(command_buffer
            .error()
            .map_or(GpuError::ExecutionWithoutError, |error| {
                GpuError::Execution(error.localizedDescription().to_string())
            }))
    } else {
        Err(GpuError::NotCompleted)
    }
}
