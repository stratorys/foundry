use std::collections::VecDeque;

use objc2::rc::Retained;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder,
    MTLBuffer,
    MTLCommandBuffer,
    MTLCommandBufferStatus,
    MTLCommandEncoder,
    MTLCommandQueue,
    MTLComputeCommandEncoder,
};

use crate::backend::metal::{
    Buffer,
    CommandBuffer,
    ComputeEncoder,
    MetalError,
    Pipeline,
    Queue,
};

const DISPATCHES_PER_BUFFER: usize = 64;

const IN_FLIGHT_MAX: usize = 3;

pub struct CommandStream {
    queue: Queue,
    open: Option<OpenBuffer>,
    in_flight: VecDeque<CommandBuffer>,
    failure: Option<String>,
}

struct OpenBuffer {
    command_buffer: CommandBuffer,
    compute: Option<Retained<ComputeEncoder>>,
    dispatch_count: usize,
}

impl CommandStream {
    pub fn new(queue: Queue) -> Self {
        Self {
            queue,
            open: None,
            in_flight: VecDeque::with_capacity(IN_FLIGHT_MAX.saturating_add(1)),
            failure: None,
        }
    }

    pub fn queue(&self) -> &Queue { &self.queue }

    pub fn compute(
        &mut self,
        pipeline: &Pipeline,
        bind: impl FnOnce(&ComputeEncoder),
    ) -> Result<(), MetalError> {
        self.check_failure()?;
        let encoder = self.open_buffer()?.compute_encoder()?;
        encoder.setComputePipelineState(pipeline);
        bind(encoder);
        self.count_dispatch()
    }

    pub fn fill_zero(
        &mut self,
        buffer: &Buffer,
    ) -> Result<(), MetalError> {
        self.check_failure()?;
        let open = self.open_buffer()?;
        open.end_compute();
        let encoder = open
            .command_buffer
            .blitCommandEncoder()
            .ok_or(MetalError::BlitEncoderCreation)?;
        encoder.fillBuffer_range_value(buffer, NSRange::new(0, buffer.length()), 0);
        encoder.endEncoding();
        self.count_dispatch()
    }

    pub fn synchronize(&mut self) -> Result<(), MetalError> {
        self.check_failure()?;
        if let Some(mut open) = self.open.take() {
            self.in_flight.push_back(open.commit());
        }
        while let Some(command_buffer) = self.in_flight.pop_front() {
            self.retire(&command_buffer)?;
        }
        Ok(())
    }

    fn check_failure(&self) -> Result<(), MetalError> {
        self.failure.as_ref().map_or(Ok(()), |message| {
            Err(MetalError::CommandBufferFailed {
                message: message.clone(),
            })
        })
    }

    fn open_buffer(&mut self) -> Result<&mut OpenBuffer, MetalError> {
        match &mut self.open {
            Some(open) => Ok(open),
            slot @ None => {
                let command_buffer = self
                    .queue
                    .commandBuffer()
                    .ok_or(MetalError::CommandBufferCreation)?;
                Ok(slot.insert(OpenBuffer {
                    command_buffer,
                    compute: None,
                    dispatch_count: 0,
                }))
            }
        }
    }

    fn count_dispatch(&mut self) -> Result<(), MetalError> {
        let Some(open) = self.open.as_mut() else {
            return Ok(());
        };
        open.dispatch_count = open.dispatch_count.saturating_add(1);
        if open.dispatch_count < DISPATCHES_PER_BUFFER {
            return Ok(());
        }
        self.commit()
    }

    fn commit(&mut self) -> Result<(), MetalError> {
        if let Some(mut open) = self.open.take() {
            self.in_flight.push_back(open.commit());
        }
        while self
            .in_flight
            .front()
            .is_some_and(|command_buffer| is_finished(command_buffer.status()))
            || self.in_flight.len() > IN_FLIGHT_MAX
        {
            if let Some(command_buffer) = self.in_flight.pop_front() {
                self.retire(&command_buffer)?;
            }
        }
        Ok(())
    }

    fn retire(
        &mut self,
        command_buffer: &CommandBuffer,
    ) -> Result<(), MetalError> {
        command_buffer.waitUntilCompleted();
        match command_buffer.status() {
            MTLCommandBufferStatus::Completed => Ok(()),
            status => {
                let message = command_buffer.error().map_or_else(
                    || format!("status {status:?}"),
                    |error| error.localizedDescription().to_string(),
                );
                self.failure = Some(message.clone());
                Err(MetalError::CommandBufferFailed {
                    message,
                })
            }
        }
    }
}

impl OpenBuffer {
    fn compute_encoder(&mut self) -> Result<&ComputeEncoder, MetalError> {
        match &mut self.compute {
            Some(encoder) => Ok(&**encoder),
            slot @ None => {
                let encoder = self
                    .command_buffer
                    .computeCommandEncoder()
                    .ok_or(MetalError::ComputeEncoderCreation)?;
                Ok(&**slot.insert(encoder))
            }
        }
    }

    fn end_compute(&mut self) {
        if let Some(encoder) = self.compute.take() {
            encoder.endEncoding();
        }
    }

    fn commit(&mut self) -> CommandBuffer {
        self.end_compute();
        self.command_buffer.commit();
        self.command_buffer.clone()
    }
}

impl Drop for OpenBuffer {
    fn drop(&mut self) { self.end_compute(); }
}

fn is_finished(status: MTLCommandBufferStatus) -> bool {
    matches!(
        status,
        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
    )
}
