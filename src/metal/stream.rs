use std::collections::VecDeque;

use objc2::rc::Retained;
use objc2_metal::{
    MTLCommandBuffer,
    MTLCommandBufferStatus,
    MTLCommandEncoder,
    MTLCommandQueue,
    MTLComputeCommandEncoder,
};
use tracing::error;

use crate::metal::{
    CommandBuffer,
    ComputeEncoder,
    MetalError,
    Pipeline,
    Queue,
};

const DISPATCHES_PER_BUFFER: usize = 64;

const MAX_IN_FLIGHT: usize = 3;

pub struct CommandStream {
    queue: Queue,
    open: Option<OpenBuffer>,
    in_flight: VecDeque<CommandBuffer>,
    failed: bool,
    gpu_intervals_s: Option<Vec<(f64, f64)>>,
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
            in_flight: VecDeque::with_capacity(MAX_IN_FLIGHT.saturating_add(1)),
            failed: false,
            gpu_intervals_s: None,
        }
    }

    pub fn with_gpu_timing(self) -> Self {
        Self {
            gpu_intervals_s: Some(Vec::new()),
            ..self
        }
    }

    pub fn queue(&self) -> &Queue { &self.queue }

    // Completed command buffers only: synchronize first to include all work.
    pub fn take_gpu_intervals(&mut self) -> Option<Vec<(f64, f64)>> {
        self.gpu_intervals_s.as_mut().map(std::mem::take)
    }

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
        if self.failed {
            Err(MetalError::CommandBufferFailed)
        } else {
            Ok(())
        }
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
        while self.in_flight.front().is_some_and(|command_buffer| {
            matches!(
                command_buffer.status(),
                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
            )
        }) || self.in_flight.len() > MAX_IN_FLIGHT
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
            MTLCommandBufferStatus::Completed => {
                if let Some(intervals) = &mut self.gpu_intervals_s {
                    intervals.push((command_buffer.GPUStartTime(), command_buffer.GPUEndTime()));
                }
                Ok(())
            }
            status => {
                let description = command_buffer
                    .error()
                    .map(|error| error.localizedDescription().to_string());
                error!(
                    message = "Metal command buffer failed.",
                    ?status,
                    error = ?description,
                );
                self.failed = true;
                Err(MetalError::CommandBufferFailed)
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
