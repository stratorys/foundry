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
use crate::nn::{
    GpuInterval,
    ProfileMode,
};

const DISPATCHES_PER_BUFFER: usize = 64;

const MAX_IN_FLIGHT: usize = 3;

pub struct CommandStream {
    queue: Queue,
    open: Option<OpenBuffer>,
    in_flight: VecDeque<InFlight>,
    failed: bool,
    profile_mode: Option<ProfileMode>,
    gpu_intervals: Vec<GpuInterval>,
}

struct OpenBuffer {
    command_buffer: CommandBuffer,
    compute: Option<Retained<ComputeEncoder>>,
    dispatch_count: usize,
    kernel: Option<&'static str>,
}

struct InFlight {
    command_buffer: CommandBuffer,
    kernel: Option<&'static str>,
}

impl CommandStream {
    pub fn new(queue: Queue) -> Self {
        Self {
            queue,
            open: None,
            in_flight: VecDeque::with_capacity(MAX_IN_FLIGHT.saturating_add(1)),
            failed: false,
            profile_mode: None,
            gpu_intervals: Vec::new(),
        }
    }

    pub fn with_gpu_timing(
        self,
        mode: ProfileMode,
    ) -> Self {
        Self {
            profile_mode: Some(mode),
            ..self
        }
    }

    pub fn queue(&self) -> &Queue { &self.queue }

    // Completed command buffers only: synchronize first to include all work.
    pub fn take_gpu_intervals(&mut self) -> Option<Vec<GpuInterval>> {
        self.profile_mode
            .map(|_| std::mem::take(&mut self.gpu_intervals))
    }

    pub fn compute(
        &mut self,
        pipeline: &Pipeline,
        kernel: &'static str,
        bind: impl FnOnce(&ComputeEncoder),
    ) -> Result<(), MetalError> {
        self.check_failure()?;
        let is_kernel_timed = self.profile_mode == Some(ProfileMode::Kernels);
        let open = self.open_buffer()?;
        if is_kernel_timed {
            open.kernel = Some(kernel);
        }
        let encoder = open.compute_encoder()?;
        encoder.setComputePipelineState(pipeline);
        bind(encoder);
        self.count_dispatch()
    }

    pub fn synchronize(&mut self) -> Result<(), MetalError> {
        self.check_failure()?;
        if let Some(mut open) = self.open.take() {
            self.in_flight.push_back(open.commit());
        }
        while let Some(in_flight) = self.in_flight.pop_front() {
            self.retire(&in_flight)?;
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
                    kernel: None,
                }))
            }
        }
    }

    // Kernel timing reads GPU time per command buffer, so each one holds a
    // single dispatch.
    fn dispatches_per_buffer(&self) -> usize {
        match self.profile_mode {
            Some(ProfileMode::Kernels) => 1,
            Some(ProfileMode::Steps) | None => DISPATCHES_PER_BUFFER,
        }
    }

    // Concurrent command buffers would share the GPU and inflate each other's
    // kernel time, so kernel timing completes each one before the next.
    fn in_flight_max(&self) -> usize {
        match self.profile_mode {
            Some(ProfileMode::Kernels) => 0,
            Some(ProfileMode::Steps) | None => MAX_IN_FLIGHT,
        }
    }

    fn count_dispatch(&mut self) -> Result<(), MetalError> {
        let dispatches_per_buffer = self.dispatches_per_buffer();
        let Some(open) = self.open.as_mut() else {
            return Ok(());
        };
        open.dispatch_count = open.dispatch_count.saturating_add(1);
        if open.dispatch_count < dispatches_per_buffer {
            return Ok(());
        }
        self.commit()
    }

    fn commit(&mut self) -> Result<(), MetalError> {
        if let Some(mut open) = self.open.take() {
            self.in_flight.push_back(open.commit());
        }
        let in_flight_max = self.in_flight_max();
        while self.in_flight.front().is_some_and(|in_flight| {
            matches!(
                in_flight.command_buffer.status(),
                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
            )
        }) || self.in_flight.len() > in_flight_max
        {
            if let Some(in_flight) = self.in_flight.pop_front() {
                self.retire(&in_flight)?;
            }
        }
        Ok(())
    }

    fn retire(
        &mut self,
        in_flight: &InFlight,
    ) -> Result<(), MetalError> {
        let command_buffer = &in_flight.command_buffer;
        command_buffer.waitUntilCompleted();
        match command_buffer.status() {
            MTLCommandBufferStatus::Completed => {
                if self.profile_mode.is_some() {
                    self.gpu_intervals.push(GpuInterval {
                        kernel: in_flight.kernel,
                        start_s: command_buffer.GPUStartTime(),
                        end_s: command_buffer.GPUEndTime(),
                    });
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

    fn commit(&mut self) -> InFlight {
        self.end_compute();
        self.command_buffer.commit();
        InFlight {
            command_buffer: self.command_buffer.clone(),
            kernel: self.kernel,
        }
    }
}

impl Drop for OpenBuffer {
    fn drop(&mut self) { self.end_compute(); }
}
