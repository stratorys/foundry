use std::mem;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer,
    MTLCommandBuffer,
    MTLCommandBufferStatus,
    MTLDevice,
    MTLEvent,
};

use crate::diagnostics::{
    GpuLog,
    Submission,
    interval,
};
use crate::error::MetalError;
use crate::pipeline::Pipeline;
use crate::tracked::TrackedBuffer;

pub(crate) type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
pub(crate) type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub(crate) type Event = Retained<ProtocolObject<dyn MTLEvent>>;
pub(crate) type Device = Retained<ProtocolObject<dyn MTLDevice>>;

struct InFlight {
    command_buffer: CommandBuffer,
    _buffers: Vec<TrackedBuffer>,
    _events: Vec<Event>,
    _pipeline: Option<Pipeline>,
    submission: Option<Submission>,
}

#[derive(Default)]
pub(crate) struct Submissions {
    in_flight: Vec<InFlight>,
    faults: Vec<MetalError>,
    log: Option<GpuLog>,
    #[cfg(test)]
    released: Vec<MTLCommandBufferStatus>,
}

impl Submissions {
    pub(crate) fn register(
        &mut self,
        command_buffer: CommandBuffer,
        buffers: Vec<TrackedBuffer>,
        events: Vec<Event>,
        pipeline: Option<Pipeline>,
        submission: Option<Submission>,
    ) {
        self.in_flight.push(InFlight {
            command_buffer,
            _buffers: buffers,
            _events: events,
            _pipeline: pipeline,
            submission,
        });
    }

    pub(crate) fn retire(&mut self) {
        let (finished, pending): (Vec<_>, Vec<_>) = mem::take(&mut self.in_flight)
            .into_iter()
            .partition(|entry| finished(&entry.command_buffer));
        self.in_flight = pending;
        for entry in finished {
            if let Err(error) = completion(&entry.command_buffer) {
                self.faults.push(error);
            }
            self.release(entry);
        }
    }

    pub(crate) fn take_fault(&mut self) -> Option<MetalError> {
        mem::take(&mut self.faults).into_iter().next()
    }

    pub(crate) fn drain(&mut self) -> Result<(), MetalError> {
        let mut errors = mem::take(&mut self.faults);
        for entry in mem::take(&mut self.in_flight) {
            entry.command_buffer.waitUntilCompleted();
            if let Err(error) = completion(&entry.command_buffer) {
                errors.push(error);
            }
            self.release(entry);
        }
        errors.into_iter().next().map_or(Ok(()), Err)
    }

    pub(crate) fn record(
        &mut self,
        log: GpuLog,
    ) {
        self.log = Some(log);
    }

    pub(crate) fn recording(&self) -> bool { self.log.is_some() }

    pub(crate) fn take_log(&mut self) -> Option<(GpuLog, usize)> {
        let pending = self
            .in_flight
            .iter()
            .filter(|entry| entry.submission.is_some())
            .count();
        self.log.take().map(|log| (log, pending))
    }

    fn release(
        &mut self,
        entry: InFlight,
    ) {
        if let (Some(log), Some(submission)) = (self.log.as_mut(), entry.submission) {
            log.push(interval(&entry.command_buffer, submission));
        }
        #[cfg(test)]
        self.released.push(entry.command_buffer.status());
        drop(entry);
    }

    #[cfg(test)]
    pub(crate) fn in_flight_len(&self) -> usize { self.in_flight.len() }

    #[cfg(test)]
    pub(crate) fn released(&self) -> &[MTLCommandBufferStatus] { &self.released }
}

impl Drop for Submissions {
    fn drop(&mut self) {
        for entry in &self.in_flight {
            entry.command_buffer.waitUntilCompleted();
        }
    }
}

fn finished(command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> bool {
    let status = command_buffer.status();
    status == MTLCommandBufferStatus::Completed || status == MTLCommandBufferStatus::Error
}

pub(crate) fn completion(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>
) -> Result<(), MetalError> {
    let status = command_buffer.status();
    if status == MTLCommandBufferStatus::Completed {
        Ok(())
    } else if status == MTLCommandBufferStatus::Error {
        Err(command_buffer
            .error()
            .map_or(MetalError::GpuWithoutError, |error| {
                MetalError::Gpu(error.localizedDescription().to_string())
            }))
    } else {
        Err(MetalError::NotCompleted)
    }
}
