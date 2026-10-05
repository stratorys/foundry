use std::error::Error;
use std::time::Duration;

use foundry_infer_core::{
    ByteSize,
    DeviceCapabilities,
    OpId,
};

pub trait Backend {
    type DeviceBuffer;
    type HostBuffer;
    type Stream;
    type Event;
    type Error: Error + 'static;

    fn capabilities(&self) -> DeviceCapabilities;

    fn allocate_device(
        &mut self,
        bytes: ByteSize,
    ) -> Result<Self::DeviceBuffer, Self::Error>;

    fn allocate_host(
        &mut self,
        contents: &[u8],
    ) -> Result<Self::HostBuffer, Self::Error>;

    fn create_stream(&mut self) -> Result<Self::Stream, Self::Error>;

    fn copy_to_device(
        &mut self,
        stream: &mut Self::Stream,
        source: &Self::HostBuffer,
        destination: &mut Self::DeviceBuffer,
    ) -> Result<Self::Event, Self::Error>;

    fn synthetic_compute(
        &mut self,
        stream: &mut Self::Stream,
        op: OpId,
        weights: &[&Self::DeviceBuffer],
        duration_hint: Option<Duration>,
    ) -> Result<Self::Event, Self::Error>;

    fn wait_stream(
        &mut self,
        stream: &mut Self::Stream,
        event: &Self::Event,
    ) -> Result<(), Self::Error>;

    fn wait_host(
        &mut self,
        event: &Self::Event,
    ) -> Result<(), Self::Error>;

    fn drain(&mut self) -> Result<(), Self::Error>;
}
