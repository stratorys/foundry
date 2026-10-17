#[cfg(target_os = "macos")]
use foundry::metal::MetalDevice;
use foundry::nn::Device;
use tracing::error;

use crate::cli::error::CliError;

pub const DEVICE: &str = "metal";

pub trait DeviceTask {
    type Output;

    #[cfg_attr(
        not(target_os = "macos"),
        expect(dead_code, reason = "no device on this platform")
    )]
    fn run<D: Device>(
        self,
        device: &mut D,
    ) -> Result<Self::Output, CliError>;
}

#[cfg(target_os = "macos")]
pub fn run_on_platform<T: DeviceTask>(task: T) -> Result<T::Output, CliError> {
    let mut device = MetalDevice::new().map_err(|error| {
        error!(message = "Creating the Metal device failed.", %error);
        CliError::Device
    })?;
    task.run(&mut device)
}

#[cfg(not(target_os = "macos"))]
pub fn run_on_platform<T: DeviceTask>(_task: T) -> Result<T::Output, CliError> {
    error!(message = "No device is available on this platform.");
    Err(CliError::DeviceUnavailable)
}
