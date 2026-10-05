use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLComputePipelineState,
    MTLDevice,
    MTLLibrary,
};

use crate::error::MetalError;

pub(crate) const CHUNK: usize = 64;

pub(crate) type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub(crate) fn compile(device: &ProtocolObject<dyn MTLDevice>) -> Result<Pipeline, MetalError> {
    let source = NSString::from_str(include_str!("checksum.metal"));
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .map_err(|error| MetalError::ShaderCompilation(error.localizedDescription().to_string()))?;
    let function = library
        .newFunctionWithName(&NSString::from_str("checksum"))
        .ok_or(MetalError::MissingKernel)?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| MetalError::PipelineCreation(error.localizedDescription().to_string()))
}
