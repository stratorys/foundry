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

pub fn reference_checksum(payload: &[u8]) -> u32 {
    (0_u32..)
        .zip(payload.chunks(CHUNK))
        .fold((0_u32, 0_u32), |(total, position), (chunk, bytes)| {
            let (sum, position) = bytes
                .iter()
                .fold((0_u32, position), |(sum, position), &byte| {
                    let position = position.wrapping_add(1);
                    (
                        sum.wrapping_add(u32::from(byte).wrapping_mul(position)),
                        position,
                    )
                });
            let total = total.wrapping_add(sum.wrapping_mul(2_654_435_761).wrapping_add(chunk));
            (total, position)
        })
        .0
}

#[cfg(test)]
mod tests {
    use super::reference_checksum;

    #[test]
    fn reference_checksums_follow_the_kernel_formula() {
        let cases: [(&[u8], u32); 4] = [
            (&[], 0),
            (&[0; 100], 1),
            (&[1, 2, 3], 2_802_362_286),
            (&[1; 65], 2_933_040_146),
        ];
        for (payload, expected) in cases {
            assert_eq!(
                reference_checksum(payload),
                expected,
                "{} byte payload",
                payload.len()
            );
        }
    }
}
