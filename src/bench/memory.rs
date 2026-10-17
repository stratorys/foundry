use serde::Serialize;

use crate::nn::AllocatedMemorySource;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PeakMemorySource {
    MetalCurrentAllocatedSize,
}

impl From<AllocatedMemorySource> for PeakMemorySource {
    fn from(source: AllocatedMemorySource) -> Self {
        match source {
            AllocatedMemorySource::MetalCurrentAllocatedSize => Self::MetalCurrentAllocatedSize,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::bench::PeakMemorySource;
    use crate::nn::AllocatedMemorySource;

    #[test]
    fn peak_memory_source_serializes_in_snake_case() {
        assert_eq!(
            serde_json::to_string(&PeakMemorySource::from(
                AllocatedMemorySource::MetalCurrentAllocatedSize
            ))
            .ok(),
            Some("\"metal_current_allocated_size\"".to_owned()),
        );
    }
}
