use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeakMemory {
    pub bytes: u64,
    pub source: PeakMemorySource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PeakMemorySource {
    MetalCurrentAllocatedSize,
}

#[cfg(test)]
mod tests {
    use crate::core::PeakMemorySource;

    #[test]
    fn peak_memory_source_serializes_in_snake_case() {
        assert_eq!(
            serde_json::to_string(&PeakMemorySource::MetalCurrentAllocatedSize).ok(),
            Some("\"metal_current_allocated_size\"".to_owned()),
        );
    }
}
