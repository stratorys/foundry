use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocatedMemory {
    pub bytes: u64,
    pub source: AllocatedMemorySource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AllocatedMemorySource {
    MetalCurrentAllocatedSize,
}

#[cfg(test)]
mod tests {
    use crate::core::AllocatedMemorySource;

    #[test]
    fn allocated_memory_source_serializes_in_snake_case() {
        assert_eq!(
            serde_json::to_string(&AllocatedMemorySource::MetalCurrentAllocatedSize).ok(),
            Some("\"metal_current_allocated_size\"".to_owned()),
        );
    }
}
