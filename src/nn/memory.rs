#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocatedMemory {
    pub bytes: u64,
    pub source: AllocatedMemorySource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocatedMemorySource {
    MetalCurrentAllocatedSize,
}
