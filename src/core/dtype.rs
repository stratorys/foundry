#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    U32,
}

impl DType {
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::U32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }

    pub const fn is_float(self) -> bool { matches!(self, Self::F32 | Self::F16 | Self::BF16) }
}
