#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    U8,
}

impl DType {
    pub const fn size_in_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DType;

    #[test]
    fn sizes_match_the_element_widths() {
        let expected = [
            (DType::F32, 4),
            (DType::F16, 2),
            (DType::BF16, 2),
            (DType::I8, 1),
            (DType::U8, 1),
        ];
        for (dtype, size) in expected {
            assert_eq!(dtype.size_in_bytes(), size, "size of {dtype:?}");
        }
    }
}
