use tracing::error;

use crate::core::CoreError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    U32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatDType {
    F32,
    F16,
    BF16,
}

pub const DTYPE_SIZE_BYTES_MAX: usize = 4;

impl DType {
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::U32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }

    pub const fn float(self) -> Option<FloatDType> {
        match self {
            Self::F32 => Some(FloatDType::F32),
            Self::F16 => Some(FloatDType::F16),
            Self::BF16 => Some(FloatDType::BF16),
            Self::U32 => None,
        }
    }

    const fn from_float(dtype: FloatDType) -> Self {
        match dtype {
            FloatDType::F32 => Self::F32,
            FloatDType::F16 => Self::F16,
            FloatDType::BF16 => Self::BF16,
        }
    }
}

impl FloatDType {
    pub const fn size_bytes(self) -> usize { DType::from_float(self).size_bytes() }
}

impl From<FloatDType> for DType {
    fn from(dtype: FloatDType) -> Self { Self::from_float(dtype) }
}

pub fn exact_f32(value: usize) -> Result<f32, CoreError> {
    u16::try_from(value).map(f32::from).map_err(|_| {
        error!(
            message = "Value exceeds the largest value converted exactly to f32.",
            value,
            value_max = u16::MAX,
        );
        CoreError::DimensionTooLargeForF32
    })
}

#[cfg(test)]
mod tests {
    use crate::core::{
        CoreError,
        DType,
        FloatDType,
        exact_f32,
    };

    #[test]
    fn size_bytes_matches_each_dtype() {
        assert_eq!(
            (
                [DType::F32, DType::F16, DType::BF16, DType::U32].map(DType::size_bytes),
                [FloatDType::F32, FloatDType::F16, FloatDType::BF16].map(FloatDType::size_bytes),
            ),
            ([4, 2, 2, 4], [4, 2, 2]),
            "f32 and u32 take four bytes, f16 and bf16 two"
        );
    }

    #[test]
    fn exact_f32_accepts_values_up_to_u16_max_only() {
        assert_eq!(
            [0, 65_535, 65_536].map(exact_f32),
            [
                Ok(0.0),
                Ok(65_535.0),
                Err(CoreError::DimensionTooLargeForF32)
            ],
            "u16::MAX is the largest value converted"
        );
    }
}
