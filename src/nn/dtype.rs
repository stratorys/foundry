use tracing::error;

use crate::nn::TensorError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    U32,
}

pub const MAX_ITEMSIZE: usize = 4;

impl DType {
    pub const fn itemsize(self) -> usize {
        match self {
            Self::F32 | Self::U32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }

    pub const fn is_float(self) -> bool { matches!(self, Self::F32 | Self::F16 | Self::BF16) }
}

pub fn exact_f32(value: usize) -> Result<f32, TensorError> {
    u16::try_from(value).map(f32::from).map_err(|_| {
        error!(
            message = "Value exceeds the largest value converted exactly to f32.",
            value,
            max_value = u16::MAX,
        );
        TensorError::DimensionTooLargeForF32
    })
}

#[cfg(test)]
mod tests {
    use crate::nn::{
        DType,
        TensorError,
        exact_f32,
    };

    #[test]
    fn size_bytes_matches_each_dtype() {
        assert_eq!(
            [DType::F32, DType::F16, DType::BF16, DType::U32].map(DType::itemsize),
            [4, 2, 2, 4],
            "f32 and u32 take four bytes, f16 and bf16 two"
        );
    }

    #[test]
    fn is_float_is_true_for_float_dtypes_only() {
        assert_eq!(
            [DType::F32, DType::F16, DType::BF16, DType::U32].map(DType::is_float),
            [true, true, true, false],
            "u32 is the only non-float dtype"
        );
    }

    #[test]
    fn exact_f32_accepts_values_up_to_u16_max_only() {
        assert_eq!(
            [0, 65_535, 65_536].map(exact_f32),
            [
                Ok(0.0),
                Ok(65_535.0),
                Err(TensorError::DimensionTooLargeForF32)
            ],
            "u16::MAX is the largest value converted"
        );
    }
}
