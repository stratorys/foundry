use half::bf16;
use serde::Serialize;
use tracing::error;

use crate::eval::EvalError;
use crate::nn::DType;

const PER_MILLE: usize = 1_000;
const CLOSE_PER_MILLE_MIN: usize = 999;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Tolerance {
    pub atol: f32,
    pub rtol: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CloseReport {
    pub numel: usize,
    pub close_count: usize,
    pub close_fraction: f64,
    pub non_finite_count: usize,
    pub max_abs_error: f32,
}

impl CloseReport {
    pub fn passed(&self) -> bool {
        let close_enough = self
            .close_count
            .checked_mul(PER_MILLE)
            .zip(self.numel.checked_mul(CLOSE_PER_MILLE_MIN))
            .is_some_and(|(close, required)| close >= required);
        self.non_finite_count == 0 && close_enough
    }
}

pub fn to_f32(
    dtype: DType,
    bytes: &[u8],
) -> Result<Vec<f32>, EvalError> {
    match dtype {
        DType::BF16 => decode(bytes, |element| bf16::from_le_bytes(element).to_f32()),
        DType::F32 => decode(bytes, f32::from_le_bytes),
        DType::U32 => {
            error!(message = "Activation dtype cannot be compared.", ?dtype);
            Err(EvalError::DTypeNotComparable)
        }
    }
}

pub fn compare_close(
    actual: &[f32],
    expected: &[f32],
    tolerance: Tolerance,
) -> Result<CloseReport, EvalError> {
    if actual.len() != expected.len() {
        error!(
            message = "Compared activations do not have the same element count.",
            actual = actual.len(),
            expected = expected.len(),
        );
        return Err(EvalError::NumelMismatch);
    }
    let (close_count, non_finite_count, max_abs_error) = actual.iter().zip(expected).fold(
        (0_usize, 0_usize, 0.0_f32),
        |(close_count, non_finite_count, max_abs_error), (&actual, &expected)| {
            if !actual.is_finite() || !expected.is_finite() {
                return (
                    close_count,
                    non_finite_count.saturating_add(1),
                    max_abs_error,
                );
            }
            let abs_error = (actual - expected).abs();
            let bound = tolerance.atol + tolerance.rtol * actual.abs().max(expected.abs());
            (
                close_count.saturating_add(usize::from(abs_error <= bound)),
                non_finite_count,
                max_abs_error.max(abs_error),
            )
        },
    );
    let numel = actual.len();
    Ok(CloseReport {
        numel,
        close_count,
        close_fraction: fraction(close_count, numel)?,
        non_finite_count,
        max_abs_error,
    })
}

fn decode<const N: usize>(
    bytes: &[u8],
    element: impl Fn([u8; N]) -> f32,
) -> Result<Vec<f32>, EvalError> {
    let (elements, rest) = bytes.as_chunks::<N>();
    if !rest.is_empty() {
        error!(
            message = "Activation bytes do not hold whole elements.",
            nbytes = bytes.len(),
            itemsize = N,
        );
        return Err(EvalError::PartialElement);
    }
    Ok(elements.iter().copied().map(element).collect())
}

fn fraction(
    count: usize,
    numel: usize,
) -> Result<f64, EvalError> {
    let exact = |value: usize| {
        u32::try_from(value).map(f64::from).map_err(|_| {
            error!(message = "Element count does not fit in u32.", value);
            EvalError::NumelOverflow
        })
    };
    if numel == 0 {
        return Ok(1.0);
    }
    Ok(exact(count)? / exact(numel)?)
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use crate::eval::{
        CloseReport,
        EvalError,
        Tolerance,
        compare_close,
        to_f32,
    };
    use crate::nn::DType;

    #[test]
    fn compare_close_counts_elements_up_to_the_absolute_bound() {
        assert_eq!(
            compare_close(
                &[0.0, 0.0],
                &[0.5, 0.75],
                Tolerance {
                    atol: 0.5,
                    rtol: 0.0,
                },
            ),
            Ok(CloseReport {
                numel: 2,
                close_count: 1,
                close_fraction: 0.5,
                non_finite_count: 0,
                max_abs_error: 0.75,
            }),
            "an error equal to atol is close, a larger one is not"
        );
    }

    #[test]
    fn compare_close_scales_the_relative_bound_by_the_larger_magnitude() {
        assert_eq!(
            compare_close(
                &[2.0, 3.0],
                &[1.0, 1.0],
                Tolerance {
                    atol: 0.0,
                    rtol: 0.5,
                },
            ),
            Ok(CloseReport {
                numel: 2,
                close_count: 1,
                close_fraction: 0.5,
                non_finite_count: 0,
                max_abs_error: 2.0,
            }),
            "|2 - 1| <= 0.5 * 2 is close, |3 - 1| > 0.5 * 3 is not"
        );
    }

    #[test]
    fn compare_close_counts_nan_and_infinity_apart() {
        assert_eq!(
            compare_close(
                &[f32::NAN, 1.0, f32::INFINITY],
                &[1.0, 1.0, 1.0],
                Tolerance {
                    atol: 1e-3,
                    rtol: 1e-2,
                },
            ),
            Ok(CloseReport {
                numel: 3,
                close_count: 1,
                close_fraction: 1.0 / 3.0,
                non_finite_count: 2,
                max_abs_error: 0.0,
            }),
            "non-finite elements are neither close nor part of the maximum error"
        );
    }

    #[test]
    fn compare_close_rejects_different_element_counts() {
        assert_eq!(
            compare_close(
                &[1.0],
                &[1.0, 2.0],
                Tolerance {
                    atol: 1e-3,
                    rtol: 1e-2,
                },
            ),
            Err(EvalError::NumelMismatch),
            "one element against two"
        );
    }

    #[test]
    fn passed_requires_999_per_mille_close_and_no_non_finite_element() {
        let report = |close_count, non_finite_count| CloseReport {
            numel: 1_000,
            close_count,
            close_fraction: 0.0,
            non_finite_count,
            max_abs_error: 0.0,
        };
        assert_eq!(
            [
                report(1_000, 0).passed(),
                report(999, 0).passed(),
                report(998, 0).passed(),
                report(999, 1).passed(),
            ],
            [true, true, false, false],
            "all close, the minimum fraction, one element short, one non-finite element"
        );
    }

    #[test]
    fn to_f32_decodes_little_endian_bf16_and_f32() {
        let bf16_bytes: Vec<u8> = [1.0_f32, -2.0, 0.5]
            .into_iter()
            .map(bf16::from_f32)
            .flat_map(bf16::to_le_bytes)
            .collect();
        let f32_bytes: Vec<u8> = [1.5_f32, -0.25]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert_eq!(
            [
                to_f32(DType::BF16, &bf16_bytes),
                to_f32(DType::F32, &f32_bytes)
            ],
            [Ok(vec![1.0, -2.0, 0.5]), Ok(vec![1.5, -0.25])],
            "values exactly representable in bf16"
        );
    }

    #[test]
    fn to_f32_rejects_partial_elements_and_integer_dtypes() {
        assert_eq!(
            [
                to_f32(DType::BF16, &[0, 0, 0]),
                to_f32(DType::F32, &[0, 0, 0, 0, 0]),
                to_f32(DType::U32, &[0, 0, 0, 0]),
            ],
            [
                Err(EvalError::PartialElement),
                Err(EvalError::PartialElement),
                Err(EvalError::DTypeNotComparable),
            ],
            "three bf16 bytes, five f32 bytes, u32 indices"
        );
    }
}
