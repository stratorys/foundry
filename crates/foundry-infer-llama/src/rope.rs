use std::f32::consts::TAU;

use crate::config::RopeConfig;

pub fn llama3_frequencies(
    rope: &RopeConfig,
    head_dim: u32,
) -> Vec<f32> {
    let dims = f32::from(u16::try_from(head_dim).unwrap_or(u16::MAX));
    let context = f32::from(u16::try_from(rope.original_context).unwrap_or(u16::MAX));
    let low_wavelength = context / rope.low_freq_factor;
    let high_wavelength = context / rope.high_freq_factor;
    (0..head_dim)
        .step_by(2)
        .map(|index| {
            let exponent = f32::from(u16::try_from(index).unwrap_or(u16::MAX)) / dims;
            let base = rope.theta.powf(exponent);
            let wavelength = TAU * base;
            let scaled = if wavelength > low_wavelength {
                base * rope.factor
            } else {
                base
            };
            let medium = wavelength > high_wavelength && wavelength < low_wavelength;
            if medium {
                let smooth = (context / wavelength - rope.low_freq_factor)
                    / (rope.high_freq_factor - rope.low_freq_factor);
                scaled / ((1.0 - smooth) / rope.factor + smooth)
            } else {
                scaled
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::llama3_frequencies;
    use crate::config::RopeConfig;

    #[test]
    fn llama3_frequencies_follow_the_three_bands() {
        let rope = RopeConfig {
            theta: 500_000.0,
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_context: 8192,
        };
        let frequencies = llama3_frequencies(&rope, 128);
        assert_eq!(frequencies.len(), 64, "one frequency per rotated pair");
        assert_eq!(
            frequencies.first().copied(),
            Some(1.0),
            "the first pair is unscaled"
        );
        let last = frequencies.last().copied().unwrap_or_default();
        let expected = 500_000_f32.powf(126.0 / 128.0) * 32.0;
        assert!(
            (last - expected).abs() <= expected * 1e-6,
            "the lowest frequency is scaled by the factor: {last} vs {expected}"
        );
        assert!(
            frequencies
                .windows(2)
                .all(|pair| matches!(pair, [left, right] if left < right)),
            "frequency divisors increase monotonically"
        );
    }
}
