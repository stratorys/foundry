use sha2::{
    Digest,
    Sha256,
};

use crate::error::ConfigError;

pub const BENCHMARK_INPUT_JSON: &str =
    include_str!("../../../tools/mlx-baseline/inputs/random-512-seed0.json");
pub const BENCHMARK_INPUT_SHA256: &str =
    "9286ad09d58db9a18214d35afb5d92d2071e1fa9f8f1ccc63d2911401886dc49";
pub const BENCHMARK_INPUT_TOKENS: usize = 512;
pub const BENCHMARK_OUTPUT_TOKENS: usize = 128;

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn benchmark_input() -> Result<Vec<u32>, ConfigError> {
    let digest = sha256_hex(BENCHMARK_INPUT_JSON.as_bytes());
    if digest != BENCHMARK_INPUT_SHA256 {
        return Err(ConfigError::Unsupported {
            field: "benchmark input sha256",
            expected: BENCHMARK_INPUT_SHA256.to_owned(),
            actual: digest,
        });
    }
    let tokens: Vec<u32> =
        serde_json::from_str(BENCHMARK_INPUT_JSON).map_err(ConfigError::Parse)?;
    if tokens.len() == BENCHMARK_INPUT_TOKENS {
        Ok(tokens)
    } else {
        Err(ConfigError::Unsupported {
            field: "benchmark input length",
            expected: BENCHMARK_INPUT_TOKENS.to_string(),
            actual: tokens.len().to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::benchmark_input;
    use crate::error::ConfigError;

    #[test]
    fn the_benchmark_input_matches_the_mlx_harness() -> Result<(), ConfigError> {
        let tokens = benchmark_input()?;
        assert_eq!(
            tokens.get(..4),
            Some([128_000, 110_680, 50_494, 99_346].as_slice()),
            "first tokens"
        );
        assert_eq!(
            tokens.get(509..),
            Some([83_904, 57_074, 48_817].as_slice()),
            "last tokens"
        );
        assert_eq!(
            tokens.iter().filter(|&&token| token == 128_000).count(),
            1,
            "a single BOS"
        );
        Ok(())
    }
}
