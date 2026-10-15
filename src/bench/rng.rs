use std::num::NonZeroU32;

use tracing::error;

use crate::bench::error::BenchError;

const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
const MIX_MULTIPLIER_1: u64 = 0xbf58_476d_1ce4_e5b9;
const MIX_MULTIPLIER_2: u64 = 0x94d0_49bb_1331_11eb;
const DRAW_COUNT_MAX: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed,
        }
    }

    pub const fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let mixed = (self.state ^ self.state.wrapping_shr(30)).wrapping_mul(MIX_MULTIPLIER_1);
        let mixed = (mixed ^ mixed.wrapping_shr(27)).wrapping_mul(MIX_MULTIPLIER_2);
        mixed ^ mixed.wrapping_shr(31)
    }

    pub fn token_ids(
        &mut self,
        count: usize,
        vocab_size: NonZeroU32,
    ) -> Result<Vec<u32>, BenchError> {
        (0..count).map(|_| self.below(vocab_size)).collect()
    }

    pub fn below(
        &mut self,
        bound: NonZeroU32,
    ) -> Result<u32, BenchError> {
        let bound = u64::from(bound.get());
        let rejected_count = u64::MAX
            .checked_rem(bound)
            .and_then(|remainder| remainder.checked_add(1))
            .and_then(|remainder| remainder.checked_rem(bound))
            .ok_or(BenchError::Random)?;
        let accepted_max = u64::MAX
            .checked_sub(rejected_count)
            .ok_or(BenchError::Random)?;
        (0..DRAW_COUNT_MAX)
            .map(|_| self.next_u64())
            .find(|&draw| draw <= accepted_max)
            .and_then(|draw| draw.checked_rem(bound))
            .and_then(|draw| u32::try_from(draw).ok())
            .ok_or_else(|| {
                error!(message = "Drawing a random token ID failed.", bound);
                BenchError::Random
            })
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use crate::bench::rng::SplitMix64;

    #[test]
    fn split_mix_64_matches_the_reference_sequence() {
        let mut rng = SplitMix64::new(0);
        assert_eq!(
            [rng.next_u64(), rng.next_u64(), rng.next_u64()],
            [
                0xe220_a839_7b1d_cdaf,
                0x6e78_9e6a_a1b9_65f4,
                0x06c4_5d18_8009_454f
            ],
            "first three outputs of seed 0"
        );
    }

    #[test]
    fn token_ids_are_below_the_vocab_size_and_repeat_with_the_seed() {
        let vocab_size = NonZeroU32::new(7).expect("7 is not zero");
        let token_ids = SplitMix64::new(42)
            .token_ids(1000, vocab_size)
            .expect("drawing succeeds");
        assert_eq!(token_ids.len(), 1000, "one ID per requested token");
        assert!(
            token_ids.iter().all(|&token_id| token_id < 7),
            "every ID is below 7"
        );
        assert_eq!(
            SplitMix64::new(42).token_ids(1000, vocab_size),
            Ok(token_ids),
            "the same seed gives the same IDs"
        );
    }

    #[test]
    fn a_vocab_of_one_always_draws_zero() {
        assert_eq!(
            SplitMix64::new(3).token_ids(4, NonZeroU32::MIN),
            Ok(vec![0; 4]),
            "the only token ID is 0"
        );
    }
}
