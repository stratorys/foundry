use std::num::NonZeroU32;

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

    pub fn below(
        &mut self,
        bound: NonZeroU32,
    ) -> Option<u32> {
        let bound = u64::from(bound.get());
        let rejected_count = u64::MAX
            .checked_rem(bound)
            .and_then(|remainder| remainder.checked_add(1))
            .and_then(|remainder| remainder.checked_rem(bound))?;
        let accepted_max = u64::MAX.checked_sub(rejected_count)?;
        (0..DRAW_COUNT_MAX)
            .map(|_| self.next_u64())
            .find(|&draw| draw <= accepted_max)
            .and_then(|draw| draw.checked_rem(bound))
            .and_then(|draw| u32::try_from(draw).ok())
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use crate::core::SplitMix64;

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
    fn a_bound_of_one_always_draws_zero() {
        let mut rng = SplitMix64::new(3);
        assert_eq!(
            [(); 4].map(|()| rng.below(NonZeroU32::MIN)),
            [Some(0); 4],
            "the only value below 1 is 0"
        );
    }
}
