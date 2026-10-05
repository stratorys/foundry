const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_FIRST: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_SECOND: u64 = 0x94D0_49BB_1331_11EB;
const REJECTIONS_MAX: u32 = 128;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) const fn new(seed: u64) -> Self {
        Self {
            state: seed,
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let first = (self.state ^ (self.state >> 30)).wrapping_mul(MIX_FIRST);
        let second = (first ^ (first >> 27)).wrapping_mul(MIX_SECOND);
        second ^ (second >> 31)
    }

    pub(crate) fn below(
        &mut self,
        bound: u64,
    ) -> Option<u64> {
        let threshold = bound.wrapping_neg().checked_rem(bound)?;
        (0..REJECTIONS_MAX)
            .map(|_| self.next_u64())
            .find(|&value| value >= threshold)
            .and_then(|value| value.checked_rem(bound))
    }

    pub(crate) fn index_below(
        &mut self,
        length: usize,
    ) -> Option<usize> {
        let bound = u64::try_from(length).ok()?;
        usize::try_from(self.below(bound)?).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::SplitMix64;

    #[test]
    fn the_generator_matches_the_reference_sequence() {
        let mut generator = SplitMix64::new(0);
        let values: Vec<u64> = (0..3).map(|_| generator.next_u64()).collect();
        assert_eq!(
            values,
            [
                0xE220_A839_7B1D_CDAF,
                0x6E78_9E6A_A1B9_65F4,
                0x06C4_5D18_8009_454F
            ],
            "SplitMix64 seeded with 0"
        );
        let mut generator = SplitMix64::new(1_234_567);
        assert_eq!(
            generator.next_u64(),
            0x599E_D017_FB08_FC85,
            "SplitMix64 seeded with 1234567"
        );
    }

    #[test]
    fn bounded_draws_reject_the_biased_range() {
        let mut generator = SplitMix64::new(7);
        assert_eq!(generator.below(0), None, "an empty range has no value");
        assert!(
            (0..1000).all(|_| generator.below(1) == Some(0)),
            "a single value is always drawn"
        );
        assert!(
            (0..1000).all(|_| generator.below(30).is_some_and(|value| value < 30)),
            "draws stay below their bound"
        );
        let bound = (u64::MAX / 2).saturating_add(2);
        let threshold = bound.wrapping_neg() % bound;
        let mut reference = SplitMix64::new(11);
        let mut drawn = SplitMix64::new(11);
        for _ in 0..100 {
            let expected = loop {
                let value = reference.next_u64();
                if value >= threshold {
                    break value % bound;
                }
            };
            assert_eq!(
                drawn.below(bound),
                Some(expected),
                "values below the threshold are rejected"
            );
        }
    }

    #[test]
    fn index_draws_cover_every_index() {
        let mut generator = SplitMix64::new(3);
        let mut seen = [false; 5];
        for _ in 0..200 {
            let index = generator.index_below(5);
            if let Some(slot) = index.and_then(|index| seen.get_mut(index)) {
                *slot = true;
            }
        }
        assert!(seen.iter().all(|&seen| seen), "every index is drawn");
        assert_eq!(generator.index_below(0), None, "no index below zero");
    }
}
