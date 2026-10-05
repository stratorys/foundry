use crate::error::MetalError;

const NANOS_PER_SECOND: f64 = 1e9;

#[repr(C)]
#[derive(Default)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

// SAFETY: both functions are part of libSystem on every macOS release,
// `mach_absolute_time` has no preconditions, and `mach_timebase_info` only
// writes the two fields of the `MachTimebaseInfo` it is given.
unsafe extern "C" {
    safe fn mach_absolute_time() -> u64;
    unsafe fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HostInstant(u64);

impl HostInstant {
    pub const fn from_nanos(nanos: u64) -> Self { Self(nanos) }

    pub const fn nanos(self) -> u64 { self.0 }

    pub const fn nanos_since(
        self,
        origin: Self,
    ) -> Option<u64> {
        self.0.checked_sub(origin.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostClock {
    numer: u32,
    denom: u32,
}

impl HostClock {
    pub fn new() -> Result<Self, MetalError> {
        let mut info = MachTimebaseInfo::default();
        // SAFETY: `info` is a valid, exclusively borrowed `MachTimebaseInfo`
        // with the C layout the function expects.
        let status = unsafe { mach_timebase_info(&raw mut info) };
        if status != 0 || info.numer == 0 || info.denom == 0 {
            return Err(MetalError::HostClock {
                status,
            });
        }
        Ok(Self {
            numer: info.numer,
            denom: info.denom,
        })
    }

    pub fn now(&self) -> HostInstant { self.instant_at(mach_absolute_time()) }

    pub const fn timebase(&self) -> (u32, u32) { (self.numer, self.denom) }

    fn instant_at(
        &self,
        ticks: u64,
    ) -> HostInstant {
        let nanos = u128::from(ticks)
            .saturating_mul(u128::from(self.numer))
            .checked_div(u128::from(self.denom))
            .unwrap_or_default();
        HostInstant(u64::try_from(nanos).unwrap_or(u64::MAX))
    }
}

#[expect(
    clippy::as_conversions,
    reason = "no lossless conversion from f64 to u64 exists; the value is checked to be finite, \
              positive and in range before the saturating cast"
)]
pub fn gpu_seconds(seconds: f64) -> Option<HostInstant> {
    let nanos = (seconds * NANOS_PER_SECOND).round();
    (nanos.is_finite() && nanos > 0.0 && nanos < u64::MAX as f64)
        .then_some(HostInstant(nanos as u64))
}

#[cfg(test)]
mod tests {
    use super::{
        HostClock,
        HostInstant,
        gpu_seconds,
    };

    #[test]
    fn gpu_seconds_convert_to_host_nanoseconds() {
        assert_eq!(
            gpu_seconds(1.5),
            Some(HostInstant::from_nanos(1_500_000_000)),
            "seconds scale to nanoseconds"
        );
        assert_eq!(
            gpu_seconds(123.000_000_001),
            Some(HostInstant::from_nanos(123_000_000_001)),
            "nanoseconds are rounded"
        );
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e30] {
            assert_eq!(gpu_seconds(invalid), None, "{invalid} is not a timestamp");
        }
    }

    #[test]
    fn the_host_clock_is_monotonic_and_scaled() -> Result<(), Box<dyn std::error::Error>> {
        let clock = HostClock::new()?;
        let (numer, denom) = clock.timebase();
        assert!(numer > 0 && denom > 0, "the timebase is valid");
        let first = clock.now();
        let second = clock.now();
        assert!(second >= first, "the clock never goes backwards");
        assert_eq!(
            clock.instant_at(u64::from(denom)),
            HostInstant::from_nanos(u64::from(numer)),
            "ticks scale by the timebase"
        );
        assert_eq!(
            second.nanos_since(first),
            second.nanos().checked_sub(first.nanos()),
            "intervals are differences of nanoseconds"
        );
        Ok(())
    }
}
