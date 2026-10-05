use std::fmt;

use crate::error::CoreError;

const KIB: u64 = 1 << 10;
const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ByteSize(u64);

impl ByteSize {
    pub const fn from_bytes(bytes: u64) -> Self { Self(bytes) }

    pub fn from_kib(kib: u64) -> Result<Self, CoreError> { Self(kib).checked_mul(KIB) }

    pub fn from_mib(mib: u64) -> Result<Self, CoreError> { Self(mib).checked_mul(MIB) }

    pub fn from_gib(gib: u64) -> Result<Self, CoreError> { Self(gib).checked_mul(GIB) }

    pub const fn bytes(self) -> u64 { self.0 }

    #[expect(
        clippy::as_conversions,
        reason = "no lossless conversion from u64 to f64 exists; rounding is acceptable for \
                  display and throughput"
    )]
    pub const fn to_f64_lossy(self) -> f64 { self.0 as f64 }

    pub fn checked_add(
        self,
        other: Self,
    ) -> Result<Self, CoreError> {
        self.0
            .checked_add(other.0)
            .map(Self)
            .ok_or(CoreError::Overflow)
    }

    pub fn checked_mul(
        self,
        factor: u64,
    ) -> Result<Self, CoreError> {
        self.0
            .checked_mul(factor)
            .map(Self)
            .ok_or(CoreError::Overflow)
    }

    pub fn align_up(
        self,
        alignment: Alignment,
    ) -> Result<Self, CoreError> {
        let mask = alignment.bytes().saturating_sub(1);
        self.0
            .checked_add(mask)
            .map(|bytes| Self(bytes & !mask))
            .ok_or(CoreError::Overflow)
    }
}

impl fmt::Display for ByteSize {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let units = [(GIB, "GiB"), (MIB, "MiB"), (KIB, "KiB")];
        match units.into_iter().find(|&(size, _)| self.0 >= size) {
            Some((size, unit)) => write!(
                formatter,
                "{:.1} {unit}",
                self.to_f64_lossy() / Self(size).to_f64_lossy()
            ),
            None => write!(formatter, "{} B", self.0),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Alignment(u64);

impl Alignment {
    pub fn new(bytes: u64) -> Result<Self, CoreError> {
        if bytes.is_power_of_two() {
            Ok(Self(bytes))
        } else {
            Err(CoreError::InvalidAlignment(bytes))
        }
    }

    pub const fn bytes(self) -> u64 { self.0 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemorySpace {
    Host,
    HostPinned,
    Device,
    Unified,
    MappedFile,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceCapabilities {
    pub device_memory: ByteSize,
    pub unified_memory: bool,
    pub async_host_to_device: bool,
    pub concurrent_copy_compute: bool,
    pub alignment: Alignment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemoryBudget {
    space: MemorySpace,
    capacity: ByteSize,
}

impl MemoryBudget {
    pub fn new(
        space: MemorySpace,
        capacity: ByteSize,
    ) -> Self {
        Self {
            space,
            capacity,
        }
    }

    pub fn space(&self) -> MemorySpace { self.space }

    pub fn capacity(&self) -> ByteSize { self.capacity }

    pub fn check(
        &self,
        requested: ByteSize,
    ) -> Result<(), CoreError> {
        if requested <= self.capacity {
            Ok(())
        } else {
            Err(CoreError::BudgetExceeded {
                requested,
                capacity: self.capacity,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Alignment,
        ByteSize,
        DeviceCapabilities,
        MemoryBudget,
        MemorySpace,
    };
    use crate::error::CoreError;

    #[test]
    fn alignment_must_be_a_non_zero_power_of_two() {
        for invalid in [0, 3, 96] {
            assert_eq!(
                Alignment::new(invalid),
                Err(CoreError::InvalidAlignment(invalid)),
                "alignment {invalid} is rejected"
            );
        }
        for valid in [1, 256, 4096] {
            assert_eq!(
                Alignment::new(valid).map(Alignment::bytes),
                Ok(valid),
                "alignment {valid} is accepted"
            );
        }
    }

    #[test]
    fn align_up_rounds_to_the_next_multiple() -> Result<(), CoreError> {
        let alignment = Alignment::new(256)?;
        let cases = [(0, 0), (1, 256), (256, 256), (257, 512)];
        for (bytes, aligned) in cases {
            assert_eq!(
                ByteSize::from_bytes(bytes).align_up(alignment),
                Ok(ByteSize::from_bytes(aligned)),
                "{bytes} aligned to 256"
            );
        }
        assert_eq!(
            ByteSize::from_bytes(u64::MAX).align_up(alignment),
            Err(CoreError::Overflow),
            "aligning u64::MAX overflows"
        );
        Ok(())
    }

    #[test]
    fn unit_constructors_are_binary_and_checked() {
        assert_eq!(
            ByteSize::from_kib(1).map(ByteSize::bytes),
            Ok(1024),
            "1 KiB"
        );
        assert_eq!(
            ByteSize::from_mib(1).map(ByteSize::bytes),
            Ok(1 << 20),
            "1 MiB"
        );
        assert_eq!(
            ByteSize::from_gib(1).map(ByteSize::bytes),
            Ok(1 << 30),
            "1 GiB"
        );
        assert_eq!(
            ByteSize::from_gib(u64::MAX),
            Err(CoreError::Overflow),
            "u64::MAX GiB overflows"
        );
    }

    #[test]
    fn checked_arithmetic_reports_overflow() {
        let max = ByteSize::from_bytes(u64::MAX);
        assert_eq!(
            max.checked_add(ByteSize::from_bytes(1)),
            Err(CoreError::Overflow),
            "addition overflow"
        );
        assert_eq!(
            max.checked_mul(2),
            Err(CoreError::Overflow),
            "multiplication overflow"
        );
        assert_eq!(
            ByteSize::from_bytes(2).checked_add(ByteSize::from_bytes(3)),
            Ok(ByteSize::from_bytes(5)),
            "2 + 3 bytes"
        );
    }

    #[test]
    fn displays_binary_units() -> Result<(), CoreError> {
        let cases = [
            (ByteSize::from_bytes(512), "512 B"),
            (ByteSize::from_kib(2)?, "2.0 KiB"),
            (ByteSize::from_mib(1536)?, "1.5 GiB"),
            (ByteSize::from_bytes(43 << 29), "21.5 GiB"),
        ];
        for (size, text) in cases {
            assert_eq!(size.to_string(), text, "display of {} bytes", size.bytes());
        }
        Ok(())
    }

    #[test]
    fn a_budget_accepts_up_to_its_capacity() -> Result<(), CoreError> {
        let capacity = ByteSize::from_gib(7)?;
        let budget = MemoryBudget::new(MemorySpace::Device, capacity);
        assert_eq!(budget.space(), MemorySpace::Device, "space is kept");
        assert_eq!(budget.capacity(), capacity, "capacity is kept");
        assert_eq!(budget.check(capacity), Ok(()), "exactly the capacity fits");
        let over = capacity.checked_add(ByteSize::from_bytes(1))?;
        assert_eq!(
            budget.check(over),
            Err(CoreError::BudgetExceeded {
                requested: over,
                capacity
            }),
            "one byte over the capacity is rejected"
        );
        Ok(())
    }

    #[test]
    fn capabilities_describe_a_device_without_a_backend() -> Result<(), CoreError> {
        let capabilities = DeviceCapabilities {
            device_memory: ByteSize::from_gib(24)?,
            unified_memory: true,
            async_host_to_device: true,
            concurrent_copy_compute: true,
            alignment: Alignment::new(16_384)?,
        };
        let budget = MemoryBudget::new(MemorySpace::Unified, capabilities.device_memory);
        assert_eq!(
            budget.check(ByteSize::from_gib(25)?),
            Err(CoreError::BudgetExceeded {
                requested: ByteSize::from_gib(25)?,
                capacity: capabilities.device_memory,
            }),
            "a budget built from the capabilities bounds requests"
        );
        Ok(())
    }
}
