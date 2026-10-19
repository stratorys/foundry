use std::collections::BTreeMap;

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ProfileMode {
    Steps,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceProfile {
    pub kernels: BTreeMap<&'static str, KernelProfile>,
    pub allocation_count: u64,
    pub allocated_bytes: u64,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct KernelProfile {
    pub dispatch_count: u64,
}

impl DeviceProfile {
    pub fn count_dispatch(
        &mut self,
        kernel: &'static str,
    ) {
        let kernel = self.kernels.entry(kernel).or_default();
        kernel.dispatch_count = kernel.dispatch_count.saturating_add(1);
    }

    pub fn count_allocation(
        &mut self,
        bytes: u64,
    ) {
        self.allocation_count = self.allocation_count.saturating_add(1);
        self.allocated_bytes = self.allocated_bytes.saturating_add(bytes);
    }

    pub fn dispatch_count(&self) -> u64 {
        self.kernels.values().fold(0, |total, kernel| {
            total.saturating_add(kernel.dispatch_count)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::nn::{
        DeviceProfile,
        KernelProfile,
    };

    #[test]
    fn counts_accumulate_per_kernel_and_in_total() {
        let mut profile = DeviceProfile::default();
        ["binary_add", "matmul", "binary_add"]
            .into_iter()
            .for_each(|kernel| profile.count_dispatch(kernel));
        [16, 0, 4096]
            .into_iter()
            .for_each(|bytes| profile.count_allocation(bytes));
        assert_eq!(
            profile,
            DeviceProfile {
                kernels: BTreeMap::from([
                    (
                        "binary_add",
                        KernelProfile {
                            dispatch_count: 2,
                        }
                    ),
                    (
                        "matmul",
                        KernelProfile {
                            dispatch_count: 1,
                        }
                    ),
                ]),
                allocation_count: 3,
                allocated_bytes: 4112,
            },
            "dispatches grouped by kernel, allocations and bytes summed"
        );
        assert_eq!(profile.dispatch_count(), 3, "total over every kernel");
    }

    #[test]
    fn counts_saturate_instead_of_wrapping() {
        let mut profile = DeviceProfile {
            kernels: BTreeMap::from([(
                "copy",
                KernelProfile {
                    dispatch_count: u64::MAX,
                },
            )]),
            allocation_count: u64::MAX,
            allocated_bytes: u64::MAX,
        };
        profile.count_dispatch("copy");
        profile.count_dispatch("matmul");
        profile.count_allocation(1);
        assert_eq!(
            (
                profile.dispatch_count(),
                profile.allocation_count,
                profile.allocated_bytes
            ),
            (u64::MAX, u64::MAX, u64::MAX),
            "every counter saturates at u64::MAX"
        );
    }
}
