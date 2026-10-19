use std::collections::BTreeMap;

use serde::Serialize;

const NS_PER_S: f64 = 1e9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ProfileMode {
    Steps,
    // One dispatch per command buffer: per-kernel GPU time, perturbed wall time.
    Kernels,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize)]
pub struct DeviceProfile {
    pub kernels: BTreeMap<String, KernelProfile>,
    pub allocation_count: u64,
    pub allocated_bytes: u64,
    pub gpu_busy_ns: Option<f64>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize)]
pub struct KernelProfile {
    pub dispatch_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_ns: Option<f64>,
}

// Host times in seconds, as Metal reports them; `kernel` is set only when the
// command buffer ran that single kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuInterval {
    pub kernel: Option<String>,
    pub start_s: f64,
    pub end_s: f64,
}

impl DeviceProfile {
    pub fn count_dispatch(
        &mut self,
        kernel: String,
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

    pub fn with_gpu_intervals(
        mut self,
        intervals: Vec<GpuInterval>,
    ) -> Self {
        self.gpu_busy_ns = Some(busy_ns(
            intervals
                .iter()
                .map(|interval| (interval.start_s, interval.end_s))
                .collect(),
        ));
        intervals.into_iter().for_each(|interval| {
            if let Some(kernel) = interval.kernel {
                let kernel = self.kernels.entry(kernel).or_default();
                let gpu_ns = (interval.end_s - interval.start_s) * NS_PER_S;
                kernel.gpu_ns = Some(kernel.gpu_ns.unwrap_or(0.0) + gpu_ns);
            }
        });
        self
    }
}

// Overlapping intervals count once.
fn busy_ns(mut intervals_s: Vec<(f64, f64)>) -> f64 {
    intervals_s.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
    let (busy_s, _) = intervals_s.into_iter().fold(
        (0.0, f64::NEG_INFINITY),
        |(busy_s, covered_end_s), (start_s, end_s)| {
            let start_s = start_s.max(covered_end_s);
            if end_s > start_s {
                (busy_s + (end_s - start_s), end_s)
            } else {
                (busy_s, covered_end_s)
            }
        },
    );
    busy_s * NS_PER_S
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::nn::profile::busy_ns;
    use crate::nn::{
        DeviceProfile,
        GpuInterval,
        KernelProfile,
    };

    #[test]
    fn gpu_intervals_add_busy_time_and_time_per_single_kernel_buffer() {
        let interval = |kernel, start_s, end_s| GpuInterval {
            kernel,
            start_s,
            end_s,
        };
        let profile = DeviceProfile {
            kernels: BTreeMap::from([(
                "binary_add".to_owned(),
                KernelProfile {
                    dispatch_count: 2,
                    gpu_ns: None,
                },
            )]),
            ..DeviceProfile::default()
        }
        .with_gpu_intervals(vec![
            interval(Some("binary_add".to_owned()), 0.0, 0.5),
            interval(Some("matmul".to_owned()), 0.25, 1.0),
            interval(Some("binary_add".to_owned()), 1.0, 1.25),
            interval(None, 2.0, 2.5),
        ]);
        assert_eq!(
            profile,
            DeviceProfile {
                kernels: BTreeMap::from([
                    (
                        "binary_add".to_owned(),
                        KernelProfile {
                            dispatch_count: 2,
                            gpu_ns: Some(0.75e9),
                        }
                    ),
                    (
                        "matmul".to_owned(),
                        KernelProfile {
                            dispatch_count: 0,
                            gpu_ns: Some(0.75e9),
                        }
                    ),
                ]),
                gpu_busy_ns: Some(1.75e9),
                ..DeviceProfile::default()
            },
            "kernel times are summed per kernel, busy time merges overlaps and keeps untagged \
             buffers"
        );
        assert_eq!(
            DeviceProfile::default().with_gpu_intervals(Vec::new()),
            DeviceProfile {
                gpu_busy_ns: Some(0.0),
                ..DeviceProfile::default()
            },
            "no completed command buffer means zero busy time and no kernel time"
        );
    }

    #[test]
    fn busy_time_counts_overlapping_intervals_once() {
        assert_eq!(
            [
                vec![],
                vec![(1.0, 1.5)],
                vec![(0.0, 0.5), (1.0, 1.25)],
                vec![(0.0, 0.5), (0.25, 1.0)],
                vec![(0.0, 1.0), (0.25, 0.5)],
                vec![(2.0, 2.5), (0.25, 1.0), (2.125, 2.25), (0.0, 0.5)],
            ]
            .map(busy_ns),
            [0.0, 0.5e9, 0.75e9, 1e9, 1e9, 1.5e9],
            "empty, single, disjoint, overlapping, nested, then unsorted intervals"
        );
    }

    #[test]
    fn counts_accumulate_per_kernel_and_in_total() {
        let mut profile = DeviceProfile::default();
        ["binary_add", "matmul", "binary_add"]
            .into_iter()
            .for_each(|kernel| profile.count_dispatch(kernel.to_owned()));
        [16, 0, 4096]
            .into_iter()
            .for_each(|bytes| profile.count_allocation(bytes));
        assert_eq!(
            profile,
            DeviceProfile {
                kernels: BTreeMap::from([
                    (
                        "binary_add".to_owned(),
                        KernelProfile {
                            dispatch_count: 2,
                            gpu_ns: None,
                        }
                    ),
                    (
                        "matmul".to_owned(),
                        KernelProfile {
                            dispatch_count: 1,
                            gpu_ns: None,
                        }
                    ),
                ]),
                allocation_count: 3,
                allocated_bytes: 4112,
                gpu_busy_ns: None,
            },
            "dispatches grouped by kernel, allocations and bytes summed"
        );
        assert_eq!(profile.dispatch_count(), 3, "total over every kernel");
    }

    #[test]
    fn counts_saturate_instead_of_wrapping() {
        let mut profile = DeviceProfile {
            kernels: BTreeMap::from([(
                "copy".to_owned(),
                KernelProfile {
                    dispatch_count: u64::MAX,
                    gpu_ns: None,
                },
            )]),
            allocation_count: u64::MAX,
            allocated_bytes: u64::MAX,
            gpu_busy_ns: None,
        };
        profile.count_dispatch("copy".to_owned());
        profile.count_dispatch("matmul".to_owned());
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
