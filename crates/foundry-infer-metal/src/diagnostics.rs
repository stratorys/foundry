use std::collections::BTreeMap;
use std::mem;

use foundry_infer_core::{
    ByteSize,
    OpId,
    TensorId,
};
use foundry_infer_plan::{
    BufferSlot,
    Command,
    EventId,
    ExecutionPlan,
    StreamId,
};
use foundry_infer_runtime::CommandContext;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer,
    MTLCommandBufferStatus,
    MTLDevice,
};

use crate::clock::{
    HostClock,
    HostInstant,
    gpu_seconds,
};
use crate::error::MetalError;
use crate::submission::Device;

pub const CHECKSUM_KERNEL: &str = "checksum";

const BOUNDARIES: usize = 3;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CommandLabel {
    pub index: Option<usize>,
    pub op: Option<OpId>,
    pub tensor: Option<TensorId>,
    pub slot: Option<BufferSlot>,
    pub stream: Option<StreamId>,
    pub event: Option<EventId>,
    pub bytes: Option<ByteSize>,
}

impl From<&CommandContext<'_>> for CommandLabel {
    fn from(context: &CommandContext<'_>) -> Self {
        Self {
            index: context.index,
            op: context.op,
            tensor: context.tensor,
            slot: context
                .slot
                .or_else(|| context.bindings.first().map(|binding| binding.slot)),
            stream: context.stream,
            event: context.event,
            bytes: context.bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuWork {
    Copy,
    Compute,
}

impl GpuWork {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Copy => "copy",
            Self::Compute => "compute",
        }
    }

    pub const fn kernel(self) -> Option<&'static str> {
        match self {
            Self::Copy => None,
            Self::Compute => Some(CHECKSUM_KERNEL),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TimestampFault {
    NotFinished,
    StartUnavailable,
    EndUnavailable,
    EndBeforeStart,
}

impl TimestampFault {
    pub const fn reason(self) -> &'static str {
        match self {
            Self::NotFinished => "the command buffer had not finished when it was retired",
            Self::StartUnavailable => "GPUStartTime was zero, negative or not finite",
            Self::EndUnavailable => "GPUEndTime was zero, negative or not finite",
            Self::EndBeforeStart => "GPUEndTime precedes GPUStartTime",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuTiming {
    Valid {
        start: HostInstant,
        end: HostInstant,
    },
    Invalid(TimestampFault),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuInterval {
    pub label: CommandLabel,
    pub work: GpuWork,
    pub submitted: HostInstant,
    pub completed: bool,
    pub timing: GpuTiming,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Submission {
    pub(crate) label: CommandLabel,
    pub(crate) work: GpuWork,
    pub(crate) submitted: HostInstant,
}

pub(crate) fn interval(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    submission: Submission,
) -> GpuInterval {
    let status = command_buffer.status();
    let completed = status == MTLCommandBufferStatus::Completed;
    let finished = completed || status == MTLCommandBufferStatus::Error;
    let timing = if finished {
        match (
            gpu_seconds(command_buffer.GPUStartTime()),
            gpu_seconds(command_buffer.GPUEndTime()),
        ) {
            (None, _) => GpuTiming::Invalid(TimestampFault::StartUnavailable),
            (_, None) => GpuTiming::Invalid(TimestampFault::EndUnavailable),
            (Some(start), Some(end)) if end < start => {
                GpuTiming::Invalid(TimestampFault::EndBeforeStart)
            }
            (Some(start), Some(end)) => GpuTiming::Valid {
                start,
                end,
            },
        }
    } else {
        GpuTiming::Invalid(TimestampFault::NotFinished)
    };
    GpuInterval {
        label: submission.label,
        work: submission.work,
        submitted: submission.submitted,
        completed,
        timing,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AllocationCategory {
    Staging,
    PrivateSlot,
    ChecksumResult,
}

impl AllocationCategory {
    pub const ALL: [Self; 3] = [Self::Staging, Self::PrivateSlot, Self::ChecksumResult];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Staging => "shared_staging",
            Self::PrivateSlot => "private_weight_slot",
            Self::ChecksumResult => "shared_checksum_result",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationChange {
    Allocated,
    Freed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocationEvent {
    pub id: u64,
    pub category: AllocationCategory,
    pub change: AllocationChange,
    pub at: HostInstant,
    pub requested: u64,
    pub allocated: u64,
    pub label: CommandLabel,
    pub device_current: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub allocations: u64,
    pub frees: u64,
    pub live_requested: u64,
    pub live_allocated: u64,
    pub peak_requested: u64,
    pub peak_allocated: u64,
}

impl Usage {
    fn allocate(
        &mut self,
        requested: u64,
        allocated: u64,
    ) {
        self.allocations = self.allocations.saturating_add(1);
        self.live_requested = self.live_requested.saturating_add(requested);
        self.live_allocated = self.live_allocated.saturating_add(allocated);
        self.peak_requested = self.peak_requested.max(self.live_requested);
        self.peak_allocated = self.peak_allocated.max(self.live_allocated);
    }

    fn free(
        &mut self,
        requested: u64,
        allocated: u64,
    ) {
        self.frees = self.frees.saturating_add(1);
        self.live_requested = self.live_requested.saturating_sub(requested);
        self.live_allocated = self.live_allocated.saturating_sub(allocated);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Boundary {
    ExecutionStart,
    AfterDrain,
    AfterChecksumRelease,
}

impl Boundary {
    pub const fn name(self) -> &'static str {
        match self {
            Self::ExecutionStart => "execution_start",
            Self::AfterDrain => "after_drain",
            Self::AfterChecksumRelease => "after_checksum_release",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemorySnapshot {
    pub boundary: Boundary,
    pub at: HostInstant,
    pub categories: BTreeMap<AllocationCategory, Usage>,
    pub tracked: Usage,
    pub live_allocations: usize,
    pub device_current: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiagnosticLimits {
    gpu_intervals: usize,
    allocation_events: usize,
}

impl DiagnosticLimits {
    pub fn for_plan(plan: &ExecutionPlan) -> Result<Self, MetalError> {
        let (allocations, submissions) = plan.commands().iter().fold(
            (0_usize, 0_usize),
            |(allocations, submissions), command| match command {
                Command::Allocate {
                    ..
                } => (allocations.saturating_add(1), submissions),
                Command::Prefetch {
                    ..
                }
                | Command::Launch {
                    ..
                } => (allocations.saturating_add(1), submissions.saturating_add(1)),
                Command::Wait {
                    ..
                }
                | Command::Release {
                    ..
                } => (allocations, submissions),
            },
        );
        let allocation_events = allocations
            .checked_mul(2)
            .ok_or(MetalError::DiagnosticLimit {
                what: "allocation events",
            })?;
        Ok(Self {
            gpu_intervals: submissions,
            allocation_events,
        })
    }

    pub const fn gpu_intervals(self) -> usize { self.gpu_intervals }

    pub const fn allocation_events(self) -> usize { self.allocation_events }
}

fn reserve<T>(
    limit: usize,
    what: &'static str,
) -> Result<Vec<T>, MetalError> {
    let mut records = Vec::new();
    records
        .try_reserve_exact(limit)
        .map_err(|_| MetalError::DiagnosticLimit {
            what,
        })?;
    Ok(records)
}

fn storage_bytes<T>(records: &Vec<T>) -> u64 {
    records
        .capacity()
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .unwrap_or(u64::MAX)
}

pub(crate) struct GpuLog {
    intervals: Vec<GpuInterval>,
    truncated: u64,
}

impl GpuLog {
    pub(crate) fn new(limits: DiagnosticLimits) -> Result<Self, MetalError> {
        Ok(Self {
            intervals: reserve(limits.gpu_intervals, "GPU intervals")?,
            truncated: 0,
        })
    }

    pub(crate) fn push(
        &mut self,
        interval: GpuInterval,
    ) {
        if self.intervals.len() < self.intervals.capacity() {
            self.intervals.push(interval);
        } else {
            self.truncated = self.truncated.saturating_add(1);
        }
    }
}

struct LiveAllocation {
    category: AllocationCategory,
    requested: u64,
    allocated: u64,
    label: CommandLabel,
}

pub(crate) struct Ledger {
    clock: HostClock,
    device: Device,
    next_id: u64,
    live: BTreeMap<u64, LiveAllocation>,
    categories: BTreeMap<AllocationCategory, Usage>,
    tracked: Usage,
    events: Vec<AllocationEvent>,
    truncated: u64,
    snapshots: Vec<MemorySnapshot>,
    device_sampled_max: u64,
    device_samples: u64,
}

impl Ledger {
    pub(crate) fn new(
        clock: HostClock,
        device: Device,
        limits: DiagnosticLimits,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            clock,
            device,
            next_id: 0,
            live: BTreeMap::new(),
            categories: AllocationCategory::ALL
                .into_iter()
                .map(|category| (category, Usage::default()))
                .collect(),
            tracked: Usage::default(),
            events: reserve(limits.allocation_events, "allocation events")?,
            truncated: 0,
            snapshots: reserve(BOUNDARIES, "memory snapshots")?,
            device_sampled_max: 0,
            device_samples: 0,
        })
    }

    fn sample_device(&mut self) -> u64 {
        let current = u64::try_from(self.device.currentAllocatedSize()).unwrap_or(u64::MAX);
        self.device_sampled_max = self.device_sampled_max.max(current);
        self.device_samples = self.device_samples.saturating_add(1);
        current
    }

    fn event(
        &mut self,
        id: u64,
        change: AllocationChange,
        allocation: &LiveAllocation,
    ) {
        let device_current = self.sample_device();
        let event = AllocationEvent {
            id,
            category: allocation.category,
            change,
            at: self.clock.now(),
            requested: allocation.requested,
            allocated: allocation.allocated,
            label: allocation.label,
            device_current,
        };
        if self.events.len() < self.events.capacity() {
            self.events.push(event);
        } else {
            self.truncated = self.truncated.saturating_add(1);
        }
    }

    pub(crate) fn allocate(
        &mut self,
        category: AllocationCategory,
        requested: u64,
        allocated: u64,
        label: CommandLabel,
    ) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let allocation = LiveAllocation {
            category,
            requested,
            allocated,
            label,
        };
        if let Some(usage) = self.categories.get_mut(&category) {
            usage.allocate(requested, allocated);
        }
        self.tracked.allocate(requested, allocated);
        self.event(id, AllocationChange::Allocated, &allocation);
        self.live.insert(id, allocation);
        id
    }

    pub(crate) fn free(
        &mut self,
        id: u64,
    ) {
        let Some(allocation) = self.live.remove(&id) else {
            return;
        };
        if let Some(usage) = self.categories.get_mut(&allocation.category) {
            usage.free(allocation.requested, allocation.allocated);
        }
        self.tracked
            .free(allocation.requested, allocation.allocated);
        self.event(id, AllocationChange::Freed, &allocation);
    }

    pub(crate) fn snapshot(
        &mut self,
        boundary: Boundary,
    ) {
        let device_current = self.sample_device();
        let snapshot = MemorySnapshot {
            boundary,
            at: self.clock.now(),
            categories: self.categories.clone(),
            tracked: self.tracked,
            live_allocations: self.live.len(),
            device_current,
        };
        if self.snapshots.len() < self.snapshots.capacity() {
            self.snapshots.push(snapshot);
        } else {
            self.truncated = self.truncated.saturating_add(1);
        }
    }

    #[cfg(test)]
    pub(crate) fn live_usage(
        &self,
        category: AllocationCategory,
    ) -> Option<(u64, u64)> {
        self.categories.get(&category).map(|usage| {
            (
                usage.allocations.saturating_sub(usage.frees),
                usage.live_requested,
            )
        })
    }

    fn storage_bytes(&self) -> u64 {
        storage_bytes(&self.events).saturating_add(storage_bytes(&self.snapshots))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalDiagnostics {
    pub gpu: Vec<GpuInterval>,
    pub gpu_truncated: u64,
    pub gpu_pending: usize,
    pub allocations: Vec<AllocationEvent>,
    pub allocations_truncated: u64,
    pub categories: BTreeMap<AllocationCategory, Usage>,
    pub tracked: Usage,
    pub snapshots: Vec<MemorySnapshot>,
    pub live_allocations: usize,
    pub device_sampled_max: u64,
    pub device_samples: u64,
    pub storage_bytes: u64,
}

impl MetalDiagnostics {
    pub(crate) fn extract(
        ledger: &mut Ledger,
        log: GpuLog,
        gpu_pending: usize,
    ) -> Self {
        let storage = ledger
            .storage_bytes()
            .saturating_add(storage_bytes(&log.intervals));
        Self {
            gpu: log.intervals,
            gpu_truncated: log.truncated,
            gpu_pending,
            allocations: mem::take(&mut ledger.events),
            allocations_truncated: ledger.truncated,
            categories: ledger.categories.clone(),
            tracked: ledger.tracked,
            snapshots: mem::take(&mut ledger.snapshots),
            live_allocations: ledger.live.len(),
            device_sampled_max: ledger.device_sampled_max,
            device_samples: ledger.device_samples,
            storage_bytes: storage,
        }
    }

    pub fn gpu_timestamps_valid(&self) -> bool {
        self.gpu_truncated == 0
            && self.gpu_pending == 0
            && self
                .gpu
                .iter()
                .all(|interval| matches!(interval.timing, GpuTiming::Valid { .. }))
    }
}
