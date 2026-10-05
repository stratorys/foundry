use foundry_infer_core::{
    ByteSize,
    OpId,
    TensorId,
};
use foundry_infer_plan::{
    BufferSlot,
    EventId,
    StreamId,
    WeightBinding,
};

use crate::error::BackendOperation;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CpuActivity {
    CreateStream,
    Allocate,
    Stage,
    SubmitCopy,
    StreamWait,
    SubmitLaunch,
    HostWait,
    Release,
    CleanupDrain,
}

impl CpuActivity {
    pub const fn name(self) -> &'static str {
        match self {
            Self::CreateStream => "create_stream",
            Self::Allocate => "allocate",
            Self::Stage => "stage",
            Self::SubmitCopy => "submit_copy",
            Self::StreamWait => "stream_wait",
            Self::SubmitLaunch => "submit_launch",
            Self::HostWait => "host_wait",
            Self::Release => "release",
            Self::CleanupDrain => "cleanup_drain",
        }
    }

    pub const fn waits(self) -> bool { matches!(self, Self::HostWait | Self::CleanupDrain) }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CommandContext<'plan> {
    pub index: Option<usize>,
    pub op: Option<OpId>,
    pub tensor: Option<TensorId>,
    pub slot: Option<BufferSlot>,
    pub bindings: &'plan [WeightBinding],
    pub stream: Option<StreamId>,
    pub event: Option<EventId>,
    pub bytes: Option<ByteSize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuRecord<'plan, I> {
    pub context: CommandContext<'plan>,
    pub activity: CpuActivity,
    pub start: I,
    pub end: I,
    pub succeeded: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionOutcome {
    Completed,
    Rejected,
    Failed {
        operation: Option<BackendOperation>,
        cleanup_failed: bool,
    },
}

pub trait ExecutionObserver {
    type Instant: Copy;

    fn now(&mut self) -> Self::Instant;

    fn record(
        &mut self,
        record: CpuRecord<'_, Self::Instant>,
    );

    fn finish(
        &mut self,
        outcome: ExecutionOutcome,
    );
}

pub(crate) struct Unobserved;

impl ExecutionObserver for Unobserved {
    type Instant = ();

    fn now(&mut self) {}

    fn record(
        &mut self,
        _record: CpuRecord<'_, ()>,
    ) {
    }

    fn finish(
        &mut self,
        _outcome: ExecutionOutcome,
    ) {
    }
}
