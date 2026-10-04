use std::error::Error;
use std::fmt;

use foundry_infer_core::{
    ByteSize,
    CoreError,
    GraphError,
    MemorySpace,
    OpId,
    TensorId,
};

use crate::id::{
    BufferSlot,
    EventId,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    Graph(GraphError),
    BudgetSpace(MemorySpace),
    ZeroAllocation {
        index: usize,
        slot: BufferSlot,
    },
    DoubleAllocation {
        index: usize,
        slot: BufferSlot,
    },
    Memory {
        index: usize,
        slot: BufferSlot,
        source: CoreError,
    },
    UnknownTensor {
        index: usize,
        tensor: TensorId,
    },
    NotAWeight {
        index: usize,
        tensor: TensorId,
    },
    UnknownSlot {
        index: usize,
        slot: BufferSlot,
    },
    UseAfterRelease {
        index: usize,
        slot: BufferSlot,
    },
    SlotTooSmall {
        index: usize,
        slot: BufferSlot,
        capacity: ByteSize,
        required: ByteSize,
    },
    PrematureOverwrite {
        index: usize,
        slot: BufferSlot,
        event: EventId,
    },
    DuplicateEvent {
        index: usize,
        event: EventId,
        first: usize,
    },
    UnknownEvent {
        index: usize,
        event: EventId,
    },
    WaitBeforeProduction {
        index: usize,
        event: EventId,
        producer: usize,
    },
    UnknownOp {
        index: usize,
        op: OpId,
    },
    RepeatedLaunch {
        index: usize,
        op: OpId,
    },
    OutOfOrderLaunch {
        index: usize,
        expected: OpId,
        found: OpId,
    },
    DuplicateBinding {
        index: usize,
        op: OpId,
        tensor: TensorId,
    },
    UnexpectedBinding {
        index: usize,
        op: OpId,
        tensor: TensorId,
    },
    MissingBinding {
        index: usize,
        op: OpId,
        tensor: TensorId,
    },
    StaleBinding {
        index: usize,
        slot: BufferSlot,
        expected: TensorId,
        found: Option<TensorId>,
    },
    TransferNotReady {
        index: usize,
        op: OpId,
        tensor: TensorId,
        event: EventId,
    },
    ProducerNotReady {
        index: usize,
        op: OpId,
        tensor: TensorId,
        producer: OpId,
        event: EventId,
    },
    DoubleRelease {
        index: usize,
        slot: BufferSlot,
    },
    ReleaseBeforeCompletion {
        index: usize,
        slot: BufferSlot,
        event: EventId,
    },
    MissingLaunch {
        op: OpId,
    },
    UnsynchronizedWork {
        event: EventId,
    },
    UnreleasedSlot {
        slot: BufferSlot,
    },
}

impl From<GraphError> for PlanError {
    fn from(error: GraphError) -> Self { Self::Graph(error) }
}

impl fmt::Display for PlanError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Graph(error) => write!(formatter, "invalid graph: {error}"),
            Self::BudgetSpace(space) => {
                write!(
                    formatter,
                    "the plan budget must cover device memory, not {space:?}"
                )
            }
            Self::ZeroAllocation {
                index,
                slot,
            } => write!(formatter, "command {index}: {slot:?} allocates zero bytes"),
            Self::DoubleAllocation {
                index,
                slot,
            } => write!(formatter, "command {index}: {slot:?} is already allocated"),
            Self::Memory {
                index,
                slot,
                source,
            } => write!(
                formatter,
                "command {index}: allocating {slot:?} breaks the slot budget: {source}"
            ),
            Self::UnknownTensor {
                index,
                tensor,
            } => write!(formatter, "command {index}: {tensor:?} is not in the graph"),
            Self::NotAWeight {
                index,
                tensor,
            } => write!(
                formatter,
                "command {index}: {tensor:?} is not a graph weight"
            ),
            Self::UnknownSlot {
                index,
                slot,
            } => write!(formatter, "command {index}: {slot:?} is not allocated"),
            Self::UseAfterRelease {
                index,
                slot,
            } => write!(formatter, "command {index}: {slot:?} is used after release"),
            Self::SlotTooSmall {
                index,
                slot,
                capacity,
                required,
            } => write!(
                formatter,
                "command {index}: {slot:?} holds {capacity}, but the weight needs {required}"
            ),
            Self::PrematureOverwrite {
                index,
                slot,
                event,
            } => write!(
                formatter,
                "command {index}: {slot:?} is overwritten before {event:?} completes"
            ),
            Self::DuplicateEvent {
                index,
                event,
                first,
            } => write!(
                formatter,
                "command {index}: {event:?} is already produced by command {first}"
            ),
            Self::UnknownEvent {
                index,
                event,
            } => write!(formatter, "command {index}: {event:?} is never produced"),
            Self::WaitBeforeProduction {
                index,
                event,
                producer,
            } => write!(
                formatter,
                "command {index}: waits on {event:?}, which command {producer} produces later"
            ),
            Self::UnknownOp {
                index,
                op,
            } => write!(formatter, "command {index}: {op:?} is not in the graph"),
            Self::RepeatedLaunch {
                index,
                op,
            } => write!(formatter, "command {index}: {op:?} is already launched"),
            Self::OutOfOrderLaunch {
                index,
                expected,
                found,
            } => write!(
                formatter,
                "command {index}: launches {found:?}, but {expected:?} is next in graph order"
            ),
            Self::DuplicateBinding {
                index,
                op,
                tensor,
            } => write!(
                formatter,
                "command {index}: {op:?} binds {tensor:?} more than once"
            ),
            Self::UnexpectedBinding {
                index,
                op,
                tensor,
            } => write!(
                formatter,
                "command {index}: {op:?} binds {tensor:?}, which is not a weight it reads"
            ),
            Self::MissingBinding {
                index,
                op,
                tensor,
            } => write!(
                formatter,
                "command {index}: {op:?} does not bind {tensor:?}"
            ),
            Self::StaleBinding {
                index,
                slot,
                expected,
                found,
            } => write!(
                formatter,
                "command {index}: {slot:?} is bound to {expected:?}, but holds {found:?}"
            ),
            Self::TransferNotReady {
                index,
                op,
                tensor,
                event,
            } => write!(
                formatter,
                "command {index}: {op:?} is not ordered after the transfer {event:?} of {tensor:?}"
            ),
            Self::ProducerNotReady {
                index,
                op,
                tensor,
                producer,
                event,
            } => write!(
                formatter,
                "command {index}: {op:?} reads {tensor:?} without waiting for {producer:?} \
                 ({event:?})"
            ),
            Self::DoubleRelease {
                index,
                slot,
            } => write!(formatter, "command {index}: {slot:?} is already released"),
            Self::ReleaseBeforeCompletion {
                index,
                slot,
                event,
            } => write!(
                formatter,
                "command {index}: {slot:?} is released before the host completes {event:?}"
            ),
            Self::MissingLaunch {
                op,
            } => write!(formatter, "{op:?} is never launched"),
            Self::UnsynchronizedWork {
                event,
            } => write!(formatter, "{event:?} is never completed by a host wait"),
            Self::UnreleasedSlot {
                slot,
            } => write!(formatter, "{slot:?} is never released"),
        }
    }
}

impl Error for PlanError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Graph(error) => Some(error),
            Self::Memory {
                source, ..
            } => Some(source),
            Self::BudgetSpace(_)
            | Self::ZeroAllocation {
                ..
            }
            | Self::DoubleAllocation {
                ..
            }
            | Self::UnknownTensor {
                ..
            }
            | Self::NotAWeight {
                ..
            }
            | Self::UnknownSlot {
                ..
            }
            | Self::UseAfterRelease {
                ..
            }
            | Self::SlotTooSmall {
                ..
            }
            | Self::PrematureOverwrite {
                ..
            }
            | Self::DuplicateEvent {
                ..
            }
            | Self::UnknownEvent {
                ..
            }
            | Self::WaitBeforeProduction {
                ..
            }
            | Self::UnknownOp {
                ..
            }
            | Self::RepeatedLaunch {
                ..
            }
            | Self::OutOfOrderLaunch {
                ..
            }
            | Self::DuplicateBinding {
                ..
            }
            | Self::UnexpectedBinding {
                ..
            }
            | Self::MissingBinding {
                ..
            }
            | Self::StaleBinding {
                ..
            }
            | Self::TransferNotReady {
                ..
            }
            | Self::ProducerNotReady {
                ..
            }
            | Self::DoubleRelease {
                ..
            }
            | Self::ReleaseBeforeCompletion {
                ..
            }
            | Self::MissingLaunch {
                ..
            }
            | Self::UnsynchronizedWork {
                ..
            }
            | Self::UnreleasedSlot {
                ..
            } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyntheticPlanError {
    UnsupportedOp { op: OpId },
    WeightCount { op: OpId, count: usize },
    SharedWeight { op: OpId, tensor: TensorId },
    Core(CoreError),
}

impl From<CoreError> for SyntheticPlanError {
    fn from(error: CoreError) -> Self { Self::Core(error) }
}

impl fmt::Display for SyntheticPlanError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::UnsupportedOp {
                op,
            } => write!(formatter, "{op:?} is not a synthetic compute operation"),
            Self::WeightCount {
                op,
                count,
            } => write!(
                formatter,
                "{op:?} reads {count} weights, but the synthetic chain needs exactly one"
            ),
            Self::SharedWeight {
                op,
                tensor,
            } => write!(
                formatter,
                "{op:?} reads {tensor:?}, which an earlier operation already reads"
            ),
            Self::Core(error) => write!(formatter, "{error}"),
        }
    }
}

impl Error for SyntheticPlanError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Core(error) => Some(error),
            Self::UnsupportedOp {
                ..
            }
            | Self::WeightCount {
                ..
            }
            | Self::SharedWeight {
                ..
            } => None,
        }
    }
}

#[derive(Debug)]
pub struct CodecError(serde_json::Error);

impl CodecError {
    pub(crate) fn new(error: serde_json::Error) -> Self { Self(error) }
}

impl fmt::Display for CodecError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(formatter, "execution plan JSON: {}", self.0)
    }
}

impl Error for CodecError {
    fn source(&self) -> Option<&(dyn Error + 'static)> { Some(&self.0) }
}
