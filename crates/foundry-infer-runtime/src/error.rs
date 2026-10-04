use std::error::Error;
use std::fmt;

use foundry_infer_core::{
    ByteSize,
    OpId,
    TensorId,
};
use foundry_infer_plan::{
    BufferSlot,
    EventId,
    PlanError,
    StreamId,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendOperation {
    CreateStream {
        stream: StreamId,
    },
    Allocate {
        index: usize,
        slot: BufferSlot,
    },
    StageHost {
        index: usize,
        tensor: TensorId,
    },
    Copy {
        index: usize,
        tensor: TensorId,
        slot: BufferSlot,
        stream: StreamId,
        event: EventId,
    },
    Launch {
        index: usize,
        op: OpId,
        stream: StreamId,
        event: EventId,
    },
    StreamWait {
        index: usize,
        stream: StreamId,
        event: EventId,
    },
    HostWait {
        index: usize,
        event: EventId,
    },
}

impl fmt::Display for BackendOperation {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::CreateStream {
                stream,
            } => write!(formatter, "creating {stream:?}"),
            Self::Allocate {
                index,
                slot,
            } => write!(formatter, "command {index}: allocating {slot:?}"),
            Self::StageHost {
                index,
                tensor,
            } => write!(
                formatter,
                "command {index}: staging {tensor:?} in host memory"
            ),
            Self::Copy {
                index,
                tensor,
                slot,
                stream,
                event,
            } => write!(
                formatter,
                "command {index}: copying {tensor:?} to {slot:?} on {stream:?} for {event:?}"
            ),
            Self::Launch {
                index,
                op,
                stream,
                event,
            } => write!(
                formatter,
                "command {index}: launching {op:?} on {stream:?} for {event:?}"
            ),
            Self::StreamWait {
                index,
                stream,
                event,
            } => write!(
                formatter,
                "command {index}: making {stream:?} wait on {event:?}"
            ),
            Self::HostWait {
                index,
                event,
            } => write!(formatter, "command {index}: waiting on {event:?}"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ExecutionFailure<E> {
    Backend {
        operation: BackendOperation,
        source: E,
    },
    MissingStream {
        index: usize,
        stream: StreamId,
    },
    MissingSlot {
        index: usize,
        slot: BufferSlot,
    },
    MissingEvent {
        index: usize,
        event: EventId,
    },
    MissingOp {
        index: usize,
        op: OpId,
    },
    MissingHostWeight {
        index: usize,
        tensor: TensorId,
    },
}

impl<E: fmt::Display> fmt::Display for ExecutionFailure<E> {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Backend {
                operation,
                source,
            } => write!(formatter, "{operation} failed: {source}"),
            Self::MissingStream {
                index,
                stream,
            } => write!(formatter, "command {index}: {stream:?} was not created"),
            Self::MissingSlot {
                index,
                slot,
            } => write!(formatter, "command {index}: {slot:?} has no device buffer"),
            Self::MissingEvent {
                index,
                event,
            } => write!(formatter, "command {index}: {event:?} was not submitted"),
            Self::MissingOp {
                index,
                op,
            } => write!(formatter, "command {index}: {op:?} has no prepared launch"),
            Self::MissingHostWeight {
                index,
                tensor,
            } => write!(
                formatter,
                "command {index}: {tensor:?} has no host weight binding"
            ),
        }
    }
}

impl<E: Error + 'static> Error for ExecutionFailure<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Backend {
                source, ..
            } => Some(source),
            Self::MissingStream {
                ..
            }
            | Self::MissingSlot {
                ..
            }
            | Self::MissingEvent {
                ..
            }
            | Self::MissingOp {
                ..
            }
            | Self::MissingHostWeight {
                ..
            } => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum RuntimeError<E> {
    Plan(PlanError),
    UnsupportedOp {
        op: OpId,
    },
    UnsupportedWeightSource {
        tensor: TensorId,
    },
    MissingHostWeight {
        index: usize,
        tensor: TensorId,
    },
    HostWeightSize {
        index: usize,
        tensor: TensorId,
        expected: ByteSize,
        actual: ByteSize,
    },
    BudgetExceedsDevice {
        budget: ByteSize,
        device: ByteSize,
    },
    Execution {
        failure: ExecutionFailure<E>,
        cleanup: Option<E>,
    },
}

impl<E> RuntimeError<E> {
    pub fn cleanup(&self) -> Option<&E> {
        match self {
            Self::Execution {
                cleanup, ..
            } => cleanup.as_ref(),
            Self::Plan(_)
            | Self::UnsupportedOp {
                ..
            }
            | Self::UnsupportedWeightSource {
                ..
            }
            | Self::MissingHostWeight {
                ..
            }
            | Self::HostWeightSize {
                ..
            }
            | Self::BudgetExceedsDevice {
                ..
            } => None,
        }
    }
}

impl<E> From<PlanError> for RuntimeError<E> {
    fn from(error: PlanError) -> Self { Self::Plan(error) }
}

impl<E: fmt::Display> fmt::Display for RuntimeError<E> {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Plan(error) => write!(formatter, "invalid execution plan: {error}"),
            Self::UnsupportedOp {
                op,
            } => write!(formatter, "{op:?} is not supported by the interpreter"),
            Self::UnsupportedWeightSource {
                tensor,
            } => write!(
                formatter,
                "weight {tensor:?} is file-backed, which the interpreter does not load"
            ),
            Self::MissingHostWeight {
                index,
                tensor,
            } => write!(
                formatter,
                "command {index}: {tensor:?} has no host weight binding"
            ),
            Self::HostWeightSize {
                index,
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "command {index}: host weight of {tensor:?} holds {actual}, but the weight needs \
                 {expected}"
            ),
            Self::BudgetExceedsDevice {
                budget,
                device,
            } => write!(
                formatter,
                "the {budget} budget exceeds the {device} of device memory"
            ),
            Self::Execution {
                failure,
                cleanup,
            } => {
                write!(formatter, "{failure}")?;
                match cleanup {
                    Some(cleanup) => write!(
                        formatter,
                        "; draining submitted work also failed: {cleanup}"
                    ),
                    None => Ok(()),
                }
            }
        }
    }
}

impl<E: Error + 'static> Error for RuntimeError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Plan(error) => Some(error),
            Self::Execution {
                failure, ..
            } => Some(failure),
            Self::UnsupportedOp {
                ..
            }
            | Self::UnsupportedWeightSource {
                ..
            }
            | Self::MissingHostWeight {
                ..
            }
            | Self::HostWeightSize {
                ..
            }
            | Self::BudgetExceedsDevice {
                ..
            } => None,
        }
    }
}
