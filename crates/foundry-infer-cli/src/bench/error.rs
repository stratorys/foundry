use std::error::Error;
use std::fmt;
use std::num::{
    NonZeroU32,
    NonZeroU64,
};

use foundry_infer_core::{
    ByteSize,
    CoreError,
    GraphError,
    Id,
    OpId,
    TensorId,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use foundry_infer_metal::MetalError;
use foundry_infer_plan::{
    PlanError,
    SyntheticPlanError,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use foundry_infer_runtime::RuntimeError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Warmup,
    Measured,
}

impl fmt::Display for Phase {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Warmup => formatter.write_str("warmup"),
            Self::Measured => formatter.write_str("measured"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Round {
    pub(crate) phase: Phase,
    pub(crate) index: u32,
    pub(crate) buffers: u32,
}

impl fmt::Display for Round {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(
            formatter,
            "{} round {} with {} buffers",
            self.phase, self.index, self.buffers
        )
    }
}

#[derive(Debug)]
pub(crate) enum BenchError {
    WeightOverflow {
        weight_mib: NonZeroU64,
    },
    TotalOverflow {
        layers: NonZeroU32,
        weight_mib: NonZeroU64,
    },
    HostAddressSpace {
        what: &'static str,
        bytes: ByteSize,
    },
    SlotAlignment {
        weight: ByteSize,
    },
    BufferLimit {
        slot: ByteSize,
        limit: ByteSize,
    },
    SlotBudgetOverflow {
        slots: u32,
        weight: ByteSize,
    },
    HostAllocation {
        layer: u32,
        bytes: ByteSize,
    },
    WeightTooLargeForKernel {
        weight: ByteSize,
    },
    BudgetExceedsWorkingSet {
        budget: ByteSize,
        working_set: ByteSize,
        device: String,
    },
    InvalidBufferCount {
        buffers: u32,
    },
    MissingWeight {
        op: OpId,
    },
    MissingPayload {
        tensor: TensorId,
    },
    UnsupportedOp {
        op: OpId,
    },
    RoundsOverflow {
        warmup: u32,
        iterations: NonZeroU32,
    },
    MissingConfiguration {
        buffers: u32,
    },
    ChecksumMismatch {
        round: Round,
        op: OpId,
        expected: u32,
        actual: Vec<u32>,
    },
    NoSamples {
        buffers: u32,
    },
    MissingBaseline,
    ZeroMedian {
        buffers: u32,
    },
    Core(CoreError),
    Graph(GraphError),
    SyntheticPlan(SyntheticPlanError),
    Plan(PlanError),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(MetalError),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Execution {
        round: Round,
        source: RuntimeError<MetalError>,
    },
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Drain {
        round: Round,
        source: MetalError,
    },
    #[cfg(all(feature = "metal", target_os = "macos"))]
    ChecksumRead {
        round: Round,
        op: OpId,
        source: MetalError,
    },
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    MetalUnavailable,
}

impl fmt::Display for BenchError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::WeightOverflow {
                weight_mib,
            } => write!(
                formatter,
                "a weight of {weight_mib} MiB overflows a 64-bit byte count"
            ),
            Self::TotalOverflow {
                layers,
                weight_mib,
            } => write!(
                formatter,
                "{layers} layers of {weight_mib} MiB overflow a 64-bit byte count"
            ),
            Self::HostAddressSpace {
                what,
                bytes,
            } => write!(
                formatter,
                "the {what} of {bytes} exceeds the host address space"
            ),
            Self::SlotAlignment {
                weight,
            } => write!(formatter, "aligning a {weight} weight slot overflows"),
            Self::BufferLimit {
                slot,
                limit,
            } => write!(
                formatter,
                "an aligned weight slot of {slot} exceeds the {limit} Metal buffer length limit"
            ),
            Self::SlotBudgetOverflow {
                slots,
                weight,
            } => write!(
                formatter,
                "the slot budget for {slots} aligned {weight} slots overflows"
            ),
            Self::HostAllocation {
                layer,
                bytes,
            } => write!(
                formatter,
                "cannot allocate {bytes} of host payload for layer {layer}"
            ),
            Self::WeightTooLargeForKernel {
                weight,
            } => write!(
                formatter,
                "the Metal checksum kernel supports weights below 4 GiB, got {weight}"
            ),
            Self::BudgetExceedsWorkingSet {
                budget,
                working_set,
                device,
            } => write!(
                formatter,
                "the device slot budget of {budget} exceeds the {working_set} recommended working \
                 set of {device}"
            ),
            Self::InvalidBufferCount {
                buffers,
            } => write!(formatter, "buffer count {buffers} is not positive"),
            Self::MissingWeight {
                op,
            } => write!(formatter, "synthetic layer {} has no weight", op.index()),
            Self::MissingPayload {
                tensor,
            } => write!(formatter, "weight tensor {} has no payload", tensor.index()),
            Self::UnsupportedOp {
                op,
            } => write!(
                formatter,
                "op {} is not a synthetic compute layer",
                op.index()
            ),
            Self::RoundsOverflow {
                warmup,
                iterations,
            } => write!(
                formatter,
                "{warmup} warmup and {iterations} measured iterations overflow"
            ),
            Self::MissingConfiguration {
                buffers,
            } => write!(formatter, "missing configuration with {buffers} buffers"),
            Self::ChecksumMismatch {
                round,
                op,
                expected,
                actual,
            } => write!(
                formatter,
                "{round}: layer {} checksum mismatch: expected {expected:#010x}, got \
                 {actual:#010x?}",
                op.index()
            ),
            Self::NoSamples {
                buffers,
            } => write!(formatter, "{buffers} buffers have no samples"),
            Self::MissingBaseline => formatter.write_str("the one-buffer baseline is missing"),
            Self::ZeroMedian {
                buffers,
            } => write!(
                formatter,
                "the median elapsed time with {buffers} buffers is zero; throughput and speedup \
                 are undefined"
            ),
            Self::Core(error) => write!(formatter, "{error}"),
            Self::Graph(error) => write!(formatter, "{error}"),
            Self::SyntheticPlan(error) => write!(formatter, "{error}"),
            Self::Plan(error) => write!(formatter, "{error}"),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(error) => write!(formatter, "{error}"),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Execution {
                round,
                source,
            } => write!(formatter, "{round} failed: {source}"),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Drain {
                round,
                source,
            } => write!(formatter, "{round} failed to drain: {source}"),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::ChecksumRead {
                round,
                op,
                source,
            } => write!(
                formatter,
                "{round}: reading the checksums of layer {} failed: {source}",
                op.index()
            ),
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            Self::MetalUnavailable => formatter.write_str(
                "Metal benchmarking requires a macOS build of foundry with `--features metal`",
            ),
        }
    }
}

impl Error for BenchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Core(error) => Some(error),
            Self::Graph(error) => Some(error),
            Self::SyntheticPlan(error) => Some(error),
            Self::Plan(error) => Some(error),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(error)
            | Self::Drain {
                source: error, ..
            }
            | Self::ChecksumRead {
                source: error, ..
            } => Some(error),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Execution {
                source, ..
            } => Some(source),
            Self::WeightOverflow {
                ..
            }
            | Self::TotalOverflow {
                ..
            }
            | Self::HostAddressSpace {
                ..
            }
            | Self::SlotAlignment {
                ..
            }
            | Self::BufferLimit {
                ..
            }
            | Self::SlotBudgetOverflow {
                ..
            }
            | Self::HostAllocation {
                ..
            }
            | Self::WeightTooLargeForKernel {
                ..
            }
            | Self::BudgetExceedsWorkingSet {
                ..
            }
            | Self::InvalidBufferCount {
                ..
            }
            | Self::MissingWeight {
                ..
            }
            | Self::MissingPayload {
                ..
            }
            | Self::UnsupportedOp {
                ..
            }
            | Self::RoundsOverflow {
                ..
            }
            | Self::MissingConfiguration {
                ..
            }
            | Self::ChecksumMismatch {
                ..
            }
            | Self::NoSamples {
                ..
            }
            | Self::MissingBaseline
            | Self::ZeroMedian {
                ..
            } => None,
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            Self::MetalUnavailable => None,
        }
    }
}

impl From<CoreError> for BenchError {
    fn from(error: CoreError) -> Self { Self::Core(error) }
}

impl From<GraphError> for BenchError {
    fn from(error: GraphError) -> Self { Self::Graph(error) }
}

impl From<SyntheticPlanError> for BenchError {
    fn from(error: SyntheticPlanError) -> Self { Self::SyntheticPlan(error) }
}

impl From<PlanError> for BenchError {
    fn from(error: PlanError) -> Self { Self::Plan(error) }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl From<MetalError> for BenchError {
    fn from(error: MetalError) -> Self { Self::Metal(error) }
}
