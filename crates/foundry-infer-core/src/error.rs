use std::error::Error;
use std::fmt;

use crate::id::{
    OpId,
    TensorId,
};
use crate::memory::ByteSize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreError {
    Overflow,
    IdsExhausted,
    InvalidAlignment(u64),
    BudgetExceeded {
        requested: ByteSize,
        capacity: ByteSize,
    },
}

impl fmt::Display for CoreError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Overflow => formatter.write_str("size computation overflows 64 bits"),
            Self::IdsExhausted => formatter.write_str("identifier space is exhausted"),
            Self::InvalidAlignment(value) => {
                write!(
                    formatter,
                    "alignment {value} is not a non-zero power of two"
                )
            }
            Self::BudgetExceeded {
                requested,
                capacity,
            } => write!(
                formatter,
                "{requested} requested exceeds the {capacity} memory budget"
            ),
        }
    }
}

impl Error for CoreError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphError {
    Core(CoreError),
    UnknownTensor(TensorId),
    UnknownOperand {
        op: OpId,
        tensor: TensorId,
    },
    DuplicateInput(TensorId),
    DuplicateWeight(TensorId),
    InputWeightConflict(TensorId),
    TensorSize {
        tensor: TensorId,
        source: CoreError,
    },
    WeightSizeMismatch {
        tensor: TensorId,
        expected: ByteSize,
        actual: ByteSize,
    },
    MappedLenMismatch {
        tensor: TensorId,
        len: ByteSize,
        bytes: ByteSize,
    },
    MappedRangeOverflow {
        tensor: TensorId,
        offset: u64,
        len: ByteSize,
    },
    UninitializedRead {
        op: OpId,
        tensor: TensorId,
    },
    OverwritesInput {
        op: OpId,
        tensor: TensorId,
    },
    OverwritesWeight {
        op: OpId,
        tensor: TensorId,
    },
    MultipleProducers {
        tensor: TensorId,
        first: OpId,
        second: OpId,
    },
}

impl From<CoreError> for GraphError {
    fn from(error: CoreError) -> Self { Self::Core(error) }
}

impl fmt::Display for GraphError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Core(error) => write!(formatter, "{error}"),
            Self::UnknownTensor(tensor) => write!(formatter, "{tensor:?} is not in the graph"),
            Self::UnknownOperand {
                op,
                tensor,
            } => write!(
                formatter,
                "{op:?} refers to {tensor:?}, which is not in the graph"
            ),
            Self::DuplicateInput(tensor) => {
                write!(formatter, "{tensor:?} is already a graph input")
            }
            Self::DuplicateWeight(tensor) => {
                write!(formatter, "{tensor:?} already has a weight description")
            }
            Self::InputWeightConflict(tensor) => {
                write!(
                    formatter,
                    "{tensor:?} cannot be both a graph input and a weight"
                )
            }
            Self::TensorSize {
                tensor,
                source,
            } => write!(
                formatter,
                "byte size of {tensor:?} is not computable: {source}"
            ),
            Self::WeightSizeMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "weight of {tensor:?} describes {actual}, but the tensor holds {expected}"
            ),
            Self::MappedLenMismatch {
                tensor,
                len,
                bytes,
            } => write!(
                formatter,
                "mapped range of {tensor:?} spans {len}, but the weight holds {bytes}"
            ),
            Self::MappedRangeOverflow {
                tensor,
                offset,
                len,
            } => write!(
                formatter,
                "mapped range of {tensor:?} at offset {offset} with {len} overflows 64 bits"
            ),
            Self::UninitializedRead {
                op,
                tensor,
            } => write!(
                formatter,
                "{op:?} reads {tensor:?} before it is initialized"
            ),
            Self::OverwritesInput {
                op,
                tensor,
            } => write!(formatter, "{op:?} overwrites graph input {tensor:?}"),
            Self::OverwritesWeight {
                op,
                tensor,
            } => write!(formatter, "{op:?} overwrites weight {tensor:?}"),
            Self::MultipleProducers {
                tensor,
                first,
                second,
            } => write!(
                formatter,
                "{second:?} writes {tensor:?}, which {first:?} already produces"
            ),
        }
    }
}

impl Error for GraphError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Core(error)
            | Self::TensorSize {
                source: error, ..
            } => Some(error),
            Self::UnknownTensor(_)
            | Self::UnknownOperand {
                ..
            }
            | Self::DuplicateInput(_)
            | Self::DuplicateWeight(_)
            | Self::InputWeightConflict(_)
            | Self::WeightSizeMismatch {
                ..
            }
            | Self::MappedLenMismatch {
                ..
            }
            | Self::MappedRangeOverflow {
                ..
            }
            | Self::UninitializedRead {
                ..
            }
            | Self::OverwritesInput {
                ..
            }
            | Self::OverwritesWeight {
                ..
            }
            | Self::MultipleProducers {
                ..
            } => None,
        }
    }
}
