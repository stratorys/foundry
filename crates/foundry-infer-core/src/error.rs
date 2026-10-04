use std::error::Error;
use std::fmt;

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
