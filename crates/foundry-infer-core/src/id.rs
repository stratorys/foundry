use std::marker::PhantomData;

use crate::error::CoreError;

mod sealed {
    pub trait Sealed {
        fn from_index(index: u32) -> Self;
    }
}

pub trait Id: sealed::Sealed + Copy {
    fn index(self) -> u32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpId(u32);

impl sealed::Sealed for TensorId {
    fn from_index(index: u32) -> Self { Self(index) }
}

impl Id for TensorId {
    fn index(self) -> u32 { self.0 }
}

impl sealed::Sealed for OpId {
    fn from_index(index: u32) -> Self { Self(index) }
}

impl Id for OpId {
    fn index(self) -> u32 { self.0 }
}

#[derive(Debug)]
pub struct IdSpace<I> {
    next: Option<u32>,
    ids: PhantomData<fn() -> I>,
}

impl<I: Id> IdSpace<I> {
    pub fn new() -> Self { Self::starting_at(0) }

    fn starting_at(index: u32) -> Self {
        Self {
            next: Some(index),
            ids: PhantomData,
        }
    }

    pub fn allocate(&mut self) -> Result<I, CoreError> {
        let index = self.next.ok_or(CoreError::IdsExhausted)?;
        self.next = index.checked_add(1);
        Ok(I::from_index(index))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{
        Id,
        IdSpace,
        OpId,
        TensorId,
    };
    use crate::error::CoreError;

    #[test]
    fn ids_of_one_space_are_unique() -> Result<(), CoreError> {
        let mut space = IdSpace::<TensorId>::new();
        let mut seen = HashSet::new();
        for _ in 0..1000 {
            assert!(seen.insert(space.allocate()?), "an id was allocated twice");
        }
        Ok(())
    }

    #[test]
    fn spaces_are_independent() -> Result<(), CoreError> {
        let mut tensors = IdSpace::<TensorId>::new();
        let mut ops = IdSpace::<OpId>::new();
        assert_eq!(tensors.allocate()?.index(), 0, "first tensor id");
        assert_eq!(tensors.allocate()?.index(), 1, "second tensor id");
        assert_eq!(ops.allocate()?.index(), 0, "first op id");
        Ok(())
    }

    #[test]
    fn an_exhausted_space_reports_it() -> Result<(), CoreError> {
        let mut space = IdSpace::<OpId>::starting_at(u32::MAX - 1);
        assert_eq!(space.allocate()?.index(), u32::MAX - 1, "next to last id");
        assert_eq!(space.allocate()?.index(), u32::MAX, "last id");
        assert_eq!(
            space.allocate(),
            Err(CoreError::IdsExhausted),
            "no id is left"
        );
        assert_eq!(
            space.allocate(),
            Err(CoreError::IdsExhausted),
            "the space stays exhausted"
        );
        Ok(())
    }
}
