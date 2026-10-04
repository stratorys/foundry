use foundry_infer_core::{
    ByteSize,
    Graph,
    MemoryBudget,
    OpId,
    TensorId,
};

use crate::error::PlanError;
use crate::id::{
    BufferSlot,
    EventId,
    StreamId,
};
use crate::validate;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WaitTarget {
    Host,
    Stream(StreamId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightBinding {
    pub tensor: TensorId,
    pub slot: BufferSlot,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Command {
    Allocate {
        slot: BufferSlot,
        bytes: ByteSize,
    },
    Prefetch {
        tensor: TensorId,
        slot: BufferSlot,
        stream: StreamId,
        event: EventId,
    },
    Wait {
        target: WaitTarget,
        event: EventId,
    },
    Launch {
        op: OpId,
        stream: StreamId,
        weight_slots: Vec<WeightBinding>,
        event: EventId,
    },
    Release {
        slot: BufferSlot,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ExecutionPlan {
    commands: Vec<Command>,
}

impl ExecutionPlan {
    pub fn new(commands: Vec<Command>) -> Self {
        Self {
            commands,
        }
    }

    pub fn commands(&self) -> &[Command] { &self.commands }

    pub fn validate(
        &self,
        graph: &Graph,
        budget: MemoryBudget,
    ) -> Result<(), PlanError> {
        validate::validate(&self.commands, graph, budget)
    }
}
