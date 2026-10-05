use std::error::Error;

use foundry_infer_core::{
    ByteSize,
    Graph,
    Id,
    MemoryBudget,
    MemorySpace,
    OpId,
    TensorId,
};

use crate::id::{
    BufferSlot,
    EventId,
    StreamId,
};
use crate::plan::{
    Command,
    ExecutionPlan,
    WaitTarget,
    WeightBinding,
};
use crate::synthetic::synthetic_chain_graph;

pub type TestResult = Result<(), Box<dyn Error>>;

pub const WEIGHT: u64 = 1024;
pub const COPY: StreamId = StreamId::new(0);
pub const COMPUTE: StreamId = StreamId::new(1);

pub fn chain(layers: u32) -> Result<Graph, Box<dyn Error>> {
    Ok(synthetic_chain_graph(
        layers,
        ByteSize::from_bytes(WEIGHT),
        None,
    )?)
}

pub fn op(layer: u32) -> OpId { OpId::from_index(layer) }

pub fn input() -> TensorId { TensorId::from_index(0) }

pub fn weight(layer: u32) -> TensorId {
    TensorId::from_index(layer.saturating_mul(2).saturating_add(1))
}

pub fn output(layer: u32) -> TensorId {
    TensorId::from_index(layer.saturating_mul(2).saturating_add(2))
}

pub fn device(bytes: u64) -> MemoryBudget {
    MemoryBudget::new(MemorySpace::Device, ByteSize::from_bytes(bytes))
}

pub fn budget() -> MemoryBudget { device(1 << 20) }

pub fn plan(commands: impl Into<Vec<Command>>) -> ExecutionPlan {
    ExecutionPlan::new(commands.into())
}

pub fn allocate(
    slot: u32,
    bytes: u64,
) -> Command {
    Command::Allocate {
        slot: BufferSlot::new(slot),
        bytes: ByteSize::from_bytes(bytes),
    }
}

pub fn prefetch(
    tensor: TensorId,
    slot: u32,
    stream: StreamId,
    event: u32,
) -> Command {
    Command::Prefetch {
        tensor,
        slot: BufferSlot::new(slot),
        stream,
        event: EventId::new(event),
    }
}

pub fn wait_on(
    stream: StreamId,
    event: u32,
) -> Command {
    Command::Wait {
        target: WaitTarget::Stream(stream),
        event: EventId::new(event),
    }
}

pub fn wait_host(event: u32) -> Command {
    Command::Wait {
        target: WaitTarget::Host,
        event: EventId::new(event),
    }
}

pub fn bind(
    tensor: TensorId,
    slot: u32,
) -> WeightBinding {
    WeightBinding {
        tensor,
        slot: BufferSlot::new(slot),
    }
}

pub fn launch(
    op: OpId,
    stream: StreamId,
    weight_slots: impl Into<Vec<WeightBinding>>,
    event: u32,
) -> Command {
    Command::Launch {
        op,
        stream,
        weight_slots: weight_slots.into(),
        event: EventId::new(event),
    }
}

pub fn release(slot: u32) -> Command {
    Command::Release {
        slot: BufferSlot::new(slot),
    }
}

pub fn one_layer() -> Vec<Command> {
    vec![
        allocate(0, WEIGHT),
        prefetch(weight(0), 0, COPY, 0),
        wait_on(COMPUTE, 0),
        launch(op(0), COMPUTE, [bind(weight(0), 0)], 1),
        wait_host(1),
        release(0),
    ]
}

pub fn two_layers_one_slot() -> Vec<Command> {
    vec![
        allocate(0, WEIGHT),
        prefetch(weight(0), 0, COPY, 0),
        wait_on(COMPUTE, 0),
        launch(op(0), COMPUTE, [bind(weight(0), 0)], 1),
        wait_on(COPY, 1),
        prefetch(weight(1), 0, COPY, 2),
        wait_on(COMPUTE, 2),
        launch(op(1), COMPUTE, [bind(weight(1), 0)], 3),
        wait_host(3),
        release(0),
    ]
}

pub fn two_layers_two_slots() -> Vec<Command> {
    vec![
        allocate(0, WEIGHT),
        allocate(1, WEIGHT),
        prefetch(weight(0), 0, COPY, 0),
        prefetch(weight(1), 1, COPY, 2),
        wait_on(COMPUTE, 0),
        launch(op(0), COMPUTE, [bind(weight(0), 0)], 1),
        wait_on(COMPUTE, 2),
        launch(op(1), COMPUTE, [bind(weight(1), 1)], 3),
        wait_host(3),
        release(0),
        release(1),
    ]
}

pub fn without(
    mut commands: Vec<Command>,
    index: usize,
) -> Vec<Command> {
    commands.remove(index);
    commands
}

pub fn inserted(
    mut commands: Vec<Command>,
    index: usize,
    command: Command,
) -> Vec<Command> {
    commands.insert(index, command);
    commands
}

pub fn replaced(
    commands: Vec<Command>,
    index: usize,
    command: Command,
) -> Vec<Command> {
    inserted(without(commands, index), index, command)
}
