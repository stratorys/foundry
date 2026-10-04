use std::collections::{
    BTreeMap,
    BTreeSet,
};

use foundry_infer_core::{
    ByteSize,
    Graph,
    MemoryBudget,
    MemorySpace,
    Op,
    OpId,
    TensorId,
};

use crate::error::PlanError;
use crate::id::{
    BufferSlot,
    EventId,
    StreamId,
};
use crate::order::{
    Access,
    Order,
};
use crate::plan::{
    Command,
    WaitTarget,
    WeightBinding,
};

struct Allocation {
    bytes: ByteSize,
    contents: Option<(TensorId, Access)>,
    accesses: Vec<Access>,
}

enum Slot {
    Allocated(Allocation),
    Released,
}

pub fn validate(
    commands: &[Command],
    graph: &Graph,
    budget: MemoryBudget,
) -> Result<(), PlanError> {
    graph.validate()?;
    if budget.space() != MemorySpace::Device {
        return Err(PlanError::BudgetSpace(budget.space()));
    }
    let mut validator = Validator::new(commands, graph, budget);
    for (index, command) in commands.iter().enumerate() {
        validator.command(index, command)?;
    }
    validator.finish()
}

fn reads(op: &Op) -> Vec<TensorId> {
    match op {
        Op::MatMul {
            input,
            weight,
            ..
        } => vec![*input, *weight],
        Op::SyntheticCompute {
            inputs, ..
        } => inputs.clone(),
    }
}

struct Validator<'graph> {
    graph: &'graph Graph,
    budget: MemoryBudget,
    ops: Vec<(OpId, &'graph Op)>,
    positions: BTreeMap<OpId, usize>,
    weights: BTreeMap<TensorId, ByteSize>,
    producers: BTreeMap<TensorId, OpId>,
    events: BTreeMap<EventId, usize>,
    order: Order,
    slots: BTreeMap<BufferSlot, Slot>,
    launched: BTreeMap<OpId, Access>,
    work: Vec<Access>,
}

impl<'graph> Validator<'graph> {
    fn new(
        commands: &[Command],
        graph: &'graph Graph,
        budget: MemoryBudget,
    ) -> Self {
        let ops = graph.ops().collect::<Vec<_>>();
        let positions = ops
            .iter()
            .enumerate()
            .map(|(position, (op, _))| (*op, position))
            .collect();
        let weights = graph
            .weights()
            .iter()
            .map(|weight| (weight.tensor, weight.bytes))
            .collect();
        let producers = ops.iter().map(|(id, op)| (op.output(), *id)).collect();
        let mut events = BTreeMap::new();
        for (index, command) in commands.iter().enumerate() {
            match command {
                Command::Prefetch {
                    event, ..
                }
                | Command::Launch {
                    event, ..
                } => {
                    events.entry(*event).or_insert(index);
                }
                Command::Allocate {
                    ..
                }
                | Command::Wait {
                    ..
                }
                | Command::Release {
                    ..
                } => {}
            }
        }
        Self {
            graph,
            budget,
            ops,
            positions,
            weights,
            producers,
            events,
            order: Order::default(),
            slots: BTreeMap::new(),
            launched: BTreeMap::new(),
            work: Vec::new(),
        }
    }

    fn command(
        &mut self,
        index: usize,
        command: &Command,
    ) -> Result<(), PlanError> {
        match command {
            Command::Allocate {
                slot,
                bytes,
            } => self.allocate(index, *slot, *bytes),
            Command::Prefetch {
                tensor,
                slot,
                stream,
                event,
            } => self.prefetch(index, *tensor, *slot, *stream, *event),
            Command::Wait {
                target,
                event,
            } => self.wait(index, *target, *event),
            Command::Launch {
                op,
                stream,
                weight_slots,
                event,
            } => self.launch(index, *op, *stream, weight_slots, *event),
            Command::Release {
                slot,
            } => self.release(index, *slot),
        }
    }

    fn allocate(
        &mut self,
        index: usize,
        slot: BufferSlot,
        bytes: ByteSize,
    ) -> Result<(), PlanError> {
        if bytes.bytes() == 0 {
            return Err(PlanError::ZeroAllocation {
                index,
                slot,
            });
        }
        if let Some(Slot::Allocated(_)) = self.slots.get(&slot) {
            return Err(PlanError::DoubleAllocation {
                index,
                slot,
            });
        }
        let memory = |source| PlanError::Memory {
            index,
            slot,
            source,
        };
        let live = self
            .slots
            .values()
            .filter_map(|state| match state {
                Slot::Allocated(allocation) => Some(allocation.bytes),
                Slot::Released => None,
            })
            .try_fold(bytes, ByteSize::checked_add)
            .map_err(memory)?;
        self.budget.check(live).map_err(memory)?;
        self.slots.insert(
            slot,
            Slot::Allocated(Allocation {
                bytes,
                contents: None,
                accesses: Vec::new(),
            }),
        );
        Ok(())
    }

    fn prefetch(
        &mut self,
        index: usize,
        tensor: TensorId,
        slot: BufferSlot,
        stream: StreamId,
        event: EventId,
    ) -> Result<(), PlanError> {
        let required = match self.weights.get(&tensor) {
            Some(bytes) => *bytes,
            None if self.graph.tensor(tensor).is_none() => {
                return Err(PlanError::UnknownTensor {
                    index,
                    tensor,
                });
            }
            None => {
                return Err(PlanError::NotAWeight {
                    index,
                    tensor,
                });
            }
        };
        let ready = self.order.ready(stream);
        let allocation = self.allocation(index, slot)?;
        if allocation.bytes < required {
            return Err(PlanError::SlotTooSmall {
                index,
                slot,
                capacity: allocation.bytes,
                required,
            });
        }
        if let Some(pending) = allocation
            .accesses
            .iter()
            .find(|access| !ready.covers(access))
        {
            return Err(PlanError::PrematureOverwrite {
                index,
                slot,
                event: pending.event,
            });
        }
        let access = self.produce(index, stream, event)?;
        let allocation = self.allocation_mut(index, slot)?;
        allocation.contents = Some((tensor, access));
        allocation.accesses.push(access);
        Ok(())
    }

    fn wait(
        &mut self,
        index: usize,
        target: WaitTarget,
        event: EventId,
    ) -> Result<(), PlanError> {
        match self.events.get(&event) {
            None => {
                return Err(PlanError::UnknownEvent {
                    index,
                    event,
                });
            }
            Some(&producer) if producer > index => {
                return Err(PlanError::WaitBeforeProduction {
                    index,
                    event,
                    producer,
                });
            }
            Some(_) => {}
        }
        match target {
            WaitTarget::Host => self.order.wait_host(event),
            WaitTarget::Stream(stream) => self.order.wait_stream(stream, event),
        }
        Ok(())
    }

    fn launch(
        &mut self,
        index: usize,
        op: OpId,
        stream: StreamId,
        bindings: &[WeightBinding],
        event: EventId,
    ) -> Result<(), PlanError> {
        let next = self.launched.len();
        let position = *self.positions.get(&op).ok_or(PlanError::UnknownOp {
            index,
            op,
        })?;
        if position < next {
            return Err(PlanError::RepeatedLaunch {
                index,
                op,
            });
        }
        if let Some((expected, _)) = self.ops.get(next).filter(|_| position > next) {
            return Err(PlanError::OutOfOrderLaunch {
                index,
                expected: *expected,
                found: op,
            });
        }
        let reads = self
            .ops
            .get(position)
            .map(|(_, graph_op)| reads(graph_op))
            .unwrap_or_default();
        let required = reads
            .iter()
            .copied()
            .filter(|tensor| self.weights.contains_key(tensor))
            .collect::<BTreeSet<_>>();

        let mut bound = BTreeSet::new();
        for binding in bindings {
            if !bound.insert(binding.tensor) {
                return Err(PlanError::DuplicateBinding {
                    index,
                    op,
                    tensor: binding.tensor,
                });
            }
            if !required.contains(&binding.tensor) {
                return Err(PlanError::UnexpectedBinding {
                    index,
                    op,
                    tensor: binding.tensor,
                });
            }
        }
        if let Some(&tensor) = required.difference(&bound).next() {
            return Err(PlanError::MissingBinding {
                index,
                op,
                tensor,
            });
        }

        let ready = self.order.ready(stream);
        for binding in bindings {
            let allocation = self.allocation(index, binding.slot)?;
            match allocation.contents {
                Some((tensor, transfer)) if tensor == binding.tensor => {
                    if !ready.covers(&transfer) {
                        return Err(PlanError::TransferNotReady {
                            index,
                            op,
                            tensor,
                            event: transfer.event,
                        });
                    }
                }
                Some(_) | None => {
                    return Err(PlanError::StaleBinding {
                        index,
                        slot: binding.slot,
                        expected: binding.tensor,
                        found: allocation.contents.map(|(tensor, _)| tensor),
                    });
                }
            }
        }
        for tensor in &reads {
            let Some(&producer) = self.producers.get(tensor) else {
                continue;
            };
            let Some(access) = self.launched.get(&producer) else {
                continue;
            };
            if !ready.covers(access) {
                return Err(PlanError::ProducerNotReady {
                    index,
                    op,
                    tensor: *tensor,
                    producer,
                    event: access.event,
                });
            }
        }

        let access = self.produce(index, stream, event)?;
        self.launched.insert(op, access);
        for binding in bindings {
            self.allocation_mut(index, binding.slot)?
                .accesses
                .push(access);
        }
        Ok(())
    }

    fn release(
        &mut self,
        index: usize,
        slot: BufferSlot,
    ) -> Result<(), PlanError> {
        match self.slots.get(&slot) {
            None => {
                return Err(PlanError::UnknownSlot {
                    index,
                    slot,
                });
            }
            Some(Slot::Released) => {
                return Err(PlanError::DoubleRelease {
                    index,
                    slot,
                });
            }
            Some(Slot::Allocated(allocation)) => {
                let host = self.order.host();
                if let Some(pending) = allocation
                    .accesses
                    .iter()
                    .find(|access| !host.covers(access))
                {
                    return Err(PlanError::ReleaseBeforeCompletion {
                        index,
                        slot,
                        event: pending.event,
                    });
                }
            }
        }
        self.slots.insert(slot, Slot::Released);
        Ok(())
    }

    fn produce(
        &mut self,
        index: usize,
        stream: StreamId,
        event: EventId,
    ) -> Result<Access, PlanError> {
        match self.events.get(&event) {
            Some(&first) if first != index => Err(PlanError::DuplicateEvent {
                index,
                event,
                first,
            }),
            Some(_) | None => {
                let access = self.order.submit(stream, event);
                self.work.push(access);
                Ok(access)
            }
        }
    }

    fn allocation(
        &self,
        index: usize,
        slot: BufferSlot,
    ) -> Result<&Allocation, PlanError> {
        match self.slots.get(&slot) {
            Some(Slot::Allocated(allocation)) => Ok(allocation),
            Some(Slot::Released) => Err(PlanError::UseAfterRelease {
                index,
                slot,
            }),
            None => Err(PlanError::UnknownSlot {
                index,
                slot,
            }),
        }
    }

    fn allocation_mut(
        &mut self,
        index: usize,
        slot: BufferSlot,
    ) -> Result<&mut Allocation, PlanError> {
        match self.slots.get_mut(&slot) {
            Some(Slot::Allocated(allocation)) => Ok(allocation),
            Some(Slot::Released) => Err(PlanError::UseAfterRelease {
                index,
                slot,
            }),
            None => Err(PlanError::UnknownSlot {
                index,
                slot,
            }),
        }
    }

    fn finish(self) -> Result<(), PlanError> {
        if let Some((op, _)) = self.ops.get(self.launched.len()) {
            return Err(PlanError::MissingLaunch {
                op: *op,
            });
        }
        let host = self.order.host();
        if let Some(access) = self.work.iter().find(|access| !host.covers(access)) {
            return Err(PlanError::UnsynchronizedWork {
                event: access.event,
            });
        }
        if let Some((slot, _)) = self
            .slots
            .iter()
            .find(|(_, state)| matches!(state, Slot::Allocated(_)))
        {
            return Err(PlanError::UnreleasedSlot {
                slot: *slot,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use foundry_infer_core::{
        ByteSize,
        CoreError,
        DType,
        Graph,
        GraphError,
        Id,
        MemoryBudget,
        MemorySpace,
        Op,
        OpId,
        Shape,
        TensorDesc,
        TensorId,
    };

    use crate::error::PlanError;
    use crate::fixtures::{
        COMPUTE,
        COPY,
        TestResult,
        WEIGHT,
        allocate,
        bind,
        budget,
        chain,
        device,
        input,
        inserted,
        launch,
        one_layer,
        op,
        output,
        plan,
        prefetch,
        release,
        replaced,
        two_layers_one_slot,
        two_layers_two_slots,
        wait_host,
        wait_on,
        weight,
        without,
    };
    use crate::id::{
        BufferSlot,
        EventId,
        StreamId,
    };
    use crate::plan::{
        Command,
        ExecutionPlan,
    };

    fn check(
        graph: &Graph,
        commands: Vec<Command>,
    ) -> Result<(), PlanError> {
        plan(commands).validate(graph, budget())
    }

    fn slot(index: u32) -> BufferSlot { BufferSlot::new(index) }

    fn event(index: u32) -> EventId { EventId::new(index) }

    #[test]
    fn an_empty_plan_validates_only_an_empty_graph() -> TestResult {
        assert_eq!(
            ExecutionPlan::default().validate(&Graph::new(), budget()),
            Ok(()),
            "nothing to schedule"
        );
        assert_eq!(
            check(&chain(1)?, Vec::new()),
            Err(PlanError::MissingLaunch {
                op: op(0)
            }),
            "the op is never launched"
        );
        Ok(())
    }

    #[test]
    fn fixture_identifiers_match_the_graph() -> TestResult {
        let graph = chain(2)?;
        assert_eq!(graph.inputs(), &[input()], "input");
        let weights = graph
            .weights()
            .iter()
            .map(|weight| weight.tensor)
            .collect::<Vec<_>>();
        assert_eq!(weights, [weight(0), weight(1)], "weights");
        let ops = graph
            .ops()
            .map(|(id, op)| (id, op.output()))
            .collect::<Vec<_>>();
        assert_eq!(
            ops,
            [(op(0), output(0)), (op(1), output(1))],
            "ops and outputs"
        );
        Ok(())
    }

    #[test]
    fn hand_written_plans_validate() -> TestResult {
        assert_eq!(check(&chain(1)?, one_layer()), Ok(()), "one layer");
        assert_eq!(
            check(&chain(2)?, two_layers_one_slot()),
            Ok(()),
            "sequential reuse of one slot"
        );
        assert_eq!(
            check(&chain(2)?, two_layers_two_slots()),
            Ok(()),
            "two resident slots"
        );
        Ok(())
    }

    #[test]
    fn unknown_identifiers_are_rejected() -> TestResult {
        let graph = chain(1)?;
        let foreign_op = OpId::from_index(9);
        let foreign_tensor = TensorId::from_index(99);
        let cases = [
            (
                replaced(
                    one_layer(),
                    3,
                    launch(foreign_op, COMPUTE, [bind(weight(0), 0)], 1),
                ),
                PlanError::UnknownOp {
                    index: 3,
                    op: foreign_op,
                },
            ),
            (
                replaced(one_layer(), 1, prefetch(foreign_tensor, 0, COPY, 0)),
                PlanError::UnknownTensor {
                    index: 1,
                    tensor: foreign_tensor,
                },
            ),
            (
                replaced(one_layer(), 1, prefetch(input(), 0, COPY, 0)),
                PlanError::NotAWeight {
                    index: 1,
                    tensor: input(),
                },
            ),
            (
                replaced(one_layer(), 1, prefetch(weight(0), 7, COPY, 0)),
                PlanError::UnknownSlot {
                    index: 1,
                    slot: slot(7),
                },
            ),
            (
                replaced(one_layer(), 5, release(7)),
                PlanError::UnknownSlot {
                    index: 5,
                    slot: slot(7),
                },
            ),
            (
                replaced(one_layer(), 2, wait_on(COMPUTE, 9)),
                PlanError::UnknownEvent {
                    index: 2,
                    event: event(9),
                },
            ),
        ];
        for (commands, error) in cases {
            assert_eq!(check(&graph, commands), Err(error), "{error}");
        }
        Ok(())
    }

    #[test]
    fn events_are_produced_once_before_any_wait() -> TestResult {
        let graph = chain(1)?;
        assert_eq!(
            check(
                &graph,
                replaced(
                    one_layer(),
                    3,
                    launch(op(0), COMPUTE, [bind(weight(0), 0)], 0)
                )
            ),
            Err(PlanError::DuplicateEvent {
                index: 3,
                event: event(0),
                first: 1
            }),
            "the launch reuses the transfer event"
        );
        assert_eq!(
            check(&graph, inserted(one_layer(), 1, wait_host(1))),
            Err(PlanError::WaitBeforeProduction {
                index: 1,
                event: event(1),
                producer: 4
            }),
            "the host waits on a launch that is not submitted yet"
        );
        let repeated = inserted(
            inserted(one_layer(), 3, wait_on(COMPUTE, 0)),
            6,
            wait_host(1),
        );
        assert_eq!(
            check(&graph, repeated),
            Ok(()),
            "repeated waits are allowed"
        );
        Ok(())
    }

    #[test]
    fn a_launch_waits_for_its_transfer() -> TestResult {
        let graph = chain(1)?;
        let not_ready = |index| PlanError::TransferNotReady {
            index,
            op: op(0),
            tensor: weight(0),
            event: event(0),
        };
        assert_eq!(
            check(&graph, without(one_layer(), 2)),
            Err(not_ready(2)),
            "the compute stream does not wait for the copy"
        );
        assert_eq!(
            check(
                &graph,
                replaced(one_layer(), 2, wait_on(StreamId::new(2), 0))
            ),
            Err(not_ready(3)),
            "a wait on another stream does not order the launch"
        );
        assert_eq!(
            check(&graph, replaced(one_layer(), 2, wait_host(0))),
            Ok(()),
            "a host-completed transfer orders every later submission"
        );
        Ok(())
    }

    #[test]
    fn a_slot_is_overwritten_only_after_its_last_use() -> TestResult {
        let graph = chain(2)?;
        let premature = |index| PlanError::PrematureOverwrite {
            index,
            slot: slot(0),
            event: event(1),
        };
        assert_eq!(
            check(&graph, without(two_layers_one_slot(), 4)),
            Err(premature(4)),
            "the copy may overwrite the weight while the compute reads it"
        );
        assert_eq!(
            check(
                &graph,
                replaced(two_layers_one_slot(), 4, wait_on(COMPUTE, 1))
            ),
            Err(premature(5)),
            "a wait on the compute stream does not order the copy stream"
        );
        assert_eq!(
            check(&graph, two_layers_one_slot()),
            Ok(()),
            "the copy stream waits for the compute completion"
        );
        Ok(())
    }

    #[test]
    fn a_consumer_waits_for_its_producer() -> TestResult {
        let graph = chain(2)?;
        let side = StreamId::new(2);
        let split = replaced(
            replaced(two_layers_two_slots(), 6, wait_on(side, 2)),
            7,
            launch(op(1), side, [bind(weight(1), 1)], 3),
        );
        assert_eq!(
            check(&graph, split.clone()),
            Err(PlanError::ProducerNotReady {
                index: 7,
                op: op(1),
                tensor: output(0),
                producer: op(0),
                event: event(1)
            }),
            "the second stream does not wait for the first op"
        );
        assert_eq!(
            check(&graph, inserted(split, 7, wait_on(side, 1))),
            Ok(()),
            "the second stream waits for the first op"
        );
        Ok(())
    }

    #[test]
    fn bindings_match_the_weights_read() -> TestResult {
        let one = chain(1)?;
        let two = chain(2)?;
        let launch_first = |bindings: Vec<_>| launch(op(0), COMPUTE, bindings, 1);
        let cases = [
            (
                &one,
                replaced(one_layer(), 3, launch_first(Vec::new())),
                PlanError::MissingBinding {
                    index: 3,
                    op: op(0),
                    tensor: weight(0),
                },
            ),
            (
                &one,
                replaced(
                    one_layer(),
                    3,
                    launch_first(vec![bind(weight(0), 0), bind(input(), 0)]),
                ),
                PlanError::UnexpectedBinding {
                    index: 3,
                    op: op(0),
                    tensor: input(),
                },
            ),
            (
                &one,
                replaced(
                    one_layer(),
                    3,
                    launch_first(vec![bind(weight(0), 0), bind(weight(0), 0)]),
                ),
                PlanError::DuplicateBinding {
                    index: 3,
                    op: op(0),
                    tensor: weight(0),
                },
            ),
            (
                &two,
                replaced(
                    two_layers_two_slots(),
                    5,
                    launch_first(vec![bind(weight(0), 0), bind(weight(1), 1)]),
                ),
                PlanError::UnexpectedBinding {
                    index: 5,
                    op: op(0),
                    tensor: weight(1),
                },
            ),
            (
                &one,
                replaced(
                    inserted(one_layer(), 1, allocate(1, WEIGHT)),
                    4,
                    launch_first(vec![bind(weight(0), 1)]),
                ),
                PlanError::StaleBinding {
                    index: 4,
                    slot: slot(1),
                    expected: weight(0),
                    found: None,
                },
            ),
            (
                &two,
                replaced(
                    two_layers_two_slots(),
                    5,
                    launch_first(vec![bind(weight(0), 1)]),
                ),
                PlanError::StaleBinding {
                    index: 5,
                    slot: slot(1),
                    expected: weight(0),
                    found: Some(weight(1)),
                },
            ),
        ];
        for (graph, commands, error) in cases {
            assert_eq!(check(graph, commands), Err(error), "{error}");
        }
        Ok(())
    }

    #[test]
    fn ops_launch_once_in_graph_order() -> TestResult {
        let graph = chain(2)?;
        assert_eq!(
            check(
                &graph,
                replaced(
                    two_layers_two_slots(),
                    5,
                    launch(op(1), COMPUTE, [bind(weight(1), 1)], 1)
                )
            ),
            Err(PlanError::OutOfOrderLaunch {
                index: 5,
                expected: op(0),
                found: op(1)
            }),
            "the second op is launched first"
        );
        assert_eq!(
            check(
                &graph,
                replaced(
                    two_layers_two_slots(),
                    7,
                    launch(op(0), COMPUTE, [bind(weight(0), 0)], 3)
                )
            ),
            Err(PlanError::RepeatedLaunch {
                index: 7,
                op: op(0)
            }),
            "the first op is launched twice"
        );
        assert_eq!(
            check(&graph, one_layer()),
            Err(PlanError::MissingLaunch {
                op: op(1)
            }),
            "the second op is skipped"
        );
        Ok(())
    }

    #[test]
    fn a_slot_must_hold_its_weight() -> TestResult {
        assert_eq!(
            check(
                &chain(1)?,
                replaced(one_layer(), 0, allocate(0, WEIGHT - 1))
            ),
            Err(PlanError::SlotTooSmall {
                index: 1,
                slot: slot(0),
                capacity: ByteSize::from_bytes(WEIGHT - 1),
                required: ByteSize::from_bytes(WEIGHT)
            }),
            "one byte short"
        );
        assert_eq!(
            check(&Graph::new(), vec![allocate(0, 0)]),
            Err(PlanError::ZeroAllocation {
                index: 0,
                slot: slot(0)
            }),
            "an empty allocation"
        );
        Ok(())
    }

    #[test]
    fn live_slots_fit_the_budget() -> TestResult {
        let graph = chain(2)?;
        assert_eq!(
            plan(two_layers_two_slots()).validate(&graph, device(2 * WEIGHT)),
            Ok(()),
            "both slots exactly fill the budget"
        );
        assert_eq!(
            plan(two_layers_two_slots()).validate(&graph, device(2 * WEIGHT - 1)),
            Err(PlanError::Memory {
                index: 1,
                slot: slot(1),
                source: CoreError::BudgetExceeded {
                    requested: ByteSize::from_bytes(2 * WEIGHT),
                    capacity: ByteSize::from_bytes(2 * WEIGHT - 1)
                }
            }),
            "one byte over the budget"
        );
        assert_eq!(
            plan(two_layers_one_slot()).validate(&graph, device(WEIGHT)),
            Ok(()),
            "a reused slot counts once"
        );
        Ok(())
    }

    #[test]
    fn live_bytes_use_checked_arithmetic() {
        let error = plan(vec![allocate(0, u64::MAX), allocate(1, 1)])
            .validate(&Graph::new(), device(u64::MAX));
        assert_eq!(
            error,
            Err(PlanError::Memory {
                index: 1,
                slot: slot(1),
                source: CoreError::Overflow
            }),
            "the live total overflows"
        );
        assert_eq!(
            error
                .err()
                .as_ref()
                .and_then(Error::source)
                .map(ToString::to_string),
            Some(CoreError::Overflow.to_string()),
            "the core error is the source"
        );
    }

    #[test]
    fn slot_lifetimes_are_enforced() -> TestResult {
        let one = chain(1)?;
        let two = chain(2)?;
        assert_eq!(
            check(&one, inserted(one_layer(), 1, allocate(0, WEIGHT))),
            Err(PlanError::DoubleAllocation {
                index: 1,
                slot: slot(0)
            }),
            "a live slot is allocated again"
        );
        assert_eq!(
            check(&one, inserted(one_layer(), 6, release(0))),
            Err(PlanError::DoubleRelease {
                index: 6,
                slot: slot(0)
            }),
            "a slot is released twice"
        );
        let mut reused = one_layer();
        reused.push(prefetch(weight(1), 0, COPY, 2));
        assert_eq!(
            check(&two, reused.clone()),
            Err(PlanError::UseAfterRelease {
                index: 6,
                slot: slot(0)
            }),
            "a released slot is written"
        );
        reused.insert(6, allocate(0, WEIGHT));
        reused.extend([
            wait_on(COMPUTE, 2),
            launch(op(1), COMPUTE, [bind(weight(1), 0)], 3),
            wait_host(3),
            release(0),
        ]);
        assert_eq!(
            check(&two, reused),
            Ok(()),
            "a released slot is allocated again"
        );
        Ok(())
    }

    #[test]
    fn release_requires_host_completion() -> TestResult {
        let graph = chain(1)?;
        let pending = |access| PlanError::ReleaseBeforeCompletion {
            index: 5,
            slot: slot(0),
            event: event(access),
        };
        assert_eq!(
            check(&graph, replaced(one_layer(), 4, wait_on(COPY, 1))),
            Err(pending(0)),
            "a stream wait does not authorize the release"
        );
        assert_eq!(
            check(&graph, replaced(one_layer(), 4, wait_host(0))),
            Err(pending(1)),
            "the transfer completed, but the launch still reads the slot"
        );
        assert_eq!(
            check(&graph, one_layer()),
            Ok(()),
            "the host waited for the last access"
        );
        Ok(())
    }

    #[test]
    fn a_complete_plan_synchronizes_and_releases_everything() -> TestResult {
        let mut graph = Graph::new();
        let x = graph.add_tensor(TensorDesc::new(DType::F32, Shape::new([4])))?;
        let y = graph.add_tensor(TensorDesc::new(DType::F32, Shape::new([4])))?;
        graph.mark_input(x)?;
        let first = graph.add_op(Op::SyntheticCompute {
            inputs: vec![x],
            output: y,
            duration_hint: None,
        })?;
        let compute = launch(first, COMPUTE, Vec::new(), 0);
        assert_eq!(
            check(&graph, vec![compute.clone()]),
            Err(PlanError::UnsynchronizedWork {
                event: event(0)
            }),
            "nobody waits for the launch"
        );
        assert_eq!(
            check(&graph, vec![compute.clone(), wait_on(COPY, 0)]),
            Err(PlanError::UnsynchronizedWork {
                event: event(0)
            }),
            "a stream wait does not complete the launch"
        );
        assert_eq!(
            check(&graph, vec![compute, wait_host(0)]),
            Ok(()),
            "the host completes the launch"
        );
        assert_eq!(
            check(&chain(1)?, without(one_layer(), 5)),
            Err(PlanError::UnreleasedSlot {
                slot: slot(0)
            }),
            "the slot is never released"
        );
        Ok(())
    }

    #[test]
    fn the_graph_and_the_budget_are_checked_first() -> TestResult {
        let graph = chain(1)?;
        assert_eq!(
            plan(one_layer()).validate(
                &graph,
                MemoryBudget::new(MemorySpace::Host, ByteSize::from_gib(1)?)
            ),
            Err(PlanError::BudgetSpace(MemorySpace::Host)),
            "a host budget"
        );
        let mut invalid = Graph::new();
        let x = invalid.add_tensor(TensorDesc::new(DType::F32, Shape::new([4])))?;
        let first = invalid.add_op(Op::SyntheticCompute {
            inputs: vec![x],
            output: x,
            duration_hint: None,
        })?;
        assert_eq!(
            check(&invalid, Vec::new()),
            Err(PlanError::Graph(GraphError::UninitializedRead {
                op: first,
                tensor: x
            })),
            "the graph error is wrapped"
        );
        Ok(())
    }

    #[test]
    fn errors_name_the_command() -> TestResult {
        let error = check(&chain(1)?, without(one_layer(), 2));
        assert_eq!(
            error.map_err(|error| error.to_string()),
            Err(format!(
                "command 2: {:?} is not ordered after the transfer {:?} of {:?}",
                op(0),
                event(0),
                weight(0)
            )),
            "the message names the command and the identifiers"
        );
        Ok(())
    }
}
