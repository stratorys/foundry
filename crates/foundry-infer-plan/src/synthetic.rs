use std::collections::{
    BTreeMap,
    BTreeSet,
};
use std::num::NonZeroU32;
use std::time::Duration;

use foundry_infer_core::{
    Alignment,
    ByteSize,
    CoreError,
    DType,
    Graph,
    GraphError,
    Op,
    OpId,
    Shape,
    TensorDesc,
    TensorId,
    WeightDesc,
    WeightSource,
};

use crate::error::SyntheticPlanError;
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

const COPY: StreamId = StreamId::new(0);
const COMPUTE: StreamId = StreamId::new(1);

pub fn synthetic_chain_graph(
    layers: u32,
    weight: ByteSize,
    duration_hint: Option<Duration>,
) -> Result<Graph, GraphError> {
    let activation = || TensorDesc::new(DType::F32, Shape::new([1, 64]));
    let mut graph = Graph::new();
    let mut previous = graph.add_tensor(activation())?;
    graph.mark_input(previous)?;
    for _ in 0..layers {
        let tensor = graph.add_tensor(TensorDesc::new(DType::U8, Shape::new([weight.bytes()])))?;
        graph.add_weight(WeightDesc {
            tensor,
            bytes: weight,
            source: WeightSource::HostMemory,
        })?;
        let output = graph.add_tensor(activation())?;
        graph.add_op(Op::SyntheticCompute {
            inputs: vec![previous, tensor],
            output,
            duration_hint,
        })?;
        previous = output;
    }
    Ok(graph)
}

struct Layer {
    op: OpId,
    weight: TensorId,
    bytes: ByteSize,
}

fn layers(graph: &Graph) -> Result<Vec<Layer>, SyntheticPlanError> {
    let weights = graph
        .weights()
        .iter()
        .map(|weight| (weight.tensor, weight.bytes))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();
    let mut layers = Vec::new();
    for (op, desc) in graph.ops() {
        let inputs = match desc {
            Op::SyntheticCompute {
                inputs, ..
            } => inputs,
            Op::MatMul {
                ..
            } => {
                return Err(SyntheticPlanError::UnsupportedOp {
                    op,
                });
            }
        };
        let read = inputs
            .iter()
            .filter_map(|tensor| weights.get(tensor).map(|bytes| (*tensor, *bytes)))
            .collect::<BTreeMap<_, _>>();
        let (weight, bytes) = match read.iter().next() {
            Some((&weight, &bytes)) if read.len() == 1 => (weight, bytes),
            Some(_) | None => {
                return Err(SyntheticPlanError::WeightCount {
                    op,
                    count: read.len(),
                });
            }
        };
        if !seen.insert(weight) {
            return Err(SyntheticPlanError::SharedWeight {
                op,
                tensor: weight,
            });
        }
        layers.push(Layer {
            op,
            weight,
            bytes,
        });
    }
    Ok(layers)
}

fn event(
    layer: usize,
    completion: u32,
) -> Result<EventId, CoreError> {
    u32::try_from(layer)
        .ok()
        .and_then(|layer| layer.checked_mul(2))
        .and_then(|index| index.checked_add(completion))
        .map(EventId::new)
        .ok_or(CoreError::IdsExhausted)
}

fn transfer(layer: usize) -> Result<EventId, CoreError> { event(layer, 0) }

fn compute(layer: usize) -> Result<EventId, CoreError> { event(layer, 1) }

pub fn synthetic_chain_plan(
    graph: &Graph,
    slots: NonZeroU32,
    alignment: Alignment,
) -> Result<ExecutionPlan, SyntheticPlanError> {
    let layers = layers(graph)?;
    let resident =
        usize::try_from(slots.get()).map_or(layers.len(), |slots| slots.min(layers.len()));
    let slot_bytes = layers
        .iter()
        .map(|layer| layer.bytes)
        .max()
        .unwrap_or_default()
        .align_up(alignment)?;
    let slot = |layer: usize| {
        layer
            .checked_rem(resident)
            .and_then(|slot| u32::try_from(slot).ok())
            .map(BufferSlot::new)
            .ok_or(CoreError::IdsExhausted)
    };
    let prefetch = |layer: usize, weight: TensorId| -> Result<Command, CoreError> {
        Ok(Command::Prefetch {
            tensor: weight,
            slot: slot(layer)?,
            stream: COPY,
            event: transfer(layer)?,
        })
    };

    let mut commands = Vec::new();
    for layer in 0..resident {
        commands.push(Command::Allocate {
            slot: slot(layer)?,
            bytes: slot_bytes,
        });
    }
    for (index, layer) in layers.iter().take(resident).enumerate() {
        commands.push(prefetch(index, layer.weight)?);
    }
    for (index, layer) in layers.iter().enumerate() {
        commands.push(Command::Wait {
            target: WaitTarget::Stream(COMPUTE),
            event: transfer(index)?,
        });
        commands.push(Command::Launch {
            op: layer.op,
            stream: COMPUTE,
            weight_slots: vec![WeightBinding {
                tensor: layer.weight,
                slot: slot(index)?,
            }],
            event: compute(index)?,
        });
        let next = index.saturating_add(resident);
        if let Some(upcoming) = layers.get(next) {
            commands.push(Command::Wait {
                target: WaitTarget::Stream(COPY),
                event: compute(index)?,
            });
            commands.push(prefetch(next, upcoming.weight)?);
        }
    }
    if let Some(last) = layers.len().checked_sub(1) {
        commands.push(Command::Wait {
            target: WaitTarget::Host,
            event: compute(last)?,
        });
    }
    for layer in 0..resident {
        commands.push(Command::Release {
            slot: slot(layer)?,
        });
    }
    Ok(ExecutionPlan::new(commands))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;
    use std::time::Duration;

    use foundry_infer_core::{
        Alignment,
        ByteSize,
        DType,
        Graph,
        MemoryBudget,
        MemorySpace,
        Op,
        Shape,
        TensorDesc,
        WeightDesc,
        WeightSource,
    };

    use super::{
        synthetic_chain_graph,
        synthetic_chain_plan,
    };
    use crate::error::SyntheticPlanError;
    use crate::fixtures::{
        TestResult,
        WEIGHT,
        budget,
        chain,
        input,
        op,
        output,
        weight,
    };
    use crate::plan::Command;

    fn slots(count: u32) -> Result<NonZeroU32, String> {
        NonZeroU32::new(count).ok_or_else(|| format!("{count} slots"))
    }

    fn count(
        commands: &[Command],
        kind: fn(&Command) -> bool,
    ) -> usize {
        commands.iter().filter(|command| kind(command)).count()
    }

    fn is_allocate(command: &Command) -> bool { matches!(command, Command::Allocate { .. }) }

    fn is_prefetch(command: &Command) -> bool { matches!(command, Command::Prefetch { .. }) }

    fn is_launch(command: &Command) -> bool { matches!(command, Command::Launch { .. }) }

    #[test]
    fn sequential_double_and_triple_buffering_validate() -> TestResult {
        let graph = chain(6)?;
        let alignment = Alignment::new(256)?;
        for resident in 1..=3 {
            let plan = synthetic_chain_plan(&graph, slots(resident)?, alignment)?;
            let commands = plan.commands();
            assert_eq!(
                plan.validate(&graph, budget()),
                Ok(()),
                "{resident} slots validate"
            );
            assert_eq!(
                count(commands, is_allocate),
                usize::try_from(resident)?,
                "{resident} slots are allocated"
            );
            assert_eq!(count(commands, is_launch), 6, "every op is launched");
            let ahead = commands
                .iter()
                .take_while(|command| !is_launch(command))
                .filter(|command| is_prefetch(command))
                .count();
            assert_eq!(
                ahead,
                usize::try_from(resident)?,
                "{resident} transfers precede the first launch"
            );
        }
        Ok(())
    }

    #[test]
    fn slots_are_padded_and_bounded_by_the_op_count() -> TestResult {
        let graph = synthetic_chain_graph(2, ByteSize::from_bytes(WEIGHT + 1), None)?;
        let plan = synthetic_chain_plan(&graph, slots(5)?, Alignment::new(256)?)?;
        let sizes = plan
            .commands()
            .iter()
            .filter_map(|command| match command {
                Command::Allocate {
                    bytes, ..
                } => Some(bytes.bytes()),
                Command::Prefetch {
                    ..
                }
                | Command::Wait {
                    ..
                }
                | Command::Launch {
                    ..
                }
                | Command::Release {
                    ..
                } => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(sizes, [WEIGHT + 256, WEIGHT + 256], "two padded slots");
        assert_eq!(
            plan.validate(&graph, budget()),
            Ok(()),
            "the plan validates"
        );
        Ok(())
    }

    #[test]
    fn an_empty_chain_has_an_empty_plan() -> TestResult {
        let plan = synthetic_chain_plan(&Graph::new(), slots(3)?, Alignment::new(256)?)?;
        assert!(plan.commands().is_empty(), "no commands");
        Ok(())
    }

    #[test]
    fn the_48_layer_workload_fits_in_three_slots() -> TestResult {
        let layer = ByteSize::from_mib(450)?;
        let graph = synthetic_chain_graph(48, layer, Some(Duration::from_millis(60)))?;
        let total = graph
            .weights()
            .iter()
            .try_fold(ByteSize::from_bytes(0), |total, weight| {
                total.checked_add(weight.bytes)
            })?;
        assert!(
            total > ByteSize::from_gib(20)?,
            "{total} of logical weights exceed 20 GiB"
        );
        let plan = synthetic_chain_plan(&graph, slots(3)?, Alignment::new(256)?)?;
        let resident = plan
            .commands()
            .iter()
            .filter_map(|command| match command {
                Command::Allocate {
                    bytes, ..
                } => Some(*bytes),
                Command::Prefetch {
                    ..
                }
                | Command::Wait {
                    ..
                }
                | Command::Launch {
                    ..
                }
                | Command::Release {
                    ..
                } => None,
            })
            .try_fold(ByteSize::from_bytes(0), ByteSize::checked_add)?;
        let budget = MemoryBudget::new(MemorySpace::Device, ByteSize::from_gib(4)?);
        assert_eq!(resident, layer.checked_mul(3)?, "three resident slots");
        assert!(
            resident <= budget.capacity(),
            "{resident} of slots fit 4 GiB"
        );
        assert_eq!(plan.validate(&graph, budget), Ok(()), "the plan validates");
        assert!(
            plan.validate(
                &graph,
                MemoryBudget::new(MemorySpace::Device, layer.checked_mul(2)?)
            )
            .is_err(),
            "three slots do not fit two layers of budget"
        );
        Ok(())
    }

    #[test]
    fn unsupported_graphs_are_rejected() -> TestResult {
        let alignment = Alignment::new(256)?;
        let host_weight = |tensor| WeightDesc {
            tensor,
            bytes: ByteSize::from_bytes(WEIGHT),
            source: WeightSource::HostMemory,
        };
        let weight_desc = || TensorDesc::new(DType::U8, Shape::new([WEIGHT]));
        let activation = || TensorDesc::new(DType::F32, Shape::new([1, 64]));

        let mut matmul = chain(0)?;
        let w = matmul.add_tensor(weight_desc())?;
        matmul.add_weight(host_weight(w))?;
        let y = matmul.add_tensor(activation())?;
        let first = matmul.add_op(Op::MatMul {
            input: input(),
            weight: w,
            output: y,
        })?;
        assert_eq!(
            synthetic_chain_plan(&matmul, slots(1)?, alignment).map(|_| ()),
            Err(SyntheticPlanError::UnsupportedOp {
                op: first
            }),
            "a matmul"
        );

        let mut two_weights = chain(1)?;
        let extra = two_weights.add_tensor(weight_desc())?;
        two_weights.add_weight(host_weight(extra))?;
        let y = two_weights.add_tensor(activation())?;
        let second = two_weights.add_op(Op::SyntheticCompute {
            inputs: vec![output(0), weight(0), extra],
            output: y,
            duration_hint: None,
        })?;
        assert_eq!(
            synthetic_chain_plan(&two_weights, slots(1)?, alignment).map(|_| ()),
            Err(SyntheticPlanError::WeightCount {
                op: second,
                count: 2
            }),
            "two weights in one op"
        );

        let mut shared = chain(1)?;
        let y = shared.add_tensor(activation())?;
        let second = shared.add_op(Op::SyntheticCompute {
            inputs: vec![output(0), weight(0)],
            output: y,
            duration_hint: None,
        })?;
        assert_eq!(
            synthetic_chain_plan(&shared, slots(1)?, alignment).map(|_| ()),
            Err(SyntheticPlanError::SharedWeight {
                op: second,
                tensor: weight(0)
            }),
            "a weight shared by two ops"
        );

        let mut unweighted = chain(0)?;
        let y = unweighted.add_tensor(activation())?;
        unweighted.add_op(Op::SyntheticCompute {
            inputs: vec![input()],
            output: y,
            duration_hint: None,
        })?;
        assert_eq!(
            synthetic_chain_plan(&unweighted, slots(1)?, alignment).map(|_| ()),
            Err(SyntheticPlanError::WeightCount {
                op: op(0),
                count: 0
            }),
            "an op without weights"
        );
        Ok(())
    }
}
