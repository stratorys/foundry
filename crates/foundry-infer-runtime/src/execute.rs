use std::collections::{
    BTreeMap,
    BTreeSet,
};
use std::mem;
use std::time::Duration;

use foundry_infer_core::{
    ByteSize,
    Graph,
    MemoryBudget,
    Op,
    OpId,
    TensorId,
    WeightSource,
};
use foundry_infer_plan::{
    BufferSlot,
    Command,
    EventId,
    ExecutionPlan,
    StreamId,
    WaitTarget,
    WeightBinding,
};

use crate::backend::Backend;
use crate::error::{
    BackendOperation,
    ExecutionFailure,
    RuntimeError,
};
use crate::observe::{
    CommandContext,
    CpuActivity,
    CpuRecord,
    ExecutionObserver,
    ExecutionOutcome,
    Unobserved,
};

pub fn execute<B: Backend>(
    backend: &mut B,
    graph: &Graph,
    plan: &ExecutionPlan,
    budget: MemoryBudget,
    weights: &BTreeMap<TensorId, &[u8]>,
) -> Result<(), RuntimeError<B::Error>> {
    execute_observed(backend, graph, plan, budget, weights, &mut Unobserved)
}

pub fn execute_observed<B: Backend, O: ExecutionObserver>(
    backend: &mut B,
    graph: &Graph,
    plan: &ExecutionPlan,
    budget: MemoryBudget,
    weights: &BTreeMap<TensorId, &[u8]>,
    observer: &mut O,
) -> Result<(), RuntimeError<B::Error>> {
    let durations = match preflight(backend, graph, plan, budget, weights) {
        Ok(durations) => durations,
        Err(error) => {
            observer.finish(ExecutionOutcome::Rejected);
            return Err(error);
        }
    };
    let mut run = Run::new(backend, observer, durations, weights);
    match run.commands(plan.commands()) {
        Ok(()) => {
            run.observer.finish(ExecutionOutcome::Completed);
            Ok(())
        }
        Err(failure) => Err(run.fail(failure)),
    }
}
fn preflight<B: Backend>(
    backend: &B,
    graph: &Graph,
    plan: &ExecutionPlan,
    budget: MemoryBudget,
    weights: &BTreeMap<TensorId, &[u8]>,
) -> Result<BTreeMap<OpId, Option<Duration>>, RuntimeError<B::Error>> {
    plan.validate(graph, budget)?;
    let durations = graph
        .ops()
        .map(|(op, desc)| match desc {
            Op::SyntheticCompute {
                duration_hint, ..
            } => Ok((op, *duration_hint)),
            Op::MatMul {
                ..
            } => Err(RuntimeError::UnsupportedOp {
                op,
            }),
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let expected = graph
        .weights()
        .iter()
        .map(|weight| match weight.source {
            WeightSource::HostMemory => Ok((weight.tensor, weight.bytes)),
            WeightSource::MappedFile {
                ..
            } => Err(RuntimeError::UnsupportedWeightSource {
                tensor: weight.tensor,
            }),
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    for (index, command) in plan.commands().iter().enumerate() {
        let Command::Prefetch {
            tensor, ..
        } = command
        else {
            continue;
        };
        let tensor = *tensor;
        let bytes = weights
            .get(&tensor)
            .ok_or(RuntimeError::MissingHostWeight {
                index,
                tensor,
            })?;
        let expected = expected.get(&tensor).copied().unwrap_or_default();
        let actual =
            u64::try_from(bytes.len()).map_or(ByteSize::from_bytes(u64::MAX), ByteSize::from_bytes);
        if actual != expected {
            return Err(RuntimeError::HostWeightSize {
                index,
                tensor,
                expected,
                actual,
            });
        }
    }
    let device = backend.capabilities().device_memory;
    if budget.capacity() > device {
        return Err(RuntimeError::BudgetExceedsDevice {
            budget: budget.capacity(),
            device,
        });
    }
    Ok(durations)
}

fn streams(commands: &[Command]) -> BTreeSet<StreamId> {
    commands
        .iter()
        .filter_map(|command| match command {
            Command::Prefetch {
                stream, ..
            }
            | Command::Launch {
                stream, ..
            }
            | Command::Wait {
                target: WaitTarget::Stream(stream),
                ..
            } => Some(*stream),
            Command::Allocate {
                ..
            }
            | Command::Wait {
                target: WaitTarget::Host,
                ..
            }
            | Command::Release {
                ..
            } => None,
        })
        .collect()
}

fn byte_len(bytes: &[u8]) -> ByteSize {
    u64::try_from(bytes.len()).map_or(ByteSize::from_bytes(u64::MAX), ByteSize::from_bytes)
}

fn observed<B: Backend, O: ExecutionObserver, T>(
    backend: &mut B,
    observer: &mut O,
    activity: CpuActivity,
    context: CommandContext<'_>,
    call: impl FnOnce(&mut B) -> Result<T, B::Error>,
) -> Result<T, B::Error> {
    backend.annotate(Some(&context));
    let start = observer.now();
    let result = call(backend);
    let end = observer.now();
    backend.annotate(None);
    observer.record(CpuRecord {
        context,
        activity,
        start,
        end,
        succeeded: result.is_ok(),
    });
    result
}

struct Run<'run, B: Backend, O: ExecutionObserver> {
    backend: &'run mut B,
    observer: &'run mut O,
    durations: BTreeMap<OpId, Option<Duration>>,
    weights: &'run BTreeMap<TensorId, &'run [u8]>,
    streams: BTreeMap<StreamId, B::Stream>,
    slots: BTreeMap<BufferSlot, B::DeviceBuffer>,
    events: BTreeMap<EventId, B::Event>,
    staging: Option<B::HostBuffer>,
    submitted: bool,
}

impl<'run, B: Backend, O: ExecutionObserver> Run<'run, B, O> {
    fn new(
        backend: &'run mut B,
        observer: &'run mut O,
        durations: BTreeMap<OpId, Option<Duration>>,
        weights: &'run BTreeMap<TensorId, &'run [u8]>,
    ) -> Self {
        Self {
            backend,
            observer,
            durations,
            weights,
            streams: BTreeMap::new(),
            slots: BTreeMap::new(),
            events: BTreeMap::new(),
            staging: None,
            submitted: false,
        }
    }

    fn commands(
        &mut self,
        commands: &[Command],
    ) -> Result<(), ExecutionFailure<B::Error>> {
        for stream in streams(commands) {
            self.create_stream(stream)?;
        }
        for (index, command) in commands.iter().enumerate() {
            self.command(index, command)?;
        }
        Ok(())
    }

    fn create_stream(
        &mut self,
        stream: StreamId,
    ) -> Result<(), ExecutionFailure<B::Error>> {
        let context = CommandContext {
            stream: Some(stream),
            ..CommandContext::default()
        };
        let created = observed(
            self.backend,
            self.observer,
            CpuActivity::CreateStream,
            context,
            |backend| backend.create_stream(),
        )
        .map_err(|source| ExecutionFailure::Backend {
            operation: BackendOperation::CreateStream {
                stream,
            },
            source,
        })?;
        self.streams.insert(stream, created);
        Ok(())
    }

    fn command(
        &mut self,
        index: usize,
        command: &Command,
    ) -> Result<(), ExecutionFailure<B::Error>> {
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
                target: WaitTarget::Stream(stream),
                event,
            } => self.wait_stream(index, *stream, *event),
            Command::Wait {
                target: WaitTarget::Host,
                event,
            } => self.wait_host(index, *event),
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
    ) -> Result<(), ExecutionFailure<B::Error>> {
        let context = CommandContext {
            index: Some(index),
            slot: Some(slot),
            bytes: Some(bytes),
            ..CommandContext::default()
        };
        let buffer = observed(
            self.backend,
            self.observer,
            CpuActivity::Allocate,
            context,
            |backend| backend.allocate_device(bytes),
        )
        .map_err(|source| ExecutionFailure::Backend {
            operation: BackendOperation::Allocate {
                index,
                slot,
            },
            source,
        })?;
        self.slots.insert(slot, buffer);
        Ok(())
    }

    fn prefetch(
        &mut self,
        index: usize,
        tensor: TensorId,
        slot: BufferSlot,
        stream: StreamId,
        event: EventId,
    ) -> Result<(), ExecutionFailure<B::Error>> {
        let contents = self
            .weights
            .get(&tensor)
            .ok_or(ExecutionFailure::MissingHostWeight {
                index,
                tensor,
            })?;
        let queue = self
            .streams
            .get_mut(&stream)
            .ok_or(ExecutionFailure::MissingStream {
                index,
                stream,
            })?;
        let destination = self
            .slots
            .get_mut(&slot)
            .ok_or(ExecutionFailure::MissingSlot {
                index,
                slot,
            })?;
        let context = CommandContext {
            index: Some(index),
            tensor: Some(tensor),
            slot: Some(slot),
            stream: Some(stream),
            event: Some(event),
            bytes: Some(byte_len(contents)),
            ..CommandContext::default()
        };
        let staging = observed(
            self.backend,
            self.observer,
            CpuActivity::Stage,
            context,
            |backend| backend.allocate_host(contents),
        )
        .map_err(|source| ExecutionFailure::Backend {
            operation: BackendOperation::StageHost {
                index,
                tensor,
            },
            source,
        })?;
        self.submitted = true;
        let source = self.staging.insert(staging);
        let completion = observed(
            self.backend,
            self.observer,
            CpuActivity::SubmitCopy,
            context,
            |backend| backend.copy_to_device(queue, source, destination),
        )
        .map_err(|source| ExecutionFailure::Backend {
            operation: BackendOperation::Copy {
                index,
                tensor,
                slot,
                stream,
                event,
            },
            source,
        })?;
        self.staging = None;
        self.events.insert(event, completion);
        Ok(())
    }

    fn wait_stream(
        &mut self,
        index: usize,
        stream: StreamId,
        event: EventId,
    ) -> Result<(), ExecutionFailure<B::Error>> {
        let queue = self
            .streams
            .get_mut(&stream)
            .ok_or(ExecutionFailure::MissingStream {
                index,
                stream,
            })?;
        let completion = self
            .events
            .get(&event)
            .ok_or(ExecutionFailure::MissingEvent {
                index,
                event,
            })?;
        self.submitted = true;
        let context = CommandContext {
            index: Some(index),
            stream: Some(stream),
            event: Some(event),
            ..CommandContext::default()
        };
        observed(
            self.backend,
            self.observer,
            CpuActivity::StreamWait,
            context,
            |backend| backend.wait_stream(queue, completion),
        )
        .map_err(|source| ExecutionFailure::Backend {
            operation: BackendOperation::StreamWait {
                index,
                stream,
                event,
            },
            source,
        })
    }

    fn wait_host(
        &mut self,
        index: usize,
        event: EventId,
    ) -> Result<(), ExecutionFailure<B::Error>> {
        let completion = self
            .events
            .get(&event)
            .ok_or(ExecutionFailure::MissingEvent {
                index,
                event,
            })?;
        let context = CommandContext {
            index: Some(index),
            event: Some(event),
            ..CommandContext::default()
        };
        observed(
            self.backend,
            self.observer,
            CpuActivity::HostWait,
            context,
            |backend| backend.wait_host(completion),
        )
        .map_err(|source| ExecutionFailure::Backend {
            operation: BackendOperation::HostWait {
                index,
                event,
            },
            source,
        })
    }

    fn launch(
        &mut self,
        index: usize,
        op: OpId,
        stream: StreamId,
        bindings: &[WeightBinding],
        event: EventId,
    ) -> Result<(), ExecutionFailure<B::Error>> {
        let duration_hint = *self.durations.get(&op).ok_or(ExecutionFailure::MissingOp {
            index,
            op,
        })?;
        let queue = self
            .streams
            .get_mut(&stream)
            .ok_or(ExecutionFailure::MissingStream {
                index,
                stream,
            })?;
        let weights = bindings
            .iter()
            .map(|binding| {
                self.slots
                    .get(&binding.slot)
                    .ok_or(ExecutionFailure::MissingSlot {
                        index,
                        slot: binding.slot,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bytes = bindings
            .iter()
            .try_fold(ByteSize::default(), |total, binding| {
                let weight = self.weights.get(&binding.tensor)?;
                total.checked_add(byte_len(weight)).ok()
            });
        self.submitted = true;
        let context = CommandContext {
            index: Some(index),
            op: Some(op),
            bindings,
            stream: Some(stream),
            event: Some(event),
            bytes,
            ..CommandContext::default()
        };
        let completion = observed(
            self.backend,
            self.observer,
            CpuActivity::SubmitLaunch,
            context,
            |backend| backend.synthetic_compute(queue, op, &weights, duration_hint),
        )
        .map_err(|source| ExecutionFailure::Backend {
            operation: BackendOperation::Launch {
                index,
                op,
                stream,
                event,
            },
            source,
        })?;
        self.events.insert(event, completion);
        Ok(())
    }

    fn release(
        &mut self,
        index: usize,
        slot: BufferSlot,
    ) -> Result<(), ExecutionFailure<B::Error>> {
        let buffer = self
            .slots
            .remove(&slot)
            .ok_or(ExecutionFailure::MissingSlot {
                index,
                slot,
            })?;
        let context = CommandContext {
            index: Some(index),
            slot: Some(slot),
            ..CommandContext::default()
        };
        let start = self.observer.now();
        drop(buffer);
        let end = self.observer.now();
        self.observer.record(CpuRecord {
            context,
            activity: CpuActivity::Release,
            start,
            end,
            succeeded: true,
        });
        Ok(())
    }

    fn fail(
        mut self,
        failure: ExecutionFailure<B::Error>,
    ) -> RuntimeError<B::Error> {
        let cleanup = if self.submitted {
            observed(
                self.backend,
                self.observer,
                CpuActivity::CleanupDrain,
                CommandContext::default(),
                |backend| backend.drain(),
            )
            .err()
        } else {
            None
        };
        if cleanup.is_some() {
            mem::forget(self.staging.take());
            mem::forget(mem::take(&mut self.slots));
            mem::forget(mem::take(&mut self.events));
            mem::forget(mem::take(&mut self.streams));
        }
        let operation = match &failure {
            ExecutionFailure::Backend {
                operation, ..
            } => Some(*operation),
            ExecutionFailure::MissingStream {
                ..
            }
            | ExecutionFailure::MissingSlot {
                ..
            }
            | ExecutionFailure::MissingEvent {
                ..
            }
            | ExecutionFailure::MissingOp {
                ..
            }
            | ExecutionFailure::MissingHostWeight {
                ..
            } => None,
        };
        self.observer.finish(ExecutionOutcome::Failed {
            operation,
            cleanup_failed: cleanup.is_some(),
        });
        RuntimeError::Execution {
            failure,
            cleanup,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::error::Error;
    use std::num::NonZeroU32;
    use std::time::Duration;

    use foundry_infer_core::{
        Alignment,
        ByteSize,
        DType,
        Graph,
        Id,
        MemoryBudget,
        MemorySpace,
        Op,
        OpId,
        Shape,
        TensorDesc,
        TensorId,
        WeightDesc,
        WeightSource,
    };
    use foundry_infer_plan::{
        BufferSlot,
        Command,
        EventId,
        ExecutionPlan,
        PlanError,
        StreamId,
        WaitTarget,
        WeightBinding,
        synthetic_chain_graph,
        synthetic_chain_plan,
    };

    use super::{
        Run,
        execute,
        execute_observed,
    };
    use crate::error::{
        BackendOperation,
        ExecutionFailure,
        RuntimeError,
    };
    use crate::fake::{
        Entry,
        FakeBackend,
        FakeError,
        Operation,
    };
    use crate::observe::{
        CpuActivity,
        CpuRecord,
        ExecutionObserver,
        ExecutionOutcome,
        Unobserved,
    };

    type TestResult = Result<(), Box<dyn Error>>;

    type Rejection<'weights> = (
        &'static str,
        Graph,
        ExecutionPlan,
        MemoryBudget,
        BTreeMap<TensorId, &'weights [u8]>,
        u64,
        RuntimeError<FakeError>,
    );

    const WEIGHT: u64 = 64;
    const SLOT: u64 = 256;
    const DEVICE: u64 = 1 << 20;
    const COPY: StreamId = StreamId::new(0);
    const COMPUTE: StreamId = StreamId::new(1);
    const HINT: Duration = Duration::from_millis(7);

    fn budget() -> MemoryBudget {
        MemoryBudget::new(MemorySpace::Device, ByteSize::from_bytes(DEVICE))
    }

    fn chain(layers: u32) -> Result<Graph, Box<dyn Error>> {
        Ok(synthetic_chain_graph(
            layers,
            ByteSize::from_bytes(WEIGHT),
            Some(HINT),
        )?)
    }

    fn buffered(
        graph: &Graph,
        slots: u32,
    ) -> Result<ExecutionPlan, Box<dyn Error>> {
        let slots = NonZeroU32::new(slots).ok_or("zero slots")?;
        Ok(synthetic_chain_plan(graph, slots, Alignment::new(SLOT)?)?)
    }

    fn weight(layer: u32) -> TensorId {
        TensorId::from_index(layer.saturating_mul(2).saturating_add(1))
    }

    fn payload(layer: u32) -> Result<Vec<u8>, Box<dyn Error>> {
        let fill = u8::try_from(layer.checked_add(1).ok_or("layer overflow")?)?;
        Ok(vec![fill; usize::try_from(WEIGHT)?])
    }

    fn resident(layer: u32) -> Result<Vec<u8>, Box<dyn Error>> {
        let mut bytes = payload(layer)?;
        bytes.resize(usize::try_from(SLOT)?, 0);
        Ok(bytes)
    }

    fn payloads(graph: &Graph) -> Result<BTreeMap<TensorId, Vec<u8>>, Box<dyn Error>> {
        graph
            .weights()
            .iter()
            .enumerate()
            .map(|(layer, desc)| Ok((desc.tensor, payload(u32::try_from(layer)?)?)))
            .collect()
    }

    fn bindings(payloads: &BTreeMap<TensorId, Vec<u8>>) -> BTreeMap<TensorId, &[u8]> {
        payloads
            .iter()
            .map(|(tensor, bytes)| (*tensor, bytes.as_slice()))
            .collect()
    }

    fn run(
        backend: &mut FakeBackend,
        graph: &Graph,
        plan: &ExecutionPlan,
    ) -> Result<Result<(), RuntimeError<FakeError>>, Box<dyn Error>> {
        let payloads = payloads(graph)?;
        Ok(execute(
            backend,
            graph,
            plan,
            budget(),
            &bindings(&payloads),
        ))
    }

    fn slot(index: u32) -> BufferSlot { BufferSlot::new(index) }

    fn event(index: u32) -> EventId { EventId::new(index) }

    fn layer_commands(
        layer: u32,
        transfer: u32,
    ) -> [Command; 6] {
        [
            Command::Allocate {
                slot: slot(0),
                bytes: ByteSize::from_bytes(SLOT),
            },
            Command::Prefetch {
                tensor: weight(layer),
                slot: slot(0),
                stream: COPY,
                event: event(transfer),
            },
            Command::Wait {
                target: WaitTarget::Stream(COMPUTE),
                event: event(transfer),
            },
            Command::Launch {
                op: OpId::from_index(layer),
                stream: COMPUTE,
                weight_slots: vec![WeightBinding {
                    tensor: weight(layer),
                    slot: slot(0),
                }],
                event: event(transfer.saturating_add(1)),
            },
            Command::Wait {
                target: WaitTarget::Host,
                event: event(transfer.saturating_add(1)),
            },
            Command::Release {
                slot: slot(0),
            },
        ]
    }

    fn device_resources(log: &[Entry]) -> Vec<u32> {
        log.iter()
            .filter_map(|entry| {
                if let Entry::AllocatedDevice {
                    resource, ..
                } = entry
                {
                    Some(*resource)
                } else {
                    None
                }
            })
            .collect()
    }

    fn check_lifetimes(log: &[Entry]) -> Result<(), String> {
        let devices = device_resources(log);
        let mut users = BTreeMap::<u32, Vec<usize>>::new();
        let mut completed = BTreeMap::new();
        for (position, entry) in log.iter().enumerate() {
            match entry {
                Entry::SubmittedCopy {
                    work,
                    source,
                    destination,
                    ..
                } => {
                    users.entry(*source).or_default().push(*work);
                    users.entry(*destination).or_default().push(*work);
                }
                Entry::SubmittedCompute {
                    work,
                    weights,
                    ..
                } => {
                    for resource in weights {
                        users.entry(*resource).or_default().push(*work);
                    }
                }
                Entry::Completed {
                    work,
                } => {
                    completed.insert(*work, position);
                }
                Entry::Freed {
                    resource,
                }
                | Entry::HandleDropped {
                    resource,
                } => {
                    let guarded =
                        matches!(entry, Entry::Freed { .. }) || devices.contains(resource);
                    let pending = users
                        .get(resource)
                        .into_iter()
                        .flatten()
                        .find(|work| !completed.contains_key(work));
                    if let (true, Some(work)) = (guarded, pending) {
                        return Err(format!(
                            "{entry:?} at {position} precedes the completion of work {work}"
                        ));
                    }
                }
                Entry::AllocatedDevice {
                    ..
                }
                | Entry::Staged {
                    ..
                }
                | Entry::CreatedStream {
                    ..
                }
                | Entry::StreamWait {
                    ..
                }
                | Entry::HostWait {
                    ..
                }
                | Entry::Computed {
                    ..
                }
                | Entry::Drained
                | Entry::Failed(_) => {}
            }
        }
        Ok(())
    }

    fn computed(log: &[Entry]) -> Vec<(OpId, Vec<Vec<u8>>)> {
        log.iter()
            .filter_map(|entry| {
                if let Entry::Computed {
                    op,
                    weights,
                } = entry
                {
                    Some((*op, weights.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    #[test]
    fn an_empty_plan_executes_without_backend_calls() -> TestResult {
        let mut backend = FakeBackend::new(DEVICE)?;
        let result = execute(
            &mut backend,
            &Graph::new(),
            &ExecutionPlan::default(),
            budget(),
            &BTreeMap::new(),
        );
        assert_eq!(result, Ok(()), "the empty plan executes");
        assert_eq!(backend.log(), [], "the backend is untouched");
        Ok(())
    }

    #[test]
    fn a_single_layer_dispatches_streams_events_and_waits() -> TestResult {
        let graph = chain(1)?;
        let plan = buffered(&graph, 1)?;
        let mut backend = FakeBackend::new(DEVICE)?;
        assert_eq!(
            run(&mut backend, &graph, &plan)?,
            Ok(()),
            "the plan executes"
        );
        let expected = [
            Entry::CreatedStream {
                stream: 0,
            },
            Entry::CreatedStream {
                stream: 1,
            },
            Entry::AllocatedDevice {
                resource: 0,
                bytes: SLOT,
            },
            Entry::Staged {
                resource: 1,
                contents: payload(0)?,
            },
            Entry::SubmittedCopy {
                work: 0,
                stream: 0,
                source: 1,
                destination: 0,
            },
            Entry::HandleDropped {
                resource: 1,
            },
            Entry::StreamWait {
                stream: 1,
                work: 0,
            },
            Entry::SubmittedCompute {
                work: 1,
                stream: 1,
                op: OpId::from_index(0),
                weights: vec![0],
                duration_hint: Some(HINT),
            },
            Entry::HostWait {
                work: 1,
            },
            Entry::Completed {
                work: 0,
            },
            Entry::Freed {
                resource: 1,
            },
            Entry::Computed {
                op: OpId::from_index(0),
                weights: vec![resident(0)?],
            },
            Entry::Completed {
                work: 1,
            },
            Entry::HandleDropped {
                resource: 0,
            },
            Entry::Freed {
                resource: 0,
            },
        ];
        assert_eq!(backend.log(), expected, "the dispatch sequence");
        Ok(())
    }

    #[test]
    fn sequential_double_and_triple_buffering_execute() -> TestResult {
        let layers = 5;
        let graph = chain(layers)?;
        for slots in 1..=3 {
            let plan = buffered(&graph, slots)?;
            let mut backend = FakeBackend::new(DEVICE)?;
            assert_eq!(
                run(&mut backend, &graph, &plan)?,
                Ok(()),
                "{slots} slots execute"
            );
            let log = backend.log();
            let expected = (0..layers)
                .map(|layer| Ok((OpId::from_index(layer), vec![resident(layer)?])))
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            assert_eq!(
                computed(&log),
                expected,
                "{slots} slots: each op sees its own weight"
            );
            assert_eq!(
                device_resources(&log).len(),
                usize::try_from(slots)?,
                "{slots} device buffers"
            );
            assert_eq!(backend.pending(), 0, "{slots} slots: no pending work");
            assert_eq!(
                backend.live_device(),
                0,
                "{slots} slots: device memory is returned"
            );
            check_lifetimes(&log)?;
        }
        Ok(())
    }

    #[test]
    fn copies_and_launches_use_their_planned_streams() -> TestResult {
        let graph = chain(4)?;
        let plan = buffered(&graph, 2)?;
        let mut backend = FakeBackend::new(DEVICE)?;
        assert_eq!(
            run(&mut backend, &graph, &plan)?,
            Ok(()),
            "the plan executes"
        );
        let log = backend.log();
        let mut copies = Vec::new();
        let mut launches = Vec::new();
        let mut waits = Vec::new();
        for entry in &log {
            if let Entry::SubmittedCopy {
                work,
                stream,
                ..
            } = entry
            {
                copies.push((*work, *stream));
            }
            if let Entry::SubmittedCompute {
                work,
                stream,
                duration_hint,
                ..
            } = entry
            {
                launches.push((*work, *stream, *duration_hint));
            }
            if let Entry::StreamWait {
                stream,
                work,
            } = entry
            {
                waits.push((*stream, *work));
            }
        }
        assert!(
            copies.iter().all(|(_, stream)| *stream == 0),
            "copies run on the copy stream: {copies:?}"
        );
        assert!(
            launches
                .iter()
                .all(|(_, stream, hint)| *stream == 1 && *hint == Some(HINT)),
            "launches run on the compute stream with their hint: {launches:?}"
        );
        let copy_works = copies.iter().map(|(work, _)| *work).collect::<Vec<_>>();
        let launch_works = launches.iter().map(|(work, ..)| *work).collect::<Vec<_>>();
        let compute_waits = waits
            .iter()
            .filter(|(stream, _)| *stream == 1)
            .map(|(_, work)| *work)
            .collect::<Vec<_>>();
        let copy_waits = waits
            .iter()
            .filter(|(stream, _)| *stream == 0)
            .map(|(_, work)| *work)
            .collect::<Vec<_>>();
        assert_eq!(compute_waits, copy_works, "compute waits on every transfer");
        assert_eq!(
            copy_waits,
            launch_works.get(..2).ok_or("two launches")?,
            "transfers into reused slots wait on the previous launch"
        );
        Ok(())
    }

    #[test]
    fn stream_waits_never_complete_work() -> TestResult {
        let graph = chain(3)?;
        let plan = buffered(&graph, 2)?;
        let mut backend = FakeBackend::new(DEVICE)?;
        assert_eq!(
            run(&mut backend, &graph, &plan)?,
            Ok(()),
            "the plan executes"
        );
        let log = backend.log();
        let host_wait = log
            .iter()
            .position(|entry| matches!(entry, Entry::HostWait { .. }))
            .ok_or("a host wait")?;
        let stream_waits = log
            .iter()
            .filter(|entry| matches!(entry, Entry::StreamWait { .. }))
            .count();
        let first_completion = log
            .iter()
            .position(|entry| matches!(entry, Entry::Completed { .. }))
            .ok_or("a completion")?;
        assert_eq!(stream_waits, 4, "three transfer waits and one reuse wait");
        assert!(
            first_completion > host_wait,
            "nothing completes before the host waits"
        );
        Ok(())
    }

    #[test]
    fn a_released_slot_can_be_allocated_again() -> TestResult {
        let graph = chain(2)?;
        let mut commands = layer_commands(0, 0).to_vec();
        commands.extend(layer_commands(1, 2));
        let plan = ExecutionPlan::new(commands);
        let mut backend = FakeBackend::new(DEVICE)?;
        assert_eq!(
            run(&mut backend, &graph, &plan)?,
            Ok(()),
            "the plan executes"
        );
        let log = backend.log();
        assert_eq!(device_resources(&log), [0, 2], "the slot gets a new buffer");
        let freed = log
            .iter()
            .position(|entry| {
                *entry
                    == Entry::Freed {
                        resource: 0,
                    }
            })
            .ok_or("the first buffer is freed")?;
        let reallocated = log
            .iter()
            .position(|entry| {
                matches!(
                    entry,
                    Entry::AllocatedDevice {
                        resource: 2,
                        ..
                    }
                )
            })
            .ok_or("the second buffer is allocated")?;
        assert!(
            freed < reallocated,
            "the first buffer is freed before the reallocation"
        );
        assert_eq!(
            computed(&log),
            [
                (OpId::from_index(0), vec![resident(0)?]),
                (OpId::from_index(1), vec![resident(1)?]),
            ],
            "each op sees its own weight"
        );
        assert_eq!(backend.live_device(), 0, "device memory is returned");
        check_lifetimes(&log)?;
        Ok(())
    }

    #[test]
    fn the_backend_retains_staging_buffers_until_their_copy_completes() -> TestResult {
        let graph = chain(3)?;
        let plan = buffered(&graph, 2)?;
        let mut backend = FakeBackend::new(DEVICE)?;
        assert_eq!(
            run(&mut backend, &graph, &plan)?,
            Ok(()),
            "the plan executes"
        );
        let log = backend.log();
        let position = |expected: &Entry| log.iter().position(|entry| entry == expected);
        let copies = log
            .iter()
            .filter_map(|entry| {
                if let Entry::SubmittedCopy {
                    work,
                    source,
                    ..
                } = entry
                {
                    Some((*work, *source))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(copies.len(), 3, "one copy per layer");
        for (work, source) in copies {
            let dropped = position(&Entry::HandleDropped {
                resource: source,
            })
            .ok_or("the handle is dropped")?;
            let completed = position(&Entry::Completed {
                work,
            })
            .ok_or("the copy completes")?;
            let freed = position(&Entry::Freed {
                resource: source,
            })
            .ok_or("the staging buffer is freed")?;
            assert!(
                dropped < completed && completed < freed,
                "staging buffer {source} outlives the interpreter handle until copy {work} \
                 completes"
            );
        }
        check_lifetimes(&log)?;
        Ok(())
    }

    fn single_layer_graph(
        matmul: bool,
        source: WeightSource,
    ) -> Result<Graph, Box<dyn Error>> {
        let mut graph = Graph::new();
        let activation = || TensorDesc::new(DType::F32, Shape::new([1, 64]));
        let input = graph.add_tensor(activation())?;
        graph.mark_input(input)?;
        let tensor = graph.add_tensor(TensorDesc::new(DType::U8, Shape::new([WEIGHT])))?;
        graph.add_weight(WeightDesc {
            tensor,
            bytes: ByteSize::from_bytes(WEIGHT),
            source,
        })?;
        let output = graph.add_tensor(activation())?;
        graph.add_op(if matmul {
            Op::MatMul {
                input,
                weight: tensor,
                output,
            }
        } else {
            Op::SyntheticCompute {
                inputs: vec![input, tensor],
                output,
                duration_hint: None,
            }
        })?;
        Ok(graph)
    }

    #[test]
    fn rejected_executions_have_no_backend_side_effects() -> TestResult {
        let graph = chain(2)?;
        let plan = buffered(&graph, 2)?;
        let payloads = payloads(&graph)?;
        let mapped = WeightSource::MappedFile {
            offset: 0,
            len: ByteSize::from_bytes(WEIGHT),
        };
        let single = ExecutionPlan::new(layer_commands(0, 0).to_vec());
        let mut truncated = plan.commands().to_vec();
        truncated.pop();
        let host = MemoryBudget::new(MemorySpace::Host, ByteSize::from_bytes(DEVICE));
        let missing = bindings(&payloads)
            .into_iter()
            .filter(|(tensor, _)| *tensor != weight(1))
            .collect::<BTreeMap<_, _>>();
        let short = [0; 63];
        let mut wrong = bindings(&payloads);
        wrong.insert(weight(1), &short);

        let cases: [Rejection<'_>; 7] = [
            (
                "an invalid plan",
                chain(2)?,
                ExecutionPlan::new(truncated),
                budget(),
                bindings(&payloads),
                DEVICE,
                RuntimeError::Plan(PlanError::UnreleasedSlot {
                    slot: slot(1),
                }),
            ),
            (
                "a host budget",
                chain(2)?,
                plan.clone(),
                host,
                bindings(&payloads),
                DEVICE,
                RuntimeError::Plan(PlanError::BudgetSpace(MemorySpace::Host)),
            ),
            (
                "a missing host weight",
                chain(2)?,
                plan.clone(),
                budget(),
                missing,
                DEVICE,
                RuntimeError::MissingHostWeight {
                    index: 3,
                    tensor: weight(1),
                },
            ),
            (
                "a short host weight",
                chain(2)?,
                plan.clone(),
                budget(),
                wrong,
                DEVICE,
                RuntimeError::HostWeightSize {
                    index: 3,
                    tensor: weight(1),
                    expected: ByteSize::from_bytes(WEIGHT),
                    actual: ByteSize::from_bytes(63),
                },
            ),
            (
                "a matmul",
                single_layer_graph(true, WeightSource::HostMemory)?,
                single.clone(),
                budget(),
                bindings(&payloads),
                DEVICE,
                RuntimeError::UnsupportedOp {
                    op: OpId::from_index(0),
                },
            ),
            (
                "a mapped-file weight",
                single_layer_graph(false, mapped)?,
                single,
                budget(),
                bindings(&payloads),
                DEVICE,
                RuntimeError::UnsupportedWeightSource {
                    tensor: weight(0),
                },
            ),
            (
                "a budget beyond the device",
                chain(2)?,
                plan.clone(),
                budget(),
                bindings(&payloads),
                DEVICE / 2,
                RuntimeError::BudgetExceedsDevice {
                    budget: ByteSize::from_bytes(DEVICE),
                    device: ByteSize::from_bytes(DEVICE / 2),
                },
            ),
        ];
        for (name, graph, plan, budget, weights, device, expected) in cases {
            let mut backend = FakeBackend::new(device)?;
            assert_eq!(
                execute(&mut backend, &graph, &plan, budget, &weights),
                Err(expected),
                "{name} is rejected"
            );
            assert_eq!(backend.log(), [], "{name} leaves the backend untouched");
        }
        Ok(())
    }

    fn backend_failure(
        operation: BackendOperation,
        source: Operation,
        cleanup: Option<Operation>,
    ) -> RuntimeError<FakeError> {
        RuntimeError::Execution {
            failure: ExecutionFailure::Backend {
                operation,
                source: FakeError(source),
            },
            cleanup: cleanup.map(FakeError),
        }
    }

    fn released(
        log: &[Entry],
        resource: u32,
    ) -> bool {
        log.iter().any(|entry| {
            *entry
                == Entry::HandleDropped {
                    resource,
                }
                || *entry
                    == Entry::Freed {
                        resource,
                    }
        })
    }

    #[test]
    fn backend_failures_carry_their_command_and_drain_submitted_work() -> TestResult {
        let graph = chain(2)?;
        let plan = buffered(&graph, 1)?;
        let cases = [
            (
                Operation::CreateStream,
                1,
                BackendOperation::CreateStream {
                    stream: COMPUTE,
                },
                false,
            ),
            (
                Operation::AllocateDevice,
                0,
                BackendOperation::Allocate {
                    index: 0,
                    slot: slot(0),
                },
                false,
            ),
            (
                Operation::Copy,
                0,
                BackendOperation::Copy {
                    index: 1,
                    tensor: weight(0),
                    slot: slot(0),
                    stream: COPY,
                    event: event(0),
                },
                true,
            ),
            (
                Operation::StreamWait,
                0,
                BackendOperation::StreamWait {
                    index: 2,
                    stream: COMPUTE,
                    event: event(0),
                },
                true,
            ),
            (
                Operation::AllocateHost,
                1,
                BackendOperation::StageHost {
                    index: 5,
                    tensor: weight(1),
                },
                true,
            ),
            (
                Operation::Copy,
                1,
                BackendOperation::Copy {
                    index: 5,
                    tensor: weight(1),
                    slot: slot(0),
                    stream: COPY,
                    event: event(2),
                },
                true,
            ),
            (
                Operation::Compute,
                1,
                BackendOperation::Launch {
                    index: 7,
                    op: OpId::from_index(1),
                    stream: COMPUTE,
                    event: event(3),
                },
                true,
            ),
            (
                Operation::HostWait,
                0,
                BackendOperation::HostWait {
                    index: 8,
                    event: event(3),
                },
                true,
            ),
        ];
        for (operation, call, expected, drained) in cases {
            let mut backend = FakeBackend::new(DEVICE)?.fail_on(operation, call);
            let error = run(&mut backend, &graph, &plan)?
                .err()
                .ok_or("the execution fails")?;
            assert_eq!(
                error,
                backend_failure(expected, operation, None),
                "{operation:?} #{call} is reported"
            );
            assert_eq!(
                error
                    .source()
                    .and_then(Error::source)
                    .map(ToString::to_string),
                Some(FakeError(operation).to_string()),
                "{operation:?} #{call}: the backend error ends the source chain"
            );
            let log = backend.log();
            assert_eq!(
                log.contains(&Entry::Drained),
                drained,
                "{operation:?} #{call}: drained only after a submission"
            );
            assert_eq!(
                backend.pending(),
                0,
                "{operation:?} #{call}: no pending work"
            );
            assert_eq!(
                backend.live_device(),
                0,
                "{operation:?} #{call}: device memory is returned"
            );
            check_lifetimes(&log)?;
        }
        Ok(())
    }

    #[test]
    fn a_failed_drain_is_reported_and_keeps_in_flight_resources() -> TestResult {
        let graph = chain(2)?;
        let plan = buffered(&graph, 1)?;
        let mut backend = FakeBackend::new(DEVICE)?
            .fail_on(Operation::Compute, 1)
            .fail_on(Operation::Drain, 0);
        let error = run(&mut backend, &graph, &plan)?
            .err()
            .ok_or("the execution fails")?;
        assert_eq!(
            error,
            backend_failure(
                BackendOperation::Launch {
                    index: 7,
                    op: OpId::from_index(1),
                    stream: COMPUTE,
                    event: event(3),
                },
                Operation::Compute,
                Some(Operation::Drain),
            ),
            "the launch failure is kept and the drain failure is attached"
        );
        assert_eq!(
            error.cleanup(),
            Some(&FakeError(Operation::Drain)),
            "the cleanup failure is exposed"
        );
        let message = error.to_string();
        assert!(
            message.contains("fake Compute failure") && message.contains("fake Drain failure"),
            "both failures are described: {message}"
        );
        let log = backend.log();
        assert!(backend.pending() > 0, "work is still in flight");
        assert!(
            !log.iter().any(|entry| matches!(
                entry,
                Entry::HandleDropped {
                    resource: 0
                } | Entry::Freed {
                    resource: 0
                }
            )),
            "the in-flight device buffer is never released"
        );
        check_lifetimes(&log)?;
        Ok(())
    }

    fn missing_stream(
        backend: &mut FakeBackend
    ) -> Result<RuntimeError<FakeError>, Box<dyn Error>> {
        let graph = chain(1)?;
        let payloads = payloads(&graph)?;
        let weights = bindings(&payloads);
        let durations = BTreeMap::from([(OpId::from_index(0), Some(HINT))]);
        let [allocate, prefetch, wait, ..] = layer_commands(0, 0);
        let mut observer = Unobserved;
        let mut run = Run::new(backend, &mut observer, durations, &weights);
        run.create_stream(COPY)?;
        run.command(0, &allocate)?;
        run.command(1, &prefetch)?;
        let failure = run
            .command(2, &wait)
            .err()
            .ok_or("the wait on the compute stream fails")?;
        assert_eq!(
            failure,
            ExecutionFailure::MissingStream {
                index: 2,
                stream: COMPUTE,
            },
            "the missing stream is named"
        );
        Ok(run.fail(failure))
    }

    #[test]
    fn a_missing_stream_is_reported_as_such() -> TestResult {
        let mut backend = FakeBackend::new(DEVICE)?;
        let error = missing_stream(&mut backend)?;
        assert_eq!(
            error,
            RuntimeError::Execution {
                failure: ExecutionFailure::MissingStream {
                    index: 2,
                    stream: COMPUTE,
                },
                cleanup: None,
            },
            "the internal failure is reported after a successful drain"
        );
        assert!(
            error.to_string().contains("StreamId(1) was not created"),
            "the message names the stream: {error}"
        );
        let log = backend.log();
        assert!(log.contains(&Entry::Drained), "submitted work is drained");
        assert_eq!(backend.pending(), 0, "no pending work");
        assert_eq!(backend.live_device(), 0, "device memory is returned");
        check_lifetimes(&log)?;
        Ok(())
    }

    #[test]
    fn an_internal_failure_after_submission_keeps_the_drain_failure() -> TestResult {
        let mut backend = FakeBackend::new(DEVICE)?.fail_on(Operation::Drain, 0);
        let error = missing_stream(&mut backend)?;
        assert_eq!(
            error,
            RuntimeError::Execution {
                failure: ExecutionFailure::MissingStream {
                    index: 2,
                    stream: COMPUTE,
                },
                cleanup: Some(FakeError(Operation::Drain)),
            },
            "the drain failure is attached to the internal failure"
        );
        let log = backend.log();
        assert!(backend.pending() > 0, "the copy is still in flight");
        assert!(
            !released(&log, 0),
            "the device buffer is neither dropped nor freed"
        );
        assert!(
            !log.contains(&Entry::Freed {
                resource: 1
            }),
            "the backend still retains the staging buffer of the pending copy"
        );
        check_lifetimes(&log)?;
        Ok(())
    }

    #[test]
    fn work_submitted_by_a_failing_call_is_drained_and_retained() -> TestResult {
        let graph = chain(2)?;
        let plan = buffered(&graph, 1)?;
        let cases = [
            (
                Operation::Copy,
                0,
                BackendOperation::Copy {
                    index: 1,
                    tensor: weight(0),
                    slot: slot(0),
                    stream: COPY,
                    event: event(0),
                },
                Some(1),
            ),
            (
                Operation::Copy,
                1,
                BackendOperation::Copy {
                    index: 5,
                    tensor: weight(1),
                    slot: slot(0),
                    stream: COPY,
                    event: event(2),
                },
                Some(2),
            ),
            (
                Operation::Compute,
                1,
                BackendOperation::Launch {
                    index: 7,
                    op: OpId::from_index(1),
                    stream: COMPUTE,
                    event: event(3),
                },
                None,
            ),
            (
                Operation::StreamWait,
                0,
                BackendOperation::StreamWait {
                    index: 2,
                    stream: COMPUTE,
                    event: event(0),
                },
                None,
            ),
        ];
        for (operation, call, expected, staging) in cases {
            let mut backend = FakeBackend::new(DEVICE)?.fail_after_submit(operation, call);
            let error = run(&mut backend, &graph, &plan)?
                .err()
                .ok_or("the execution fails")?;
            assert_eq!(
                error,
                backend_failure(expected, operation, None),
                "{operation:?} #{call} is reported"
            );
            let log = backend.log();
            let position = |expected: &Entry| log.iter().position(|entry| entry == expected);
            let drained = position(&Entry::Drained).ok_or("submitted work is drained")?;
            let failed = position(&Entry::Failed(operation)).ok_or("the failure is logged")?;
            assert!(
                failed < drained,
                "{operation:?} #{call}: the drain follows the failure"
            );
            if let Some(resource) = staging {
                let work = log
                    .iter()
                    .find_map(|entry| {
                        if let Entry::SubmittedCopy {
                            work,
                            source,
                            ..
                        } = entry
                        {
                            (*source == resource).then_some(*work)
                        } else {
                            None
                        }
                    })
                    .ok_or("the failing copy was submitted")?;
                let completed = position(&Entry::Completed {
                    work,
                })
                .ok_or("the failing copy completes during the drain")?;
                let dropped = position(&Entry::HandleDropped {
                    resource,
                })
                .ok_or("the staging handle is dropped")?;
                let freed = position(&Entry::Freed {
                    resource,
                })
                .ok_or("the staging buffer is freed")?;
                assert!(
                    failed < completed && completed < freed && drained < dropped,
                    "{operation:?} #{call}: staging buffer {resource} outlives its failed copy"
                );
            }
            assert_eq!(
                backend.pending(),
                0,
                "{operation:?} #{call}: no pending work"
            );
            assert_eq!(
                backend.live_device(),
                0,
                "{operation:?} #{call}: device memory is returned"
            );
            check_lifetimes(&log)?;
        }
        Ok(())
    }

    #[test]
    fn a_failing_submission_with_a_failed_drain_frees_nothing_in_flight() -> TestResult {
        let graph = chain(2)?;
        let plan = buffered(&graph, 1)?;
        let mut backend = FakeBackend::new(DEVICE)?
            .fail_after_submit(Operation::Copy, 1)
            .fail_on(Operation::Drain, 0);
        let error = run(&mut backend, &graph, &plan)?
            .err()
            .ok_or("the execution fails")?;
        assert_eq!(
            error,
            backend_failure(
                BackendOperation::Copy {
                    index: 5,
                    tensor: weight(1),
                    slot: slot(0),
                    stream: COPY,
                    event: event(2),
                },
                Operation::Copy,
                Some(Operation::Drain),
            ),
            "both failures are reported"
        );
        let log = backend.log();
        assert!(backend.pending() > 0, "the failed copy is still in flight");
        assert!(
            !released(&log, 0) && !released(&log, 2),
            "neither the slot nor the staging buffer of the failed copy is released"
        );
        check_lifetimes(&log)?;
        Ok(())
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Recorded {
        activity: CpuActivity,
        index: Option<usize>,
        op: Option<OpId>,
        tensor: Option<TensorId>,
        slot: Option<BufferSlot>,
        bindings: Vec<WeightBinding>,
        stream: Option<StreamId>,
        event: Option<EventId>,
        bytes: Option<ByteSize>,
        start: u64,
        end: u64,
        succeeded: bool,
    }

    type Observed = (Result<(), RuntimeError<FakeError>>, Recorder);

    #[derive(Default)]
    struct Recorder {
        clock: u64,
        records: Vec<Recorded>,
        outcomes: Vec<ExecutionOutcome>,
    }

    impl ExecutionObserver for Recorder {
        type Instant = u64;

        fn now(&mut self) -> u64 {
            self.clock = self.clock.saturating_add(1);
            self.clock
        }

        fn record(
            &mut self,
            record: CpuRecord<'_, u64>,
        ) {
            let context = record.context;
            self.records.push(Recorded {
                activity: record.activity,
                index: context.index,
                op: context.op,
                tensor: context.tensor,
                slot: context.slot,
                bindings: context.bindings.to_vec(),
                stream: context.stream,
                event: context.event,
                bytes: context.bytes,
                start: record.start,
                end: record.end,
                succeeded: record.succeeded,
            });
        }

        fn finish(
            &mut self,
            outcome: ExecutionOutcome,
        ) {
            self.outcomes.push(outcome);
        }
    }

    fn observe(
        backend: &mut FakeBackend,
        graph: &Graph,
        plan: &ExecutionPlan,
    ) -> Result<Observed, Box<dyn Error>> {
        let payloads = payloads(graph)?;
        let mut recorder = Recorder::default();
        let result = execute_observed(
            backend,
            graph,
            plan,
            budget(),
            &bindings(&payloads),
            &mut recorder,
        );
        Ok((result, recorder))
    }

    fn activity(command: &Command) -> Vec<CpuActivity> {
        match command {
            Command::Allocate {
                ..
            } => vec![CpuActivity::Allocate],
            Command::Prefetch {
                ..
            } => vec![CpuActivity::Stage, CpuActivity::SubmitCopy],
            Command::Wait {
                target: WaitTarget::Stream(_),
                ..
            } => vec![CpuActivity::StreamWait],
            Command::Wait {
                target: WaitTarget::Host,
                ..
            } => vec![CpuActivity::HostWait],
            Command::Launch {
                ..
            } => vec![CpuActivity::SubmitLaunch],
            Command::Release {
                ..
            } => vec![CpuActivity::Release],
        }
    }

    #[test]
    fn observed_and_unobserved_executions_are_equivalent() -> TestResult {
        let graph = chain(4)?;
        for slots in 1..=3 {
            let plan = buffered(&graph, slots)?;
            let mut plain = FakeBackend::new(DEVICE)?;
            let expected = run(&mut plain, &graph, &plan)?;
            let mut watched = FakeBackend::new(DEVICE)?;
            let (result, recorder) = observe(&mut watched, &graph, &plan)?;
            assert_eq!(result, expected, "{slots} slots: same result");
            assert_eq!(
                watched.log(),
                plain.log(),
                "{slots} slots: same backend calls"
            );
            assert_eq!(
                recorder.outcomes,
                [ExecutionOutcome::Completed],
                "{slots} slots: one completed outcome"
            );
            let activities: Vec<CpuActivity> = recorder
                .records
                .iter()
                .map(|record| record.activity)
                .collect();
            let expected: Vec<CpuActivity> = [CpuActivity::CreateStream, CpuActivity::CreateStream]
                .into_iter()
                .chain(plan.commands().iter().flat_map(activity))
                .collect();
            assert_eq!(activities, expected, "{slots} slots: one record per call");
            assert!(
                recorder
                    .records
                    .windows(2)
                    .all(|pair| matches!(pair, [first, second] if first.end < second.start)),
                "{slots} slots: records are chronological and disjoint"
            );
            assert!(
                recorder
                    .records
                    .iter()
                    .all(|record| record.start < record.end && record.succeeded),
                "{slots} slots: every interval is closed and successful"
            );
        }
        Ok(())
    }

    #[test]
    fn records_carry_their_command_context() -> TestResult {
        let graph = chain(1)?;
        let plan = buffered(&graph, 1)?;
        let mut backend = FakeBackend::new(DEVICE)?;
        let (result, recorder) = observe(&mut backend, &graph, &plan)?;
        assert_eq!(result, Ok(()), "the plan executes");
        let base = Recorded {
            activity: CpuActivity::CreateStream,
            index: None,
            op: None,
            tensor: None,
            slot: None,
            bindings: Vec::new(),
            stream: None,
            event: None,
            bytes: None,
            start: 0,
            end: 0,
            succeeded: true,
        };
        let copy = Recorded {
            activity: CpuActivity::Stage,
            index: Some(1),
            tensor: Some(weight(0)),
            slot: Some(slot(0)),
            stream: Some(COPY),
            event: Some(event(0)),
            bytes: Some(ByteSize::from_bytes(WEIGHT)),
            ..base.clone()
        };
        let expected = [
            Recorded {
                stream: Some(COPY),
                ..base.clone()
            },
            Recorded {
                stream: Some(COMPUTE),
                ..base.clone()
            },
            Recorded {
                activity: CpuActivity::Allocate,
                index: Some(0),
                slot: Some(slot(0)),
                bytes: Some(ByteSize::from_bytes(SLOT)),
                ..base.clone()
            },
            copy.clone(),
            Recorded {
                activity: CpuActivity::SubmitCopy,
                ..copy
            },
            Recorded {
                activity: CpuActivity::StreamWait,
                index: Some(2),
                stream: Some(COMPUTE),
                event: Some(event(0)),
                ..base.clone()
            },
            Recorded {
                activity: CpuActivity::SubmitLaunch,
                index: Some(3),
                op: Some(OpId::from_index(0)),
                bindings: vec![WeightBinding {
                    tensor: weight(0),
                    slot: slot(0),
                }],
                stream: Some(COMPUTE),
                event: Some(event(1)),
                bytes: Some(ByteSize::from_bytes(WEIGHT)),
                ..base.clone()
            },
            Recorded {
                activity: CpuActivity::HostWait,
                index: Some(4),
                event: Some(event(1)),
                ..base.clone()
            },
            Recorded {
                activity: CpuActivity::Release,
                index: Some(5),
                slot: Some(slot(0)),
                ..base
            },
        ];
        let actual: Vec<Recorded> = recorder
            .records
            .iter()
            .map(|record| Recorded {
                start: 0,
                end: 0,
                ..record.clone()
            })
            .collect();
        assert_eq!(actual, expected, "every record names its command");
        Ok(())
    }

    #[test]
    fn observed_failures_report_their_outcome_and_cleanup() -> TestResult {
        let graph = chain(2)?;
        let plan = buffered(&graph, 2)?;
        let cases = [
            (
                FakeBackend::new(DEVICE)?.fail_on(Operation::AllocateDevice, 0),
                CpuActivity::Allocate,
                false,
                false,
            ),
            (
                FakeBackend::new(DEVICE)?.fail_on(Operation::AllocateHost, 1),
                CpuActivity::Stage,
                true,
                false,
            ),
            (
                FakeBackend::new(DEVICE)?.fail_after_submit(Operation::Copy, 0),
                CpuActivity::SubmitCopy,
                true,
                false,
            ),
            (
                FakeBackend::new(DEVICE)?.fail_on(Operation::Compute, 0),
                CpuActivity::SubmitLaunch,
                true,
                false,
            ),
            (
                FakeBackend::new(DEVICE)?
                    .fail_on(Operation::HostWait, 0)
                    .fail_on(Operation::Drain, 0),
                CpuActivity::HostWait,
                true,
                true,
            ),
        ];
        for (mut backend, failed, drained, cleanup_failed) in cases {
            let (result, recorder) = observe(&mut backend, &graph, &plan)?;
            let error = result.err().ok_or("the execution fails")?;
            let operation = match &error {
                RuntimeError::Execution {
                    failure:
                        ExecutionFailure::Backend {
                            operation, ..
                        },
                    ..
                } => Some(*operation),
                RuntimeError::Plan(_)
                | RuntimeError::UnsupportedOp {
                    ..
                }
                | RuntimeError::UnsupportedWeightSource {
                    ..
                }
                | RuntimeError::MissingHostWeight {
                    ..
                }
                | RuntimeError::HostWeightSize {
                    ..
                }
                | RuntimeError::BudgetExceedsDevice {
                    ..
                }
                | RuntimeError::Execution {
                    ..
                } => None,
            };
            assert!(operation.is_some(), "{failed:?}: a backend operation fails");
            assert_eq!(
                recorder.outcomes,
                [ExecutionOutcome::Failed {
                    operation,
                    cleanup_failed,
                }],
                "{failed:?}: the outcome names the failure"
            );
            let unsuccessful: Vec<CpuActivity> = recorder
                .records
                .iter()
                .filter(|record| !record.succeeded)
                .map(|record| record.activity)
                .collect();
            let expected = if cleanup_failed {
                vec![failed, CpuActivity::CleanupDrain]
            } else {
                vec![failed]
            };
            assert_eq!(
                unsuccessful, expected,
                "{failed:?}: failed calls are marked"
            );
            let last = recorder.records.last().map(|record| record.activity);
            assert_eq!(
                last == Some(CpuActivity::CleanupDrain),
                drained,
                "{failed:?}: submitted work is drained under observation"
            );
        }
        Ok(())
    }

    #[test]
    fn rejected_executions_are_observed_without_records() -> TestResult {
        let graph = chain(1)?;
        let plan = buffered(&graph, 1)?;
        let mut backend = FakeBackend::new(1)?;
        let (result, recorder) = observe(&mut backend, &graph, &plan)?;
        assert!(result.is_err(), "the budget exceeds the device");
        assert_eq!(recorder.records, [], "nothing is recorded");
        assert_eq!(
            recorder.outcomes,
            [ExecutionOutcome::Rejected],
            "the rejection is reported"
        );
        Ok(())
    }
}
