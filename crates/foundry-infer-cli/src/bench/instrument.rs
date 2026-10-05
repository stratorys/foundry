use std::collections::BTreeMap;

use foundry_infer_core::Id;
use foundry_infer_metal::{
    AllocationCategory,
    AllocationChange,
    CommandLabel,
    GpuTiming as Timing,
    GpuWork,
    HostClock,
    HostInstant,
    MetalDiagnostics,
};
use foundry_infer_plan::{
    Command,
    ExecutionPlan,
};
use foundry_infer_runtime::{
    CpuRecord,
    ExecutionObserver,
    ExecutionOutcome,
};
use serde_json::Value;

use crate::bench::error::BenchError;
use crate::bench::report::{
    CategoryPeak,
    CategoryUsage,
    Clock,
    ConfigurationMemory,
    ExecutionMemory,
    GpuExecution,
    GpuTiming,
    Memory,
    PhaseDto,
    Snapshot,
    TraceInfo,
};
use crate::bench::trace::{
    TraceDocument,
    TraceEvent,
    Track,
    counter,
    instant,
    lifetime,
    metadata,
    separate_overlaps,
    span,
};

pub(crate) const HARNESS_SPANS: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CpuSpan {
    pub(crate) activity: &'static str,
    pub(crate) track: Track,
    pub(crate) label: CommandLabel,
    pub(crate) start: HostInstant,
    pub(crate) end: HostInstant,
    pub(crate) succeeded: bool,
}

pub(crate) struct TraceObserver {
    clock: HostClock,
    spans: Vec<CpuSpan>,
    truncated: u64,
    outcome: Option<ExecutionOutcome>,
}

impl TraceObserver {
    pub(crate) fn new(
        clock: HostClock,
        capacity: usize,
    ) -> Result<Self, BenchError> {
        let mut spans = Vec::new();
        spans
            .try_reserve_exact(capacity)
            .map_err(|_| BenchError::DiagnosticBudget {
                what: "CPU spans",
            })?;
        Ok(Self {
            clock,
            spans,
            truncated: 0,
            outcome: None,
        })
    }

    pub(crate) fn push(
        &mut self,
        span: CpuSpan,
    ) {
        if self.spans.len() < self.spans.capacity() {
            self.spans.push(span);
        } else {
            self.truncated = self.truncated.saturating_add(1);
        }
    }

    pub(crate) fn harness(
        &mut self,
        activity: &'static str,
        track: Track,
        start: HostInstant,
        succeeded: bool,
    ) {
        let end = self.clock.now();
        self.push(CpuSpan {
            activity,
            track,
            label: CommandLabel::default(),
            start,
            end,
            succeeded,
        });
    }

    pub(crate) fn finish_trace(
        self,
        execution: ExecutionTrace,
    ) -> ExecutionTrace {
        let storage = self
            .spans
            .capacity()
            .checked_mul(size_of::<CpuSpan>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .unwrap_or(u64::MAX);
        ExecutionTrace {
            cpu: self.spans,
            cpu_truncated: self.truncated,
            cpu_storage_bytes: storage,
            outcome: self.outcome,
            ..execution
        }
    }
}

impl ExecutionObserver for TraceObserver {
    type Instant = HostInstant;

    fn now(&mut self) -> HostInstant { self.clock.now() }

    fn record(
        &mut self,
        record: CpuRecord<'_, HostInstant>,
    ) {
        let track = if record.activity.waits() {
            Track::CpuWaits
        } else {
            Track::CpuRuntime
        };
        self.push(CpuSpan {
            activity: record.activity.name(),
            track,
            label: CommandLabel::from(&record.context),
            start: record.start,
            end: record.end,
            succeeded: record.succeeded,
        });
    }

    fn finish(
        &mut self,
        outcome: ExecutionOutcome,
    ) {
        self.outcome = Some(outcome);
    }
}

pub(crate) fn cpu_capacity(plan: &ExecutionPlan) -> Option<usize> {
    let streams = plan
        .commands()
        .iter()
        .filter(|command| {
            matches!(
                command,
                Command::Prefetch { .. } | Command::Launch { .. } | Command::Wait { .. }
            )
        })
        .count();
    plan.commands()
        .len()
        .checked_mul(2)?
        .checked_add(streams)?
        .checked_add(HARNESS_SPANS)?
        .checked_add(2)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExecutionTrace {
    pub(crate) sequence: u32,
    pub(crate) phase: PhaseDto,
    pub(crate) round: u32,
    pub(crate) position: u32,
    pub(crate) buffers: u32,
    pub(crate) planned_peak: u64,
    pub(crate) timed: Option<(HostInstant, HostInstant)>,
    pub(crate) cpu: Vec<CpuSpan>,
    pub(crate) cpu_truncated: u64,
    pub(crate) cpu_storage_bytes: u64,
    pub(crate) outcome: Option<ExecutionOutcome>,
    pub(crate) diagnostics: Option<MetalDiagnostics>,
}

fn relative(
    instant: HostInstant,
    origin: HostInstant,
) -> u64 {
    instant.nanos_since(origin).unwrap_or_default()
}

fn usage_of(
    diagnostics: &MetalDiagnostics,
    category: AllocationCategory,
) -> foundry_infer_metal::Usage {
    diagnostics
        .categories
        .get(&category)
        .copied()
        .unwrap_or_default()
}

fn execution_memory(
    trace: &ExecutionTrace,
    diagnostics: &MetalDiagnostics,
    origin: HostInstant,
) -> ExecutionMemory {
    let private = usage_of(diagnostics, AllocationCategory::PrivateSlot);
    let categories = AllocationCategory::ALL
        .into_iter()
        .map(|category| {
            let usage = usage_of(diagnostics, category);
            CategoryUsage {
                category: category.name().to_owned(),
                allocations: usage.allocations,
                frees: usage.frees,
                peak_requested_bytes: usage.peak_requested,
                peak_allocated_bytes: usage.peak_allocated,
                live_requested_bytes_at_extraction: usage.live_requested,
                live_allocated_bytes_at_extraction: usage.live_allocated,
            }
        })
        .collect();
    let snapshots = diagnostics
        .snapshots
        .iter()
        .map(|snapshot| {
            let per_category = |bytes: fn(&foundry_infer_metal::Usage) -> u64| {
                snapshot
                    .categories
                    .iter()
                    .map(|(category, usage)| (category.name().to_owned(), bytes(usage)))
                    .collect()
            };
            Snapshot {
                boundary: snapshot.boundary.name().to_owned(),
                at_ns: snapshot.at.nanos_since(origin),
                live_allocations: u64::try_from(snapshot.live_allocations).unwrap_or(u64::MAX),
                live_requested_bytes: per_category(|usage| usage.live_requested),
                live_allocated_bytes: per_category(|usage| usage.live_allocated),
                device_current_allocated_bytes: snapshot.device_current,
            }
        })
        .collect();
    ExecutionMemory {
        sequence: trace.sequence,
        buffers: trace.buffers,
        planned_slot_peak_bytes: trace.planned_peak,
        observed_private_slot_peak_requested_bytes: private.peak_requested,
        retained_private_slot_bytes_beyond_plan: private
            .peak_requested
            .saturating_sub(trace.planned_peak),
        categories,
        tracked_peak_requested_bytes: diagnostics.tracked.peak_requested,
        tracked_peak_allocated_bytes: diagnostics.tracked.peak_allocated,
        snapshots,
        live_allocations_at_extraction: u64::try_from(diagnostics.live_allocations)
            .unwrap_or(u64::MAX),
        device_current_allocated_sampled_max_bytes: diagnostics.device_sampled_max,
        device_samples: diagnostics.device_samples,
        diagnostic_storage_bytes: diagnostics
            .storage_bytes
            .saturating_add(trace.cpu_storage_bytes),
        diagnostic_records_truncated: diagnostics
            .gpu_truncated
            .saturating_add(diagnostics.allocations_truncated)
            .saturating_add(trace.cpu_truncated),
    }
}

fn configuration_memory(
    buffers: u32,
    planned_peak: u64,
    slot_budget: u64,
    executions: &[&ExecutionMemory],
) -> ConfigurationMemory {
    let max = |value: fn(&ExecutionMemory) -> u64| {
        executions
            .iter()
            .map(|execution| value(execution))
            .max()
            .unwrap_or_default()
    };
    let categories = AllocationCategory::ALL
        .into_iter()
        .map(|category| {
            let usages: Vec<&CategoryUsage> = executions
                .iter()
                .filter_map(|execution| {
                    execution
                        .categories
                        .iter()
                        .find(|usage| usage.category == category.name())
                })
                .collect();
            let max = |value: fn(&CategoryUsage) -> u64| {
                usages
                    .iter()
                    .map(|usage| value(usage))
                    .max()
                    .unwrap_or_default()
            };
            CategoryPeak {
                category: category.name().to_owned(),
                peak_requested_bytes_max: max(|usage| usage.peak_requested_bytes),
                peak_allocated_bytes_max: max(|usage| usage.peak_allocated_bytes),
                allocations_max: max(|usage| usage.allocations),
            }
        })
        .collect();
    ConfigurationMemory {
        buffers,
        executions: u32::try_from(executions.len()).unwrap_or(u32::MAX),
        planned_slot_peak_bytes: planned_peak,
        planned_peak_within_budget: planned_peak <= slot_budget,
        observed_private_slot_peak_requested_bytes_max: max(|execution| {
            execution.observed_private_slot_peak_requested_bytes
        }),
        observed_private_slot_peak_allocated_bytes_max: max(|execution| {
            execution
                .categories
                .iter()
                .find(|usage| usage.category == AllocationCategory::PrivateSlot.name())
                .map(|usage| usage.peak_allocated_bytes)
                .unwrap_or_default()
        }),
        retained_private_slot_bytes_beyond_plan_max: max(|execution| {
            execution.retained_private_slot_bytes_beyond_plan
        }),
        categories,
        device_current_allocated_sampled_max_bytes: max(|execution| {
            execution.device_current_allocated_sampled_max_bytes
        }),
        diagnostic_storage_bytes_max: max(|execution| execution.diagnostic_storage_bytes),
    }
}

pub(crate) fn memory(
    traces: &[ExecutionTrace],
    plans: &[(u32, u64)],
    payload_bytes: u64,
    slot_budget: u64,
    origin: HostInstant,
) -> Memory {
    let executions: Vec<(PhaseDto, ExecutionMemory)> = traces
        .iter()
        .filter_map(|trace| {
            trace
                .diagnostics
                .as_ref()
                .map(|diagnostics| (trace.phase, execution_memory(trace, diagnostics, origin)))
        })
        .collect();
    let configurations = plans
        .iter()
        .map(|&(buffers, planned_peak)| {
            let measured: Vec<&ExecutionMemory> = executions
                .iter()
                .filter(|(phase, execution)| {
                    *phase == PhaseDto::Measured && execution.buffers == buffers
                })
                .map(|(_, execution)| execution)
                .collect();
            configuration_memory(buffers, planned_peak, slot_budget, &measured)
        })
        .collect();
    Memory {
        cpu_payload_bytes: payload_bytes,
        cpu_payload_lifetime: "generated in host memory before the first round and kept until the \
                               last execution; each execution copies it into fresh shared staging \
                               buffers"
            .to_owned(),
        slot_budget_bytes: slot_budget,
        accounting: [
            "requested bytes are the lengths Foundry asked Metal for; allocated bytes are \
             MTLAllocation.allocatedSize of the same buffer",
            "each native buffer is counted once, from creation until Foundry drops its last \
             reference, including references held by in-flight command buffers and checksum \
             results; Metal may return memory later",
            "a plan Release command drops the runtime's handle but does not prove the buffer was \
             freed",
            "configuration maxima cover measured executions only",
            "device_current_allocated values sample MTLDevice.currentAllocatedSize at allocation \
             transitions and boundaries; their maximum is a sampled maximum, not an exact process \
             peak, and is never summed with other measurements",
            "CPU payload bytes, Metal buffers, driver allocations and diagnostic storage are \
             reported separately and not added into an application memory peak",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        configurations,
        executions: executions
            .into_iter()
            .map(|(_, execution)| execution)
            .collect(),
    }
}

fn busy(
    diagnostics: &MetalDiagnostics,
    work: GpuWork,
) -> (u64, u64) {
    diagnostics
        .gpu
        .iter()
        .filter(|interval| interval.work == work)
        .fold((0_u64, 0_u64), |(count, total), interval| {
            let duration = match interval.timing {
                Timing::Valid {
                    start,
                    end,
                } => end.nanos_since(start).unwrap_or_default(),
                Timing::Invalid(_) => 0,
            };
            (count.saturating_add(1), total.saturating_add(duration))
        })
}

fn overlaps(
    diagnostics: &MetalDiagnostics,
    work: GpuWork,
) -> u64 {
    let mut spans: Vec<(HostInstant, HostInstant)> = diagnostics
        .gpu
        .iter()
        .filter(|interval| interval.work == work)
        .filter_map(|interval| match interval.timing {
            Timing::Valid {
                start,
                end,
            } => Some((start, end)),
            Timing::Invalid(_) => None,
        })
        .collect();
    spans.sort_unstable();
    spans
        .iter()
        .fold(
            (0_u64, None),
            |(count, busy): (u64, Option<HostInstant>), &(start, end)| {
                let overlapping = busy.is_some_and(|busy| start < busy);
                (
                    count.saturating_add(u64::from(overlapping)),
                    Some(busy.map_or(end, |busy| busy.max(end))),
                )
            },
        )
        .0
}

pub(crate) fn gpu_timing(traces: &[ExecutionTrace]) -> GpuTiming {
    let diagnosed: Vec<(&ExecutionTrace, &MetalDiagnostics)> = traces
        .iter()
        .filter_map(|trace| {
            trace
                .diagnostics
                .as_ref()
                .map(|diagnostics| (trace, diagnostics))
        })
        .collect();
    let intervals = diagnosed
        .iter()
        .flat_map(|(_, diagnostics)| diagnostics.gpu.iter());
    let (valid, invalid_reasons) = intervals.fold(
        (0_u64, BTreeMap::<String, u64>::new()),
        |(valid, mut reasons), interval| match interval.timing {
            Timing::Valid {
                ..
            } => (valid.saturating_add(1), reasons),
            Timing::Invalid(fault) => {
                let count = reasons.entry(fault.reason().to_owned()).or_default();
                *count = count.saturating_add(1);
                (valid, reasons)
            }
        },
    );
    let invalid = invalid_reasons
        .values()
        .fold(0_u64, |total, count| total.saturating_add(*count));
    let truncated = diagnosed.iter().fold(0_u64, |total, (_, diagnostics)| {
        total.saturating_add(diagnostics.gpu_truncated)
    });
    let pending = diagnosed.iter().fold(0_u64, |total, (_, diagnostics)| {
        total.saturating_add(u64::try_from(diagnostics.gpu_pending).unwrap_or(u64::MAX))
    });
    let complete = diagnosed.len() == traces.len();
    let passed = complete && valid > 0 && invalid == 0 && truncated == 0 && pending == 0;
    let executions = diagnosed
        .iter()
        .map(|(trace, diagnostics)| {
            let (copy_command_buffers, copy_busy_ns) = busy(diagnostics, GpuWork::Copy);
            let (compute_command_buffers, compute_busy_ns) = busy(diagnostics, GpuWork::Compute);
            let bounds = diagnostics
                .gpu
                .iter()
                .filter_map(|interval| match interval.timing {
                    Timing::Valid {
                        start,
                        end,
                    } => Some((start, end)),
                    Timing::Invalid(_) => None,
                })
                .fold(
                    None,
                    |bounds: Option<(HostInstant, HostInstant)>, (start, end)| {
                        Some(bounds.map_or((start, end), |(first, last)| {
                            (first.min(start), last.max(end))
                        }))
                    },
                );
            GpuExecution {
                sequence: trace.sequence,
                buffers: trace.buffers,
                copy_command_buffers,
                copy_busy_ns,
                compute_command_buffers,
                compute_busy_ns,
                overlapping_command_buffers: overlaps(diagnostics, GpuWork::Copy)
                    .saturating_add(overlaps(diagnostics, GpuWork::Compute)),
                first_start_to_last_end_ns: bounds
                    .and_then(|(first, last)| last.nanos_since(first)),
            }
        })
        .collect();
    GpuTiming {
        granularity: "command_buffer".to_owned(),
        semantics: "GPUStartTime to GPUEndTime of each completed command buffer, read after \
                    completion; an interval covers the whole command buffer, including encoded \
                    event waits, not an isolated kernel; CPU submission time is never substituted"
            .to_owned(),
        valid,
        invalid,
        truncated,
        pending_at_extraction: pending,
        invalid_reasons,
        gate: if passed { "passed" } else { "failed" }.to_owned(),
        executions,
    }
}

pub(crate) fn clock(
    clock: HostClock,
    origin: HostInstant,
) -> Clock {
    let (numer, denom) = clock.timebase();
    Clock {
        source: "mach_absolute_time scaled to nanoseconds by mach_timebase_info, the host time \
                 base of Metal command-buffer timestamps"
            .to_owned(),
        timebase_numer: numer,
        timebase_denom: denom,
        origin_host_ns: origin.nanos(),
        gpu_conversion: "MTLCommandBuffer GPUStartTime and GPUEndTime are host-time seconds; they \
                         are rounded to whole nanoseconds (seconds x 1e9)"
            .to_owned(),
        trace_unit: "trace ts and dur are microseconds since origin_host_ns".to_owned(),
    }
}

pub(crate) fn trace_info(
    file: String,
    host: HostClock,
    origin: HostInstant,
) -> TraceInfo {
    TraceInfo {
        file,
        format: "Chrome Trace Event JSON (Perfetto compatible)".to_owned(),
        clock: clock(host, origin),
        tracks: Track::ALL
            .into_iter()
            .map(|track| track.name().to_owned())
            .collect(),
    }
}

fn insert(
    args: &mut BTreeMap<String, Value>,
    key: &str,
    value: Option<impl Into<Value>>,
) {
    if let Some(value) = value {
        args.insert(key.to_owned(), value.into());
    }
}

fn label_args(
    args: &mut BTreeMap<String, Value>,
    label: &CommandLabel,
) {
    insert(
        args,
        "command",
        label.index.and_then(|index| u64::try_from(index).ok()),
    );
    insert(args, "op", label.op.map(Id::index));
    insert(args, "tensor", label.tensor.map(Id::index));
    insert(args, "slot", label.slot.map(|slot| slot.index()));
    insert(args, "stream", label.stream.map(|stream| stream.index()));
    insert(args, "event", label.event.map(|event| event.index()));
    insert(args, "bytes", label.bytes.map(|bytes| bytes.bytes()));
}

fn execution_args(
    run: &str,
    trace: &ExecutionTrace,
) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("run".to_owned(), Value::from(run)),
        ("sequence".to_owned(), Value::from(trace.sequence)),
        (
            "phase".to_owned(),
            Value::from(match trace.phase {
                PhaseDto::Warmup => "warmup",
                PhaseDto::Measured => "measured",
            }),
        ),
        ("round".to_owned(), Value::from(trace.round)),
        ("position".to_owned(), Value::from(trace.position)),
        ("buffers".to_owned(), Value::from(trace.buffers)),
    ])
}

fn execution_events(
    run: &str,
    trace: &ExecutionTrace,
    origin: HostInstant,
) -> Vec<TraceEvent> {
    let base = execution_args(run, trace);
    let timed = trace.timed.map(|(start, end)| {
        span(
            &format!("timed execution: {} buffers", trace.buffers),
            "harness",
            Track::CpuHarness,
            relative(start, origin),
            relative(end, origin),
            base.clone(),
        )
    });
    let cpu = trace.cpu.iter().map(|cpu| {
        let mut args = base.clone();
        label_args(&mut args, &cpu.label);
        args.insert("activity".to_owned(), Value::from(cpu.activity));
        args.insert("succeeded".to_owned(), Value::from(cpu.succeeded));
        span(
            cpu.activity,
            "cpu",
            cpu.track,
            relative(cpu.start, origin),
            relative(cpu.end, origin),
            args,
        )
    });
    let gpu = trace
        .diagnostics
        .iter()
        .flat_map(|diagnostics| diagnostics.gpu.iter())
        .map(|interval| {
            let mut args = base.clone();
            label_args(&mut args, &interval.label);
            args.insert("work".to_owned(), Value::from(interval.work.name()));
            insert(&mut args, "kernel", interval.work.kernel());
            args.insert("granularity".to_owned(), Value::from("command_buffer"));
            args.insert("completed".to_owned(), Value::from(interval.completed));
            args.insert(
                "submitted_us".to_owned(),
                Value::from(crate::bench::trace::micros(relative(
                    interval.submitted,
                    origin,
                ))),
            );
            let track = match interval.work {
                GpuWork::Copy => Track::GpuCopy,
                GpuWork::Compute => Track::GpuCompute,
            };
            let name = interval.work.kernel().unwrap_or("copy");
            match interval.timing {
                Timing::Valid {
                    start,
                    end,
                } => span(
                    name,
                    "gpu",
                    track,
                    relative(start, origin),
                    relative(end, origin),
                    args,
                ),
                Timing::Invalid(fault) => {
                    args.insert("unavailable".to_owned(), Value::from(fault.reason()));
                    instant(
                        &format!("{name}: GPU timing unavailable"),
                        "gpu",
                        track,
                        relative(interval.submitted, origin),
                        args,
                    )
                }
            }
        });
    let memory = trace.diagnostics.iter().flat_map(|diagnostics| {
        let mut live: BTreeMap<AllocationCategory, u64> = AllocationCategory::ALL
            .into_iter()
            .map(|category| (category, 0))
            .collect();
        diagnostics.allocations.iter().flat_map(move |allocation| {
            let at = relative(allocation.at, origin);
            let begin = allocation.change == AllocationChange::Allocated;
            if let Some(bytes) = live.get_mut(&allocation.category) {
                *bytes = if begin {
                    bytes.saturating_add(allocation.allocated)
                } else {
                    bytes.saturating_sub(allocation.allocated)
                };
            }
            let mut args = BTreeMap::from([
                (
                    "requested_bytes".to_owned(),
                    Value::from(allocation.requested),
                ),
                (
                    "allocated_bytes".to_owned(),
                    Value::from(allocation.allocated),
                ),
            ]);
            label_args(&mut args, &allocation.label);
            let counters = live
                .iter()
                .map(|(category, bytes)| (category.name().to_owned(), Value::from(*bytes)))
                .collect();
            [
                lifetime(
                    allocation.category.name(),
                    begin,
                    format!("{run}:{}:{}", trace.sequence, allocation.id),
                    at,
                    args,
                ),
                counter("tracked Metal buffers (allocated bytes)", at, counters),
                counter(
                    "MTLDevice.currentAllocatedSize (sampled bytes)",
                    at,
                    BTreeMap::from([("bytes".to_owned(), Value::from(allocation.device_current))]),
                ),
            ]
        })
    });
    timed
        .into_iter()
        .chain(cpu)
        .chain(gpu)
        .chain(memory)
        .collect()
}

pub(crate) fn trace_document(
    run: &str,
    host: HostClock,
    origin: HostInstant,
    traces: &[ExecutionTrace],
) -> Result<TraceDocument, serde_json::Error> {
    let mut events: Vec<TraceEvent> = metadata()
        .into_iter()
        .chain(
            traces
                .iter()
                .flat_map(|trace| execution_events(run, trace, origin)),
        )
        .collect();
    let lanes: Vec<TraceEvent> = [Track::GpuCopy, Track::GpuCompute]
        .into_iter()
        .flat_map(|track| separate_overlaps(&mut events, track))
        .collect();
    events.extend(lanes);
    Ok(TraceDocument {
        trace_events: events,
        display_time_unit: "ns".to_owned(),
        other_data: BTreeMap::from([
            ("run".to_owned(), Value::from(run)),
            (
                "clock".to_owned(),
                serde_json::to_value(clock(host, origin))?,
            ),
        ]),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use foundry_infer_core::{
        ByteSize,
        Id,
        OpId,
        TensorId,
    };
    use foundry_infer_metal::{
        AllocationCategory,
        AllocationChange,
        AllocationEvent,
        CommandLabel,
        GpuInterval,
        GpuTiming as Timing,
        GpuWork,
        HostClock,
        HostInstant,
        MetalDiagnostics,
        TimestampFault,
        Usage,
    };
    use foundry_infer_plan::{
        BufferSlot,
        EventId,
        StreamId,
    };

    use super::{
        CpuSpan,
        ExecutionTrace,
        gpu_timing,
        memory,
        trace_document,
    };
    use crate::bench::report::PhaseDto;
    use crate::bench::trace::Track;

    const ORIGIN: u64 = 1_000_000;

    fn at(offset: u64) -> HostInstant { HostInstant::from_nanos(ORIGIN.saturating_add(offset)) }

    fn label() -> CommandLabel {
        CommandLabel {
            index: Some(4),
            op: Some(OpId::from_index(2)),
            tensor: Some(TensorId::from_index(5)),
            slot: Some(BufferSlot::new(1)),
            stream: Some(StreamId::new(0)),
            event: Some(EventId::new(3)),
            bytes: Some(ByteSize::from_bytes(4096)),
        }
    }

    fn diagnostics(timing: Timing) -> MetalDiagnostics {
        let usage = Usage {
            allocations: 1,
            frees: 1,
            live_requested: 0,
            live_allocated: 0,
            peak_requested: 8192,
            peak_allocated: 16384,
        };
        let allocation = |change, offset| AllocationEvent {
            id: 0,
            category: AllocationCategory::PrivateSlot,
            change,
            at: at(offset),
            requested: 8192,
            allocated: 16384,
            label: label(),
            device_current: 16384,
        };
        MetalDiagnostics {
            gpu: vec![
                GpuInterval {
                    label: label(),
                    work: GpuWork::Copy,
                    submitted: at(1_000),
                    completed: true,
                    timing,
                },
                GpuInterval {
                    label: label(),
                    work: GpuWork::Compute,
                    submitted: at(3_000),
                    completed: true,
                    timing: Timing::Valid {
                        start: at(4_000),
                        end: at(6_000),
                    },
                },
                GpuInterval {
                    label: label(),
                    work: GpuWork::Compute,
                    submitted: at(3_500),
                    completed: true,
                    timing: Timing::Valid {
                        start: at(5_000),
                        end: at(6_500),
                    },
                },
            ],
            gpu_truncated: 0,
            gpu_pending: 0,
            allocations: vec![
                allocation(AllocationChange::Allocated, 500),
                allocation(AllocationChange::Freed, 7_000),
            ],
            allocations_truncated: 0,
            categories: BTreeMap::from([(AllocationCategory::PrivateSlot, usage)]),
            tracked: usage,
            snapshots: Vec::new(),
            live_allocations: 0,
            device_sampled_max: 16384,
            device_samples: 2,
            storage_bytes: 512,
        }
    }

    fn execution(timing: Timing) -> ExecutionTrace {
        ExecutionTrace {
            sequence: 7,
            phase: PhaseDto::Measured,
            round: 2,
            position: 1,
            buffers: 2,
            planned_peak: 4096,
            timed: Some((at(0), at(8_000))),
            cpu: vec![CpuSpan {
                activity: "submit_copy",
                track: Track::CpuRuntime,
                label: label(),
                start: at(800),
                end: at(1_200),
                succeeded: true,
            }],
            cpu_truncated: 0,
            cpu_storage_bytes: 64,
            outcome: None,
            diagnostics: Some(diagnostics(timing)),
        }
    }

    #[test]
    fn traces_separate_cpu_and_gpu_tracks_on_one_clock() -> Result<(), Box<dyn std::error::Error>> {
        let trace = execution(Timing::Valid {
            start: at(1_500),
            end: at(2_500),
        });
        let document = trace_document("run-1", HostClock::new()?, at(0), &[trace])?;
        let named = |name: &str| {
            document
                .trace_events
                .iter()
                .find(|event| event.name == name && event.ph == "X")
                .cloned()
        };
        let submit = named("submit_copy").ok_or("no CPU submission")?;
        let copy = named("copy").ok_or("no GPU copy")?;
        let compute = named("checksum").ok_or("no GPU compute")?;
        assert_eq!(submit.tid, Track::CpuRuntime.id(), "CPU submission track");
        assert_eq!(copy.tid, Track::GpuCopy.id(), "GPU copy track");
        assert_eq!(compute.tid, Track::GpuCompute.id(), "GPU compute track");
        assert!(
            (submit.ts - 0.8).abs() < 1e-9 && (submit.dur.unwrap_or_default() - 0.4).abs() < 1e-9,
            "CPU spans convert nanoseconds since origin to microseconds"
        );
        assert!(
            (copy.ts - 1.5).abs() < 1e-9 && (copy.dur.unwrap_or_default() - 1.0).abs() < 1e-9,
            "GPU spans share the host clock"
        );
        for key in [
            "run", "sequence", "round", "buffers", "command", "op", "tensor", "slot", "stream",
            "event", "bytes",
        ] {
            assert!(copy.args.contains_key(key), "GPU spans carry {key}");
        }
        assert_eq!(
            compute
                .args
                .get("kernel")
                .and_then(|kernel| kernel.as_str()),
            Some("checksum"),
            "compute spans name their kernel"
        );
        assert_eq!(
            copy.args
                .get("granularity")
                .and_then(|granularity| granularity.as_str()),
            Some("command_buffer"),
            "GPU spans keep command-buffer granularity"
        );
        assert!(
            !copy.args.contains_key("kernel"),
            "copies have no kernel field"
        );
        let phases: Vec<&str> = document
            .trace_events
            .iter()
            .filter(|event| event.tid == Track::Memory.id())
            .map(|event| event.ph.as_str())
            .collect();
        assert!(
            phases.contains(&"b") && phases.contains(&"e") && phases.contains(&"C"),
            "allocations have lifetimes and counters"
        );
        let json = serde_json::to_value(&document)?;
        assert!(
            json.get("traceEvents")
                .and_then(|events| events.as_array())
                .is_some_and(|events| !events.is_empty()),
            "the document uses the Chrome traceEvents key"
        );
        Ok(())
    }

    #[test]
    fn invalid_gpu_timestamps_are_explicit_and_fail_the_gate()
    -> Result<(), Box<dyn std::error::Error>> {
        let valid = execution(Timing::Valid {
            start: at(1_500),
            end: at(2_500),
        });
        let invalid = execution(Timing::Invalid(TimestampFault::StartUnavailable));
        let summary = gpu_timing(std::slice::from_ref(&valid));
        assert_eq!(summary.gate, "passed", "valid timestamps pass");
        assert_eq!(
            summary
                .executions
                .first()
                .map(|execution| execution.overlapping_command_buffers),
            Some(1),
            "overlapping command buffers on one queue are counted"
        );
        let summary = gpu_timing(std::slice::from_ref(&invalid));
        assert_eq!(
            summary.gate, "failed",
            "an invalid timestamp fails the gate"
        );
        assert_eq!((summary.valid, summary.invalid), (2, 1), "both are counted");
        let document = trace_document("run-2", HostClock::new()?, at(0), &[invalid])?;
        assert!(
            document
                .trace_events
                .iter()
                .any(|event| event.ph == "i" && event.args.contains_key("unavailable")),
            "the missing interval is marked, never replaced by CPU time"
        );
        assert!(
            !document
                .trace_events
                .iter()
                .any(|event| event.name == "copy" && event.ph == "X"),
            "no GPU span is invented"
        );
        Ok(())
    }

    #[test]
    fn memory_separates_plan_and_observation() {
        let trace = execution(Timing::Invalid(TimestampFault::NotFinished));
        let report = memory(&[trace], &[(2, 4096)], 1 << 30, 12288, at(0));
        let execution = report.executions.first();
        assert_eq!(
            execution.map(|execution| execution.retained_private_slot_bytes_beyond_plan),
            Some(4096),
            "private slots retained beyond the plan are reported"
        );
        assert_eq!(
            execution.map(|execution| execution.diagnostic_storage_bytes),
            Some(576),
            "Metal and CPU diagnostic storage are counted"
        );
        let configuration = report.configurations.first();
        assert_eq!(
            configuration.map(|configuration| (
                configuration.planned_slot_peak_bytes,
                configuration.planned_peak_within_budget,
                configuration.observed_private_slot_peak_requested_bytes_max
            )),
            Some((4096, true, 8192)),
            "the plan, the budget and the observation are separate"
        );
        assert_eq!(report.cpu_payload_bytes, 1 << 30, "CPU payload bytes");
    }
}
