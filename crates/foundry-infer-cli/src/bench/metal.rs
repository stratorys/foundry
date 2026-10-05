use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::process;
use std::time::{
    Instant,
    SystemTime,
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
use foundry_infer_metal::{
    Boundary,
    DiagnosticLimits,
    HostClock,
    HostInstant,
    MetalBackend,
    reference_checksum,
};
use foundry_infer_plan::{
    ExecutionPlan,
    synthetic_chain_graph,
    synthetic_chain_plan,
};
use foundry_infer_runtime::{
    Backend,
    execute,
    execute_observed,
};

use crate::bench::analysis::{
    protocol,
    statistics,
};
use crate::bench::error::{
    BenchError,
    Phase,
    Round,
};
use crate::bench::instrument::{
    ExecutionTrace,
    TraceObserver,
    cpu_capacity,
    gpu_timing,
    memory,
    trace_document,
    trace_info,
};
use crate::bench::report::{
    Available,
    ChecksumStatus,
    ConfigurationPlan,
    DeviceLimits,
    Execution,
    ExecutionOutcomeDto,
    Failure,
    Foundry,
    Instrumentation,
    Invocation,
    PhaseDto,
    Report,
    Run,
    SCHEMA_VERSION,
    Statistics,
    Status,
    Workload,
    run_id,
    utc_timestamp,
};
use crate::bench::trace::Track;
use crate::bench::workload::{
    BUFFER_COUNTS,
    Sizes,
    aligned_slot,
    check_buffer_limit,
    payloads,
    planned_peak,
    round_order,
    slot_budget,
};
use crate::bench::{
    Settings,
    provenance,
};

struct Configuration {
    buffers: u32,
    plan: ExecutionPlan,
    peak: ByteSize,
    limits: DiagnosticLimits,
    cpu_capacity: usize,
}

struct Prepared {
    graph: Graph,
    payloads: BTreeMap<TensorId, Vec<u8>>,
    expected: Vec<(OpId, u32)>,
    configurations: Vec<Configuration>,
    budget: MemoryBudget,
    workload: Workload,
}

struct Campaign {
    traced: bool,
    clock: HostClock,
    executions: Vec<Execution>,
    traces: Vec<ExecutionTrace>,
    failure: Option<Failure>,
}

impl Campaign {
    fn fail(
        &mut self,
        stage: &str,
        error: &BenchError,
        execution: Option<&Execution>,
    ) {
        self.failure = Some(Failure {
            stage: stage.to_owned(),
            message: error.to_string(),
            phase: execution.map(|execution| execution.phase),
            round: execution.map(|execution| execution.round),
            position: execution.map(|execution| execution.position),
            buffers: execution.map(|execution| execution.buffers),
            completed_executions: u32::try_from(
                self.executions
                    .iter()
                    .filter(|execution| execution.outcome == ExecutionOutcomeDto::Completed)
                    .count(),
            )
            .unwrap_or(u32::MAX),
        });
    }
}

pub(crate) fn run(settings: &Settings) -> Result<(), BenchError> {
    let started = SystemTime::now();
    let build = provenance::build().map_err(BenchError::BuildProvenance)?;
    let repository = provenance::repository(settings.working_directory.as_deref());
    let backend = MetalBackend::new();
    let environment = provenance::environment(
        backend
            .as_ref()
            .map(MetalBackend::device_name)
            .map_err(ToString::to_string),
    );
    let mut backend = backend?;
    let clock = backend.host_clock();
    let origin = clock.now();
    let id = run_id(started, process::id());
    let mut campaign = Campaign {
        traced: settings.outputs.trace.is_some(),
        clock,
        executions: Vec::new(),
        traces: Vec::new(),
        failure: None,
    };
    let prepared = prepare(&backend, &settings.sizes, settings);
    let workload = prepared
        .as_ref()
        .map(|prepared| prepared.workload.clone())
        .map_err(ToString::to_string);
    let measured = match prepared {
        Ok(prepared) => {
            header(&backend, &prepared, settings);
            measure(&mut backend, &prepared, settings, &mut campaign)
        }
        Err(error) => {
            campaign.fail("preparation", &error, None);
            Err(error)
        }
    };
    let summary = measured
        .as_ref()
        .map_err(ToString::to_string)
        .and_then(|()| {
            statistics(
                &campaign.executions,
                settings.sizes.total,
                settings.resamples,
                settings.seed,
            )
        });
    let timing = campaign.traced.then(|| gpu_timing(&campaign.traces));
    let gate = match (&measured, &timing) {
        (Ok(()), Some(timing)) if timing.gate != "passed" => {
            let error = BenchError::GpuTimingGate {
                invalid: timing
                    .invalid
                    .saturating_add(timing.truncated)
                    .saturating_add(timing.pending_at_extraction),
            };
            campaign.fail("gpu_timing_gate", &error, None);
            Err(error)
        }
        (Ok(()), _) => summary.as_ref().map(|_| ()).map_err(|reason| {
            let error = BenchError::Statistics(reason.clone());
            campaign.fail("statistics", &error, None);
            error
        }),
        (Err(_), _) => Ok(()),
    };
    if let Ok(statistics) = &summary {
        print_statistics(statistics, campaign.traced);
    }
    let outcome = measured.and(gate);
    let status = if outcome.is_ok() {
        Status::Succeeded
    } else {
        Status::Failed
    };
    let trace_file = settings.outputs.trace.as_ref();
    let report = Report {
        schema_version: SCHEMA_VERSION,
        run: Run {
            id: id.clone(),
            started_at_utc: utc_timestamp(started).unwrap_or_default(),
            status,
            instrumentation: if campaign.traced {
                Instrumentation::Traced
            } else {
                Instrumentation::Untraced
            },
        },
        foundry: Foundry {
            version: env!("CARGO_PKG_VERSION").to_owned(),
            build,
        },
        runtime_repository: repository,
        invocation: Invocation {
            arguments: settings.arguments.clone(),
            working_directory: Available::from_option(
                settings
                    .working_directory
                    .as_ref()
                    .map(|directory| directory.display().to_string()),
                "the working directory is unavailable",
            ),
            parameters: settings.parameters.clone(),
        },
        environment,
        workload: Available::from_result(workload.clone()),
        protocol: protocol(settings.warmup, settings.iterations),
        executions: campaign.executions.clone(),
        statistics: match (&summary, &status) {
            (Ok(statistics), _) if campaign.failure.is_none() => {
                Available::known(statistics.clone())
            }
            (Ok(_), _) | (Err(_), _) => Available::unknown(
                "the campaign did not complete successfully; no comparison is summarized",
            ),
        },
        memory: if campaign.traced {
            workload.map_or_else(Available::unknown, |workload| {
                let plans: Vec<(u32, u64)> = workload
                    .configurations
                    .iter()
                    .map(|configuration| {
                        (configuration.buffers, configuration.planned_slot_peak_bytes)
                    })
                    .collect();
                Available::known(memory(
                    &campaign.traces,
                    &plans,
                    workload.payload_bytes,
                    workload.slot_budget_bytes,
                    origin,
                ))
            })
        } else {
            Available::unknown("allocation accounting is enabled only with --trace")
        },
        gpu_timing: timing.map_or_else(
            || Available::unknown("GPU timestamps are collected only with --trace"),
            Available::known,
        ),
        trace: trace_file.map_or_else(
            || Available::unknown("no trace was requested"),
            |trace| Available::known(trace_info(trace.file_name(), clock, origin)),
        ),
        limitations: limitations(campaign.traced),
        failure: campaign.failure.clone(),
    };
    export(settings, &report, &id, clock, origin, &campaign, outcome)
}

fn export(
    settings: &Settings,
    report: &Report,
    id: &str,
    clock: HostClock,
    origin: HostInstant,
    campaign: &Campaign,
    outcome: Result<(), BenchError>,
) -> Result<(), BenchError> {
    let written = settings
        .outputs
        .trace
        .as_ref()
        .map_or(Ok(()), |destination| {
            let document =
                trace_document(id, clock, origin, &campaign.traces).map_err(|source| {
                    BenchError::OutputSerialize {
                        path: destination.requested().to_path_buf(),
                        source,
                    }
                })?;
            destination.write(&document, false)
        })
        .and_then(|()| {
            settings
                .outputs
                .report
                .as_ref()
                .map_or(Ok(()), |destination| destination.write(report, true))
        });
    match (outcome, written) {
        (Ok(()), written) => written,
        (Err(source), Ok(())) => match &settings.outputs.report {
            Some(destination) => Err(BenchError::Reported {
                source: Box::new(source),
                report: destination.requested().to_path_buf(),
            }),
            None => Err(source),
        },
        (Err(source), Err(export)) => Err(BenchError::ExportAfterFailure {
            source: Box::new(source),
            export: Box::new(export),
        }),
    }
}

fn limitations(traced: bool) -> Vec<String> {
    let common = [
        "the workload computes checksums over transferred weights; it is not inference",
        "the Metal backend ignores duration hints and does not claim concurrent copy and compute",
        "throughput is end-to-end logical payload bytes over the median elapsed time, not raw \
         transfer bandwidth",
        "these results do not establish transfer/compute overlap or inference performance",
        "independent invocations are not pooled; compare campaigns explicitly",
    ];
    let traced_only = [
        "tracing adds host clock reads, diagnostic records and allocation accounting inside the \
         timed region; use untraced runs for timing comparisons",
        "GPU intervals cover whole command buffers, including encoded event waits, not isolated \
         kernels",
        "MTLDevice.currentAllocatedSize is sampled at allocation transitions and boundaries; its \
         maximum is not an exact process peak",
    ];
    let untraced_only = ["GPU timestamps and allocation accounting require --trace"];
    let specific: &[&str] = if traced { &traced_only } else { &untraced_only };
    common
        .iter()
        .chain(specific)
        .map(|limitation| (*limitation).to_owned())
        .collect()
}

fn prepare(
    backend: &MetalBackend,
    sizes: &Sizes,
    settings: &Settings,
) -> Result<Prepared, BenchError> {
    if u32::try_from(sizes.weight.bytes()).is_err() {
        return Err(BenchError::WeightTooLargeForKernel {
            weight: sizes.weight,
        });
    }
    let alignment = backend.capabilities().alignment;
    let slot = aligned_slot(sizes.weight, alignment)?;
    check_buffer_limit(slot, backend.buffer_length_max())?;
    let budget = slot_budget(sizes.weight, alignment)?;
    let working_set = backend.working_set_recommended();
    if budget > working_set {
        return Err(BenchError::BudgetExceedsWorkingSet {
            budget,
            working_set,
            device: backend.device_name(),
        });
    }
    let budget = MemoryBudget::new(MemorySpace::Device, budget);
    let graph = synthetic_chain_graph(sizes.layers, sizes.weight, None)?;
    let mut configurations = Vec::new();
    for buffers in BUFFER_COUNTS {
        let slots = NonZeroU32::new(buffers).ok_or(BenchError::InvalidBufferCount {
            buffers,
        })?;
        let plan = synthetic_chain_plan(&graph, slots, alignment)?;
        plan.validate(&graph, budget)?;
        let peak = planned_peak(&plan)?;
        let limits = DiagnosticLimits::for_plan(&plan)?;
        let cpu_capacity = cpu_capacity(&plan).ok_or(BenchError::DiagnosticBudget {
            what: "CPU spans",
        })?;
        configurations.push(Configuration {
            buffers,
            plan,
            peak,
            limits,
            cpu_capacity,
        });
    }
    if settings.outputs.trace.is_some() {
        check_diagnostic_budget(&configurations, settings)?;
    }
    let workload = Workload {
        layers: sizes.layers,
        weight_bytes: sizes.weight.bytes(),
        payload_bytes: sizes.total.bytes(),
        alignment_bytes: alignment.bytes(),
        slot_bytes: slot.bytes(),
        slot_budget_bytes: budget.capacity().bytes(),
        configurations: configurations
            .iter()
            .map(|configuration| ConfigurationPlan {
                buffers: configuration.buffers,
                commands: u64::try_from(configuration.plan.commands().len()).unwrap_or(u64::MAX),
                planned_slot_peak_bytes: configuration.peak.bytes(),
            })
            .collect(),
        device: DeviceLimits {
            recommended_working_set_bytes: working_set.bytes(),
            max_buffer_length_bytes: backend.buffer_length_max().bytes(),
            unified_memory: backend.capabilities().unified_memory,
        },
    };
    let payloads = payloads(&graph, sizes.weight_len)?;
    let expected = expected(&graph, &payloads)?;
    Ok(Prepared {
        graph,
        payloads,
        expected,
        configurations,
        budget,
        workload,
    })
}

fn check_diagnostic_budget(
    configurations: &[Configuration],
    settings: &Settings,
) -> Result<(), BenchError> {
    let budget = || BenchError::DiagnosticBudget {
        what: "the traced campaign",
    };
    let rounds = usize::try_from(
        settings
            .warmup
            .checked_add(settings.iterations.get())
            .ok_or_else(budget)?,
    )
    .map_err(|_| budget())?;
    configurations
        .iter()
        .try_fold(0_usize, |total, configuration| {
            let records = configuration
                .limits
                .gpu_intervals()
                .checked_add(configuration.limits.allocation_events())?
                .checked_add(configuration.cpu_capacity)?
                .checked_mul(rounds)?;
            total.checked_add(records)
        })
        .ok_or_else(budget)
        .map(drop)
}

fn expected(
    graph: &Graph,
    payloads: &BTreeMap<TensorId, Vec<u8>>,
) -> Result<Vec<(OpId, u32)>, BenchError> {
    graph
        .ops()
        .map(|(op, desc)| match desc {
            Op::SyntheticCompute {
                inputs, ..
            } => {
                let tensor = inputs.get(1).ok_or(BenchError::MissingWeight {
                    op,
                })?;
                let payload = payloads.get(tensor).ok_or(BenchError::MissingPayload {
                    tensor: *tensor,
                })?;
                Ok((op, reference_checksum(payload)))
            }
            Op::MatMul {
                ..
            } => Err(BenchError::UnsupportedOp {
                op,
            }),
        })
        .collect()
}

fn header(
    backend: &MetalBackend,
    prepared: &Prepared,
    settings: &Settings,
) {
    let sizes = &settings.sizes;
    println!("Foundry synthetic streaming benchmark");
    println!("GPU: {}", backend.device_name());
    println!(
        "Workload: {} layers x {} weights, checksum compute, no duration hint",
        sizes.layers, sizes.weight
    );
    println!(
        "Host payload: {} ({} bytes), generated in memory",
        sizes.total,
        sizes.total.bytes()
    );
    println!(
        "Device slot budget: {} for three aligned weight slots, not total application memory",
        prepared.budget.capacity()
    );
    println!(
        "Runs per buffer count: {} warmup, {} measured, order rotated every round",
        settings.warmup, settings.iterations
    );
    println!(
        "Timed: from the runtime interpreter call until all submitted GPU work has drained, \
         including runtime validation, allocations, host staging, transfers, checksum kernels, \
         synchronization, and runtime cleanup (slot release and a final drain of submitted GPU \
         work)"
    );
    println!(
        "Untimed: graph and plan construction, payload generation, backend setup, expected \
         checksums, result verification, release of checksum results via clear_checksums(), and \
         output"
    );
    println!(
        "Statistics: 95% percentile bootstrap, {} resamples, SplitMix64 seed {}",
        settings.resamples, settings.seed
    );
    if settings.outputs.trace.is_some() {
        println!(
            "Instrumentation: traced; CPU spans, GPU command-buffer timestamps and allocation \
             accounting are recorded, so use untraced runs for timing comparisons"
        );
    }
    println!();
}

struct Failed {
    stage: &'static str,
    error: Box<BenchError>,
}

impl Failed {
    fn new(
        stage: &'static str,
        error: BenchError,
    ) -> Self {
        Self {
            stage,
            error: Box::new(error),
        }
    }
}

type Timed = (Result<u64, Failed>, (HostInstant, HostInstant));

struct Step<'step> {
    prepared: &'step Prepared,
    configuration: &'step Configuration,
    weights: &'step BTreeMap<TensorId, &'step [u8]>,
    round: Round,
}

fn timed_untraced(
    backend: &mut MetalBackend,
    step: &Step<'_>,
) -> Result<u64, Failed> {
    let start = Instant::now();
    execute(
        backend,
        &step.prepared.graph,
        &step.configuration.plan,
        step.prepared.budget,
        step.weights,
    )
    .map_err(|source| {
        Failed::new(
            "execution",
            BenchError::Execution {
                round: step.round,
                source,
            },
        )
    })?;
    backend.drain().map_err(|source| {
        Failed::new(
            "drain",
            BenchError::Drain {
                round: step.round,
                source,
            },
        )
    })?;
    Ok(nanos(start))
}

fn nanos(start: Instant) -> u64 { u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX) }

fn timed_traced(
    backend: &mut MetalBackend,
    step: &Step<'_>,
    observer: &mut TraceObserver,
    clock: HostClock,
) -> Timed {
    let host_start = clock.now();
    let start = Instant::now();
    let executed = execute_observed(
        backend,
        &step.prepared.graph,
        &step.configuration.plan,
        step.prepared.budget,
        step.weights,
        observer,
    )
    .map_err(|source| {
        Failed::new(
            "execution",
            BenchError::Execution {
                round: step.round,
                source,
            },
        )
    });
    let drained = executed.and_then(|()| {
        let drain_start = clock.now();
        let drained = backend.drain();
        observer.harness("final_drain", Track::CpuWaits, drain_start, drained.is_ok());
        drained.map_err(|source| {
            Failed::new(
                "drain",
                BenchError::Drain {
                    round: step.round,
                    source,
                },
            )
        })
    });
    let elapsed = nanos(start);
    let host_end = clock.now();
    (drained.map(|()| elapsed), (host_start, host_end))
}

fn measure(
    backend: &mut MetalBackend,
    prepared: &Prepared,
    settings: &Settings,
    campaign: &mut Campaign,
) -> Result<(), BenchError> {
    let weights: BTreeMap<TensorId, &[u8]> = prepared
        .payloads
        .iter()
        .map(|(&tensor, payload)| (tensor, payload.as_slice()))
        .collect();
    let rounds = settings
        .warmup
        .checked_add(settings.iterations.get())
        .ok_or(BenchError::RoundsOverflow {
            warmup: settings.warmup,
            iterations: settings.iterations,
        })?;
    let executions = usize::try_from(rounds)
        .ok()
        .and_then(|rounds| rounds.checked_mul(BUFFER_COUNTS.len()))
        .ok_or(BenchError::RoundsOverflow {
            warmup: settings.warmup,
            iterations: settings.iterations,
        })?;
    campaign.executions.reserve_exact(executions);
    if campaign.traced {
        campaign.traces.reserve_exact(executions);
    }
    for round in 0..rounds {
        let (phase, phase_dto) = if round < settings.warmup {
            (Phase::Warmup, PhaseDto::Warmup)
        } else {
            (Phase::Measured, PhaseDto::Measured)
        };
        for (position, buffers) in (0_u32..).zip(round_order(round)) {
            let configuration = prepared
                .configurations
                .iter()
                .find(|configuration| configuration.buffers == buffers)
                .ok_or(BenchError::MissingConfiguration {
                    buffers,
                })?;
            let step = Step {
                prepared,
                configuration,
                weights: &weights,
                round: Round {
                    phase,
                    index: round,
                    buffers,
                },
            };
            let record = Execution {
                sequence: u32::try_from(campaign.executions.len()).unwrap_or(u32::MAX),
                phase: phase_dto,
                round,
                position,
                buffers,
                elapsed_ns: None,
                outcome: ExecutionOutcomeDto::Failed,
                checksums: ChecksumStatus::NotVerified,
            };
            let (result, record) = if campaign.traced {
                traced_step(backend, &step, record, campaign)
            } else {
                untraced_step(backend, &step, record)
            };
            campaign.executions.push(record);
            if let Err(failed) = result {
                campaign.fail(failed.stage, &failed.error, Some(&record));
                return Err(*failed.error);
            }
        }
    }
    Ok(())
}

fn finish(
    backend: &mut MetalBackend,
    step: &Step<'_>,
    mut record: Execution,
    timed: Result<u64, Failed>,
) -> (Result<(), Failed>, Execution) {
    let elapsed = match timed {
        Ok(elapsed) => elapsed,
        Err(error) => return (Err(error), record),
    };
    record.elapsed_ns = Some(elapsed);
    let verified = verify(backend, step.prepared, step.round);
    record.checksums = match &verified {
        Ok(()) => ChecksumStatus::Verified,
        Err(failed) if failed.stage == "checksum_mismatch" => ChecksumStatus::Mismatch,
        Err(_) => ChecksumStatus::NotVerified,
    };
    if verified.is_ok() {
        record.outcome = ExecutionOutcomeDto::Completed;
    }
    (verified, record)
}

fn untraced_step(
    backend: &mut MetalBackend,
    step: &Step<'_>,
    record: Execution,
) -> (Result<(), Failed>, Execution) {
    let timed = timed_untraced(backend, step);
    let finished = finish(backend, step, record, timed);
    backend.clear_checksums();
    finished
}

fn traced_step(
    backend: &mut MetalBackend,
    step: &Step<'_>,
    record: Execution,
    campaign: &mut Campaign,
) -> (Result<(), Failed>, Execution) {
    let clock = campaign.clock;
    let setup = backend
        .begin_diagnostics(step.configuration.limits)
        .map_err(BenchError::from)
        .and_then(|()| TraceObserver::new(clock, step.configuration.cpu_capacity));
    let mut observer = match setup {
        Ok(observer) => observer,
        Err(error) => {
            drop(backend.take_diagnostics());
            return (Err(Failed::new("diagnostics", error)), record);
        }
    };
    let (timed, bounds) = timed_traced(backend, step, &mut observer, clock);
    backend.snapshot(Boundary::AfterDrain);
    let verify_start = clock.now();
    let (result, record) = finish(backend, step, record, timed);
    observer.harness(
        "verify_checksums",
        Track::CpuHarness,
        verify_start,
        result.is_ok(),
    );
    let clear_start = clock.now();
    backend.clear_checksums();
    observer.harness("clear_checksums", Track::CpuHarness, clear_start, true);
    backend.snapshot(Boundary::AfterChecksumRelease);
    let diagnostics = backend.take_diagnostics();
    let trace = observer.finish_trace(ExecutionTrace {
        sequence: record.sequence,
        phase: record.phase,
        round: record.round,
        position: record.position,
        buffers: record.buffers,
        planned_peak: step.configuration.peak.bytes(),
        timed: Some(bounds),
        cpu: Vec::new(),
        cpu_truncated: 0,
        cpu_storage_bytes: 0,
        outcome: None,
        diagnostics,
    });
    campaign.traces.push(trace);
    (result, record)
}

fn verify(
    backend: &MetalBackend,
    prepared: &Prepared,
    round: Round,
) -> Result<(), Failed> {
    prepared.expected.iter().try_for_each(|&(op, expected)| {
        let actual = backend.checksums(op).map_err(|source| {
            Failed::new(
                "checksum_read",
                BenchError::ChecksumRead {
                    round,
                    op,
                    source,
                },
            )
        })?;
        if actual == [expected] {
            Ok(())
        } else {
            Err(Failed::new(
                "checksum_mismatch",
                BenchError::ChecksumMismatch {
                    round,
                    op,
                    expected,
                    actual,
                },
            ))
        }
    })
}

fn milliseconds(nanos: f64) -> f64 { nanos / 1e6 }

fn print_statistics(
    statistics: &Statistics,
    traced: bool,
) {
    println!(
        "{:>7}  {:>10}  {:>10}  {:>9}  {:>10}  {:>10}  {:>21}  {:>21}  {:>8}  {:>8}  {:>17}",
        "buffers",
        "median ms",
        "mean ms",
        "sd ms",
        "min ms",
        "max ms",
        "median 95% CI ms",
        "mean 95% CI ms",
        "GiB/s",
        "speedup",
        "speedup 95% CI"
    );
    let number = |value: Option<f64>, scale: fn(f64) -> f64| {
        value.map_or_else(|| "n/a".to_owned(), |value| format!("{:.2}", scale(value)))
    };
    let range = |value: Option<crate::bench::report::IntervalDto>, scale: fn(f64) -> f64| {
        value.map_or_else(
            || "n/a".to_owned(),
            |interval| {
                format!(
                    "[{:.2}, {:.2}]",
                    scale(interval.lower),
                    scale(interval.upper)
                )
            },
        )
    };
    let identity = |value: f64| value;
    for row in &statistics.configurations {
        println!(
            "{:>7}  {:>10.2}  {:>10}  {:>9}  {:>10.2}  {:>10.2}  {:>21}  {:>21}  {:>8}  {:>8}  \
             {:>17}",
            row.buffers,
            milliseconds(crate::bench::stats::lossy(u128::from(row.median_ns))),
            number(row.mean_ns.value, milliseconds),
            number(row.standard_deviation_ns.value, milliseconds),
            milliseconds(crate::bench::stats::lossy(u128::from(row.min_ns))),
            milliseconds(crate::bench::stats::lossy(u128::from(row.max_ns))),
            range(row.median_ci_ns.value, milliseconds),
            range(row.mean_ci_ns.value, milliseconds),
            number(row.throughput_gib_s.value, identity),
            number(row.speedup.value, identity),
            range(row.speedup_ci.value, identity),
        );
    }
    println!();
    println!(
        "GiB/s is end-to-end effective payload throughput: logical weight bytes divided by the \
         median elapsed time, with 1 GiB = 2^30 bytes."
    );
    println!(
        "Speedup is the one-buffer median divided by the configuration median; its interval \
         resamples complete measured rounds in pairs."
    );
    println!(
        "The workload computes checksums over the transferred weights, and the Metal backend \
         ignores duration hints."
    );
    println!("These results do not establish transfer/compute overlap or inference performance.");
    if traced {
        println!("Traced timings include instrumentation overhead.");
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::error::Error;
    use std::num::{
        NonZeroU32,
        NonZeroU64,
    };

    use foundry_infer_core::ByteSize;
    use foundry_infer_metal::{
        MetalBackend,
        MetalError,
    };

    use super::{
        Campaign,
        measure,
        prepare,
    };
    use crate::bench::Settings;
    use crate::bench::analysis::measured;
    use crate::bench::report::{
        ChecksumStatus,
        ExecutionOutcomeDto,
        PhaseDto,
    };
    use crate::bench::workload::{
        Sizes,
        round_order,
    };

    type TestResult = Result<(), Box<dyn Error>>;

    fn gpu() -> Result<Option<MetalBackend>, Box<dyn Error>> {
        match MetalBackend::new() {
            Ok(backend) => {
                eprintln!("metal: running on {}", backend.device_name());
                Ok(Some(backend))
            }
            Err(
                error @ (MetalError::DeviceUnavailable
                | MetalError::UnsupportedDevice {
                    ..
                }),
            ) if env::var_os("FOUNDRY_REQUIRE_METAL").is_none() => {
                eprintln!("metal: SKIPPED, GPU behavior not verified: {error}");
                Ok(None)
            }
            Err(error) => Err(error.into()),
        }
    }

    fn settings(
        warmup: u32,
        iterations: u32,
    ) -> Result<Settings, Box<dyn Error>> {
        Ok(Settings::for_tests(
            Sizes::new(
                NonZeroU32::new(4).ok_or("zero layers")?,
                NonZeroU64::new(1).ok_or("zero weight")?,
            )?,
            warmup,
            NonZeroU32::new(iterations).ok_or("zero iterations")?,
        ))
    }

    fn campaign(
        backend: &MetalBackend,
        traced: bool,
    ) -> Campaign {
        Campaign {
            traced,
            clock: backend.host_clock(),
            executions: Vec::new(),
            traces: Vec::new(),
            failure: None,
        }
    }

    #[test]
    fn measures_every_buffer_configuration_in_rotated_order() -> TestResult {
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        let settings = settings(1, 2)?;
        let prepared = prepare(&backend, &settings.sizes, &settings)?;
        let mut campaign = campaign(&backend, false);
        measure(&mut backend, &prepared, &settings, &mut campaign)?;
        let order: Vec<(u32, u32, u32)> = campaign
            .executions
            .iter()
            .map(|execution| (execution.round, execution.position, execution.buffers))
            .collect();
        let expected: Vec<(u32, u32, u32)> = (0..3)
            .flat_map(|round| {
                (0_u32..)
                    .zip(round_order(round))
                    .map(move |(position, buffers)| (round, position, buffers))
            })
            .collect();
        assert_eq!(order, expected, "executions follow the rotated order");
        assert!(
            campaign
                .executions
                .iter()
                .zip(0_u32..)
                .all(|(execution, sequence)| execution.sequence == sequence),
            "executions are chronological"
        );
        assert_eq!(
            campaign
                .executions
                .iter()
                .filter(|execution| execution.phase == PhaseDto::Warmup)
                .count(),
            3,
            "warmup executions are recorded"
        );
        assert!(
            campaign.executions.iter().all(|execution| {
                execution.outcome == ExecutionOutcomeDto::Completed
                    && execution.checksums == ChecksumStatus::Verified
                    && execution.elapsed_ns.is_some_and(|elapsed| elapsed > 0)
            }),
            "every execution completes with verified checksums"
        );
        let samples = measured(&campaign.executions);
        let buffers: Vec<u32> = samples.iter().map(|(buffers, _)| *buffers).collect();
        assert_eq!(buffers, [1, 2, 3], "every buffer count is measured");
        assert!(
            samples.iter().all(|(_, samples)| samples.len() == 2),
            "warmup executions are excluded"
        );
        for configuration in &prepared.configurations {
            assert_eq!(
                configuration.peak,
                ByteSize::from_mib(u64::from(configuration.buffers))?,
                "{} buffers hold that many 1 MiB slots",
                configuration.buffers
            );
        }
        Ok(())
    }

    #[test]
    fn traced_executions_release_every_tracked_buffer() -> TestResult {
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        let settings = settings(1, 2)?;
        let prepared = prepare(&backend, &settings.sizes, &settings)?;
        let mut campaign = campaign(&backend, true);
        measure(&mut backend, &prepared, &settings, &mut campaign)?;
        assert_eq!(campaign.traces.len(), 9, "every execution is traced");
        for trace in &campaign.traces {
            let diagnostics = trace.diagnostics.as_ref().ok_or("no diagnostics")?;
            assert!(
                diagnostics.gpu_timestamps_valid(),
                "execution {} has valid GPU timestamps",
                trace.sequence
            );
            assert_eq!(
                diagnostics.live_allocations, 0,
                "execution {} releases every tracked buffer",
                trace.sequence
            );
            assert!(
                trace.cpu.iter().any(|span| span.activity == "final_drain")
                    && trace.cpu.iter().any(|span| span.activity == "submit_copy"),
                "execution {} records runtime and harness spans",
                trace.sequence
            );
        }
        assert_eq!(
            super::gpu_timing(&campaign.traces).gate,
            "passed",
            "the GPU timing gate passes"
        );
        Ok(())
    }

    #[test]
    fn checksum_mismatches_abort_the_benchmark() -> TestResult {
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        let settings = settings(0, 1)?;
        let mut prepared = prepare(&backend, &settings.sizes, &settings)?;
        let (_, checksum) = prepared.expected.last_mut().ok_or("no layers")?;
        *checksum = checksum.wrapping_add(1);
        let mut campaign = campaign(&backend, false);
        let error = measure(&mut backend, &prepared, &settings, &mut campaign)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(
            error.contains("layer 3 checksum mismatch"),
            "a wrong checksum fails the run, got {error:?}"
        );
        let last = campaign.executions.last().ok_or("no record")?;
        assert_eq!(
            (last.outcome, last.checksums),
            (ExecutionOutcomeDto::Failed, ChecksumStatus::Mismatch),
            "the failed execution is recorded"
        );
        let failure = campaign.failure.as_ref().ok_or("no failure")?;
        assert_eq!(
            (
                failure.stage.as_str(),
                failure.buffers,
                failure.completed_executions
            ),
            ("checksum_mismatch", Some(1), 0),
            "the failure names its stage and configuration"
        );
        Ok(())
    }
}
