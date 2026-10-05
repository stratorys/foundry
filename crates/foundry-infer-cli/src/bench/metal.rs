use std::collections::BTreeMap;
use std::error::Error;
use std::num::NonZeroU32;
use std::time::{
    Duration,
    Instant,
};

use foundry_infer_core::{
    ByteSize,
    Graph,
    Id,
    MemoryBudget,
    MemorySpace,
    Op,
    OpId,
    TensorId,
};
use foundry_infer_metal::{
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
};

use crate::bench::stats::{
    Summary,
    speedup,
    throughput_gib_s,
};
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

struct Configuration {
    buffers: u32,
    plan: ExecutionPlan,
    peak: ByteSize,
}

struct Prepared {
    graph: Graph,
    payloads: BTreeMap<TensorId, Vec<u8>>,
    expected: Vec<(OpId, u32)>,
    configurations: Vec<Configuration>,
    budget: MemoryBudget,
}

struct Measurement {
    buffers: u32,
    peak: ByteSize,
    samples: Vec<Duration>,
}

pub(crate) fn run(
    sizes: &Sizes,
    warmup: u32,
    iterations: NonZeroU32,
) -> Result<(), Box<dyn Error>> {
    let mut backend = MetalBackend::new()?;
    let prepared = prepare(&backend, sizes)?;
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
        "Runs per buffer count: {warmup} warmup, {iterations} measured, order rotated every round"
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
    println!();
    let measurements = measure(&mut backend, &prepared, warmup, iterations)?;
    report(sizes.total, &measurements)
}

fn prepare(
    backend: &MetalBackend,
    sizes: &Sizes,
) -> Result<Prepared, Box<dyn Error>> {
    if u32::try_from(sizes.weight.bytes()).is_err() {
        return Err(format!(
            "the Metal checksum kernel supports weights below 4 GiB, got {}",
            sizes.weight
        )
        .into());
    }
    let alignment = backend.capabilities().alignment;
    let slot = aligned_slot(sizes.weight, alignment)?;
    check_buffer_limit(slot, backend.max_buffer_length())?;
    let budget = slot_budget(sizes.weight, alignment)?;
    let working_set = backend.recommended_working_set();
    if budget > working_set {
        return Err(format!(
            "the device slot budget of {budget} exceeds the {working_set} recommended working set \
             of {}",
            backend.device_name()
        )
        .into());
    }
    let budget = MemoryBudget::new(MemorySpace::Device, budget);
    let graph = synthetic_chain_graph(sizes.layers, sizes.weight, None)?;
    let mut configurations = Vec::new();
    for buffers in BUFFER_COUNTS {
        let slots = NonZeroU32::new(buffers).ok_or("buffer counts are positive")?;
        let plan = synthetic_chain_plan(&graph, slots, alignment)?;
        plan.validate(&graph, budget)?;
        let peak = planned_peak(&plan)?;
        configurations.push(Configuration {
            buffers,
            plan,
            peak,
        });
    }
    let payloads = payloads(&graph, sizes.weight_len)?;
    let expected = expected(&graph, &payloads)?;
    Ok(Prepared {
        graph,
        payloads,
        expected,
        configurations,
        budget,
    })
}

fn expected(
    graph: &Graph,
    payloads: &BTreeMap<TensorId, Vec<u8>>,
) -> Result<Vec<(OpId, u32)>, Box<dyn Error>> {
    graph
        .ops()
        .map(|(op, desc)| match desc {
            Op::SyntheticCompute {
                inputs, ..
            } => {
                let tensor = inputs.get(1).ok_or("a synthetic layer has no weight")?;
                let payload = payloads
                    .get(tensor)
                    .ok_or("a layer weight has no payload")?;
                Ok((op, reference_checksum(payload)))
            }
            Op::MatMul {
                ..
            } => Err("the synthetic workload has no matmul".into()),
        })
        .collect()
}

fn measure(
    backend: &mut MetalBackend,
    prepared: &Prepared,
    warmup: u32,
    iterations: NonZeroU32,
) -> Result<Vec<Measurement>, Box<dyn Error>> {
    let weights: BTreeMap<TensorId, &[u8]> = prepared
        .payloads
        .iter()
        .map(|(&tensor, payload)| (tensor, payload.as_slice()))
        .collect();
    let rounds = warmup
        .checked_add(iterations.get())
        .ok_or("warmup and measured iterations overflow")?;
    let mut samples: BTreeMap<u32, Vec<Duration>> = BTreeMap::new();
    for round in 0..rounds {
        let phase = if round < warmup { "warmup" } else { "measured" };
        for buffers in round_order(round) {
            let configuration = prepared
                .configurations
                .iter()
                .find(|configuration| configuration.buffers == buffers)
                .ok_or("missing buffer configuration")?;
            let start = Instant::now();
            execute(
                backend,
                &prepared.graph,
                &configuration.plan,
                prepared.budget,
                &weights,
            )
            .map_err(|error| {
                format!("{phase} round {round} with {buffers} buffers failed: {error}")
            })?;
            backend.drain().map_err(|error| {
                format!("{phase} round {round} with {buffers} buffers failed to drain: {error}")
            })?;
            let elapsed = start.elapsed();
            verify(backend, prepared).map_err(|error| {
                format!("{phase} round {round} with {buffers} buffers: {error}")
            })?;
            backend.clear_checksums();
            if round >= warmup {
                samples.entry(buffers).or_default().push(elapsed);
            }
        }
    }
    Ok(prepared
        .configurations
        .iter()
        .map(|configuration| Measurement {
            buffers: configuration.buffers,
            peak: configuration.peak,
            samples: samples.remove(&configuration.buffers).unwrap_or_default(),
        })
        .collect())
}

fn verify(
    backend: &MetalBackend,
    prepared: &Prepared,
) -> Result<(), Box<dyn Error>> {
    for &(op, expected) in &prepared.expected {
        let actual = backend.checksums(op)?;
        if actual != [expected] {
            return Err(format!(
                "layer {} checksum mismatch: expected {expected:#010x}, got {actual:#010x?}",
                op.index()
            )
            .into());
        }
    }
    Ok(())
}

fn report(
    total: ByteSize,
    measurements: &[Measurement],
) -> Result<(), Box<dyn Error>> {
    let summaries = measurements
        .iter()
        .map(|measurement| {
            Summary::new(&measurement.samples)
                .map(|summary| (measurement, summary))
                .ok_or_else(|| format!("{} buffers have no samples", measurement.buffers))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let baseline = summaries
        .iter()
        .find(|(measurement, _)| measurement.buffers == 1)
        .map(|(_, summary)| summary.median)
        .ok_or("the one-buffer baseline is missing")?;
    let rows = summaries
        .into_iter()
        .map(|(measurement, summary)| {
            let undefined = || {
                format!(
                    "the median elapsed time with {} buffers is zero; throughput and speedup are \
                     undefined",
                    measurement.buffers
                )
            };
            let throughput = throughput_gib_s(total, summary.median).ok_or_else(undefined)?;
            let speedup = speedup(baseline, summary.median).ok_or_else(undefined)?;
            Ok((measurement, summary, throughput, speedup))
        })
        .collect::<Result<Vec<_>, String>>()?;
    println!(
        "{:>7}  {:>12}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}",
        "buffers", "peak slots", "median ms", "min ms", "max ms", "GiB/s", "speedup"
    );
    for (measurement, summary, throughput, speedup) in rows {
        println!(
            "{:>7}  {:>12}  {:>10.2}  {:>10.2}  {:>10.2}  {:>8.2}  {:>7.2}x",
            measurement.buffers,
            measurement.peak.to_string(),
            summary.median.as_secs_f64() * 1e3,
            summary.min.as_secs_f64() * 1e3,
            summary.max.as_secs_f64() * 1e3,
            throughput,
            speedup
        );
    }
    println!();
    println!(
        "GiB/s is end-to-end effective payload throughput: {} bytes of logical weights divided by \
         the median elapsed time, with 1 GiB = 2^30 bytes.",
        total.bytes()
    );
    println!(
        "The workload computes checksums over the transferred weights, and the Metal backend \
         ignores duration hints."
    );
    println!("These results do not establish transfer/compute overlap or inference performance.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::error::Error;
    use std::num::{
        NonZeroU32,
        NonZeroU64,
    };
    use std::time::Duration;

    use foundry_infer_core::ByteSize;
    use foundry_infer_metal::{
        MetalBackend,
        MetalError,
    };

    use super::{
        measure,
        prepare,
    };
    use crate::bench::workload::Sizes;

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

    fn small() -> Result<Sizes, Box<dyn Error>> {
        Ok(Sizes::new(
            NonZeroU32::new(4).ok_or("zero layers")?,
            NonZeroU64::new(1).ok_or("zero weight")?,
        )?)
    }

    #[test]
    fn measures_every_buffer_configuration() -> TestResult {
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        let prepared = prepare(&backend, &small()?)?;
        let iterations = NonZeroU32::new(2).ok_or("zero iterations")?;
        let measurements = measure(&mut backend, &prepared, 1, iterations)?;
        let buffers: Vec<u32> = measurements
            .iter()
            .map(|measurement| measurement.buffers)
            .collect();
        assert_eq!(buffers, [1, 2, 3], "every buffer count is measured");
        for measurement in &measurements {
            assert_eq!(
                measurement.peak,
                ByteSize::from_mib(u64::from(measurement.buffers))?,
                "{} buffers hold that many 1 MiB slots",
                measurement.buffers
            );
            assert_eq!(
                measurement.samples.len(),
                2,
                "{} buffers keep only measured samples",
                measurement.buffers
            );
            assert!(
                measurement
                    .samples
                    .iter()
                    .all(|&sample| sample > Duration::ZERO),
                "{} buffers have elapsed time",
                measurement.buffers
            );
        }
        Ok(())
    }

    #[test]
    fn checksum_mismatches_abort_the_benchmark() -> TestResult {
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        let mut prepared = prepare(&backend, &small()?)?;
        let (_, checksum) = prepared.expected.last_mut().ok_or("no layers")?;
        *checksum = checksum.wrapping_add(1);
        let iterations = NonZeroU32::new(1).ok_or("zero iterations")?;
        let error = measure(&mut backend, &prepared, 0, iterations)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(
            error.contains("layer 3 checksum mismatch"),
            "a wrong checksum fails the run, got {error:?}"
        );
        Ok(())
    }
}
