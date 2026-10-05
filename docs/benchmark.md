# Synthetic benchmark

`foundry bench` measures end-to-end runtime execution of a synthetic streaming
workload with 1, 2 and 3 resident GPU buffers. It is available on macOS with the
`metal` feature.

```sh
cargo run --release -p foundry-infer-cli --features metal -- \
  bench --layers 48 --weight-mib 64 --warmup 2 --iterations 30 \
  --report run.json --trace run.trace.json
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `--backend` | `metal` | GPU backend. |
| `--layers` | `48` | Synthetic layers, each with one weight. Positive. |
| `--weight-mib` | `64` | Weight size per layer, in MiB. Positive, below 4 GiB. |
| `--iterations` | `10` | Measured rounds. Positive. |
| `--warmup` | `2` | Unmeasured rounds before measuring. |
| `--bootstrap-resamples` | `10000` | Bootstrap replicates, from 1 to 1,000,000. |
| `--seed` | `0` | Unsigned 64-bit seed of the bootstrap generator. |
| `--report PATH` | none | Write the JSON report to a new file. |
| `--trace PATH` | none | Write a Chrome Trace JSON file to a new file and enable detailed instrumentation. Requires `--report`. |

Output paths are validated before any payload is generated. The parent
directory must exist, the destination must not exist, and the report and trace
must differ. Existing files are never overwritten. Reports and traces are
serialized and written after the last execution, outside the timed region.

Raw durations and statistics are always collected. Without `--trace`, the
instrumentation mode is `untraced`. With `--trace`, it is `traced`: CPU spans,
GPU command-buffer timestamps and allocation accounting are recorded. Tracing
adds host clock reads and bookkeeping inside the timed region. Use untraced runs
for timing comparisons and traced runs for diagnostics.

If any execution fails, the run exits with a nonzero status. When `--report`
was given, a report with status `failed` is still written. It holds the
completed executions, the failure context and, when traced, the diagnostics
collected so far. A failed run has no statistics. A traced run also fails when
any command buffer lacks valid GPU timestamps.

## Protocol

- Untimed setup: Metal device and checksum pipeline, the graph (no duration
  hint), validated 1/2/3-buffer plans, deterministic in-memory weights with
  distinct content per layer, and expected CPU checksums.
- Each round runs every configuration once. Round `r` runs `[1, 2, 3]` rotated
  left by `r mod 3`. Warmup rounds come first and are recorded but excluded from
  statistics.
- A monotonic timer starts right before the runtime interpreter call and stops
  after all submitted GPU work has drained. It covers runtime validation,
  allocations, host staging, transfers, checksum kernels, synchronization, and
  runtime cleanup.
- After each execution, outside timing, every layer checksum is verified and
  checksum results are released with `clear_checksums()`.

## Report (`schema_version: 1`)

Durations are integer nanoseconds (`_ns`) and sizes are integer bytes
(`_bytes`). An unavailable value is written as `{"value": null, "reason": "..."}`
and an available one as `{"value": ...}`.

| Field | Content |
|---|---|
| `run` | Identifier, UTC start time, `succeeded` or `failed`, `untraced` or `traced`. |
| `foundry` | Version and build provenance embedded at compile time: commit, dirty state (tracked modifications and untracked non-ignored files, counted), rustc, target, profile, opt-level, debug assertions, panic strategy, features, rustflags. |
| `runtime_repository` | Commit and dirty state of the working directory at run time, which may differ from the build. |
| `invocation` | Command arguments, working directory and resolved parameters. |
| `environment` | Machine model, CPU, GPU, physical memory, OS name, version and build, architecture. |
| `workload` | Layers, weight, payload, alignment, slot and slot-budget bytes; planned slot peak per configuration; device limits. |
| `protocol` | Configuration order, warmup and measured rounds, timer, timed and untimed work, units. |
| `executions` | Every execution in order: sequence, phase, round, position in the round, buffers, elapsed time, outcome and checksum status. |
| `statistics` | Method and per-configuration summary, only for a successful run. |
| `memory` | Traced runs: per-execution and per-configuration accounting. |
| `gpu_timing` | Traced runs: valid and invalid GPU intervals, the acceptance gate, and busy times per execution. |
| `trace` | Trace file name and clock conversion. |
| `limitations` | Measurement limits that apply to the run. |
| `failure` | Stage, message, round, position and configuration of a failure. |

## Statistics

Only completed measured executions with verified checksums count.

- Median: the middle sample, or `lower + (upper - lower) / 2` in integer
  nanoseconds for an even count. Minimum, maximum and arithmetic mean are also
  reported.
- Standard deviation: sample deviation with `n - 1`, unavailable for one
  sample.
- Confidence intervals: 95% percentile bootstrap of the median and the mean.
  Each replicate draws the original number of observations with replacement.
- Speedup: one-buffer median divided by the configuration median. Its interval
  resamples complete measured rounds in pairs, so both configurations use the
  same rounds.
- Quantiles: sort the replicates, then interpolate linearly at index
  `(n - 1) x p` with `p = 0.025` and `0.975`.
- Generator: SplitMix64 with wrapping 64-bit arithmetic, seeded once. Bounded
  indices use rejection sampling: draw until the value is at least
  `(2^64 - n) mod n`, then take it modulo `n`.
- Traversal: for each configuration in ascending order and each replicate,
  draw `n` indices and compute the median and mean; then, for configurations 2
  and 3 and each replicate, draw `R` round indices. Configurations with fewer
  than two samples draw nothing and have no interval.
- Zero durations make throughput and speedup unavailable, never infinite.

`scripts/bench_summary.py` recalculates these statistics from a report without
a GPU and checks them against the report within a relative tolerance of `1e-9`:

```sh
uv run --script scripts/bench_summary.py run.json
```

## Trace

The trace is Chrome Trace Event JSON and opens in [Perfetto](https://ui.perfetto.dev).
It has separate tracks for:

- CPU runtime commands, staging and submission;
- CPU host waits, the cleanup drain and the final drain;
- the benchmark harness: timed region, checksum verification and release;
- GPU copy command buffers;
- GPU checksum compute command buffers;
- allocation lifetimes and counters.

Events carry the run, sequence, phase, round, position and buffer count, plus
the command index, op, tensor, slot, stream, event and bytes when they apply.

All timestamps share the host clock: `mach_absolute_time` scaled by
`mach_timebase_info`, the time base of Metal command-buffer timestamps. Trace
times are microseconds since an origin taken at run start. The report records
the timebase and origin.

GPU intervals are `GPUStartTime` to `GPUEndTime` of completed command buffers,
read after completion. An interval covers a whole command buffer, including
encoded event waits, not an isolated kernel. When consecutive command buffers
on one queue report overlapping intervals, the later one is drawn on an overlap
lane of the same track, and the report counts it. Missing or invalid timestamps
are marked as such; CPU submission time is never substituted.

## Memory

Traced reports separate:

- CPU payload bytes, kept for the whole run;
- shared staging buffers;
- private weight-slot buffers;
- shared checksum result buffers;
- diagnostic storage.

For each Metal category, the report gives requested bytes and
`MTLAllocation.allocatedSize`, allocation and free counts, and current and peak
live bytes. Each native buffer is counted once, from creation until Foundry
drops its last reference, including references held by in-flight command
buffers and checksum results. A plan `Release` command does not prove that a
buffer was freed. Snapshots are taken at execution start, after the drain, and
after checksum release.

The planned slot peak is reported separately from the observed private-slot
peak, along with any retention beyond the plan and the configured slot budget.
`MTLDevice.currentAllocatedSize` is sampled at allocation transitions and
boundaries; its maximum is a sampled maximum, not a process peak. Payload,
Metal buffers, driver allocations and diagnostic storage are never summed into
an application memory peak.

## Limitations

- The workload computes checksums over transferred weights. It is not
  inference.
- The Metal backend ignores duration hints and does not claim concurrent copy
  and compute.
- Throughput is logical payload bytes over the median end-to-end time, not raw
  transfer bandwidth.
- The results do not establish transfer/compute overlap or inference
  performance. More resident buffers do not guarantee a speedup.
- Independent invocations are not pooled.

## Campaigns

`scripts/bench-campaign.sh ARCHIVE_DIR` builds the release binary and runs
48 layers with 16, 64 and 128 MiB weights, 2 warmup and 30 measured rounds,
10000 resamples and seed 0. For each size, it runs three untraced invocations
and one traced invocation. It refuses an existing directory and records the
commands, provenance, logs, outcomes and recalculated summaries next to the
reports and traces.
