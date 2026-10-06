# Foundry

An experimental out-of-core GPU runtime, written in Rust.

Foundry aims to run workloads whose weights do not fit in device memory by
streaming them through a fixed device-memory budget, overlapping transfers
with compute, behind a backend-neutral core. Nothing here is stable.

## Capabilities

`foundry-infer-core` describes workloads as metadata:

- tensors, with data types, shapes, and checked byte sizes;
- memory spaces, budgets, alignment, and device capabilities;
- ordered graphs of operations, with weights described as host memory or file
  ranges but never loaded, and structural validation of references, weight
  sizes, and initialization order.

`foundry-infer-plan` provides serializable execution plans with static validation
of memory budgets, buffer lifetimes, transfers, and synchronization. Synthetic
chains support sequential, double-buffered, and triple-buffered plans.

`foundry-infer-runtime` interprets validated plans through a backend-neutral
contract, with a fake backend for tests and cleanup of submitted work on failure.
The Metal backend executes synthetic workloads on the GPU.

`foundry-infer-llama` runs one model, `mlx-community/Llama-3.2-3B-Instruct-4bit`,
with fully resident weights and Foundry's own Metal kernels. There is no
general-purpose scheduler. A synthetic streaming benchmark measures end-to-end runtime execution on Metal and exports
raw samples, provenance, statistics, traces and memory accounting.
Transfer/compute overlap has not been established.

## Metal

`foundry-infer-metal` is available on macOS. It copies host staging buffers into
private GPU memory and runs a checksum kernel over the transferred weights.
Command queues and events provide synchronization, and submitted resources are
retained until their work completes. Tests cover execution with one, two, and
three resident buffers, plus cleanup after submission failures.

The backend does not yet claim concurrent copy and compute.

## Workspace

- `foundry-infer-core`: backend-neutral tensor, memory, and graph metadata.
- `foundry-infer-plan`: execution plans, JSON serialization, and static validation.
- `foundry-infer-runtime`: backend contract, plan interpreter, and fake backend.
- `foundry-infer-metal`: macOS Metal backend for synthetic GPU workloads.
- `foundry-infer-llama`: checkpoint loader and resident Llama 3.2 3B 4-bit
  inference on Metal (batch 1, up to 512 prompt tokens and 640 positions).
- `foundry-infer-cli`: the `foundry` binary, with an optional `metal` feature.

## CLI

Export a validated triple-buffer plan for a synthetic 48-layer workload:

```sh
cargo run -p foundry-infer-cli -- plan --dump plan.json
```

The workload describes 450 MiB of weights per layer and uses three resident
slots within a 4 GiB device budget. This command exports metadata without
loading weights or executing GPU work, and refuses to overwrite an existing
file.

## Benchmark

Compare end-to-end runtime execution of one synthetic workload with 1, 2, and 3
resident GPU buffers. The Metal backend is behind the `metal` feature, which is
off by default and only takes effect on macOS:

```sh
cargo run --release -p foundry-infer-cli --features metal -- \
  bench --backend metal --layers 48 --weight-mib 64 --iterations 10
```

Defaults are `--backend metal --layers 48 --weight-mib 64 --iterations 10
--warmup 2 --bootstrap-resamples 10000 --seed 0`. Layers, weight size,
iterations and resamples must be positive; warmup may be zero. Without the
feature, or off macOS, `bench` exits with an error.

Protocol:

- Untimed setup: Metal device and checksum pipeline, the graph (no duration
  hint), validated 1/2/3-buffer plans, deterministic in-memory weights with
  distinct content per layer, and expected CPU checksums. Nothing is read from
  disk.
- Each round runs every configuration once, in an order that rotates between
  rounds. `warmup` rounds are discarded, then `iterations` rounds are measured.
- A monotonic timer starts right before the runtime interpreter call and stops
  after all submitted GPU work has drained. It covers runtime validation,
  allocations, host staging, transfers, checksum kernels, synchronization, and
  runtime cleanup (slot release and a final drain of submitted GPU work).
- After every execution, outside timing, the checksum of every layer is verified
  and the checksum results are released with `clear_checksums()`. Any execution
  or checksum error aborts the benchmark.
- For each buffer count, the console gives median, mean, standard deviation,
  minimum and maximum time, 95% bootstrap intervals, end-to-end effective payload
  throughput (logical weight bytes / median, in GiB of 2^30 bytes), and speedup
  against the one-buffer median with a paired 95% interval.

`--report PATH` writes a versioned JSON report with every execution, build and
runtime provenance, environment, protocol and statistics. `--trace PATH`
(requires `--report`) also writes a Chrome Trace for Perfetto, with CPU and GPU
command-buffer tracks and allocation accounting. Neither overwrites an existing
file. See [docs/benchmark.md](docs/benchmark.md) for the schema, the statistical
method, timing semantics and memory limitations, and
`scripts/bench_summary.py` to recalculate a report offline with
`uv run --script`.

Memory: the host keeps `layers x weight` bytes of payload (3 GiB by default),
plus transient shared staging buffers during execution. The device slot budget
is three aligned weight slots (192 MiB by default) and must fit the GPU's
recommended working set. It bounds the weight slots, not total application
memory. Each weight must be below 4 GiB, and each aligned slot must fit the
device's maximum Metal buffer length. Both limits are checked before any
payload is generated.

The workload computes checksums, not inference, and the Metal backend ignores
duration hints. The results do not establish transfer/compute overlap or
inference performance. More resident buffers do not guarantee overlap or a
speedup.

## Llama inference benchmark

Measure 512 input token IDs → 128 greedy tokens, with 2 warmup and 10 measured
executions. Use the pinned snapshot and manifest (kept outside version control):

```sh
cargo run --release -p foundry-infer-cli --features metal -- infer-bench \
  --snapshot "$snapshot" --manifest "$manifest" --report report.json
```

The report uses the same `token-ids-512x128-v1` protocol as
`tools/mlx-baseline/bench.py --protocol token-ids`.
`tools/mlx-baseline/compare.py` compares the two reports.

Full-model numerical validation against private MLX fixtures, generated with
`tools/mlx-baseline/fixtures.py`:

```sh
FOUNDRY_REQUIRE_METAL=1 FOUNDRY_REQUIRE_LLAMA=1 \
FOUNDRY_LLAMA_SNAPSHOT="$snapshot" FOUNDRY_LLAMA_FIXTURES="$fixtures" \
  cargo test --release -p foundry-infer-llama --features probe --test mlx_fixtures
```

## Build and validate

```sh
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo +nightly fmt --all --check
```

On macOS, GPU tests may skip when a compatible Metal device is unavailable.
Require actual GPU execution when validating the backend:

```sh
FOUNDRY_REQUIRE_METAL=1 cargo test -p foundry-infer-metal
FOUNDRY_REQUIRE_METAL=1 cargo test -p foundry-infer-llama --all-features
FOUNDRY_REQUIRE_METAL=1 cargo test -p foundry-infer-cli --features metal
```

The offline recalculation tests run `scripts/bench_summary.py` with
[uv](https://docs.astral.sh/uv/) and skip when it is missing; set
`FOUNDRY_REQUIRE_UV=1` to require them.

## License

Mozilla Public License 2.0. See [LICENSE](LICENSE).
