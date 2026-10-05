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

There are no inference kernels, general-purpose scheduler, or streaming
benchmarks yet. Transfer/compute overlap and performance relative to the legacy
engine have not been measured.

## Metal

`foundry-infer-metal` is available on macOS. It copies host staging buffers into
private GPU memory and runs a checksum kernel over the transferred weights.
Command queues and events provide synchronization, and submitted resources are
retained until their work completes. Tests cover execution with one, two, and
three resident buffers, plus cleanup after submission failures.

The backend does not yet claim concurrent copy and compute. Foundry does not
depend on the legacy `metal-infer-rs` repository; any future reuse of its code
will be evaluated against a performance baseline measured in that repository.

## Workspace

- `foundry-infer-core`: backend-neutral tensor, memory, and graph metadata.
- `foundry-infer-plan`: execution plans, JSON serialization, and static validation.
- `foundry-infer-runtime`: backend contract, plan interpreter, and fake backend.
- `foundry-infer-metal`: macOS Metal backend for synthetic GPU workloads.
- `foundry-infer-cli`: the `foundry` binary.

## CLI

Export a validated triple-buffer plan for a synthetic 48-layer workload:

```sh
cargo run -p foundry-infer-cli -- plan --dump plan.json
```

The workload describes 450 MiB of weights per layer and uses three resident
slots within a 4 GiB device budget. This command exports metadata without
loading weights or executing GPU work, and refuses to overwrite an existing
file. GPU execution is currently exercised through the library tests.

## Build and validate

```sh
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo +nightly fmt --all --check
```

On macOS, GPU tests may skip when a compatible Metal device is unavailable.
Require actual GPU execution when validating the backend:

```sh
FOUNDRY_REQUIRE_METAL=1 cargo test -p foundry-infer-metal
```

## License

Mozilla Public License 2.0. See [LICENSE](LICENSE).
