# Foundry

An experimental out-of-core GPU runtime, written in Rust.

Foundry aims to execute workloads whose weights do not fit in device memory by
streaming them through a fixed device-memory budget, overlapping transfers
with compute. It is designed to be backend-neutral. Nothing here is stable.

## Current status

The workspace contains only the `foundry` command-line binary, which supports
`--help` and `--version`. There is no runtime, backend, or benchmark yet.

## Build and validation

```sh
cargo build --workspace
cargo run -p foundry-infer-cli -- --help
cargo run -p foundry-infer-cli -- --version

cargo fmt --all --check
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## Workspace layout

Crates use the `foundry-infer-*` namespace. Only one exists today:

- `foundry-infer-cli`: the `foundry` binary.

New crates are added when a milestone needs them, never as placeholders.

## Milestone 1

### Objective

Run a synthetic workload that represents more than 20 GB of weights within a
device-memory budget below 8 GB, with:

- asynchronous host-to-device transfers;
- copy and compute overlapping;
- fixed buffering (a constant number of preallocated device buffers);
- tracing that shows the overlap and the memory budget.

The workload is synthetic, so M1 needs no real model.

### Out of scope

M1 excludes model integration, custom GEMM, quantization, adaptive planning,
activation spilling, multi-GPU, and NVMe streaming.

### Metal integration

Metal integration is deferred to M1.4. It will reuse parts of the legacy
`metal-infer-rs` engine selectively; Foundry does not depend on that
repository.

Before M1.4 starts, a fresh legacy baseline is captured separately in
`metal-infer-rs`, with its own benchmark harness. Its protocol is:
metal-infer only, `Qwen/Qwen3-0.6B`, 512 prompt tokens, 128 generated tokens,
3 rounds of 5 offline iterations, the synthetic server workload at
concurrency 1 with 8 requests per round, and a 30-second cooldown. The model is
used only to measure the legacy engine; it is not a Foundry dependency. The
historical 171 tokens/second decode result is context, not the baseline.

### Regression gate

When Metal is integrated, any prefill or decode regression exceeding 5 %
against that baseline, under the same protocol, must be investigated. This is
a review rule; there is no automated performance gate.

## License

Mozilla Public License 2.0. See [LICENSE](LICENSE).
