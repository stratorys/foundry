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

Foundry does not execute graphs yet. There is no GPU backend, scheduler, or
streaming benchmark.

## Metal

A Metal backend will reuse parts of the legacy `metal-infer-rs` engine
selectively; Foundry does not depend on that repository. The legacy engine's
performance is measured in its own repository first, and the integrated
backend is compared against that baseline before any code is adopted.

## Workspace

- `foundry-infer-core`: backend-neutral tensor, memory, and graph metadata.
- `foundry-infer-cli`: the `foundry` binary.

## Build and validate

```sh
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo +nightly fmt --all --check
```

## License

Mozilla Public License 2.0. See [LICENSE](LICENSE).
