# Foundry

An experimental out-of-core GPU runtime, written in Rust.

## Crates

- `foundry-infer-core`: tensor and memory primitives.
- `foundry-infer-cli`: the `foundry` binary.

## Build

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo +nightly fmt --all --check
```

## License

Mozilla Public License 2.0. See [LICENSE](LICENSE).
