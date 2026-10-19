# Foundry

A cross-platform LLM inference engine written in Rust.

The current inference backend is Metal on macOS.

Use Rust nightly with `rustfmt` and `clippy` installed.

## Build and test

- Format: `cargo +nightly fmt --all --check`.
- Lint: `cargo clippy --locked --all-targets`.
- Build for release: `cargo build --locked --release`.
- Show help: `cargo run --locked -- --help`.

Run both test commands below, one at a time. The memory test samples device-wide Metal
allocations, so it runs in a separate process without concurrent Metal tests.

```sh
cargo +nightly test --locked -- --skip metal::tests::memory_peaks_reset_per_run_and_preserve_resident_allocations
cargo +nightly test --locked --lib metal::tests::memory_peaks_reset_per_run_and_preserve_resident_allocations -- --exact
```

## Profiles

Development builds use less debug information to reduce build time.
Build with `cargo build --locked`.

Release builds use Cargo's default optimized settings.

For profiling, use `cargo build --locked --profile profiling`.
This adds source line information to an optimized build.
The executable is `target/profiling/foundry`.

## Performance

Measure build time with `cargo build --locked --timings`.
The report is saved under `target/cargo-timings/`.

See [evaluation and benchmarks](tools/reference/README.md) for model tests and speed measurements.

In CI, cache Cargo downloads and build files. Keep caches separate by OS, CPU,
Rust version, build profile, and lockfile.
