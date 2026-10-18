# Foundry

A cross-platform LLM inference engine written in Rust.

The current inference backend is Metal on macOS.

Use Rust nightly with `rustfmt` and `clippy` installed.

## Build and test

- Format: `cargo +nightly fmt --all --check`.
- Lint: `cargo clippy --locked --all-targets`.
- Test: `cargo test --locked`.
- Build for release: `cargo build --locked --release`.
- Show help: `cargo run --locked -- --help`.

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
