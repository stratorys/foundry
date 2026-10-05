#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
mod error;
#[cfg(all(feature = "metal", target_os = "macos"))]
#[expect(
    clippy::print_stdout,
    reason = "the benchmark prints its report to stdout"
)]
mod metal;
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
mod stats;
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
mod workload;

use std::num::{
    NonZeroU32,
    NonZeroU64,
};

use clap::{
    Args,
    ValueEnum,
};

use crate::bench::error::BenchError;
use crate::bench::workload::Sizes;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum BackendKind {
    Metal,
}

#[derive(Args)]
pub(crate) struct BenchArgs {
    #[arg(
        long,
        value_enum,
        default_value_t = BackendKind::Metal,
        help = "GPU backend to benchmark"
    )]
    backend: BackendKind,
    #[arg(
        long,
        value_name = "COUNT",
        default_value = "48",
        help = "Number of synthetic layers, each with one weight"
    )]
    layers: NonZeroU32,
    #[arg(
        long,
        value_name = "MIB",
        default_value = "64",
        help = "Weight size of each layer, in MiB"
    )]
    weight_mib: NonZeroU64,
    #[arg(
        long,
        value_name = "COUNT",
        default_value = "10",
        help = "Measured executions per buffer configuration"
    )]
    iterations: NonZeroU32,
    #[arg(
        long,
        value_name = "COUNT",
        default_value = "2",
        help = "Unmeasured executions per buffer configuration before measuring"
    )]
    warmup: u32,
}

pub(crate) fn run(args: &BenchArgs) -> Result<(), BenchError> {
    let sizes = Sizes::new(args.layers, args.weight_mib)?;
    match args.backend {
        BackendKind::Metal => run_metal(&sizes, args.warmup, args.iterations),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_metal(
    sizes: &Sizes,
    warmup: u32,
    iterations: NonZeroU32,
) -> Result<(), BenchError> {
    metal::run(sizes, warmup, iterations)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_metal(
    _sizes: &Sizes,
    _warmup: u32,
    _iterations: NonZeroU32,
) -> Result<(), BenchError> {
    Err(BenchError::MetalUnavailable)
}
