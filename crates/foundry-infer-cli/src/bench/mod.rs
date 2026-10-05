#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
mod analysis;
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
mod bootstrap;
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
mod error;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod instrument;
#[cfg(all(feature = "metal", target_os = "macos"))]
#[expect(
    clippy::print_stdout,
    reason = "the benchmark prints its report to stdout"
)]
mod metal;
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
mod output;
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
mod provenance;
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
mod random;
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
mod report;
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
mod stats;
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
mod trace;
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
mod workload;

use std::env;
use std::num::{
    NonZeroU32,
    NonZeroU64,
};
use std::path::PathBuf;

use clap::{
    Args,
    ValueEnum,
};

use crate::bench::error::BenchError;
use crate::bench::output::Outputs;
use crate::bench::report::Parameters;
use crate::bench::workload::Sizes;

const RESAMPLES_MAX: u32 = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum BackendKind {
    Metal,
}

impl BackendKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Metal => "metal",
        }
    }
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
    #[arg(
        long,
        value_name = "PATH",
        help = "Write a versioned JSON report with raw samples, provenance and statistics to a \
                new file"
    )]
    report: Option<PathBuf>,
    #[arg(
        long,
        value_name = "PATH",
        requires = "report",
        help = "Write a Chrome Trace (Perfetto) JSON file to a new file and enable GPU timestamps \
                and allocation accounting; requires --report"
    )]
    trace: Option<PathBuf>,
    #[arg(
        long,
        value_name = "COUNT",
        default_value = "10000",
        help = "Bootstrap resamples for 95% confidence intervals"
    )]
    bootstrap_resamples: NonZeroU32,
    #[arg(
        long,
        value_name = "SEED",
        default_value = "0",
        help = "Seed of the SplitMix64 bootstrap generator"
    )]
    seed: u64,
}

#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
pub(crate) struct Settings {
    sizes: Sizes,
    warmup: u32,
    iterations: NonZeroU32,
    resamples: NonZeroU32,
    seed: u64,
    outputs: Outputs,
    parameters: Parameters,
    arguments: Vec<String>,
    working_directory: Option<PathBuf>,
}

impl Settings {
    #[cfg(all(test, feature = "metal", target_os = "macos"))]
    fn for_tests(
        sizes: Sizes,
        warmup: u32,
        iterations: NonZeroU32,
    ) -> Self {
        Self {
            sizes,
            warmup,
            iterations,
            resamples: NonZeroU32::MIN,
            seed: 0,
            outputs: Outputs {
                report: None,
                trace: None,
            },
            parameters: Parameters {
                backend: BackendKind::Metal.name().to_owned(),
                layers: sizes.layers,
                weight_mib: 1,
                iterations: iterations.get(),
                warmup,
                bootstrap_resamples: 1,
                seed: 0,
                report: None,
                trace: None,
            },
            arguments: Vec::new(),
            working_directory: None,
        }
    }
}

pub(crate) fn run(args: &BenchArgs) -> Result<(), BenchError> {
    let outputs = Outputs::new(args.report.as_deref(), args.trace.as_deref())?;
    if args.bootstrap_resamples.get() > RESAMPLES_MAX {
        return Err(BenchError::ResamplesTooLarge {
            count: args.bootstrap_resamples,
            max: RESAMPLES_MAX,
        });
    }
    let sizes = Sizes::new(args.layers, args.weight_mib)?;
    let display = |path: &Option<PathBuf>| path.as_ref().map(|path| path.display().to_string());
    let settings = Settings {
        sizes,
        warmup: args.warmup,
        iterations: args.iterations,
        resamples: args.bootstrap_resamples,
        seed: args.seed,
        outputs,
        parameters: Parameters {
            backend: args.backend.name().to_owned(),
            layers: args.layers.get(),
            weight_mib: args.weight_mib.get(),
            iterations: args.iterations.get(),
            warmup: args.warmup,
            bootstrap_resamples: args.bootstrap_resamples.get(),
            seed: args.seed,
            report: display(&args.report),
            trace: display(&args.trace),
        },
        arguments: env::args_os()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect(),
        working_directory: env::current_dir().ok(),
    };
    match args.backend {
        BackendKind::Metal => run_metal(&settings),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_metal(settings: &Settings) -> Result<(), BenchError> { metal::run(settings) }

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_metal(_settings: &Settings) -> Result<(), BenchError> { Err(BenchError::MetalUnavailable) }
