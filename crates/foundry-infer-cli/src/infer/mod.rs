#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
mod error;
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
mod report;
#[cfg(all(feature = "metal", target_os = "macos"))]
#[expect(
    clippy::print_stdout,
    reason = "the benchmark prints its summary to stdout"
)]
mod run;
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
mod summary;

use std::path::PathBuf;

use clap::Args;

pub(crate) use crate::infer::error::InferError;

#[derive(Args)]
pub(crate) struct InferBenchArgs {
    #[arg(
        long,
        value_name = "DIR",
        help = "Pinned Hugging Face snapshot directory of the model"
    )]
    snapshot: PathBuf,
    #[arg(
        long,
        value_name = "PATH",
        help = "Model manifest pinning the repository, revision, files and tensors"
    )]
    manifest: PathBuf,
    #[arg(
        long,
        value_name = "PATH",
        help = "Write the versioned JSON report to a new file"
    )]
    report: PathBuf,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn run(args: &InferBenchArgs) -> Result<(), InferError> { run::run(args) }

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub(crate) fn run(args: &InferBenchArgs) -> Result<(), InferError> {
    let _ = (&args.snapshot, &args.manifest, &args.report);
    Err(InferError::MetalUnavailable)
}
