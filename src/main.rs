#[cfg_attr(
    not(target_os = "macos"),
    expect(dead_code, reason = "no device on this platform")
)]
mod cli;

use std::process::ExitCode;

use clap::Parser;
use tracing::error;

use crate::cli::{
    Cli,
    execute,
};

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();
    match execute(Cli::parse().command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            error!(message = "Command failed.", %error);
            ExitCode::FAILURE
        }
    }
}
