use clap::Parser;

/// Experimental out-of-core GPU runtime.
#[derive(Parser)]
#[command(name = "foundry", version)]
struct Cli {}

fn main() { Cli::parse(); }
