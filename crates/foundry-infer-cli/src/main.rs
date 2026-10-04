use clap::Parser;

#[derive(Parser)]
#[command(
    name = "foundry",
    version,
    about = "Experimental out-of-core GPU runtime"
)]
struct Cli {}

fn main() { Cli::parse(); }
