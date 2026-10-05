mod bench;

use std::error::Error;
use std::fs::OpenOptions;
use std::io::{
    ErrorKind,
    Write,
};
use std::num::NonZeroU32;
use std::path::{
    Path,
    PathBuf,
};
use std::process::ExitCode;
use std::time::Duration;

use clap::{
    Parser,
    Subcommand,
};
use foundry_infer_core::{
    Alignment,
    ByteSize,
    MemoryBudget,
    MemorySpace,
};
use foundry_infer_plan::{
    synthetic_chain_graph,
    synthetic_chain_plan,
};

const LAYERS: u32 = 48;
const LAYER_WEIGHT_MIB: u64 = 450;
const LAYER_DURATION: Duration = Duration::from_millis(60);
const RESIDENT_SLOTS: NonZeroU32 = NonZeroU32::new(3).expect("three slots are non-zero");
const DEVICE_BUDGET_GIB: u64 = 4;
const SLOT_ALIGNMENT: u64 = 256;

#[derive(Parser)]
#[command(
    name = "foundry",
    version,
    about = "Experimental out-of-core GPU runtime"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    #[command(
        about = "Export the validated triple-buffer execution plan of a synthetic 48-layer \
                 workload"
    )]
    Plan {
        #[arg(
            long,
            value_name = "PATH",
            help = "Write the plan as JSON to a new file"
        )]
        dump: PathBuf,
    },
    #[command(
        about = "Benchmark end-to-end runtime execution of a synthetic streaming workload with 1, \
                 2 and 3 resident GPU buffers"
    )]
    Bench(bench::BenchArgs),
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Some(Command::Plan {
            dump,
        }) => dump_plan(&dump),
        Some(Command::Bench(args)) => bench::run(&args),
        None => Ok(()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn dump_plan(path: &Path) -> Result<(), Box<dyn Error>> {
    let graph = synthetic_chain_graph(
        LAYERS,
        ByteSize::from_mib(LAYER_WEIGHT_MIB)?,
        Some(LAYER_DURATION),
    )?;
    let plan = synthetic_chain_plan(&graph, RESIDENT_SLOTS, Alignment::new(SLOT_ALIGNMENT)?)?;
    let budget = MemoryBudget::new(MemorySpace::Device, ByteSize::from_gib(DEVICE_BUDGET_GIB)?);
    plan.validate(&graph, budget)?;
    let json = plan.to_json()?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            if error.kind() == ErrorKind::AlreadyExists {
                format!("{} already exists", path.display())
            } else {
                format!("cannot create {}: {error}", path.display())
            }
        })?;
    writeln!(file, "{json}")
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    Ok(())
}
