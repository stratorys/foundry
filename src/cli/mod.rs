mod command;
mod error;

use std::fmt;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::str::FromStr;

use clap::{
    Args,
    Parser,
    Subcommand,
};
use foundry::bench::TimingScope;
#[cfg(target_os = "macos")]
use foundry::metal::MetalDevice;
use foundry::nn::ProfileMode;
use tracing::error;

#[cfg(target_os = "macos")]
use self::command::{
    bench_command,
    eval_command,
    run,
};
pub use self::error::CliError;
use self::error::ModelIdError;

#[derive(Parser)]
#[command(about = "Foundry LLM inference engine.")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
pub enum Command {
    #[command(
        about = "Generate token IDs greedily from prompt token IDs on Metal and print them as \
                 JSON."
    )]
    Run {
        #[arg(long, help = "Hugging Face repository id, as owner/name.")]
        model: ModelId,
        #[arg(
            long,
            value_delimiter = ',',
            required = true,
            help = "Prompt token IDs, comma-separated."
        )]
        prompt_ids: Vec<u32>,
        #[arg(long, help = "Maximum number of generated tokens.")]
        max_new_tokens: usize,
    },
    #[command(
        about = "Benchmark prefill and greedy decode on random token IDs and write report.json."
    )]
    Bench(BenchArgs),
    #[command(
        about = "Compare each submodule with reference activations on Metal and write eval.json."
    )]
    Eval(EvalArgs),
}

#[derive(Debug, PartialEq, Eq, Args)]
pub struct BenchArgs {
    #[arg(long, help = "Hugging Face repository id, as owner/name.")]
    model: ModelId,
    #[arg(
        long,
        help = "Prompt tokens processed in one prefill; 0 benchmarks generation only."
    )]
    prompt_tokens: u32,
    #[arg(
        long,
        help = "Exactly N new tokens, including the prefill output; 0 measures only prompt \
                processing."
    )]
    new_tokens: u32,
    #[arg(
        long,
        value_enum,
        default_value = "generation",
        help = "Time forwards through logits or generation through host token availability."
    )]
    timing_scope: TimingScope,
    #[arg(
        long,
        default_value_t = 0,
        help = "Tokens filled into the cache before each measured run."
    )]
    depth: u32,
    #[arg(long, default_value_t = 1, help = "Minimum number of warmup runs.")]
    warmup_runs: u32,
    #[arg(
        long,
        default_value_t = 0,
        help = "Minimum warmup duration in milliseconds."
    )]
    warmup_ms: u64,
    #[arg(long, default_value = "10", help = "Number of measured runs.")]
    runs: NonZeroU32,
    #[arg(
        long,
        default_value = "1",
        help = "Sequences generated together, each with its own random prompt of the same length."
    )]
    batch_size: NonZeroU32,
    #[arg(long, default_value_t = 0, help = "Seed of the random token IDs.")]
    seed: u64,
    #[arg(
        long,
        value_enum,
        help = "Record device dispatches, allocations and GPU busy time per step; steps keeps \
                execution unchanged, kernels also times each kernel with one dispatch per command \
                buffer, which slows wall times."
    )]
    profile: Option<ProfileMode>,
    #[arg(long, help = "Directory that receives report.json.")]
    out: PathBuf,
}

#[derive(Debug, PartialEq, Eq, Args)]
pub struct EvalArgs {
    #[arg(long, help = "Hugging Face repository id, as owner/name.")]
    model: ModelId,
    #[arg(
        long = "reference",
        required = true,
        help = "Reference directory holding generation.json, last_logits.safetensors and, for at \
                least one reference, activations.safetensors. Repeatable."
    )]
    references: Vec<PathBuf>,
    #[arg(long, help = "Directory that receives eval.json.")]
    out: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelId {
    owner: String,
    name: String,
}

impl FromStr for ModelId {
    type Err = ModelIdError;

    fn from_str(model: &str) -> Result<Self, Self::Err> {
        match model.split_once('/') {
            Some((owner, name)) if !owner.is_empty() && !name.is_empty() && !name.contains('/') => {
                Ok(Self {
                    owner: owner.to_owned(),
                    name: name.to_owned(),
                })
            }
            _ => {
                error!(message = "Model id is not of the form owner/name.", model);
                Err(ModelIdError)
            }
        }
    }
}

impl fmt::Display for ModelId {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(formatter, "{}/{}", self.owner, self.name)
    }
}

#[cfg(target_os = "macos")]
pub fn execute(command: Command) -> Result<(), CliError> {
    let open_device = |profile: Option<ProfileMode>| {
        profile
            .map_or_else(MetalDevice::new, MetalDevice::profiled)
            .map_err(|error| {
                error!(message = "Creating the Metal device failed.", %error);
                CliError::Device
            })
    };
    match command {
        Command::Run {
            model,
            prompt_ids,
            max_new_tokens,
        } => run(|| open_device(None), &model, &prompt_ids, max_new_tokens),
        Command::Bench(args) => bench_command(|| open_device(args.profile), &args),
        Command::Eval(args) => eval_command(|| open_device(None), &args),
    }
}

#[cfg(not(target_os = "macos"))]
pub fn execute(_command: Command) -> Result<(), CliError> {
    error!(message = "No device is available on this platform.");
    Err(CliError::DeviceUnavailable)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;
    use std::path::PathBuf;

    use clap::error::ErrorKind;
    use clap::{
        CommandFactory,
        Parser,
    };
    use foundry::bench::TimingScope;

    use crate::cli::error::ModelIdError;
    use crate::cli::{
        BenchArgs,
        Cli,
        Command,
        EvalArgs,
        ModelId,
    };

    #[test]
    fn model_id_parses_owner_and_name() {
        let model = "a/b".parse::<ModelId>();
        assert_eq!(
            model,
            Ok(ModelId {
                owner: "a".to_owned(),
                name: "b".to_owned(),
            }),
            "owner and name"
        );
        assert_eq!(
            model.map(|model| model.to_string()),
            Ok("a/b".to_owned()),
            "display gives back owner/name"
        );
    }

    #[test]
    fn model_id_rejects_malformed_values() {
        ["ab", "/b", "a/", "a/b/c", ""]
            .into_iter()
            .for_each(|model| {
                assert_eq!(
                    model.parse::<ModelId>(),
                    Err(ModelIdError),
                    "model id {model:?} is rejected"
                );
            });
    }

    #[test]
    fn cli_definition_is_consistent() { Cli::command().debug_assert(); }

    #[test]
    fn run_parses_model_prompt_ids_and_max_new_tokens() {
        assert_eq!(
            Cli::try_parse_from([
                "foundry",
                "run",
                "--model",
                "a/b",
                "--prompt-ids",
                "1,2,3",
                "--max-new-tokens",
                "4",
            ])
            .map(|cli| cli.command)
            .map_err(|error| error.kind()),
            Ok(Command::Run {
                model: ModelId {
                    owner: "a".to_owned(),
                    name: "b".to_owned(),
                },
                prompt_ids: vec![1, 2, 3],
                max_new_tokens: 4,
            }),
            "comma-separated prompt IDs"
        );
    }

    #[test]
    fn run_rejects_missing_prompt_ids() {
        assert_eq!(
            Cli::try_parse_from(["foundry", "run", "--model", "a/b", "--max-new-tokens", "4"])
                .map(|cli| cli.command)
                .map_err(|error| error.kind()),
            Err(ErrorKind::MissingRequiredArgument),
            "prompt IDs are required"
        );
    }

    #[test]
    fn bench_parses_counts_with_defaults() {
        let cli = Cli::try_parse_from(
            "foundry bench --model a/b --prompt-tokens 512 --new-tokens 128 --out reports"
                .split_whitespace(),
        );
        assert_eq!(
            cli.map(|cli| cli.command).map_err(|e| e.kind()),
            Ok(Command::Bench(BenchArgs {
                model: ModelId {
                    owner: "a".to_owned(),
                    name: "b".to_owned()
                },
                prompt_tokens: 512,
                new_tokens: 128,
                timing_scope: TimingScope::Generation,
                depth: 0,
                warmup_runs: 1,
                warmup_ms: 0,
                runs: NonZeroU32::new(10).expect("positive run count"),
                batch_size: NonZeroU32::MIN,
                seed: 0,
                profile: None,
                out: PathBuf::from("reports"),
            })),
            "exact generation with one warmup and ten runs"
        );
    }

    #[test]
    fn bench_parses_the_forward_scope() {
        let forward = Cli::try_parse_from(
            "foundry bench --model a/b --prompt-tokens 4 --out r --new-tokens 1 --timing-scope \
             forward"
                .split_whitespace(),
        );
        assert!(
            matches!(
                forward.map(|cli| cli.command),
                Ok(Command::Bench(BenchArgs {
                    timing_scope: TimingScope::Forward,
                    new_tokens: 1,
                    ..
                }))
            ),
            "forward and exact count parsed"
        );
    }

    #[test]
    fn bench_parses_the_batch_size() {
        let cli = Cli::try_parse_from(
            "foundry bench --model a/b --prompt-tokens 512 --new-tokens 128 --batch-size 8 --out \
             reports"
                .split_whitespace(),
        );
        let batch = cli
            .map(|cli| match cli.command {
                Command::Bench(args) => Some(args.batch_size),
                Command::Run {
                    ..
                }
                | Command::Eval(_) => None,
            })
            .map_err(|e| e.kind());
        assert_eq!(batch, Ok(NonZeroU32::new(8)), "batch of eight sequences");
    }

    #[test]
    fn bench_rejects_missing_arguments() {
        for (options, expected) in [
            (
                "--prompt-tokens 1 --new-tokens 1",
                ErrorKind::MissingRequiredArgument,
            ),
            ("--out r", ErrorKind::MissingRequiredArgument),
        ] {
            let cli = Cli::try_parse_from(
                "foundry bench --model a/b"
                    .split_whitespace()
                    .chain(options.split_whitespace()),
            );
            assert_eq!(
                cli.err().map(|e| e.kind()),
                Some(expected),
                "invalid options {options}"
            );
        }
    }

    #[test]
    fn eval_parses_repeated_references() {
        assert_eq!(
            Cli::try_parse_from([
                "foundry",
                "eval",
                "--model",
                "a/b",
                "--reference",
                "fixtures/transformers",
                "--reference",
                "fixtures/mlx",
                "--out",
                "reports",
            ])
            .map(|cli| cli.command)
            .map_err(|error| error.kind()),
            Ok(Command::Eval(EvalArgs {
                model: ModelId {
                    owner: "a".to_owned(),
                    name: "b".to_owned(),
                },
                references: vec![
                    PathBuf::from("fixtures/transformers"),
                    PathBuf::from("fixtures/mlx")
                ],
                out: PathBuf::from("reports"),
            })),
            "two reference directories in the given order"
        );
    }

    #[test]
    fn eval_rejects_missing_references_or_out() {
        [
            vec!["foundry", "eval", "--model", "a/b", "--out", "r"],
            vec!["foundry", "eval", "--model", "a/b", "--reference", "ref"],
        ]
        .into_iter()
        .for_each(|args| {
            assert_eq!(
                Cli::try_parse_from(&args)
                    .map(|cli| cli.command)
                    .map_err(|error| error.kind()),
                Err(ErrorKind::MissingRequiredArgument),
                "args {args:?}"
            );
        });
    }
}
