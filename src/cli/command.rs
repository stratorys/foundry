use std::fs::File;
use std::io::{
    BufWriter,
    Write,
};
use std::num::NonZeroU32;
use std::path::{
    Path,
    PathBuf,
};
use std::time::{
    Duration,
    Instant,
    SystemTime,
    UNIX_EPOCH,
};

use foundry::bench::{
    BenchConfig,
    Metadata,
    Report,
    bench,
    git_commit,
    rfc3339_utc,
};
use foundry::eval::{
    BaselineReport,
    Reference,
    ReferenceReport,
    baselines,
    eval_reference,
    max_cache_len,
    read_reference,
};
use foundry::generate::generate;
use foundry::models::llama::{
    LlamaConfig,
    LlamaForCausalLm,
    module_atol,
};
use foundry::nn::Device;
use foundry::weights::Weights;
use hf_hub::HFClientSync;
use serde::Serialize;
use tracing::{
    error,
    info,
    warn,
};

use crate::cli::error::CliError;
use crate::cli::{
    BenchArgs,
    EvalArgs,
    ModelId,
};

const DEVICE: &str = "metal";
const BENCH_REPORT_FILE: &str = "report.json";
const EVAL_REPORT_FILE: &str = "eval.json";

#[derive(Debug, Serialize)]
struct BenchRecord {
    metadata: Metadata,
    #[serde(flatten)]
    report: Report,
}

#[derive(Debug, Serialize)]
struct EvalRecord {
    metadata: Metadata,
    references: Vec<ReferenceReport>,
    baseline: Vec<BaselineReport>,
}

pub fn run<D: Device>(
    open_device: impl FnOnce() -> Result<D, CliError>,
    model: &ModelId,
    prompt_ids: &[u32],
    max_new_tokens: usize,
) -> Result<(), CliError> {
    let (config, weights) = open_snapshot(model)?;
    check_token_ids(prompt_ids, vocab_size(&config)?)?;
    let mut device = open_device()?;
    let max_cache_len = prompt_ids
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| {
            error!(
                message = "Prompt and generated token counts overflow usize.",
                prompt_tokens = prompt_ids.len(),
                max_new_tokens,
            );
            CliError::SeqLenOverflow
        })?;
    let llama = load_llama(&mut device, model, &config, &weights, max_cache_len)?;
    let generation = generate(
        &mut device,
        &llama,
        &[prompt_ids],
        max_new_tokens,
        config.eos_token_id(),
    )
    .map_err(|error| {
        error!(message = "Generation failed.", %model, %error);
        CliError::Generate
    })?;
    let token_ids = generation.token_ids.concat();
    info!(
        message = "Generated tokens.",
        device = DEVICE,
        prompt_tokens = prompt_ids.len(),
        generated_tokens = token_ids.len(),
        prefill_ms = %generation.prefill.as_millis(),
        decode_steps = generation.decode_steps.len(),
        decode_ms_total = %generation.decode_steps.iter().sum::<Duration>().as_millis(),
    );
    write_token_ids(&mut std::io::stdout().lock(), &token_ids).map_err(|error| {
        error!(message = "Writing the token IDs to stdout failed.", %error);
        CliError::Output
    })
}

pub fn bench_command<D: Device>(
    open_device: impl FnOnce() -> Result<D, CliError>,
    args: &BenchArgs,
) -> Result<(), CliError> {
    let test_time = test_time()?;
    let git_commit = read_git_commit();
    let (config, weights) = open_snapshot(&args.model)?;
    let metadata = metadata(&args.model, &weights, test_time, git_commit)?;
    let vocab_size = vocab_size(&config)?;
    let bench_config = BenchConfig {
        prompt_token_count: args.prompt_tokens,
        gen_token_count: args.gen_tokens,
        depth_token_count: args.depth,
        warmup_run_count: args.warmup_runs,
        warmup_ms: args.warmup_ms,
        run_count: args.runs,
        batch_size: args.batch_size,
        seed: args.seed,
    };
    let mut device = open_device()?;
    let max_cache_len = [args.depth, args.prompt_tokens, args.gen_tokens]
        .into_iter()
        .try_fold(0_usize, |max_cache_len, token_count| {
            usize::try_from(token_count)
                .ok()
                .and_then(|token_count| max_cache_len.checked_add(token_count))
        })
        .ok_or_else(|| {
            error!(
                message = "Benchmark token counts overflow usize.",
                ?bench_config
            );
            CliError::SeqLenOverflow
        })?;
    let device_name = device.device_name();
    info!(message = "Selected the device.", %device_name);
    let llama = load_llama(&mut device, &args.model, &config, &weights, max_cache_len)?;
    let report = bench(&mut device, &llama, vocab_size, &bench_config).map_err(|error| {
        error!(message = "Benchmark failed.", model = %args.model, %error);
        CliError::Bench
    })?;
    let record = BenchRecord {
        metadata: Metadata {
            device_name,
            ..metadata
        },
        report,
    };
    let report_path = write_report(&args.out, BENCH_REPORT_FILE, &record)?;
    info!(
        message = "Benchmarked the model.",
        device = DEVICE,
        device_name = %record.metadata.device_name,
        git_commit = ?record.metadata.git_commit,
        mode = ?record.report.mode,
        llama_cpp_avg_ts = record.report.metrics.llama_cpp.avg_ts,
        mlx_prompt_tps = ?record.report.metrics.mlx.as_ref().map(|mlx| mlx.prompt_tps),
        mlx_generation_tps = ?record.report.metrics.mlx.as_ref().and_then(|mlx| mlx.generation_tps),
        peak_memory_bytes = ?record.report.peak_memory_bytes,
        peak_memory_source = ?record.report.peak_memory_source,
        report = %report_path.display(),
    );
    Ok(())
}

pub fn eval_command<D: Device>(
    open_device: impl FnOnce() -> Result<D, CliError>,
    args: &EvalArgs,
) -> Result<(), CliError> {
    let test_time = test_time()?;
    let git_commit = read_git_commit();
    let references = args
        .references
        .iter()
        .map(|dir| {
            read_reference(dir).map_err(|error| {
                error!(message = "Reading a reference failed.", dir = %dir.display(), %error);
                CliError::Reference
            })
        })
        .collect::<Result<Vec<Reference>, CliError>>()?;
    if references
        .iter()
        .all(|reference| reference.activations.is_none())
    {
        error!(message = "No reference holds layer activations.");
        return Err(CliError::NoActivations);
    }
    let max_cache_len = references
        .iter()
        .map(|reference| max_cache_len(&reference.generation))
        .try_fold(0, |max_cache_len, cache_len| {
            Some(max_cache_len.max(cache_len?))
        })
        .ok_or_else(|| {
            error!(message = "Prompt and generated token counts overflow usize.");
            CliError::SeqLenOverflow
        })?;
    let baseline = baselines(&references).map_err(|error| {
        error!(message = "Comparing the references failed.", %error);
        CliError::Eval
    })?;
    let (config, weights) = open_snapshot(&args.model)?;
    let vocab_size = vocab_size(&config)?;
    references
        .iter()
        .flat_map(|reference| &reference.generation.prompts)
        .try_for_each(|prompt| check_token_ids(&prompt.prompt_ids, vocab_size))?;
    let metadata = metadata(&args.model, &weights, test_time, git_commit)?;
    let mut device = open_device()?;
    let device_name = device.device_name();
    info!(message = "Selected the device.", %device_name);
    let llama = load_llama(&mut device, &args.model, &config, &weights, max_cache_len)?;
    let modules = llama.named_modules();
    let reference_reports = references
        .iter()
        .map(|reference| {
            eval_reference(
                &mut device,
                &llama,
                &modules,
                module_atol,
                config.eos_token_id(),
                reference,
            )
            .map_err(|error| {
                error!(
                    message = "Evaluation failed.",
                    model = %args.model,
                    dir = %reference.dir.display(),
                    %error,
                );
                CliError::Eval
            })
        })
        .collect::<Result<Vec<ReferenceReport>, CliError>>()?;
    let record = EvalRecord {
        metadata: Metadata {
            device_name,
            ..metadata
        },
        references: reference_reports,
        baseline,
    };
    let report_path = write_report(&args.out, EVAL_REPORT_FILE, &record)?;
    let failed_engines: Vec<&str> = record
        .references
        .iter()
        .filter(|reference| reference.is_failure())
        .map(|reference| reference.engine.as_str())
        .collect();
    info!(
        message = "Evaluated the model against the references.",
        device = DEVICE,
        device_name = %record.metadata.device_name,
        reference_count = record.references.len(),
        ?failed_engines,
        report = %report_path.display(),
    );
    if !failed_engines.is_empty() {
        return Err(CliError::EvalMismatch);
    }
    Ok(())
}

fn open_snapshot(model: &ModelId) -> Result<(LlamaConfig, Weights), CliError> {
    let resolve_error = |error| {
        error!(
            message = "Model cannot be resolved from the Hugging Face cache.",
            %model,
            %error,
        );
        CliError::Resolve
    };
    let directory = HFClientSync::new()
        .map_err(resolve_error)?
        .model(model.owner.as_str(), model.name.as_str())
        .snapshot_download()
        .local_files_only(true)
        .send()
        .map_err(resolve_error)?;
    let config = LlamaConfig::open(&directory).map_err(|error| {
        error!(
            message = "Opening the model config failed.",
            directory = %directory.display(),
            %error,
        );
        CliError::Config
    })?;
    let weights = Weights::open(&directory).map_err(|error| {
        error!(
            message = "Opening the model weights failed.",
            directory = %directory.display(),
            %error,
        );
        CliError::Weights
    })?;
    Ok((config, weights))
}

fn vocab_size(config: &LlamaConfig) -> Result<NonZeroU32, CliError> {
    u32::try_from(config.vocab_size())
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| {
            error!(
                message = "Vocabulary size is zero or does not fit in u32.",
                vocab_size = config.vocab_size()
            );
            CliError::VocabSize
        })
}

fn check_token_ids(
    token_ids: &[u32],
    vocab_size: NonZeroU32,
) -> Result<(), CliError> {
    let Some(&token_id) = token_ids
        .iter()
        .find(|&&token_id| token_id >= vocab_size.get())
    else {
        return Ok(());
    };
    error!(
        message = "Token id is outside the vocabulary.",
        token_id,
        vocab_size = vocab_size.get()
    );
    Err(CliError::TokenIdOutOfVocab)
}

fn load_llama<D: Device>(
    device: &mut D,
    model: &ModelId,
    config: &LlamaConfig,
    weights: &Weights,
    max_cache_len: usize,
) -> Result<LlamaForCausalLm<D>, CliError> {
    info!(message = "Loading the model.", %model);
    let load_start = Instant::now();
    let llama =
        LlamaForCausalLm::load(device, config, weights, max_cache_len).map_err(|error| {
            error!(message = "Loading the model failed.", %model, max_cache_len, %error);
            CliError::Load
        })?;
    info!(
        message = "Loaded the model.",
        %model,
        max_cache_len,
        load_ms = %load_start.elapsed().as_millis(),
    );
    Ok(llama)
}

fn metadata(
    model: &ModelId,
    weights: &Weights,
    test_time: String,
    git_commit: Option<String>,
) -> Result<Metadata, CliError> {
    let size_error = |error| {
        error!(message = "Measuring the model weights failed.", %model, %error);
        CliError::WeightsSize
    };
    Ok(Metadata {
        git_commit,
        test_time,
        device: DEVICE.to_owned(),
        device_name: String::new(),
        model_id: model.to_string(),
        model_bytes: weights.nbytes().map_err(size_error)?,
        model_parameter_count: weights.numel().map_err(size_error)?,
    })
}

fn test_time() -> Result<String, CliError> {
    let since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            error!(message = "System clock is before the Unix epoch.", %error);
            CliError::Timestamp
        })?;
    rfc3339_utc(since_epoch.as_secs()).ok_or_else(|| {
        error!(
            message = "System clock cannot be formatted as an RFC 3339 time.",
            unix_s = since_epoch.as_secs()
        );
        CliError::Timestamp
    })
}

fn read_git_commit() -> Option<String> {
    let output = match std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            warn!(message = "Running git failed; the commit is not recorded.", %error);
            return None;
        }
    };
    let commit = output
        .status
        .success()
        .then(|| git_commit(&output.stdout))
        .flatten();
    if commit.is_none() {
        warn!(
            message = "Git returned no commit; the commit is not recorded.",
            status = %output.status,
        );
    }
    commit
}

fn write_report(
    out: &Path,
    file_name: &str,
    record: &impl Serialize,
) -> Result<PathBuf, CliError> {
    let report_path = out.join(file_name);
    let write_error = |error: &dyn std::error::Error| {
        error!(
            message = "Writing the report failed.",
            path = %report_path.display(),
            %error,
        );
        CliError::Report
    };
    std::fs::create_dir_all(out).map_err(|error| write_error(&error))?;
    let mut writer =
        BufWriter::new(File::create(&report_path).map_err(|error| write_error(&error))?);
    serde_json::to_writer_pretty(&mut writer, record).map_err(|error| write_error(&error))?;
    writeln!(writer)
        .and_then(|()| writer.flush())
        .map_err(|error| write_error(&error))?;
    Ok(report_path)
}

fn write_token_ids(
    out: &mut impl Write,
    token_ids: &[u32],
) -> std::io::Result<()> {
    let token_ids = token_ids
        .iter()
        .map(u32::to_string)
        .collect::<Vec<String>>()
        .join(",");
    writeln!(out, "[{token_ids}]")
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use crate::cli::command::{
        check_token_ids,
        write_token_ids,
    };
    use crate::cli::error::CliError;

    #[test]
    fn check_token_ids_rejects_ids_outside_the_vocabulary() {
        let vocab_size = NonZeroU32::new(8).expect("8 is not zero");
        assert_eq!(
            [
                check_token_ids(&[0, 7], vocab_size),
                check_token_ids(&[0, 8], vocab_size),
                check_token_ids(&[u32::MAX], vocab_size),
            ],
            [
                Ok(()),
                Err(CliError::TokenIdOutOfVocab),
                Err(CliError::TokenIdOutOfVocab),
            ],
            "the first and last ids, the vocabulary size, then u32::MAX"
        );
    }

    #[test]
    fn write_token_ids_prints_a_json_array() {
        let mut out = Vec::new();
        write_token_ids(&mut out, &[1, 2, 3]).expect("writing to a vector succeeds");
        assert_eq!(
            String::from_utf8(out),
            Ok("[1,2,3]\n".to_owned()),
            "token IDs as a JSON array"
        );
    }
}
