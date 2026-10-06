use std::process::Command as Process;
use std::time::{
    Instant,
    SystemTime,
};
use std::{
    env,
    fs,
};

use foundry_infer_llama::input::{
    BENCHMARK_INPUT_SHA256,
    BENCHMARK_OUTPUT_TOKENS,
    benchmark_input,
    sha256_hex,
};
use foundry_infer_llama::manifest::{
    Manifest,
    WEIGHTS_FILE,
};
use foundry_infer_llama::metal::LlamaModel;

use crate::bench::output::Destination;
use crate::bench::provenance;
use crate::bench::report::utc_timestamp;
use crate::infer::InferBenchArgs;
use crate::infer::error::InferError;
use crate::infer::report::{
    Checkpoint,
    Command,
    Execution,
    Failure,
    Foundry,
    Input,
    Loading,
    MEASURED_EXECUTIONS,
    Memory,
    Report,
    SCHEMA_VERSION,
    Status,
    Verification,
    WARMUP_EXECUTIONS,
    memory_definitions,
    metrics,
    protocol,
    timing_boundaries,
};
use crate::infer::summary::{
    consistency,
    decode_tokens_per_s,
    statistics,
};

const VERIFIER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/verify-model-manifest.py"
);

fn nanos(start: Instant) -> u64 { u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX) }

fn bytes(value: usize) -> Option<u64> { u64::try_from(value).ok() }

fn now() -> String { utc_timestamp(SystemTime::now()).unwrap_or_default() }

fn initial_report(args: &InferBenchArgs) -> Report {
    let working_directory = env::current_dir().ok();
    Report {
        schema_version: SCHEMA_VERSION,
        engine: "foundry",
        status: Status::Running,
        failure: None,
        started_utc: now(),
        finished_utc: None,
        command: Command {
            argv: env::args_os()
                .map(|argument| argument.to_string_lossy().into_owned())
                .collect(),
            working_directory: working_directory
                .as_ref()
                .map(|directory| directory.display().to_string()),
            snapshot: args.snapshot.display().to_string(),
            manifest: args.manifest.display().to_string(),
            report: args.report.display().to_string(),
        },
        foundry: Foundry {
            build: provenance::build().ok(),
            repository: provenance::repository(working_directory.as_deref()),
        },
        environment: None,
        protocol: protocol(),
        timing_boundaries: timing_boundaries(),
        metrics: metrics(),
        checkpoint: None,
        verification: None,
        input: None,
        loading: None,
        memory: Memory {
            definitions: memory_definitions(),
            ..Memory::default()
        },
        executions: Vec::new(),
        statistics: None,
        consistency: None,
    }
}

fn verify(args: &InferBenchArgs) -> Result<Verification, InferError> {
    let manifest = args.manifest.display().to_string();
    let snapshot = args.snapshot.display().to_string();
    let command = vec![
        "uv".to_owned(),
        "run".to_owned(),
        "--script".to_owned(),
        VERIFIER.to_owned(),
        "--manifest".to_owned(),
        manifest,
        "--snapshot".to_owned(),
        snapshot,
    ];
    let output = Process::new("uv")
        .args(command.iter().skip(1))
        .output()
        .map_err(|source| InferError::VerifierLaunch {
            program: "uv".to_owned(),
            source,
        })?;
    Ok(Verification {
        command,
        returncode: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

fn execute(
    args: &InferBenchArgs,
    report: &mut Report,
    stage: &mut String,
) -> Result<(), InferError> {
    "identity".clone_into(stage);
    let manifest_text =
        fs::read_to_string(&args.manifest).map_err(|error| InferError::Manifest {
            path: args.manifest.clone(),
            reason: error.to_string(),
        })?;
    let manifest = Manifest::parse(&manifest_text).map_err(|error| InferError::Manifest {
        path: args.manifest.clone(),
        reason: error.to_string(),
    })?;
    let weights = manifest.file(WEIGHTS_FILE);
    report.checkpoint = Some(Checkpoint {
        repository: manifest.repository.clone(),
        revision: manifest.revision.clone(),
        manifest_sha256: sha256_hex(manifest_text.as_bytes()),
        weights_file_bytes: weights.map(|file| file.size_bytes),
        weights_file_sha256_manifest: weights.map(|file| file.sha256.clone()),
    });

    "verification".clone_into(stage);
    let verification = verify(args)?;
    let passed = verification.returncode == Some(0);
    let failure = InferError::Verification {
        status: verification.returncode,
        stderr: verification.stderr.clone(),
    };
    report.verification = Some(verification);
    if !passed {
        return Err(failure);
    }

    "input".clone_into(stage);
    let input = benchmark_input().map_err(|error| InferError::Manifest {
        path: args.manifest.clone(),
        reason: error.to_string(),
    })?;
    report.input = Some(Input {
        token_count: input.len(),
        sha256: BENCHMARK_INPUT_SHA256.to_owned(),
        token_ids: input.clone(),
    });

    "load".clone_into(stage);
    let started = Instant::now();
    let model = LlamaModel::load(&args.snapshot, &args.manifest)?;
    let load_ns = nanos(started);
    report.loading = Some(Loading {
        device: model.device_name(),
        load_ns,
        definition: "Wall time of LlamaModel::load: manifest and header validation, Metal device \
                     and kernel compilation, and reading every weight into the resident buffer.",
    });
    report.environment = Some(provenance::environment(Ok(model.device_name())));
    let memory = model.memory();
    report.memory.weight_tensor_bytes = Some(memory.weight_tensor_bytes);
    report.memory.weight_buffer_allocated_bytes = bytes(memory.weight_buffer_bytes);
    report.memory.rope_buffer_allocated_bytes = bytes(memory.rope_buffer_bytes);
    report.memory.device_allocated_bytes_after_load = bytes(model.device_allocated_bytes());

    let schedule = (0..WARMUP_EXECUTIONS)
        .map(|index| ("warmup", index))
        .chain((0..MEASURED_EXECUTIONS).map(|index| ("measured", index)));
    for (phase, index) in schedule {
        *stage = format!("{phase}[{index}]");
        let mut session = model.session()?;
        let session_memory = session.memory();
        report.memory.kv_cache_allocated_bytes = bytes(session_memory.kv_cache_bytes);
        report.memory.activation_allocated_bytes = bytes(session_memory.activation_bytes);
        report.memory.device_allocated_bytes_after_session = bytes(model.device_allocated_bytes());
        session.prepare(&input)?;
        model.synchronize()?;
        let start = Instant::now();
        let mut token_available_ns = Vec::with_capacity(BENCHMARK_OUTPUT_TOKENS);
        let token_ids = session.generate(BENCHMARK_OUTPUT_TOKENS, |_, _| {
            token_available_ns.push(nanos(start));
        })?;
        model.synchronize()?;
        let elapsed_ns = nanos(start);
        let device_allocated_bytes = bytes(model.device_allocated_bytes()).unwrap_or(u64::MAX);
        drop(session);
        report.memory.device_allocated_bytes_max_after_execution = Some(
            report
                .memory
                .device_allocated_bytes_max_after_execution
                .map_or(device_allocated_bytes, |max| {
                    max.max(device_allocated_bytes)
                }),
        );
        let generated = token_ids.len();
        report.executions.push(Execution {
            phase,
            index,
            generated_count: generated,
            token_ids,
            elapsed_ns,
            first_token_ns: token_available_ns.first().copied(),
            decode_tokens_per_s: decode_tokens_per_s(&token_available_ns),
            token_available_ns,
            device_allocated_bytes,
        });
        if generated != BENCHMARK_OUTPUT_TOKENS {
            return Err(InferError::TokenCount {
                phase,
                index,
                generated,
                expected: BENCHMARK_OUTPUT_TOKENS,
            });
        }
    }

    "statistics".clone_into(stage);
    report.statistics = Some(statistics(&report.executions)?);
    Ok(())
}

pub(crate) fn run(args: &InferBenchArgs) -> Result<(), InferError> {
    let destination = Destination::new(&args.report)?;
    let mut report = initial_report(args);
    let mut stage = "start".to_owned();
    let outcome = execute(args, &mut report, &mut stage);
    report.consistency = Some(consistency(&report.executions));
    report.finished_utc = Some(now());
    match &outcome {
        Ok(()) => report.status = Status::Succeeded,
        Err(error) => {
            report.status = Status::Failed;
            report.statistics = None;
            report.failure = Some(Failure {
                stage,
                message: error.to_string(),
            });
        }
    }
    destination.write(&report, true)?;
    if let (Ok(()), Some(statistics)) = (&outcome, &report.statistics) {
        let median = |name| {
            statistics
                .get(name)
                .map_or(f64::NAN, |summary| summary.median)
        };
        println!(
            "OK: {} executions ({WARMUP_EXECUTIONS} warmup), elapsed median {:.3} ms, first token \
             median {:.3} ms, decode median {:.1} tok/s, consistent outputs: {:?}",
            report.executions.len(),
            median("elapsed_ns") / 1e6,
            median("first_token_ns") / 1e6,
            median("decode_tokens_per_s"),
            report
                .consistency
                .as_ref()
                .and_then(|result| result.consistent),
        );
        println!("report: {}", args.report.display());
    }
    outcome.map_err(|source| InferError::Reported {
        source: Box::new(source),
        report: args.report.clone(),
    })
}
