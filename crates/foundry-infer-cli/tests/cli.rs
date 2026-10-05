use std::error::Error;
use std::num::NonZeroU32;
use std::path::{
    Path,
    PathBuf,
};
use std::process::{
    Command,
    Output,
};
use std::time::Duration;
use std::{
    fs,
    io,
};

use foundry_infer_core::{
    Alignment,
    ByteSize,
    MemoryBudget,
    MemorySpace,
};
use foundry_infer_plan::{
    ExecutionPlan,
    synthetic_chain_graph,
    synthetic_chain_plan,
};

fn foundry(arguments: &[&str]) -> io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_foundry"))
        .args(arguments)
        .output()
}

fn fresh_path(name: &str) -> Result<PathBuf, Box<dyn Error>> {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    if path.exists() {
        fs::remove_file(&path)?;
    }
    Ok(path)
}

fn dump(path: &Path) -> Result<Output, Box<dyn Error>> {
    let path = path.to_str().ok_or("the temporary path is not UTF-8")?;
    Ok(foundry(&["plan", "--dump", path])?)
}

#[test]
fn prints_the_version() -> Result<(), Box<dyn Error>> {
    let output = foundry(&["--version"])?;
    assert!(output.status.success(), "--version should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.trim(),
        format!("foundry {}", env!("CARGO_PKG_VERSION")),
        "--version should print the binary name and the package version"
    );
    Ok(())
}

#[test]
fn prints_the_help() -> Result<(), Box<dyn Error>> {
    let output = foundry(&["--help"])?;
    assert!(output.status.success(), "--help should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Usage: foundry"),
        "--help should print the usage line, got:\n{stdout}"
    );
    Ok(())
}

#[test]
fn exports_a_valid_triple_buffer_plan() -> Result<(), Box<dyn Error>> {
    let path = fresh_path("exported-plan.json")?;
    let output = dump(&path)?;
    assert!(
        output.status.success(),
        "plan --dump should succeed, stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan = ExecutionPlan::from_json(&fs::read_to_string(&path)?)?;
    let graph = synthetic_chain_graph(
        48,
        ByteSize::from_mib(450)?,
        Some(Duration::from_millis(60)),
    )?;
    let budget = MemoryBudget::new(MemorySpace::Device, ByteSize::from_gib(4)?);
    plan.validate(&graph, budget)?;
    let slots = NonZeroU32::new(3).ok_or("three slots")?;
    let expected = synthetic_chain_plan(&graph, slots, Alignment::new(256)?)?;
    assert_eq!(plan, expected, "the artifact is the triple-buffer plan");
    fs::remove_file(&path)?;
    Ok(())
}

#[test]
fn refuses_to_overwrite_an_existing_file() -> Result<(), Box<dyn Error>> {
    let path = fresh_path("existing-plan.json")?;
    fs::write(&path, "keep me")?;
    let output = dump(&path)?;
    assert!(!output.status.success(), "plan --dump should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("already exists"),
        "the error should name the conflict, got:\n{stderr}"
    );
    assert_eq!(
        fs::read_to_string(&path)?,
        "keep me",
        "the existing file is unchanged"
    );
    fs::remove_file(&path)?;
    Ok(())
}

fn rejected(arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = foundry(arguments)?;
    assert!(
        !output.status.success(),
        "{arguments:?} should fail, stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(String::from_utf8_lossy(&output.stderr).into_owned())
}

#[test]
fn bench_help_lists_the_defaults() -> Result<(), Box<dyn Error>> {
    let output = foundry(&["bench", "--help"])?;
    assert!(output.status.success(), "bench --help should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    for default in [
        "[default: metal]",
        "[default: 48]",
        "[default: 64]",
        "[default: 10]",
        "[default: 2]",
        "[default: 10000]",
        "[default: 0]",
    ] {
        assert!(
            stdout.contains(default),
            "bench --help should show {default}, got:\n{stdout}"
        );
    }
    Ok(())
}

#[test]
fn bench_rejects_zero_sizes_and_iterations() -> Result<(), Box<dyn Error>> {
    for argument in ["--layers", "--weight-mib", "--iterations"] {
        let stderr = rejected(&["bench", argument, "0"])?;
        assert!(
            stderr.contains(argument) && stderr.contains("zero"),
            "{argument} 0 should be rejected as zero, got:\n{stderr}"
        );
    }
    Ok(())
}

#[test]
fn bench_rejects_unknown_backends() -> Result<(), Box<dyn Error>> {
    let stderr = rejected(&["bench", "--backend", "cuda"])?;
    assert!(
        stderr.contains("possible values: metal"),
        "only metal is offered, got:\n{stderr}"
    );
    Ok(())
}

#[test]
fn bench_reports_size_overflow() -> Result<(), Box<dyn Error>> {
    let stderr = rejected(&["bench", "--weight-mib", "18446744073709551615"])?;
    assert!(
        stderr.contains("overflows a 64-bit byte count"),
        "the weight size overflow should be reported, got:\n{stderr}"
    );
    let stderr = rejected(&[
        "bench",
        "--layers",
        "4294967295",
        "--weight-mib",
        "4398046511104",
    ])?;
    assert!(
        stderr.contains("overflow a 64-bit byte count"),
        "the payload size overflow should be reported, got:\n{stderr}"
    );
    Ok(())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
#[test]
fn bench_requires_the_metal_feature() -> Result<(), Box<dyn Error>> {
    let stderr = rejected(&[
        "bench",
        "--warmup",
        "0",
        "--layers",
        "1",
        "--weight-mib",
        "1",
    ])?;
    assert!(
        stderr.contains("requires a macOS build of foundry with `--features metal`"),
        "the missing feature should be explained, got:\n{stderr}"
    );
    Ok(())
}

fn fresh_directory(name: &str) -> Result<PathBuf, Box<dyn Error>> {
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    if directory.exists() {
        fs::remove_dir_all(&directory)?;
    }
    fs::create_dir_all(&directory)?;
    Ok(directory)
}

fn text(path: &Path) -> Result<&str, Box<dyn Error>> {
    Ok(path.to_str().ok_or("the temporary path is not UTF-8")?)
}

#[test]
fn bench_rejects_invalid_statistics_parameters() -> Result<(), Box<dyn Error>> {
    let stderr = rejected(&["bench", "--bootstrap-resamples", "0"])?;
    assert!(
        stderr.contains("--bootstrap-resamples") && stderr.contains("zero"),
        "zero resamples are rejected, got:\n{stderr}"
    );
    let stderr = rejected(&["bench", "--bootstrap-resamples", "2000000"])?;
    assert!(
        stderr.contains("exceeds the supported maximum of 1000000"),
        "too many resamples are rejected, got:\n{stderr}"
    );
    let stderr = rejected(&["bench", "--seed=-1"])?;
    assert!(
        stderr.contains("--seed"),
        "a negative seed is rejected, got:\n{stderr}"
    );
    let stderr = rejected(&["bench", "--seed", "18446744073709551616"])?;
    assert!(
        stderr.contains("--seed"),
        "a seed beyond 64 bits is rejected, got:\n{stderr}"
    );
    Ok(())
}

#[test]
fn bench_trace_requires_a_report() -> Result<(), Box<dyn Error>> {
    let directory = fresh_directory("trace-without-report")?;
    let trace = directory.join("run.trace.json");
    let stderr = rejected(&["bench", "--trace", text(&trace)?])?;
    assert!(
        stderr.contains("--report"),
        "the missing report is named, got:\n{stderr}"
    );
    assert!(!trace.exists(), "nothing is written");
    Ok(())
}

#[test]
fn bench_refuses_existing_or_conflicting_outputs() -> Result<(), Box<dyn Error>> {
    let directory = fresh_directory("bench-outputs")?;
    let existing = directory.join("existing.json");
    fs::write(&existing, "keep me")?;
    let stderr = rejected(&["bench", "--report", text(&existing)?])?;
    assert!(
        stderr.contains("already exists"),
        "an existing report is refused, got:\n{stderr}"
    );
    assert_eq!(
        fs::read_to_string(&existing)?,
        "keep me",
        "the existing file is unchanged"
    );
    let report = directory.join("report.json");
    let stderr = rejected(&[
        "bench",
        "--report",
        text(&report)?,
        "--trace",
        text(&existing)?,
    ])?;
    assert!(
        stderr.contains("already exists"),
        "an existing trace is refused, got:\n{stderr}"
    );
    let stderr = rejected(&[
        "bench",
        "--report",
        text(&report)?,
        "--trace",
        text(&directory.join(".").join("report.json"))?,
    ])?;
    assert!(
        stderr.contains("same file"),
        "identical outputs are refused, got:\n{stderr}"
    );
    let stderr = rejected(&[
        "bench",
        "--report",
        text(&directory.join("missing").join("report.json"))?,
    ])?;
    assert!(
        stderr.contains("directory"),
        "a missing directory is refused, got:\n{stderr}"
    );
    assert!(!report.exists(), "nothing is written");
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn at<'json>(
    json: &'json serde_json::Value,
    pointer: &str,
) -> &'json serde_json::Value {
    json.pointer(pointer).unwrap_or(&serde_json::Value::Null)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn uv_available() -> bool {
    let available = Command::new("uv").arg("--version").output().is_ok();
    assert!(
        available || std::env::var_os("FOUNDRY_REQUIRE_UV").is_none(),
        "uv is required but unavailable"
    );
    available
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_available(output: &Output) -> bool {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let unavailable = stderr.contains("no Metal device is available")
        || stderr.contains("does not support non-uniform");
    assert!(
        !unavailable || std::env::var_os("FOUNDRY_REQUIRE_METAL").is_none(),
        "a Metal device is required, stderr:\n{stderr}"
    );
    !unavailable
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn bench_exports_a_traced_report() -> Result<(), Box<dyn Error>> {
    let directory = fresh_directory("bench-traced")?;
    let report = directory.join("report.json");
    let trace = directory.join("report.trace.json");
    let output = foundry(&[
        "bench",
        "--layers",
        "2",
        "--weight-mib",
        "1",
        "--warmup",
        "1",
        "--iterations",
        "3",
        "--bootstrap-resamples",
        "200",
        "--seed",
        "5",
        "--report",
        text(&report)?,
        "--trace",
        text(&trace)?,
    ])?;
    if !metal_available(&output) {
        return Ok(());
    }
    assert!(
        output.status.success(),
        "the traced run succeeds, stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value = serde_json::from_str(&fs::read_to_string(&report)?)?;
    let field = |pointer: &str| json.pointer(pointer).cloned().unwrap_or_default();
    assert_eq!(field("/schema_version"), 1, "the schema version");
    assert_eq!(field("/run/status"), "succeeded", "the status");
    assert_eq!(field("/run/instrumentation"), "traced", "the mode");
    assert_eq!(field("/invocation/parameters/seed"), 5, "the resolved seed");
    assert_eq!(
        field("/trace/value/file"),
        "report.trace.json",
        "the trace file"
    );
    assert_eq!(
        field("/gpu_timing/value/gate"),
        "passed",
        "valid GPU timestamps"
    );
    let executions = field("/executions");
    let executions = executions.as_array().ok_or("no executions")?;
    assert_eq!(executions.len(), 12, "4 rounds of 3 configurations");
    let order: Vec<(u64, u64)> = executions
        .iter()
        .map(|execution| {
            (
                at(execution, "/sequence").as_u64().unwrap_or(u64::MAX),
                at(execution, "/buffers").as_u64().unwrap_or(u64::MAX),
            )
        })
        .collect();
    let expected: Vec<(u64, u64)> = [1, 2, 3, 2, 3, 1, 3, 1, 2, 1, 2, 3]
        .into_iter()
        .zip(0..)
        .map(|(buffers, sequence)| (sequence, buffers))
        .collect();
    assert_eq!(order, expected, "chronological records in rotated order");
    assert_eq!(
        executions
            .iter()
            .filter(|execution| at(execution, "/phase") == "warmup")
            .count(),
        3,
        "warmup executions are recorded"
    );
    let statistics = field("/statistics/value/configurations");
    let samples: Vec<u64> = statistics
        .as_array()
        .ok_or("no statistics")?
        .iter()
        .map(|row| at(row, "/samples").as_u64().unwrap_or_default())
        .collect();
    assert_eq!(samples, [3, 3, 3], "warmup executions are excluded");
    let memory = field("/memory/value/executions");
    assert!(
        memory
            .as_array()
            .is_some_and(|executions| executions.len() == 12
                && executions
                    .iter()
                    .all(|execution| at(execution, "/live_allocations_at_extraction") == 0)),
        "every traced execution releases its tracked buffers"
    );
    let document: serde_json::Value = serde_json::from_str(&fs::read_to_string(&trace)?)?;
    let events = at(&document, "/traceEvents")
        .as_array()
        .ok_or("no trace events")?;
    let spans = |tid: u64| {
        events
            .iter()
            .filter(|event| {
                let track = at(event, "/tid").as_u64().unwrap_or_default();
                at(event, "/ph") == "X" && (track == tid || track / 100 == tid)
            })
            .count()
    };
    assert_eq!(spans(10), 24, "one GPU copy span per layer and execution");
    assert_eq!(
        spans(11),
        24,
        "one GPU compute span per layer and execution"
    );
    assert!(spans(1) > 0 && spans(2) > 0, "CPU runtime and wait spans");
    if uv_available() {
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/bench_summary.py");
        let recalculated = Command::new("uv")
            .args(["run", "--quiet", "--script"])
            .arg(script)
            .arg(&report)
            .output()?;
        assert!(
            recalculated.status.success(),
            "the offline recalculation matches, stderr:\n{}",
            String::from_utf8_lossy(&recalculated.stderr)
        );
    }
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn bench_keeps_a_failure_report() -> Result<(), Box<dyn Error>> {
    let directory = fresh_directory("bench-failure")?;
    let report = directory.join("failed.json");
    let output = foundry(&[
        "bench",
        "--layers",
        "1",
        "--weight-mib",
        "4096",
        "--report",
        text(&report)?,
    ])?;
    if !metal_available(&output) {
        return Ok(());
    }
    assert!(!output.status.success(), "the run fails");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("failure report written to"),
        "the failure report is announced, got:\n{stderr}"
    );
    let json: serde_json::Value = serde_json::from_str(&fs::read_to_string(&report)?)?;
    assert_eq!(
        at(&json, "/run/status"),
        "failed",
        "the report is not successful"
    );
    assert_eq!(
        at(&json, "/failure/stage"),
        "preparation",
        "the failure names its stage"
    );
    assert!(
        at(&json, "/statistics/value").is_null() && at(&json, "/statistics/reason").is_string(),
        "an incomplete campaign has no comparison"
    );
    Ok(())
}
