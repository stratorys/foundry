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
