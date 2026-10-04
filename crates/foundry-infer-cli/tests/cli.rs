use std::error::Error;
use std::fs;
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

fn foundry(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_foundry"))
        .args(arguments)
        .output()
        .expect("the foundry binary should run")
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
    Ok(foundry(&["plan", "--dump", path]))
}

#[test]
fn prints_the_version() {
    let output = foundry(&["--version"]);
    assert!(output.status.success(), "--version should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.trim(),
        format!("foundry {}", env!("CARGO_PKG_VERSION")),
        "--version should print the binary name and the package version"
    );
}

#[test]
fn prints_the_help() {
    let output = foundry(&["--help"]);
    assert!(output.status.success(), "--help should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Usage: foundry"),
        "--help should print the usage line, got:\n{stdout}"
    );
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
