use std::process::{
    Command,
    Output,
};

fn foundry(argument: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_foundry"))
        .arg(argument)
        .output()
        .expect("the foundry binary should run")
}

#[test]
fn prints_the_version() {
    let output = foundry("--version");
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
    let output = foundry("--help");
    assert!(output.status.success(), "--help should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Usage: foundry"),
        "--help should print the usage line, got:\n{stdout}"
    );
}
