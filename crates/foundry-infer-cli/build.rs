use std::path::{
    Path,
    PathBuf,
};
use std::process::Command;
use std::{
    env,
    fs,
};

use serde_json::{
    Value,
    json,
};

const SOURCES: [&str; 10] = [
    "crates",
    "scripts",
    "docs",
    "Cargo.toml",
    "Cargo.lock",
    "README.md",
    "rustfmt.toml",
    "clippy.toml",
    ".gitignore",
    "LICENSE",
];

const GIT_STATE: [&str; 4] = ["HEAD", "index", "refs", "packed-refs"];

fn available(value: Result<Value, String>) -> Value {
    match value {
        Ok(value) => json!({ "value": value }),
        Err(reason) => json!({ "value": null, "reason": reason }),
    }
}

fn run(
    program: &str,
    arguments: &[&str],
    directory: &Path,
) -> Result<String, String> {
    let output = Command::new(program)
        .args(arguments)
        .current_dir(directory)
        .output()
        .map_err(|error| format!("cannot run {program}: {error}"))?;
    if output.status.success() {
        String::from_utf8(output.stdout)
            .map_err(|_| format!("{program} {} printed non-UTF-8 output", arguments.join(" ")))
    } else {
        Err(format!(
            "{program} {} failed: {}",
            arguments.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn dirty(workspace: &Path) -> Result<Value, String> {
    let status = run(
        "git",
        &["status", "--porcelain=v1", "--untracked-files=normal"],
        workspace,
    )?;
    let (untracked, modified) = status.lines().filter(|line| !line.is_empty()).fold(
        (0_u64, 0_u64),
        |(untracked, modified), line| {
            if line.starts_with("??") {
                (untracked.saturating_add(1), modified)
            } else {
                (untracked, modified.saturating_add(1))
            }
        },
    );
    Ok(json!({
        "dirty": untracked > 0 || modified > 0,
        "tracked_modifications": modified,
        "untracked_files": untracked,
    }))
}

fn variable(name: &str) -> Result<Value, String> {
    env::var(name)
        .map(Value::String)
        .map_err(|_| format!("Cargo did not set {name}"))
}

fn features() -> Vec<String> {
    let mut features: Vec<String> = env::vars_os()
        .filter_map(|(name, _)| {
            name.to_str()?
                .strip_prefix("CARGO_FEATURE_")
                .map(|feature| feature.to_lowercase().replace('_', "-"))
        })
        .collect();
    features.sort_unstable();
    features
}

fn rustflags() -> Vec<String> {
    env::var("CARGO_ENCODED_RUSTFLAGS")
        .map(|flags| {
            flags
                .split('\u{1f}')
                .filter(|flag| !flag.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn watch(workspace: &Path) {
    for source in SOURCES {
        println!(
            "cargo:rerun-if-changed={}",
            workspace.join(source).display()
        );
    }
    if let Ok(git_dir) = run("git", &["rev-parse", "--absolute-git-dir"], workspace) {
        let git_dir = PathBuf::from(git_dir.trim());
        for state in GIT_STATE {
            println!("cargo:rerun-if-changed={}", git_dir.join(state).display());
        }
    }
    for name in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
}

fn main() -> Result<(), String> {
    let manifest = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").ok_or("Cargo did not set CARGO_MANIFEST_DIR")?,
    );
    let workspace = manifest
        .join("../..")
        .canonicalize()
        .map_err(|error| format!("cannot resolve the workspace root: {error}"))?;
    watch(&workspace);
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let info = json!({
        "foundry_version": env::var("CARGO_PKG_VERSION").unwrap_or_default(),
        "commit": available(
            run("git", &["rev-parse", "HEAD"], &workspace)
                .map(|commit| Value::String(commit.trim().to_owned()))
        ),
        "repository_state": available(dirty(&workspace)),
        "rustc": available(
            run(&rustc, &["-vV"], &workspace)
                .map(|version| Value::String(version.trim().to_owned()))
        ),
        "target": available(variable("TARGET")),
        "host": available(variable("HOST")),
        "profile": available(variable("PROFILE")),
        "opt_level": available(variable("OPT_LEVEL")),
        "debug": available(variable("DEBUG")),
        "debug_assertions": env::var_os("CARGO_CFG_DEBUG_ASSERTIONS").is_some(),
        "panic": available(variable("CARGO_CFG_PANIC")),
        "features": features(),
        "rustflags": rustflags(),
    });
    let out = PathBuf::from(env::var_os("OUT_DIR").ok_or("Cargo did not set OUT_DIR")?);
    fs::write(out.join("build-info.json"), info.to_string())
        .map_err(|error| format!("cannot write the build provenance: {error}"))
}
