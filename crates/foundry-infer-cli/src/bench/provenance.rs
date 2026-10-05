use std::env;
use std::path::Path;
use std::process::Command;

use crate::bench::report::{
    Available,
    BuildProvenance,
    Environment,
    RepositoryProvenance,
    RepositoryState,
};

const BUILD_INFO: &str = include_str!(concat!(env!("OUT_DIR"), "/build-info.json"));
const SYSCTL: &str = "/usr/sbin/sysctl";
const SW_VERS: &str = "/usr/bin/sw_vers";

pub(crate) fn build() -> Result<BuildProvenance, serde_json::Error> {
    serde_json::from_str(BUILD_INFO)
}

fn output(
    program: &str,
    arguments: &[&str],
    directory: Option<&Path>,
) -> Result<String, String> {
    let mut command = Command::new(program);
    command.args(arguments);
    if let Some(directory) = directory {
        command.current_dir(directory);
    }
    let output = command
        .output()
        .map_err(|error| format!("cannot run {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} {} failed: {}",
            arguments.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let text = String::from_utf8(output.stdout)
        .map_err(|_| format!("{program} printed non-UTF-8 output"))?;
    let text = text.trim();
    if text.is_empty() {
        Err(format!("{program} {} printed nothing", arguments.join(" ")))
    } else {
        Ok(text.to_owned())
    }
}

pub(crate) fn parse_status(status: &str) -> RepositoryState {
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
    RepositoryState {
        dirty: untracked > 0 || modified > 0,
        tracked_modifications: modified,
        untracked_files: untracked,
    }
}

pub(crate) fn repository(directory: Option<&Path>) -> RepositoryProvenance {
    let Some(directory) = directory else {
        let reason = "the working directory is unavailable";
        return RepositoryProvenance {
            directory: Available::unknown(reason),
            commit: Available::unknown(reason),
            repository_state: Available::unknown(reason),
        };
    };
    let root = output("git", &["rev-parse", "--show-toplevel"], Some(directory));
    let commit = output("git", &["rev-parse", "HEAD"], Some(directory));
    let state = match output(
        "git",
        &["status", "--porcelain=v1", "--untracked-files=normal"],
        Some(directory),
    ) {
        Ok(status) => Ok(parse_status(&status)),
        Err(error) if error.ends_with("printed nothing") => Ok(parse_status("")),
        Err(error) => Err(error),
    };
    RepositoryProvenance {
        directory: Available::from_result(root),
        commit: Available::from_result(commit),
        repository_state: Available::from_result(state),
    }
}

fn sysctl(name: &str) -> Result<String, String> { output(SYSCTL, &["-n", name], None) }

fn sw_vers(flag: &str) -> Result<String, String> { output(SW_VERS, &[flag], None) }

pub(crate) fn environment(gpu: Result<String, String>) -> Environment {
    let memory = sysctl("hw.memsize").and_then(|bytes| {
        bytes
            .parse::<u64>()
            .map_err(|error| format!("hw.memsize is not a byte count: {error}"))
    });
    Environment {
        machine_model: Available::from_result(sysctl("hw.model")),
        cpu: Available::from_result(sysctl("machdep.cpu.brand_string")),
        gpu: Available::from_result(gpu),
        physical_memory_bytes: Available::from_result(memory),
        os_name: Available::from_result(sw_vers("-productName")),
        os_version: Available::from_result(sw_vers("-productVersion")),
        os_build: Available::from_result(sw_vers("-buildVersion")),
        architecture: env::consts::ARCH.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build,
        parse_status,
        repository,
    };
    use crate::bench::report::RepositoryState;

    #[test]
    fn build_provenance_is_embedded() -> Result<(), Box<dyn std::error::Error>> {
        let provenance = build()?;
        assert_eq!(
            provenance.foundry_version,
            env!("CARGO_PKG_VERSION"),
            "the package version is embedded"
        );
        assert!(
            provenance.target.value.is_some() && provenance.profile.value.is_some(),
            "Cargo describes the target and profile"
        );
        assert!(
            provenance.commit.value.is_some() || provenance.commit.reason.is_some(),
            "a missing commit is explained"
        );
        Ok(())
    }

    #[test]
    fn porcelain_status_counts_modifications_and_untracked_files() {
        assert_eq!(
            parse_status(""),
            RepositoryState {
                dirty: false,
                tracked_modifications: 0,
                untracked_files: 0,
            },
            "a clean tree"
        );
        assert_eq!(
            parse_status(" M src/a.rs\nA  src/b.rs\n?? notes.txt\n?? scratch/\n"),
            RepositoryState {
                dirty: true,
                tracked_modifications: 2,
                untracked_files: 2,
            },
            "modified, staged and untracked entries"
        );
    }

    #[test]
    fn a_directory_outside_git_is_reported_unavailable() {
        let provenance = repository(Some(std::path::Path::new("/")));
        assert!(
            provenance.commit.value.is_none() && provenance.commit.reason.is_some(),
            "the root directory has no commit"
        );
        let provenance = repository(None);
        assert!(
            provenance.directory.reason.is_some(),
            "a missing directory is explained"
        );
    }
}
