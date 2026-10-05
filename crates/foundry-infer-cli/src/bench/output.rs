use std::fs::{
    self,
    OpenOptions,
};
use std::io::{
    BufWriter,
    ErrorKind,
    Write,
};
use std::path::{
    Path,
    PathBuf,
};

use serde::Serialize;

use crate::bench::error::BenchError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Destination {
    requested: PathBuf,
    resolved: PathBuf,
}

impl Destination {
    pub(crate) fn new(requested: &Path) -> Result<Self, BenchError> {
        let name = requested
            .file_name()
            .ok_or_else(|| BenchError::OutputName {
                path: requested.to_path_buf(),
            })?;
        let parent = match requested.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent,
            Some(_) | None => Path::new("."),
        };
        let directory = parent
            .canonicalize()
            .map_err(|source| BenchError::OutputDirectory {
                path: requested.to_path_buf(),
                source,
            })?;
        if !directory.is_dir() {
            return Err(BenchError::OutputDirectory {
                path: requested.to_path_buf(),
                source: ErrorKind::NotADirectory.into(),
            });
        }
        let resolved = directory.join(name);
        if fs::symlink_metadata(&resolved).is_ok() {
            return Err(BenchError::OutputExists {
                path: requested.to_path_buf(),
            });
        }
        Ok(Self {
            requested: requested.to_path_buf(),
            resolved,
        })
    }

    pub(crate) fn requested(&self) -> &Path { &self.requested }

    pub(crate) fn file_name(&self) -> String {
        self.resolved
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    pub(crate) fn write<T: Serialize>(
        &self,
        value: &T,
        pretty: bool,
    ) -> Result<(), BenchError> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.resolved)
            .map_err(|source| {
                if source.kind() == ErrorKind::AlreadyExists {
                    BenchError::OutputExists {
                        path: self.requested.clone(),
                    }
                } else {
                    BenchError::OutputWrite {
                        path: self.requested.clone(),
                        source,
                    }
                }
            })?;
        let mut writer = BufWriter::new(file);
        let serialized = if pretty {
            serde_json::to_writer_pretty(&mut writer, value)
        } else {
            serde_json::to_writer(&mut writer, value)
        };
        serialized.map_err(|source| BenchError::OutputSerialize {
            path: self.requested.clone(),
            source,
        })?;
        writer
            .write_all(b"\n")
            .and_then(|()| writer.flush())
            .map_err(|source| BenchError::OutputWrite {
                path: self.requested.clone(),
                source,
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Outputs {
    pub(crate) report: Option<Destination>,
    pub(crate) trace: Option<Destination>,
}

impl Outputs {
    pub(crate) fn new(
        report: Option<&Path>,
        trace: Option<&Path>,
    ) -> Result<Self, BenchError> {
        let report = report.map(Destination::new).transpose()?;
        let trace = trace.map(Destination::new).transpose()?;
        if let (Some(report), Some(trace)) = (&report, &trace) {
            if report.resolved == trace.resolved {
                return Err(BenchError::OutputConflict {
                    path: trace.requested.clone(),
                });
            }
        }
        if trace.is_some() && report.is_none() {
            return Err(BenchError::TraceWithoutReport);
        }
        Ok(Self {
            report,
            trace,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::fs;
    use std::path::{
        Path,
        PathBuf,
    };

    use super::{
        Destination,
        Outputs,
    };

    fn directory(name: &str) -> Result<PathBuf, Box<dyn Error>> {
        let directory = PathBuf::from(env!("OUT_DIR"))
            .join("output-tests")
            .join(name);
        if directory.exists() {
            fs::remove_dir_all(&directory)?;
        }
        fs::create_dir_all(&directory)?;
        Ok(directory)
    }

    fn message<T>(result: Result<T, super::BenchError>) -> String {
        result
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default()
    }

    #[test]
    fn new_files_are_written_once() -> Result<(), Box<dyn Error>> {
        let directory = directory("written")?;
        let path = directory.join("report.json");
        let destination = Destination::new(&path)?;
        destination.write(&serde_json::json!({ "a": 1 }), true)?;
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&fs::read_to_string(&path)?)?,
            serde_json::json!({ "a": 1 }),
            "the value is written"
        );
        let error = message(destination.write(&serde_json::json!({ "a": 2 }), true));
        assert!(
            error.contains("already exists"),
            "no overwrite, got {error:?}"
        );
        assert!(
            message(Destination::new(&path)).contains("already exists"),
            "existing destinations are rejected before writing"
        );
        Ok(())
    }

    #[test]
    fn invalid_destinations_are_rejected() -> Result<(), Box<dyn Error>> {
        let directory = directory("invalid")?;
        let missing = directory.join("missing").join("report.json");
        assert!(
            message(Destination::new(&missing)).contains("directory"),
            "a missing parent directory is rejected"
        );
        let file = directory.join("file");
        fs::write(&file, "x")?;
        assert!(
            !message(Destination::new(&file.join("report.json"))).is_empty(),
            "a file is not a directory"
        );
        assert!(
            message(Destination::new(Path::new("/"))).contains("file name"),
            "a destination needs a file name"
        );
        Ok(())
    }

    #[test]
    fn report_and_trace_must_differ() -> Result<(), Box<dyn Error>> {
        let directory = directory("conflict")?;
        let report = directory.join("run.json");
        let alias = directory.join(".").join("run.json");
        assert!(
            message(Outputs::new(Some(&report), Some(&alias))).contains("same file"),
            "the same file through another spelling is rejected"
        );
        assert!(
            message(Outputs::new(None, Some(&report))).contains("--report"),
            "a trace needs a report"
        );
        let outputs = Outputs::new(Some(&report), Some(&directory.join("run.trace.json")))?;
        assert!(
            outputs.report.is_some() && outputs.trace.is_some(),
            "distinct destinations are accepted"
        );
        Ok(())
    }
}
