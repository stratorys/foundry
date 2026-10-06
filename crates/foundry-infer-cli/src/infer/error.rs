use std::error::Error;
use std::path::PathBuf;
use std::{
    fmt,
    io,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
use foundry_infer_llama::LlamaError;

use crate::bench::error::BenchError;

#[derive(Debug)]
pub(crate) enum InferError {
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    MetalUnavailable,
    Output(Box<BenchError>),
    VerifierLaunch {
        program: String,
        source: io::Error,
    },
    Verification {
        status: Option<i32>,
        stderr: String,
    },
    Manifest {
        path: PathBuf,
        reason: String,
    },
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Model(LlamaError),
    TokenCount {
        phase: &'static str,
        index: u32,
        generated: usize,
        expected: usize,
    },
    Statistics(&'static str),
    Reported {
        source: Box<InferError>,
        report: PathBuf,
    },
}

impl fmt::Display for InferError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            Self::MetalUnavailable => write!(
                formatter,
                "infer-bench requires a macOS build of foundry with `--features metal`"
            ),
            Self::Output(error) => error.fmt(formatter),
            Self::VerifierLaunch {
                program,
                source,
            } => write!(
                formatter,
                "cannot run the manifest verifier with {program}: {source}"
            ),
            Self::Verification {
                status,
                stderr,
            } => write!(
                formatter,
                "the manifest verifier failed with status {status:?}: {}",
                stderr.trim()
            ),
            Self::Manifest {
                path,
                reason,
            } => write!(
                formatter,
                "cannot use manifest {}: {reason}",
                path.display()
            ),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Model(error) => error.fmt(formatter),
            Self::TokenCount {
                phase,
                index,
                generated,
                expected,
            } => write!(
                formatter,
                "{phase} execution {index} generated {generated} tokens, expected {expected}"
            ),
            Self::Statistics(reason) => write!(formatter, "cannot summarize executions: {reason}"),
            Self::Reported {
                source,
                report,
            } => write!(formatter, "{source} (report: {})", report.display()),
        }
    }
}

impl Error for InferError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Output(error) => Some(error.as_ref()),
            Self::VerifierLaunch {
                source, ..
            } => Some(source),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Model(error) => Some(error),
            Self::Reported {
                source, ..
            } => Some(source.as_ref()),
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            Self::MetalUnavailable => None,
            Self::Verification {
                ..
            }
            | Self::Manifest {
                ..
            }
            | Self::TokenCount {
                ..
            }
            | Self::Statistics(_) => None,
        }
    }
}

impl From<BenchError> for InferError {
    fn from(error: BenchError) -> Self { Self::Output(Box::new(error)) }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl From<LlamaError> for InferError {
    fn from(error: LlamaError) -> Self { Self::Model(error) }
}
