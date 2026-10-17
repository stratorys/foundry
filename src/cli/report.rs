use std::fs::File;
use std::io::{
    BufWriter,
    Write,
};
use std::path::{
    Path,
    PathBuf,
};
use std::time::{
    SystemTime,
    UNIX_EPOCH,
};

use foundry::bench::{
    Metadata,
    Report,
    git_commit,
    rfc3339_utc,
};
use foundry::eval::{
    BaselineReport,
    ReferenceReport,
};
use foundry::weights::Weights;
use serde::Serialize;
use tracing::{
    error,
    warn,
};

use crate::cli::ModelId;
use crate::cli::error::CliError;
use crate::cli::platform::DEVICE;

pub const BENCH_REPORT_FILE: &str = "report.json";
pub const EVAL_REPORT_FILE: &str = "eval.json";

#[derive(Debug, Serialize)]
pub struct BenchRecord {
    pub metadata: Metadata,
    #[serde(flatten)]
    pub report: Report,
}

#[derive(Debug, Serialize)]
pub struct EvalRecord {
    pub metadata: Metadata,
    pub references: Vec<ReferenceReport>,
    pub baseline: Vec<BaselineReport>,
}

pub struct ReportStamp {
    test_time: String,
    git_commit: Option<String>,
}

pub struct ModelSize {
    bytes: u64,
    parameter_count: u64,
}

impl ReportStamp {
    pub fn capture() -> Result<Self, CliError> {
        let test_time = test_time()?;
        let git_commit = read_git_commit();
        Ok(Self {
            test_time,
            git_commit,
        })
    }
}

impl ModelSize {
    pub fn measure(
        model: &ModelId,
        weights: &Weights,
    ) -> Result<Self, CliError> {
        let bytes = model_bytes(model, weights)?;
        let parameter_count = parameter_count(model, weights)?;
        Ok(Self {
            bytes,
            parameter_count,
        })
    }
}

pub fn metadata(
    stamp: ReportStamp,
    model: &ModelId,
    size: ModelSize,
    device_name: String,
) -> Metadata {
    Metadata {
        git_commit: stamp.git_commit,
        test_time: stamp.test_time,
        device: DEVICE.to_owned(),
        device_name,
        model_id: model.to_string(),
        model_bytes: size.bytes,
        model_parameter_count: size.parameter_count,
    }
}

pub fn write_report(
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

fn model_bytes<S: AsRef<[u8]>>(
    model: &ModelId,
    weights: &Weights<S>,
) -> Result<u64, CliError> {
    let bytes = weights.tensors().try_fold(0_u64, |bytes, view| {
        u64::try_from(view.bytes.len())
            .ok()
            .and_then(|tensor_bytes| bytes.checked_add(tensor_bytes))
            .ok_or_else(|| {
                error!(message = "Total tensor byte count overflows u64.", %model);
                CliError::TotalByteCountOverflow
            })
    })?;
    if let Some(total_size) = weights.total_size()
        && bytes != total_size
    {
        error!(
            message = "Tensor byte count does not match the index declaration.",
            %model,
            bytes,
            total_size,
        );
        return Err(CliError::BytesMismatch);
    }
    Ok(bytes)
}

fn parameter_count<S: AsRef<[u8]>>(
    model: &ModelId,
    weights: &Weights<S>,
) -> Result<u64, CliError> {
    weights.tensors().try_fold(0_u64, |parameter_count, view| {
        u64::try_from(view.shape.numel())
            .ok()
            .and_then(|numel| parameter_count.checked_add(numel))
            .ok_or_else(|| {
                error!(message = "Parameter count overflows u64.", %model);
                CliError::ParameterCountOverflow
            })
    })
}

#[cfg(test)]
mod tests {
    use foundry::weights::{
        Shard,
        Weights,
    };
    use serde_json::json;

    use crate::cli::ModelId;
    use crate::cli::error::CliError;
    use crate::cli::report::{
        model_bytes,
        parameter_count,
    };

    #[test]
    fn model_bytes_sums_tensor_bytes() {
        let header = serde_json::to_vec(&json!({
            "a": { "dtype": "BF16", "shape": [2], "data_offsets": [0, 4] },
            "b": { "dtype": "F32", "shape": [1], "data_offsets": [4, 8] },
        }))
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..8)
            .collect();
        let index = serde_json::to_vec(&json!({
            "metadata": { "total_size": 8 },
            "weight_map": { "a": "s1.safetensors", "b": "s1.safetensors" },
        }))
        .expect("the index serializes");
        let weights = Weights::from_shards(
            Some(&index),
            vec![Shard {
                name: "s1.safetensors".to_owned(),
                bytes: file,
            }],
        )
        .expect("the shard is valid");
        assert_eq!(
            model_bytes(
                &ModelId {
                    owner: "a".to_owned(),
                    name: "b".to_owned(),
                },
                &weights,
            ),
            Ok(8),
            "two tensors of four bytes matching the declared size"
        );
    }

    #[test]
    fn model_bytes_rejects_a_declared_size_mismatch() {
        let header = serde_json::to_vec(
            &json!({ "a": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] } }),
        )
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..4)
            .collect();
        let index = serde_json::to_vec(&json!({
            "metadata": { "total_size": 5 },
            "weight_map": { "a": "s1.safetensors" },
        }))
        .expect("the index serializes");
        let weights = Weights::from_shards(
            Some(&index),
            vec![Shard {
                name: "s1.safetensors".to_owned(),
                bytes: file,
            }],
        )
        .expect("the shard is valid");
        assert_eq!(
            model_bytes(
                &ModelId {
                    owner: "a".to_owned(),
                    name: "b".to_owned(),
                },
                &weights,
            ),
            Err(CliError::BytesMismatch),
            "the index declares one byte more than the tensors hold"
        );
    }

    #[test]
    fn parameter_count_sums_the_elements_of_every_tensor() {
        let header = serde_json::to_vec(&json!({
            "a": { "dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12] },
            "b": { "dtype": "F32", "shape": [2], "data_offsets": [12, 20] },
        }))
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..20)
            .collect();
        let weights = Weights::from_shards(
            None,
            vec![Shard {
                name: "model.safetensors".to_owned(),
                bytes: file,
            }],
        )
        .expect("the file is valid");
        assert_eq!(
            parameter_count(
                &ModelId {
                    owner: "a".to_owned(),
                    name: "b".to_owned(),
                },
                &weights,
            ),
            Ok(8),
            "six bf16 and two f32 elements"
        );
    }
}
