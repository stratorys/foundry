use std::collections::BTreeMap;

use serde::Deserialize;

use crate::error::ManifestError;
use crate::safetensors::Header;

pub const REPOSITORY: &str = "mlx-community/Llama-3.2-3B-Instruct-4bit";
pub const REVISION: &str = "7f0dc925e0d0afb0322d96f9255cfddf2ba5636e";
pub const WEIGHTS_FILE: &str = "model.safetensors";
pub const CONFIG_FILE: &str = "config.json";

#[derive(Deserialize)]
struct ModelDto {
    repository: String,
    revision: String,
}

#[derive(Deserialize)]
struct FileDto {
    path: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Deserialize)]
struct TensorDto {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Deserialize)]
struct SafetensorsDto {
    path: String,
    header_bytes: u64,
    data_bytes: u64,
    tensors: Vec<TensorDto>,
}

#[derive(Deserialize)]
struct ManifestDto {
    schema_version: u32,
    model: ModelDto,
    files: Vec<FileDto>,
    safetensors: Vec<SafetensorsDto>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileIdentity {
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ManifestTensor {
    dtype: String,
    shape: Vec<u64>,
    start: u64,
    end: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Manifest {
    pub repository: String,
    pub revision: String,
    files: BTreeMap<String, FileIdentity>,
    header_bytes: u64,
    data_bytes: u64,
    tensors: BTreeMap<String, ManifestTensor>,
}

impl Manifest {
    pub fn parse(text: &str) -> Result<Self, ManifestError> {
        let dto: ManifestDto = serde_json::from_str(text).map_err(ManifestError::Parse)?;
        Self::try_from(dto)
    }

    pub fn file(
        &self,
        path: &str,
    ) -> Option<&FileIdentity> {
        self.files.get(path)
    }

    pub fn check_file_size(
        &self,
        path: &str,
        actual_bytes: u64,
    ) -> Result<(), ManifestError> {
        let expected_bytes = self.file(path).map_or(0, |file| file.size_bytes);
        if expected_bytes == actual_bytes {
            Ok(())
        } else {
            Err(ManifestError::FileSize {
                path: path.to_owned(),
                expected_bytes,
                actual_bytes,
            })
        }
    }

    pub fn check_header(
        &self,
        header: &Header,
    ) -> Result<(), ManifestError> {
        let mismatch = |name: &str| ManifestError::TensorTable {
            name: name.to_owned(),
        };
        if header.header_bytes != self.header_bytes || header.data_bytes != self.data_bytes {
            return Err(mismatch("<header size>"));
        }
        if header.tensors.len() != self.tensors.len() {
            return Err(mismatch("<tensor count>"));
        }
        header.tensors.iter().try_for_each(|tensor| {
            let recorded = self
                .tensors
                .get(&tensor.name)
                .ok_or_else(|| mismatch(&tensor.name))?;
            let end = tensor.start.checked_add(tensor.len);
            let same = recorded.dtype == tensor.dtype.name()
                && recorded.shape == tensor.shape
                && recorded.start == tensor.start
                && Some(recorded.end) == end;
            if same {
                Ok(())
            } else {
                Err(mismatch(&tensor.name))
            }
        })
    }
}

impl TryFrom<ManifestDto> for Manifest {
    type Error = ManifestError;

    fn try_from(dto: ManifestDto) -> Result<Self, Self::Error> {
        let identity = |field, expected: &str, actual: &str| {
            if expected == actual {
                Ok(())
            } else {
                Err(ManifestError::Identity {
                    field,
                    expected: expected.to_owned(),
                    actual: actual.to_owned(),
                })
            }
        };
        identity("schema_version", "1", &dto.schema_version.to_string())?;
        identity("model.repository", REPOSITORY, &dto.model.repository)?;
        identity("model.revision", REVISION, &dto.model.revision)?;
        let [safetensors] = <[SafetensorsDto; 1]>::try_from(dto.safetensors).map_err(|files| {
            ManifestError::Identity {
                field: "safetensors",
                expected: "one shard".to_owned(),
                actual: format!("{} shards", files.len()),
            }
        })?;
        identity("safetensors.path", WEIGHTS_FILE, &safetensors.path)?;
        let files = dto
            .files
            .into_iter()
            .map(|file| {
                (
                    file.path,
                    FileIdentity {
                        size_bytes: file.size_bytes,
                        sha256: file.sha256,
                    },
                )
            })
            .collect();
        let tensors = safetensors
            .tensors
            .into_iter()
            .map(|tensor| {
                let [start, end] = tensor.data_offsets;
                (
                    tensor.name,
                    ManifestTensor {
                        dtype: tensor.dtype,
                        shape: tensor.shape,
                        start,
                        end,
                    },
                )
            })
            .collect();
        Ok(Self {
            repository: dto.model.repository,
            revision: dto.model.revision,
            files,
            header_bytes: safetensors.header_bytes,
            data_bytes: safetensors.data_bytes,
            tensors,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{
        Value,
        json,
    };

    use super::{
        Manifest,
        REPOSITORY,
        REVISION,
    };
    use crate::error::ManifestError;
    use crate::safetensors::parse_header;
    use crate::test_support::set;

    fn manifest(header_bytes: usize) -> Value {
        json!({
            "schema_version": 1,
            "model": {"repository": REPOSITORY, "revision": REVISION},
            "files": [{"path": "model.safetensors", "size_bytes": 100, "sha256": "00"}],
            "safetensors": [{
                "path": "model.safetensors",
                "header_bytes": header_bytes,
                "data_bytes": 8,
                "tensors": [{"name": "a", "dtype": "F16", "shape": [4], "data_offsets": [0, 8], "data_bytes": 8}]
            }]
        })
    }

    #[test]
    fn manifests_are_cross_checked_against_headers() -> Result<(), Box<dyn std::error::Error>> {
        let text = json!({"a": {"dtype": "F16", "shape": [4], "data_offsets": [0, 8]}})
            .to_string()
            .into_bytes();
        let total = u64::try_from(text.len())?.saturating_add(16);
        let header = parse_header(&text, total)?;
        let parsed = Manifest::parse(&manifest(text.len()).to_string())?;
        parsed.check_header(&header)?;
        parsed.check_file_size("model.safetensors", 100)?;
        assert!(
            parsed.check_file_size("model.safetensors", 99).is_err(),
            "size mismatch"
        );
        let mut different = manifest(text.len());
        set(
            &mut different,
            "/safetensors/0/tensors/0/shape",
            json!([2, 2]),
        );
        assert!(
            matches!(
                Manifest::parse(&different.to_string())?.check_header(&header),
                Err(ManifestError::TensorTable { .. })
            ),
            "shape mismatch"
        );
        let mut revision = manifest(text.len());
        set(&mut revision, "/model/revision", json!("main"));
        assert!(
            matches!(
                Manifest::parse(&revision.to_string()),
                Err(ManifestError::Identity { .. })
            ),
            "unpinned revision"
        );
        Ok(())
    }
}
