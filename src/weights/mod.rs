mod error;

use std::collections::{
    BTreeMap,
    BTreeSet,
    HashMap,
};
use std::ffi::OsStr;
use std::fs::{
    self,
    File,
};
use std::ops::Range;
use std::path::Path;

use memmap2::Mmap;
use serde::Deserialize;

pub use self::error::WeightsError;
use crate::core::{
    DType,
    Shape,
};

const INDEX_FILE: &str = "model.safetensors.index.json";
const SINGLE_FILE: &str = "model.safetensors";
const METADATA_KEY: &str = "__metadata__";
const HEADER_LEN_BYTES: usize = 8;
const HEADER_BYTES_MAX: usize = 100 * 1024 * 1024;
const INDEX_BYTES_MAX: u64 = 100 * 1024 * 1024;

pub struct Weights {
    shards: Vec<Mmap>,
    tensors: HashMap<String, TensorEntry>,
}

#[derive(Debug, Clone, Copy)]
pub struct WeightView<'weights> {
    pub dtype: DType,
    pub shape: Shape,
    pub bytes: &'weights [u8],
}

struct TensorEntry {
    dtype: DType,
    shape: Shape,
    shard_index: usize,
    byte_range: Range<usize>,
}

struct ShardTensor {
    name: String,
    dtype: DType,
    shape: Shape,
    byte_range: Range<usize>,
}

#[derive(Deserialize)]
struct IndexFile {
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct TensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

impl Weights {
    pub fn open(directory: &Path) -> Result<Self, WeightsError> {
        let index_path = directory.join(INDEX_FILE);
        let weight_map = if index_path.is_file() {
            Some(read_index(&index_path)?)
        } else {
            None
        };
        let shard_names: Vec<String> = match &weight_map {
            Some(weight_map) => weight_map
                .values()
                .cloned()
                .collect::<BTreeSet<String>>()
                .into_iter()
                .collect(),
            None => vec![SINGLE_FILE.to_owned()],
        };
        let (shards, shard_tensors): (Vec<Mmap>, Vec<Vec<ShardTensor>>) = shard_names
            .iter()
            .map(|name| open_shard(directory, name))
            .collect::<Result<Vec<_>, WeightsError>>()?
            .into_iter()
            .unzip();
        let tensors = shard_tensors
            .into_iter()
            .enumerate()
            .flat_map(|(shard_index, tensors)| {
                tensors.into_iter().map(move |tensor| (shard_index, tensor))
            })
            .try_fold(HashMap::new(), |mut tensors, (shard_index, tensor)| {
                if tensors.contains_key(&tensor.name) {
                    return Err(WeightsError::DuplicateTensor {
                        name: tensor.name,
                    });
                }
                tensors.insert(
                    tensor.name,
                    TensorEntry {
                        dtype: tensor.dtype,
                        shape: tensor.shape,
                        shard_index,
                        byte_range: tensor.byte_range,
                    },
                );
                Ok(tensors)
            })?;
        if let Some(weight_map) = weight_map {
            check_index(&weight_map, &shard_names, &tensors)?;
        }
        Ok(Self {
            shards,
            tensors,
        })
    }

    pub fn get(
        &self,
        name: &str,
    ) -> Result<WeightView<'_>, WeightsError> {
        let not_found = || WeightsError::TensorNotFound {
            name: name.to_owned(),
        };
        let entry = self.tensors.get(name).ok_or_else(not_found)?;
        let bytes = self
            .shards
            .get(entry.shard_index)
            .and_then(|shard| shard.get(entry.byte_range.clone()))
            .ok_or_else(not_found)?;
        Ok(WeightView {
            dtype: entry.dtype,
            shape: entry.shape,
            bytes,
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> { self.tensors.keys().map(String::as_str) }
}

fn read_index(path: &Path) -> Result<BTreeMap<String, String>, WeightsError> {
    let io_error = |error| WeightsError::Io {
        path: path.to_path_buf(),
        error,
    };
    let bytes = fs::metadata(path).map_err(io_error)?.len();
    if bytes > INDEX_BYTES_MAX {
        return Err(WeightsError::IndexTooLarge {
            path: path.to_path_buf(),
            bytes,
            bytes_max: INDEX_BYTES_MAX,
        });
    }
    let contents = fs::read(path).map_err(io_error)?;
    let index: IndexFile =
        serde_json::from_slice(&contents).map_err(|error| WeightsError::IndexJson {
            path: path.to_path_buf(),
            error,
        })?;
    Ok(index.weight_map)
}

fn open_shard(
    directory: &Path,
    name: &str,
) -> Result<(Mmap, Vec<ShardTensor>), WeightsError> {
    if Path::new(name).file_name() != Some(OsStr::new(name)) {
        return Err(WeightsError::InvalidShardName {
            name: name.to_owned(),
        });
    }
    let path = directory.join(name);
    let io_error = |error| WeightsError::Io {
        path: path.clone(),
        error,
    };
    let file = File::open(&path).map_err(io_error)?;
    // SAFETY: the file is opened read-only and Foundry never writes to it. The
    // model files are required to stay unmodified and untruncated while the
    // mapping, owned by `Weights`, is alive.
    let mmap = unsafe { Mmap::map(&file) }.map_err(io_error)?;
    let tensors = parse_shard(&path, &mmap)?;
    Ok((mmap, tensors))
}

fn parse_shard(
    path: &Path,
    bytes: &[u8],
) -> Result<Vec<ShardTensor>, WeightsError> {
    let truncated = || WeightsError::TruncatedHeader {
        path: path.to_path_buf(),
        bytes: bytes.len(),
    };
    let header_len: [u8; HEADER_LEN_BYTES] = bytes
        .get(..HEADER_LEN_BYTES)
        .and_then(|prefix| prefix.try_into().ok())
        .ok_or_else(truncated)?;
    let header_bytes_declared = u64::from_le_bytes(header_len);
    let header_bytes = usize::try_from(header_bytes_declared)
        .ok()
        .filter(|&header_bytes| header_bytes <= HEADER_BYTES_MAX)
        .ok_or_else(|| WeightsError::HeaderTooLarge {
            path: path.to_path_buf(),
            bytes: header_bytes_declared,
            bytes_max: HEADER_BYTES_MAX,
        })?;
    let data_start = HEADER_LEN_BYTES
        .checked_add(header_bytes)
        .ok_or_else(truncated)?;
    let header = bytes
        .get(HEADER_LEN_BYTES..data_start)
        .ok_or_else(truncated)?;
    let bytes_data = bytes.len().checked_sub(data_start).ok_or_else(truncated)?;
    let entries: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(header).map_err(|error| WeightsError::HeaderJson {
            path: path.to_path_buf(),
            error,
        })?;
    let tensors = entries
        .into_iter()
        .filter(|(name, _)| name != METADATA_KEY)
        .map(|(name, value)| parse_tensor(path, name, value, data_start, bytes_data))
        .collect::<Result<Vec<ShardTensor>, WeightsError>>()?;
    check_overlaps(&tensors)?;
    Ok(tensors)
}

fn parse_tensor(
    path: &Path,
    name: String,
    value: serde_json::Value,
    data_start: usize,
    bytes_data: usize,
) -> Result<ShardTensor, WeightsError> {
    let header: TensorHeader = match serde_json::from_value(value) {
        Ok(header) => header,
        Err(error) => {
            return Err(WeightsError::TensorHeaderJson {
                path: path.to_path_buf(),
                name,
                error,
            });
        }
    };
    let Some(dtype) = parse_dtype(&header.dtype) else {
        return Err(WeightsError::UnknownDType {
            name,
            dtype: header.dtype,
        });
    };
    let shape = match Shape::try_from(header.shape.as_slice()) {
        Ok(shape) => shape,
        Err(error) => {
            return Err(WeightsError::InvalidShape {
                name,
                error,
            });
        }
    };
    let [begin, end] = header.data_offsets;
    let Some(bytes) = end.checked_sub(begin) else {
        return Err(WeightsError::InvalidRange {
            name,
            begin,
            end,
        });
    };
    if end > bytes_data {
        return Err(WeightsError::OutOfDataRegion {
            name,
            end,
            bytes_data,
        });
    }
    let Some(bytes_expected) = shape.element_count().checked_mul(dtype.size_bytes()) else {
        return Err(WeightsError::ByteCountOverflow {
            name,
        });
    };
    if bytes != bytes_expected {
        return Err(WeightsError::SizeMismatch {
            name,
            bytes,
            bytes_expected,
        });
    }
    let (Some(start), Some(stop)) = (data_start.checked_add(begin), data_start.checked_add(end))
    else {
        return Err(WeightsError::OutOfDataRegion {
            name,
            end,
            bytes_data,
        });
    };
    Ok(ShardTensor {
        name,
        dtype,
        shape,
        byte_range: start..stop,
    })
}

fn parse_dtype(dtype: &str) -> Option<DType> {
    match dtype {
        "F32" => Some(DType::F32),
        "F16" => Some(DType::F16),
        "BF16" => Some(DType::BF16),
        "U32" => Some(DType::U32),
        _ => None,
    }
}

fn check_overlaps(tensors: &[ShardTensor]) -> Result<(), WeightsError> {
    let mut sorted: Vec<&ShardTensor> = tensors.iter().collect();
    sorted.sort_by_key(|tensor| (tensor.byte_range.start, tensor.byte_range.end));
    sorted
        .windows(2)
        .find_map(|pair| match pair {
            [first, second] if second.byte_range.start < first.byte_range.end => {
                Some(WeightsError::OverlappingRanges {
                    first: first.name.clone(),
                    second: second.name.clone(),
                })
            }
            _ => None,
        })
        .map_or(Ok(()), Err)
}

fn check_index(
    weight_map: &BTreeMap<String, String>,
    shard_names: &[String],
    tensors: &HashMap<String, TensorEntry>,
) -> Result<(), WeightsError> {
    weight_map.iter().try_for_each(|(name, shard)| {
        let shard_found = tensors
            .get(name)
            .and_then(|entry| shard_names.get(entry.shard_index));
        if shard_found == Some(shard) {
            Ok(())
        } else {
            Err(WeightsError::IndexMismatch {
                name: name.clone(),
                shard: shard.clone(),
            })
        }
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{
        Path,
        PathBuf,
    };

    use serde_json::json;

    use crate::core::DType;
    use crate::weights::{
        Weights,
        WeightsError,
    };

    fn test_dir(name: &str) -> PathBuf {
        let directory =
            std::env::temp_dir().join(format!("foundry-weights-{name}-{}", std::process::id()));
        if directory.exists() {
            fs::remove_dir_all(&directory).expect("the old test directory is removed");
        }
        fs::create_dir_all(&directory).expect("the test directory is created");
        directory
    }

    fn safetensors_bytes(
        header: &serde_json::Value,
        data: &[u8],
    ) -> Vec<u8> {
        let header = serde_json::to_vec(header).expect("the header serializes");
        let header_len = u64::try_from(header.len()).expect("the header length fits in u64");
        header_len
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(data.iter().copied())
            .collect()
    }

    fn write_safetensors(
        path: &Path,
        header: &serde_json::Value,
        data: &[u8],
    ) {
        fs::write(path, safetensors_bytes(header, data)).expect("the safetensors file is written");
    }

    fn data(len: u8) -> Vec<u8> { (0..len).collect() }

    fn open_single(
        name: &str,
        header: &serde_json::Value,
        data: &[u8],
    ) -> Result<Weights, WeightsError> {
        let directory = test_dir(name);
        write_safetensors(&directory.join("model.safetensors"), header, data);
        Weights::open(&directory)
    }

    #[test]
    fn valid_file_returns_dtype_shape_and_bytes() {
        let header = json!({
            "__metadata__": { "format": "pt" },
            "a": { "dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12] },
            "b": { "dtype": "F32", "shape": [4], "data_offsets": [12, 28] },
        });
        let bytes = data(28);
        let weights = open_single("valid", &header, &bytes).expect("the file is valid");

        let a = weights.get("a").expect("tensor a exists");
        assert_eq!(a.dtype, DType::BF16, "dtype of a");
        assert_eq!(a.shape.dims(), &[2, 3], "shape of a");
        assert_eq!(Some(a.bytes), bytes.get(0..12), "bytes of a");

        let b = weights.get("b").expect("tensor b exists");
        assert_eq!(b.dtype, DType::F32, "dtype of b");
        assert_eq!(b.shape.dims(), &[4], "shape of b");
        assert_eq!(Some(b.bytes), bytes.get(12..28), "bytes of b");

        assert_eq!(weights.names().count(), 2, "metadata is not a tensor");
    }

    #[test]
    fn index_with_two_shards_loads_both() {
        let directory = test_dir("index");
        write_safetensors(
            &directory.join("s1.safetensors"),
            &json!({ "a": { "dtype": "BF16", "shape": [2], "data_offsets": [0, 4] } }),
            &data(4),
        );
        write_safetensors(
            &directory.join("s2.safetensors"),
            &json!({ "b": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] } }),
            &[9, 8, 7, 6],
        );
        let index = json!({
            "metadata": { "total_size": 8 },
            "weight_map": { "a": "s1.safetensors", "b": "s2.safetensors" },
        });
        fs::write(
            directory.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).expect("the index serializes"),
        )
        .expect("the index is written");

        let weights = Weights::open(&directory).expect("the shards are valid");
        assert_eq!(
            weights.get("a").expect("tensor a exists").bytes,
            &[0, 1, 2, 3],
            "bytes of a"
        );
        assert_eq!(
            weights.get("b").expect("tensor b exists").bytes,
            &[9, 8, 7, 6],
            "bytes of b"
        );
    }

    #[test]
    fn index_naming_the_wrong_shard_is_rejected() {
        let directory = test_dir("index-mismatch");
        write_safetensors(
            &directory.join("s1.safetensors"),
            &json!({ "a": { "dtype": "BF16", "shape": [2], "data_offsets": [0, 4] } }),
            &data(4),
        );
        let index = json!({ "weight_map": { "a": "s1.safetensors", "c": "s1.safetensors" } });
        fs::write(
            directory.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).expect("the index serializes"),
        )
        .expect("the index is written");

        let result = Weights::open(&directory);
        assert!(
            matches!(&result, Err(WeightsError::IndexMismatch { name, .. }) if name == "c"),
            "got {:?}",
            result.err()
        );
    }

    #[test]
    fn shard_name_outside_the_directory_is_rejected() {
        let directory = test_dir("shard-name");
        let index = json!({ "weight_map": { "a": "../s1.safetensors" } });
        fs::write(
            directory.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).expect("the index serializes"),
        )
        .expect("the index is written");

        let result = Weights::open(&directory);
        assert!(
            matches!(result, Err(WeightsError::InvalidShardName { .. })),
            "got {:?}",
            result.err()
        );
    }

    #[test]
    fn file_shorter_than_the_length_prefix_is_truncated() {
        let directory = test_dir("truncated-prefix");
        fs::write(directory.join("model.safetensors"), [1, 0, 0, 0]).expect("the file is written");

        let result = Weights::open(&directory);
        assert!(
            matches!(
                result,
                Err(WeightsError::TruncatedHeader {
                    bytes: 4,
                    ..
                })
            ),
            "got {:?}",
            result.err()
        );
    }

    #[test]
    fn header_longer_than_the_file_is_truncated() {
        let directory = test_dir("truncated-header");
        let bytes: Vec<u8> = 100_u64.to_le_bytes().into_iter().chain(*b"{}").collect();
        fs::write(directory.join("model.safetensors"), bytes).expect("the file is written");

        let result = Weights::open(&directory);
        assert!(
            matches!(result, Err(WeightsError::TruncatedHeader { .. })),
            "got {:?}",
            result.err()
        );
    }

    #[test]
    fn header_above_the_limit_is_rejected() {
        let directory = test_dir("header-too-large");
        let header_bytes = 100_u64 * 1024 * 1024 + 1;
        fs::write(
            directory.join("model.safetensors"),
            header_bytes.to_le_bytes(),
        )
        .expect("the file is written");

        let result = Weights::open(&directory);
        assert!(
            matches!(result, Err(WeightsError::HeaderTooLarge { bytes, .. }) if bytes == header_bytes),
            "got {:?}",
            result.err()
        );
    }

    #[test]
    fn overlapping_ranges_are_rejected() {
        let header = json!({
            "a": { "dtype": "F32", "shape": [2], "data_offsets": [0, 8] },
            "b": { "dtype": "F32", "shape": [2], "data_offsets": [4, 12] },
        });
        let result = open_single("overlap", &header, &data(12));
        assert!(
            matches!(&result, Err(WeightsError::OverlappingRanges { first, second }) if first == "a" && second == "b"),
            "got {:?}",
            result.err()
        );
    }

    #[test]
    fn unknown_dtype_is_rejected() {
        let header = json!({ "a": { "dtype": "Q4", "shape": [2], "data_offsets": [0, 1] } });
        let result = open_single("unknown-dtype", &header, &data(1));
        assert!(
            matches!(&result, Err(WeightsError::UnknownDType { dtype, .. }) if dtype == "Q4"),
            "got {:?}",
            result.err()
        );
    }

    #[test]
    fn range_outside_the_data_region_is_rejected() {
        let header = json!({ "a": { "dtype": "F32", "shape": [2], "data_offsets": [0, 8] } });
        let result = open_single("out-of-region", &header, &data(4));
        assert!(
            matches!(
                result,
                Err(WeightsError::OutOfDataRegion {
                    end: 8,
                    bytes_data: 4,
                    ..
                })
            ),
            "got {:?}",
            result.err()
        );
    }

    #[test]
    fn range_not_matching_shape_and_dtype_is_rejected() {
        let header = json!({ "a": { "dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 8] } });
        let result = open_single("size-mismatch", &header, &data(8));
        assert!(
            matches!(
                result,
                Err(WeightsError::SizeMismatch {
                    bytes: 8,
                    bytes_expected: 12,
                    ..
                })
            ),
            "got {:?}",
            result.err()
        );
    }

    #[test]
    fn missing_tensor_is_not_found() {
        let header = json!({ "a": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] } });
        let weights = open_single("missing", &header, &data(4)).expect("the file is valid");
        let result = weights.get("b");
        assert!(
            matches!(&result, Err(WeightsError::TensorNotFound { name }) if name == "b"),
            "got {:?}",
            result.err()
        );
    }

    #[test]
    #[ignore = "needs the bf16 snapshot of Llama-3.2-3B-Instruct in FOUNDRY_LLAMA_DIR"]
    fn llama_snapshot_loads() {
        let directory = std::env::var("FOUNDRY_LLAMA_DIR").expect("FOUNDRY_LLAMA_DIR is set");
        let weights = Weights::open(Path::new(&directory)).expect("the snapshot loads");

        assert_eq!(
            weights.names().count(),
            254,
            "28 layers × 9 + embeddings + norm"
        );
        weights.names().for_each(|name| {
            let view = weights.get(name).expect("every listed tensor is readable");
            assert_eq!(view.dtype, DType::BF16, "dtype of {name}");
        });
        ["model.embed_tokens.weight", "model.norm.weight"]
            .iter()
            .for_each(|name| {
                assert!(weights.get(name).is_ok(), "{name} is present");
            });
    }
}
