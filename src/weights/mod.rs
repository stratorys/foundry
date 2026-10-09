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
use tracing::error;

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

pub struct Weights<S = Mmap> {
    shards: Vec<S>,
    tensors: HashMap<String, TensorEntry>,
    bytes_declared: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    #[serde(default)]
    metadata: Option<IndexMetadata>,
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct IndexMetadata {
    #[serde(default)]
    total_size: Option<u64>,
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
        let index = if index_path.is_file() {
            Some(parse_index(&index_path, &read_index(&index_path)?)?)
        } else {
            None
        };
        let shard_names: Vec<String> = match &index {
            Some(index) => index
                .weight_map
                .values()
                .cloned()
                .collect::<BTreeSet<String>>()
                .into_iter()
                .collect(),
            None => vec![SINGLE_FILE.to_owned()],
        };
        let shards = shard_names
            .into_iter()
            .map(|name| {
                let shard = open_shard(directory, &name)?;
                Ok((name, shard))
            })
            .collect::<Result<Vec<(String, Mmap)>, WeightsError>>()?;
        Self::assemble(directory, index, shards)
    }
}

impl<S: AsRef<[u8]>> Weights<S> {
    pub fn from_shards(
        index: Option<&[u8]>,
        shards: Vec<(String, S)>,
    ) -> Result<Self, WeightsError> {
        let index = index
            .map(|bytes| parse_index(Path::new(INDEX_FILE), bytes))
            .transpose()?;
        Self::assemble(Path::new(""), index, shards)
    }

    pub fn get(
        &self,
        name: &str,
    ) -> Option<WeightView<'_>> {
        self.view(self.tensors.get(name)?)
    }

    pub fn tensors(&self) -> impl Iterator<Item = (&str, WeightView<'_>)> {
        self.tensors
            .iter()
            .filter_map(|(name, entry)| Some((name.as_str(), self.view(entry)?)))
    }

    pub fn shard_count(&self) -> usize { self.shards.len() }

    pub fn bytes_declared(&self) -> Option<u64> { self.bytes_declared }

    fn view(
        &self,
        entry: &TensorEntry,
    ) -> Option<WeightView<'_>> {
        let bytes = self
            .shards
            .get(entry.shard_index)?
            .as_ref()
            .get(entry.byte_range.clone())?;
        Some(WeightView {
            dtype: entry.dtype,
            shape: entry.shape,
            bytes,
        })
    }

    fn assemble(
        directory: &Path,
        index: Option<IndexFile>,
        shards: Vec<(String, S)>,
    ) -> Result<Self, WeightsError> {
        let (shard_names, shards): (Vec<String>, Vec<S>) = shards.into_iter().unzip();
        let shard_tensors = shard_names
            .iter()
            .zip(&shards)
            .map(|(name, shard)| parse_shard(&directory.join(name), shard.as_ref()))
            .collect::<Result<Vec<Vec<ShardTensor>>, WeightsError>>()?;
        let tensors = shard_tensors
            .into_iter()
            .enumerate()
            .flat_map(|(shard_index, tensors)| {
                tensors.into_iter().map(move |tensor| (shard_index, tensor))
            })
            .try_fold(HashMap::new(), |mut tensors, (shard_index, tensor)| {
                if tensors.contains_key(&tensor.name) {
                    error!(
                        message = "Tensor appears in more than one shard.",
                        name = tensor.name,
                    );
                    return Err(WeightsError::DuplicateTensor);
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
        if let Some(index) = &index {
            check_index(&index.weight_map, &shard_names, &tensors)?;
        }
        let bytes_declared = index
            .and_then(|index| index.metadata)
            .and_then(|metadata| metadata.total_size);
        Ok(Self {
            shards,
            tensors,
            bytes_declared,
        })
    }
}

fn read_index(path: &Path) -> Result<Vec<u8>, WeightsError> {
    let read_error = |error| {
        error!(
            message = "Reading the safetensors index failed.",
            path = %path.display(),
            %error,
        );
        WeightsError::IndexRead
    };
    let bytes = fs::metadata(path).map_err(read_error)?.len();
    if bytes > INDEX_BYTES_MAX {
        error!(
            message = "Safetensors index exceeds the maximum size.",
            path = %path.display(),
            bytes,
            bytes_max = INDEX_BYTES_MAX,
        );
        return Err(WeightsError::IndexTooLarge);
    }
    fs::read(path).map_err(read_error)
}

fn parse_index(
    path: &Path,
    bytes: &[u8],
) -> Result<IndexFile, WeightsError> {
    serde_json::from_slice(bytes).map_err(|error| {
        error!(
            message = "Safetensors index is not valid JSON.",
            path = %path.display(),
            %error,
        );
        WeightsError::IndexJson
    })
}

fn check_shard_name(name: &str) -> Result<(), WeightsError> {
    if Path::new(name).file_name() == Some(OsStr::new(name)) {
        Ok(())
    } else {
        error!(message = "Shard name is not a plain file name.", name);
        Err(WeightsError::InvalidShardName)
    }
}

fn open_shard(
    directory: &Path,
    name: &str,
) -> Result<Mmap, WeightsError> {
    check_shard_name(name)?;
    let path = directory.join(name);
    let open_error = |error| {
        error!(
            message = "Opening a safetensors shard failed.",
            path = %path.display(),
            %error,
        );
        WeightsError::ShardOpen
    };
    let file = File::open(&path).map_err(open_error)?;
    // SAFETY: the file is opened read-only and Foundry never writes to it. The
    // model files are required to stay unmodified and untruncated while the
    // mapping, owned by `Weights`, is alive.
    unsafe { Mmap::map(&file) }.map_err(open_error)
}

fn parse_shard(
    path: &Path,
    bytes: &[u8],
) -> Result<Vec<ShardTensor>, WeightsError> {
    let truncated = || {
        error!(
            message = "Safetensors header is truncated.",
            path = %path.display(),
            bytes = bytes.len(),
        );
        WeightsError::TruncatedHeader
    };
    let header_len: [u8; HEADER_LEN_BYTES] = bytes
        .get(..HEADER_LEN_BYTES)
        .and_then(|prefix| prefix.try_into().ok())
        .ok_or_else(truncated)?;
    let header_bytes_declared = u64::from_le_bytes(header_len);
    let header_bytes = usize::try_from(header_bytes_declared)
        .ok()
        .filter(|&header_bytes| header_bytes <= HEADER_BYTES_MAX)
        .ok_or_else(|| {
            error!(
                message = "Safetensors header exceeds the maximum size.",
                path = %path.display(),
                bytes = header_bytes_declared,
                bytes_max = HEADER_BYTES_MAX,
            );
            WeightsError::HeaderTooLarge
        })?;
    let data_start = HEADER_LEN_BYTES
        .checked_add(header_bytes)
        .ok_or_else(truncated)?;
    let header = bytes
        .get(HEADER_LEN_BYTES..data_start)
        .ok_or_else(truncated)?;
    let bytes_data = bytes.len().checked_sub(data_start).ok_or_else(truncated)?;
    let entries: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(header).map_err(|error| {
            error!(
                message = "Safetensors header is not valid JSON.",
                path = %path.display(),
                %error,
            );
            WeightsError::HeaderJson
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
            error!(
                message = "Tensor header entry is malformed.",
                path = %path.display(),
                name,
                %error,
            );
            return Err(WeightsError::TensorHeaderJson);
        }
    };
    let Some(dtype) = parse_dtype(&header.dtype) else {
        error!(
            message = "Tensor has an unknown dtype.",
            name,
            dtype = header.dtype
        );
        return Err(WeightsError::UnknownDType);
    };
    let shape = Shape::try_from(header.shape.as_slice()).map_err(|error| {
        error!(message = "Tensor has an invalid shape.", name, %error);
        WeightsError::InvalidShape
    })?;
    let [begin, end] = header.data_offsets;
    let Some(bytes) = end.checked_sub(begin) else {
        error!(
            message = "Tensor data offsets begin after they end.",
            name, begin, end,
        );
        return Err(WeightsError::InvalidRange);
    };
    let out_of_data_region = || {
        error!(
            message = "Tensor ends outside the data region.",
            name, end, bytes_data,
        );
        WeightsError::OutOfDataRegion
    };
    if end > bytes_data {
        return Err(out_of_data_region());
    }
    let Some(bytes_expected) = shape.element_count().checked_mul(dtype.size_bytes()) else {
        error!(message = "Tensor byte count overflows usize.", name);
        return Err(WeightsError::TensorByteCountOverflow);
    };
    if bytes != bytes_expected {
        error!(
            message = "Tensor byte count does not match its shape and dtype.",
            name, bytes, bytes_expected,
        );
        return Err(WeightsError::SizeMismatch);
    }
    let (Some(start), Some(stop)) = (data_start.checked_add(begin), data_start.checked_add(end))
    else {
        return Err(out_of_data_region());
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
                error!(
                    message = "Tensors have overlapping byte ranges.",
                    first = first.name,
                    second = second.name,
                );
                Some(WeightsError::OverlappingRanges)
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
            error!(
                message = "Index maps a tensor to a shard that does not contain it.",
                name, shard,
            );
            Err(WeightsError::IndexMismatch)
        }
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::core::{
        DType,
        Shape,
    };
    use crate::weights::{
        HEADER_BYTES_MAX,
        WeightView,
        Weights,
        WeightsError,
        check_shard_name,
    };

    #[test]
    fn valid_file_returns_dtype_shape_and_bytes() {
        let header = serde_json::to_vec(&json!({
            "__metadata__": { "format": "pt" },
            "a": { "dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12] },
            "b": { "dtype": "F32", "shape": [4], "data_offsets": [12, 28] },
        }))
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..28)
            .collect();
        let weights = Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)])
            .expect("the file is valid");
        let a_bytes: Vec<u8> = (0..12).collect();
        let b_bytes: Vec<u8> = (12..28).collect();
        assert_eq!(
            weights.get("a"),
            Some(WeightView {
                dtype: DType::BF16,
                shape: Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
                bytes: &a_bytes,
            }),
            "tensor a"
        );
        assert_eq!(
            weights.get("b"),
            Some(WeightView {
                dtype: DType::F32,
                shape: Shape::try_from([4].as_slice()).expect("the shape is valid"),
                bytes: &b_bytes,
            }),
            "tensor b"
        );
        assert_eq!(weights.tensors().count(), 2, "metadata is not a tensor");
        assert_eq!(weights.shard_count(), 1, "single file is one shard");
        assert_eq!(weights.bytes_declared(), None, "no index, no declared size");
    }

    #[test]
    fn index_with_two_shards_loads_both() {
        let [first, second] = [
            (
                json!({ "a": { "dtype": "BF16", "shape": [2], "data_offsets": [0, 4] } }),
                vec![0, 1, 2, 3],
            ),
            (
                json!({ "b": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] } }),
                vec![9, 8, 7, 6],
            ),
        ]
        .map(|(header, data)| {
            let header = serde_json::to_vec(&header).expect("the header serializes");
            u64::try_from(header.len())
                .expect("the header length fits in u64")
                .to_le_bytes()
                .into_iter()
                .chain(header)
                .chain(data)
                .collect::<Vec<u8>>()
        });
        let index = serde_json::to_vec(&json!({
            "metadata": { "total_size": 8 },
            "weight_map": { "a": "s1.safetensors", "b": "s2.safetensors" },
        }))
        .expect("the index serializes");
        let weights = Weights::from_shards(
            Some(&index),
            vec![
                ("s1.safetensors".to_owned(), first),
                ("s2.safetensors".to_owned(), second),
            ],
        )
        .expect("the shards are valid");
        assert_eq!(
            weights.get("a").map(|view| view.bytes),
            Some([0, 1, 2, 3].as_slice()),
            "bytes of a"
        );
        assert_eq!(
            weights.get("b").map(|view| view.bytes),
            Some([9, 8, 7, 6].as_slice()),
            "bytes of b"
        );
        assert_eq!(weights.shard_count(), 2, "two shards");
        assert_eq!(weights.bytes_declared(), Some(8), "declared total size");
    }

    #[test]
    fn index_naming_the_wrong_shard_is_rejected() {
        let header = serde_json::to_vec(
            &json!({ "a": { "dtype": "BF16", "shape": [2], "data_offsets": [0, 4] } }),
        )
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..4)
            .collect();
        let index = serde_json::to_vec(
            &json!({ "weight_map": { "a": "s1.safetensors", "c": "s1.safetensors" } }),
        )
        .expect("the index serializes");
        assert_eq!(
            Weights::from_shards(Some(&index), vec![("s1.safetensors".to_owned(), file)]).err(),
            Some(WeightsError::IndexMismatch),
            "the index names a tensor the shard does not hold"
        );
    }

    #[test]
    fn index_naming_another_existing_shard_is_rejected() {
        let [first, second] = ["a", "b"].map(|name| {
            let header = serde_json::to_vec(
                &json!({ name: { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] } }),
            )
            .expect("the header serializes");
            u64::try_from(header.len())
                .expect("the header length fits in u64")
                .to_le_bytes()
                .into_iter()
                .chain(header)
                .chain(0..4)
                .collect::<Vec<u8>>()
        });
        let index = serde_json::to_vec(
            &json!({ "weight_map": { "a": "s2.safetensors", "b": "s2.safetensors" } }),
        )
        .expect("the index serializes");
        assert_eq!(
            Weights::from_shards(
                Some(&index),
                vec![
                    ("s1.safetensors".to_owned(), first),
                    ("s2.safetensors".to_owned(), second),
                ],
            )
            .err(),
            Some(WeightsError::IndexMismatch),
            "tensor a lives in s1, not in s2"
        );
    }

    #[test]
    fn index_without_metadata_declares_no_size() {
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
        let index = serde_json::to_vec(&json!({ "weight_map": { "a": "s1.safetensors" } }))
            .expect("the index serializes");
        let weights = Weights::from_shards(Some(&index), vec![("s1.safetensors".to_owned(), file)])
            .expect("the shard is valid");
        assert_eq!(
            weights.bytes_declared(),
            None,
            "no metadata, no declared size"
        );
    }

    #[test]
    fn empty_weight_map_gives_no_tensors() {
        let index = serde_json::to_vec(&json!({ "weight_map": {} })).expect("the index serializes");
        let weights =
            Weights::<Vec<u8>>::from_shards(Some(&index), vec![]).expect("the index is valid");
        assert_eq!(weights.tensors().count(), 0, "no tensors");
        assert_eq!(weights.shard_count(), 0, "no shards");
    }

    #[test]
    fn invalid_index_json_is_rejected() {
        let index = b"{";
        assert_eq!(
            Weights::<Vec<u8>>::from_shards(Some(index), vec![]).err(),
            Some(WeightsError::IndexJson),
            "an unterminated index is rejected"
        );
    }

    #[test]
    fn duplicate_tensor_across_shards_is_rejected() {
        let [first, second] = [(); 2].map(|()| {
            let header = serde_json::to_vec(
                &json!({ "a": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] } }),
            )
            .expect("the header serializes");
            u64::try_from(header.len())
                .expect("the header length fits in u64")
                .to_le_bytes()
                .into_iter()
                .chain(header)
                .chain(0..4)
                .collect::<Vec<u8>>()
        });
        assert_eq!(
            Weights::from_shards(
                None,
                vec![
                    ("s1.safetensors".to_owned(), first),
                    ("s2.safetensors".to_owned(), second),
                ],
            )
            .err(),
            Some(WeightsError::DuplicateTensor),
            "tensor a in two shards is rejected"
        );
    }

    #[test]
    fn shard_names_with_a_path_are_rejected() {
        ["a/b", "", "..", "../s1.safetensors"]
            .into_iter()
            .for_each(|name| {
                assert_eq!(
                    check_shard_name(name),
                    Err(WeightsError::InvalidShardName),
                    "shard name {name:?} is rejected"
                );
            });
        assert_eq!(
            check_shard_name("s1.safetensors"),
            Ok(()),
            "a plain file name is accepted"
        );
    }

    #[test]
    fn file_shorter_than_the_length_prefix_is_truncated() {
        assert_eq!(
            Weights::from_shards(
                None,
                vec![("model.safetensors".to_owned(), vec![1, 0, 0, 0])]
            )
            .err(),
            Some(WeightsError::TruncatedHeader),
            "four bytes cannot hold the length prefix"
        );
    }

    #[test]
    fn header_longer_than_the_file_is_truncated() {
        let file: Vec<u8> = 100_u64.to_le_bytes().into_iter().chain(*b"{}").collect();
        assert_eq!(
            Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)]).err(),
            Some(WeightsError::TruncatedHeader),
            "a header of 100 bytes does not fit in 10 bytes"
        );
    }

    #[test]
    fn header_above_the_limit_is_rejected() {
        let header_bytes = 100_u64 * 1024 * 1024 + 1;
        assert_eq!(
            Weights::from_shards(
                None,
                vec![(
                    "model.safetensors".to_owned(),
                    header_bytes.to_le_bytes().to_vec()
                )]
            )
            .err(),
            Some(WeightsError::HeaderTooLarge),
            "a header one byte above the limit is rejected"
        );
    }

    #[test]
    fn header_at_the_limit_is_not_too_large() {
        let header_bytes = u64::try_from(HEADER_BYTES_MAX).expect("the limit fits in u64");
        assert_eq!(
            Weights::from_shards(
                None,
                vec![(
                    "model.safetensors".to_owned(),
                    header_bytes.to_le_bytes().to_vec()
                )]
            )
            .err(),
            Some(WeightsError::TruncatedHeader),
            "a header at the limit passes the size check and is then truncated"
        );
    }

    #[test]
    fn invalid_header_json_is_rejected() {
        [b"{".as_slice(), b"[]", b"null", &[0xFF]]
            .into_iter()
            .for_each(|header| {
                let file: Vec<u8> = u64::try_from(header.len())
                    .expect("the header length fits in u64")
                    .to_le_bytes()
                    .into_iter()
                    .chain(header.iter().copied())
                    .collect();
                assert_eq!(
                    Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)]).err(),
                    Some(WeightsError::HeaderJson),
                    "header {header:?} is rejected"
                );
            });
    }

    #[test]
    fn malformed_tensor_entry_is_rejected() {
        [
            json!({ "dtype": "F32", "shape": [1] }),
            json!({ "dtype": "F32", "shape": [1], "data_offsets": [0, 4, 8] }),
        ]
        .into_iter()
        .for_each(|entry| {
            let header = serde_json::to_vec(&json!({ "a": entry })).expect("the header serializes");
            let file: Vec<u8> = u64::try_from(header.len())
                .expect("the header length fits in u64")
                .to_le_bytes()
                .into_iter()
                .chain(header)
                .chain(0..8)
                .collect();
            assert_eq!(
                Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)]).err(),
                Some(WeightsError::TensorHeaderJson),
                "entry {entry} is rejected"
            );
        });
    }

    #[test]
    fn unknown_dtype_is_rejected() {
        let header = serde_json::to_vec(
            &json!({ "a": { "dtype": "Q4", "shape": [2], "data_offsets": [0, 1] } }),
        )
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..1)
            .collect();
        assert_eq!(
            Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)]).err(),
            Some(WeightsError::UnknownDType),
            "dtype Q4 is unknown"
        );
    }

    #[test]
    fn lowercase_dtype_is_rejected() {
        let header = serde_json::to_vec(
            &json!({ "a": { "dtype": "bf16", "shape": [2], "data_offsets": [0, 4] } }),
        )
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..4)
            .collect();
        assert_eq!(
            Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)]).err(),
            Some(WeightsError::UnknownDType),
            "dtype names are case sensitive"
        );
    }

    #[test]
    fn f16_and_u32_dtypes_are_accepted() {
        let header = serde_json::to_vec(&json!({
            "a": { "dtype": "F16", "shape": [2], "data_offsets": [0, 4] },
            "b": { "dtype": "U32", "shape": [1], "data_offsets": [4, 8] },
        }))
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..8)
            .collect();
        let weights = Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)])
            .expect("the file is valid");
        assert_eq!(
            weights.get("a").map(|view| view.dtype),
            Some(DType::F16),
            "dtype of a"
        );
        assert_eq!(
            weights.get("b").map(|view| view.dtype),
            Some(DType::U32),
            "dtype of b"
        );
    }

    #[test]
    fn invalid_tensor_shape_is_rejected() {
        let header = serde_json::to_vec(
            &json!({ "a": { "dtype": "F32", "shape": [1, 1, 1, 1, 1], "data_offsets": [0, 4] } }),
        )
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..4)
            .collect();
        assert_eq!(
            Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)]).err(),
            Some(WeightsError::InvalidShape),
            "a rank 5 shape is rejected"
        );
    }

    #[test]
    fn reversed_range_is_rejected() {
        let header = serde_json::to_vec(
            &json!({ "a": { "dtype": "F32", "shape": [1], "data_offsets": [8, 4] } }),
        )
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..8)
            .collect();
        assert_eq!(
            Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)]).err(),
            Some(WeightsError::InvalidRange),
            "a range that begins after its end is rejected"
        );
    }

    #[test]
    fn range_outside_the_data_region_is_rejected() {
        let header = serde_json::to_vec(
            &json!({ "a": { "dtype": "F32", "shape": [2], "data_offsets": [0, 8] } }),
        )
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..4)
            .collect();
        assert_eq!(
            Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)]).err(),
            Some(WeightsError::OutOfDataRegion),
            "a range ending after the data is rejected"
        );
    }

    #[test]
    fn range_not_matching_shape_and_dtype_is_rejected() {
        let header = serde_json::to_vec(
            &json!({ "a": { "dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 8] } }),
        )
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..8)
            .collect();
        assert_eq!(
            Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)]).err(),
            Some(WeightsError::SizeMismatch),
            "8 bytes do not hold six bf16 values"
        );
    }

    #[test]
    fn empty_header_has_no_tensors() {
        let file: Vec<u8> = 2_u64.to_le_bytes().into_iter().chain(*b"{}").collect();
        let weights = Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)])
            .expect("an empty header is valid");
        assert_eq!(weights.tensors().count(), 0, "no tensors");
    }

    #[test]
    fn scalar_and_zero_dim_tensors_are_accepted() {
        let header = serde_json::to_vec(&json!({
            "s": { "dtype": "F32", "shape": [], "data_offsets": [0, 4] },
            "z": { "dtype": "F32", "shape": [0, 3], "data_offsets": [4, 4] },
        }))
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..4)
            .collect();
        let weights = Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)])
            .expect("the file is valid");
        assert_eq!(
            weights.get("s"),
            Some(WeightView {
                dtype: DType::F32,
                shape: Shape::try_from([].as_slice()).expect("the shape is valid"),
                bytes: &[0, 1, 2, 3],
            }),
            "a scalar holds one element"
        );
        assert_eq!(
            weights.get("z"),
            Some(WeightView {
                dtype: DType::F32,
                shape: Shape::try_from([0, 3].as_slice()).expect("the shape is valid"),
                bytes: &[],
            }),
            "a zero dim holds no element"
        );
    }

    #[test]
    fn overlapping_ranges_are_rejected() {
        let header = serde_json::to_vec(&json!({
            "a": { "dtype": "F32", "shape": [2], "data_offsets": [0, 8] },
            "b": { "dtype": "F32", "shape": [2], "data_offsets": [4, 12] },
        }))
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..12)
            .collect();
        assert_eq!(
            Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)]).err(),
            Some(WeightsError::OverlappingRanges),
            "ranges [0, 8] and [4, 12] overlap"
        );
    }

    #[test]
    fn ranges_listed_out_of_offset_order_are_checked() {
        let header = serde_json::to_vec(&json!({
            "a": { "dtype": "F32", "shape": [2], "data_offsets": [4, 12] },
            "b": { "dtype": "F32", "shape": [2], "data_offsets": [0, 8] },
        }))
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..12)
            .collect();
        assert_eq!(
            Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)]).err(),
            Some(WeightsError::OverlappingRanges),
            "ranges are compared in offset order, not in name order"
        );
    }

    #[test]
    fn empty_ranges_at_the_same_offset_are_accepted() {
        let header = serde_json::to_vec(&json!({
            "a": { "dtype": "F32", "shape": [0], "data_offsets": [4, 4] },
            "b": { "dtype": "F32", "shape": [0], "data_offsets": [4, 4] },
            "c": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] },
        }))
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..4)
            .collect();
        let weights = Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)])
            .expect("empty ranges do not overlap");
        assert_eq!(weights.tensors().count(), 3, "three tensors");
    }

    #[test]
    fn empty_range_inside_another_is_rejected() {
        let header = serde_json::to_vec(&json!({
            "a": { "dtype": "F32", "shape": [2], "data_offsets": [0, 8] },
            "b": { "dtype": "F32", "shape": [0], "data_offsets": [4, 4] },
        }))
        .expect("the header serializes");
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(0..8)
            .collect();
        assert_eq!(
            Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)]).err(),
            Some(WeightsError::OverlappingRanges),
            "an empty range that starts inside another range is rejected"
        );
    }

    #[test]
    fn missing_tensor_is_not_found() {
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
        let weights = Weights::from_shards(None, vec![("model.safetensors".to_owned(), file)])
            .expect("the file is valid");
        assert_eq!(weights.get("b"), None, "tensor b is not in the file");
    }
}
