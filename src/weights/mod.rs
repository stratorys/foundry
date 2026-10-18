mod error;

use std::collections::hash_map::Entry;
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
use safetensors::tensor::TensorInfo;
use safetensors::{
    Dtype,
    SafeTensors,
};
use serde::Deserialize;
use tracing::error;

pub use self::error::WeightsError;
use crate::nn::{
    DType,
    Shape,
};

const INDEX_FILE: &str = "model.safetensors.index.json";
const SINGLE_FILE: &str = "model.safetensors";
const HEADER_LEN_BYTES: usize = 8;
const MAX_INDEX_BYTES: u64 = 100 * 1024 * 1024;

pub struct Shard<S> {
    pub name: String,
    pub bytes: S,
}

pub struct Weights<S = Mmap> {
    shards: Vec<Shard<S>>,
    tensors: HashMap<String, TensorEntry>,
    total_size: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightView<'weights> {
    pub name: &'weights str,
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

impl Weights {
    pub fn open(directory: &Path) -> Result<Self, WeightsError> {
        let index_path = directory.join(INDEX_FILE);
        let index = if index_path.is_file() {
            Some(parse_index(&index_path, &read_index(&index_path)?)?)
        } else {
            None
        };
        let shard_names: BTreeSet<String> = match &index {
            Some(index) => index.weight_map.values().cloned().collect(),
            None => BTreeSet::from([SINGLE_FILE.to_owned()]),
        };
        let shards = shard_names
            .into_iter()
            .map(|name| {
                Ok(Shard {
                    bytes: open_shard(directory, &name)?,
                    name,
                })
            })
            .collect::<Result<Vec<Shard<Mmap>>, WeightsError>>()?;
        Self::assemble(directory, index, shards)
    }
}

impl<S: AsRef<[u8]>> Weights<S> {
    pub fn from_shards(
        index: Option<&[u8]>,
        shards: Vec<Shard<S>>,
    ) -> Result<Self, WeightsError> {
        let index = index
            .map(|bytes| parse_index(Path::new(INDEX_FILE), bytes))
            .transpose()?;
        Self::assemble(Path::new(""), index, shards)
    }

    pub fn tensor(
        &self,
        name: &str,
    ) -> Option<WeightView<'_>> {
        let (name, entry) = self.tensors.get_key_value(name)?;
        self.view(name, entry)
    }

    pub fn tensors(&self) -> impl Iterator<Item = WeightView<'_>> {
        self.tensors
            .iter()
            .filter_map(|(name, entry)| self.view(name, entry))
    }

    pub fn shard_count(&self) -> usize { self.shards.len() }

    pub fn total_size(&self) -> Option<u64> { self.total_size }

    pub fn nbytes(&self) -> Result<u64, WeightsError> {
        let nbytes = self.tensors().try_fold(0_u64, |nbytes, view| {
            u64::try_from(view.bytes.len())
                .ok()
                .and_then(|tensor_nbytes| nbytes.checked_add(tensor_nbytes))
                .ok_or_else(|| {
                    error!(message = "Total tensor byte count overflows u64.");
                    WeightsError::ByteCountOverflow
                })
        })?;
        if let Some(total_size) = self.total_size
            && nbytes != total_size
        {
            error!(
                message = "Tensor byte count does not match the index declaration.",
                nbytes, total_size,
            );
            return Err(WeightsError::ByteCountMismatch);
        }
        Ok(nbytes)
    }

    pub fn numel(&self) -> Result<u64, WeightsError> {
        self.tensors().try_fold(0_u64, |numel, view| {
            u64::try_from(view.shape.numel())
                .ok()
                .and_then(|tensor_numel| numel.checked_add(tensor_numel))
                .ok_or_else(|| {
                    error!(message = "Parameter count overflows u64.");
                    WeightsError::ParameterCountOverflow
                })
        })
    }

    fn view<'weights>(
        &'weights self,
        name: &'weights str,
        entry: &TensorEntry,
    ) -> Option<WeightView<'weights>> {
        let bytes = self
            .shards
            .get(entry.shard_index)?
            .bytes
            .as_ref()
            .get(entry.byte_range.clone())?;
        Some(WeightView {
            name,
            dtype: entry.dtype,
            shape: entry.shape,
            bytes,
        })
    }

    fn assemble(
        directory: &Path,
        index: Option<IndexFile>,
        shards: Vec<Shard<S>>,
    ) -> Result<Self, WeightsError> {
        let tensors =
            shards
                .iter()
                .zip(0..)
                .try_fold(HashMap::new(), |tensors, (shard, shard_index)| {
                    parse_shard(
                        &directory.join(&shard.name),
                        shard.bytes.as_ref(),
                        shard_index,
                    )?
                    .into_iter()
                    .try_fold(tensors, insert_tensor)
                })?;
        if let Some(index) = &index {
            check_index(&index.weight_map, &shards, &tensors)?;
        }
        let total_size = index
            .and_then(|index| index.metadata)
            .and_then(|metadata| metadata.total_size);
        Ok(Self {
            shards,
            tensors,
            total_size,
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
    if bytes > MAX_INDEX_BYTES {
        error!(
            message = "Safetensors index exceeds the maximum size.",
            path = %path.display(),
            bytes,
            max_bytes = MAX_INDEX_BYTES,
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
    shard_index: usize,
) -> Result<Vec<(String, TensorEntry)>, WeightsError> {
    let (header_bytes, metadata) = SafeTensors::read_metadata(bytes).map_err(|error| {
        error!(
            message = "Safetensors header is invalid.",
            path = %path.display(),
            %error,
        );
        WeightsError::InvalidHeader
    })?;
    let data_start = HEADER_LEN_BYTES
        .checked_add(header_bytes)
        .ok_or(WeightsError::InvalidHeader)?;
    metadata
        .tensors()
        .into_iter()
        .map(|(name, info)| parse_tensor(name, info, data_start, shard_index))
        .collect()
}

fn parse_tensor(
    name: String,
    info: &TensorInfo,
    data_start: usize,
    shard_index: usize,
) -> Result<(String, TensorEntry), WeightsError> {
    let Some(dtype) = parse_dtype(info.dtype) else {
        error!(
            message = "Tensor has an unknown dtype.",
            name,
            dtype = ?info.dtype,
        );
        return Err(WeightsError::UnknownDType);
    };
    let shape = Shape::try_from(info.shape.as_slice()).map_err(|error| {
        error!(message = "Tensor has an invalid shape.", name, %error);
        WeightsError::InvalidShape
    })?;
    let (begin, end) = info.data_offsets;
    let (Some(start), Some(stop)) = (data_start.checked_add(begin), data_start.checked_add(end))
    else {
        return Err(WeightsError::InvalidHeader);
    };
    Ok((
        name,
        TensorEntry {
            dtype,
            shape,
            shard_index,
            byte_range: start..stop,
        },
    ))
}

fn insert_tensor(
    mut tensors: HashMap<String, TensorEntry>,
    (name, tensor): (String, TensorEntry),
) -> Result<HashMap<String, TensorEntry>, WeightsError> {
    match tensors.entry(name) {
        Entry::Occupied(occupied) => {
            error!(
                message = "Tensor appears in more than one shard.",
                name = occupied.key(),
            );
            return Err(WeightsError::DuplicateTensor);
        }
        Entry::Vacant(vacant) => {
            vacant.insert(tensor);
        }
    }
    Ok(tensors)
}

fn parse_dtype(dtype: Dtype) -> Option<DType> {
    [
        (Dtype::F32, DType::F32),
        (Dtype::BF16, DType::BF16),
        (Dtype::U32, DType::U32),
    ]
    .into_iter()
    .find_map(|(file_dtype, foundry_dtype)| (file_dtype == dtype).then_some(foundry_dtype))
}

fn check_index<S>(
    weight_map: &BTreeMap<String, String>,
    shards: &[Shard<S>],
    tensors: &HashMap<String, TensorEntry>,
) -> Result<(), WeightsError> {
    weight_map.iter().try_for_each(|(name, shard)| {
        let shard_found = tensors
            .get(name)
            .and_then(|entry| shards.get(entry.shard_index))
            .map(|found| &found.name);
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
    use safetensors::tensor::TensorView;
    use safetensors::{
        Dtype,
        serialize,
    };
    use serde_json::json;

    use crate::nn::{
        DType,
        Shape,
    };
    use crate::weights::{
        Shard,
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
        let weights = Weights::from_shards(
            None,
            vec![Shard {
                name: "model.safetensors".to_owned(),
                bytes: file,
            }],
        )
        .expect("the file is valid");
        let a_bytes: Vec<u8> = (0..12).collect();
        let b_bytes: Vec<u8> = (12..28).collect();
        assert_eq!(
            weights.tensor("a"),
            Some(WeightView {
                name: "a",
                dtype: DType::BF16,
                shape: Shape::try_from([2, 3]).expect("the shape is valid"),
                bytes: &a_bytes,
            }),
            "tensor a"
        );
        assert_eq!(
            weights.tensor("b"),
            Some(WeightView {
                name: "b",
                dtype: DType::F32,
                shape: Shape::try_from([4]).expect("the shape is valid"),
                bytes: &b_bytes,
            }),
            "tensor b"
        );
        assert_eq!(weights.tensors().count(), 2, "metadata is not a tensor");
        assert_eq!(weights.shard_count(), 1, "single file is one shard");
        assert_eq!(weights.total_size(), None, "no index, no declared size");
    }

    #[test]
    fn nbytes_sums_tensor_bytes() {
        let file = serialize(
            [
                (
                    "a",
                    TensorView::new(Dtype::BF16, vec![2], &[0; 4]).expect("the view is valid"),
                ),
                (
                    "b",
                    TensorView::new(Dtype::F32, vec![1], &[0; 4]).expect("the view is valid"),
                ),
            ],
            None,
        )
        .expect("the file serializes");
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
            weights.nbytes(),
            Ok(8),
            "two tensors of four bytes matching the declared size"
        );
    }

    #[test]
    fn nbytes_rejects_a_declared_size_mismatch() {
        let file = serialize(
            [(
                "a",
                TensorView::new(Dtype::F32, vec![1], &[0; 4]).expect("the view is valid"),
            )],
            None,
        )
        .expect("the file serializes");
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
            weights.nbytes(),
            Err(WeightsError::ByteCountMismatch),
            "the index declares one byte more than the tensors hold"
        );
    }

    #[test]
    fn numel_sums_the_elements_of_every_tensor() {
        let file = serialize(
            [
                (
                    "a",
                    TensorView::new(Dtype::BF16, vec![2, 3], &[0; 12]).expect("the view is valid"),
                ),
                (
                    "b",
                    TensorView::new(Dtype::F32, vec![2], &[0; 8]).expect("the view is valid"),
                ),
            ],
            None,
        )
        .expect("the file serializes");
        let weights = Weights::from_shards(
            None,
            vec![Shard {
                name: "model.safetensors".to_owned(),
                bytes: file,
            }],
        )
        .expect("the file is valid");
        assert_eq!(weights.numel(), Ok(8), "six bf16 and two f32 elements");
    }

    #[test]
    fn index_with_two_shards_loads_both() {
        let first = serialize(
            [(
                "a",
                TensorView::new(Dtype::BF16, vec![2], &[0, 1, 2, 3]).expect("the view is valid"),
            )],
            None,
        )
        .expect("the file serializes");
        let second = serialize(
            [(
                "b",
                TensorView::new(Dtype::F32, vec![1], &[9, 8, 7, 6]).expect("the view is valid"),
            )],
            None,
        )
        .expect("the file serializes");
        let index = serde_json::to_vec(&json!({
            "metadata": { "total_size": 8 },
            "weight_map": { "a": "s1.safetensors", "b": "s2.safetensors" },
        }))
        .expect("the index serializes");
        let weights = Weights::from_shards(
            Some(&index),
            vec![
                Shard {
                    name: "s1.safetensors".to_owned(),
                    bytes: first,
                },
                Shard {
                    name: "s2.safetensors".to_owned(),
                    bytes: second,
                },
            ],
        )
        .expect("the shards are valid");
        assert_eq!(
            weights.tensor("a").map(|view| view.bytes),
            Some([0, 1, 2, 3].as_slice()),
            "bytes of a"
        );
        assert_eq!(
            weights.tensor("b").map(|view| view.bytes),
            Some([9, 8, 7, 6].as_slice()),
            "bytes of b"
        );
        assert_eq!(weights.shard_count(), 2, "two shards");
        assert_eq!(weights.total_size(), Some(8), "declared total size");
    }

    #[test]
    fn index_naming_the_wrong_shard_is_rejected() {
        let file = serialize(
            [(
                "a",
                TensorView::new(Dtype::BF16, vec![2], &[0; 4]).expect("the view is valid"),
            )],
            None,
        )
        .expect("the file serializes");
        let index = serde_json::to_vec(
            &json!({ "weight_map": { "a": "s1.safetensors", "c": "s1.safetensors" } }),
        )
        .expect("the index serializes");
        assert_eq!(
            Weights::from_shards(
                Some(&index),
                vec![Shard {
                    name: "s1.safetensors".to_owned(),
                    bytes: file
                }]
            )
            .err(),
            Some(WeightsError::IndexMismatch),
            "the index names a tensor the shard does not hold"
        );
    }

    #[test]
    fn index_naming_another_existing_shard_is_rejected() {
        let [first, second] = ["a", "b"].map(|name| {
            serialize(
                [(
                    name,
                    TensorView::new(Dtype::F32, vec![1], &[0; 4]).expect("the view is valid"),
                )],
                None,
            )
            .expect("the file serializes")
        });
        let index = serde_json::to_vec(
            &json!({ "weight_map": { "a": "s2.safetensors", "b": "s2.safetensors" } }),
        )
        .expect("the index serializes");
        assert_eq!(
            Weights::from_shards(
                Some(&index),
                vec![
                    Shard {
                        name: "s1.safetensors".to_owned(),
                        bytes: first
                    },
                    Shard {
                        name: "s2.safetensors".to_owned(),
                        bytes: second
                    },
                ],
            )
            .err(),
            Some(WeightsError::IndexMismatch),
            "tensor a lives in s1, not in s2"
        );
    }

    #[test]
    fn index_without_metadata_declares_no_size() {
        let file = serialize(
            [(
                "a",
                TensorView::new(Dtype::F32, vec![1], &[0; 4]).expect("the view is valid"),
            )],
            None,
        )
        .expect("the file serializes");
        let index = serde_json::to_vec(&json!({ "weight_map": { "a": "s1.safetensors" } }))
            .expect("the index serializes");
        let weights = Weights::from_shards(
            Some(&index),
            vec![Shard {
                name: "s1.safetensors".to_owned(),
                bytes: file,
            }],
        )
        .expect("the shard is valid");
        assert_eq!(weights.total_size(), None, "no metadata, no declared size");
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
        let [first, second] = [0_u8; 2].map(|_| {
            serialize(
                [(
                    "a",
                    TensorView::new(Dtype::F32, vec![1], &[0; 4]).expect("the view is valid"),
                )],
                None,
            )
            .expect("the file serializes")
        });
        assert_eq!(
            Weights::from_shards(
                None,
                vec![
                    Shard {
                        name: "s1.safetensors".to_owned(),
                        bytes: first
                    },
                    Shard {
                        name: "s2.safetensors".to_owned(),
                        bytes: second
                    },
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
    fn header_rejected_by_the_parser_is_invalid() {
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
                    Weights::from_shards(
                        None,
                        vec![Shard {
                            name: "model.safetensors".to_owned(),
                            bytes: file
                        }]
                    )
                    .err(),
                    Some(WeightsError::InvalidHeader),
                    "header {header:?} is rejected"
                );
            });
    }

    #[test]
    fn unknown_dtype_is_rejected() {
        [("I64", 1), ("F16", 4)]
            .into_iter()
            .for_each(|(dtype, size)| {
                let header = serde_json::to_vec(
                    &json!({ "a": { "dtype": dtype, "shape": [size], "data_offsets": [0, 8] } }),
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
                    Weights::from_shards(
                        None,
                        vec![Shard {
                            name: "model.safetensors".to_owned(),
                            bytes: file
                        }]
                    )
                    .err(),
                    Some(WeightsError::UnknownDType),
                    "dtype {dtype} is not supported"
                );
            });
    }

    #[test]
    fn u32_dtype_is_accepted() {
        let file = serialize(
            [(
                "b",
                TensorView::new(Dtype::U32, vec![1], &[0; 4]).expect("the view is valid"),
            )],
            None,
        )
        .expect("the file serializes");
        let weights = Weights::from_shards(
            None,
            vec![Shard {
                name: "model.safetensors".to_owned(),
                bytes: file,
            }],
        )
        .expect("the file is valid");
        assert_eq!(
            weights.tensor("b").map(|view| view.dtype),
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
            Weights::from_shards(
                None,
                vec![Shard {
                    name: "model.safetensors".to_owned(),
                    bytes: file
                }]
            )
            .err(),
            Some(WeightsError::InvalidShape),
            "a rank 5 shape is rejected"
        );
    }

    #[test]
    fn empty_header_has_no_tensors() {
        let file: Vec<u8> = 2_u64.to_le_bytes().into_iter().chain(*b"{}").collect();
        let weights = Weights::from_shards(
            None,
            vec![Shard {
                name: "model.safetensors".to_owned(),
                bytes: file,
            }],
        )
        .expect("an empty header is valid");
        assert_eq!(weights.tensors().count(), 0, "no tensors");
    }

    #[test]
    fn scalar_and_zero_dim_tensors_are_accepted() {
        let file = serialize(
            [
                (
                    "s",
                    TensorView::new(Dtype::F32, vec![], &[0, 1, 2, 3]).expect("the view is valid"),
                ),
                (
                    "z",
                    TensorView::new(Dtype::F32, vec![0, 3], &[]).expect("the view is valid"),
                ),
            ],
            None,
        )
        .expect("the file serializes");
        let weights = Weights::from_shards(
            None,
            vec![Shard {
                name: "model.safetensors".to_owned(),
                bytes: file,
            }],
        )
        .expect("the file is valid");
        assert_eq!(
            weights.tensor("s"),
            Some(WeightView {
                name: "s",
                dtype: DType::F32,
                shape: Shape::try_from([]).expect("the shape is valid"),
                bytes: &[0, 1, 2, 3],
            }),
            "a scalar holds one element"
        );
        assert_eq!(
            weights.tensor("z"),
            Some(WeightView {
                name: "z",
                dtype: DType::F32,
                shape: Shape::try_from([0, 3]).expect("the shape is valid"),
                bytes: &[],
            }),
            "a zero dim holds no element"
        );
    }

    #[test]
    fn empty_ranges_at_the_same_offset_are_accepted() {
        let file = serialize(
            [
                (
                    "a",
                    TensorView::new(Dtype::F32, vec![0], &[]).expect("the view is valid"),
                ),
                (
                    "b",
                    TensorView::new(Dtype::F32, vec![0], &[]).expect("the view is valid"),
                ),
                (
                    "c",
                    TensorView::new(Dtype::F32, vec![1], &[0; 4]).expect("the view is valid"),
                ),
            ],
            None,
        )
        .expect("the file serializes");
        let weights = Weights::from_shards(
            None,
            vec![Shard {
                name: "model.safetensors".to_owned(),
                bytes: file,
            }],
        )
        .expect("empty ranges do not overlap");
        assert_eq!(weights.tensors().count(), 3, "three tensors");
    }

    #[test]
    fn missing_tensor_is_not_found() {
        let file = serialize(
            [(
                "a",
                TensorView::new(Dtype::F32, vec![1], &[0; 4]).expect("the view is valid"),
            )],
            None,
        )
        .expect("the file serializes");
        let weights = Weights::from_shards(
            None,
            vec![Shard {
                name: "model.safetensors".to_owned(),
                bytes: file,
            }],
        )
        .expect("the file is valid");
        assert_eq!(weights.tensor("b"), None, "tensor b is not in the file");
    }
}
