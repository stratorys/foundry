use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::error::{
    LlamaError,
    SafetensorsError,
};

pub const HEADER_BYTES_MAX: u64 = 100 * 1024 * 1024;
const LENGTH_BYTES: u64 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    U32,
    F16,
    F32,
}

impl Dtype {
    pub const fn size_bytes(self) -> u64 {
        match self {
            Self::F16 => 2,
            Self::U32 | Self::F32 => 4,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::U32 => "U32",
            Self::F16 => "F16",
            Self::F32 => "F32",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "U32" => Some(Self::U32),
            "F16" => Some(Self::F16),
            "F32" => Some(Self::F32),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub start: u64,
    pub len: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub header_bytes: u64,
    pub data_start: u64,
    pub data_bytes: u64,
    pub metadata: BTreeMap<String, String>,
    pub tensors: Vec<TensorEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorDto {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

pub fn read_header(path: &Path) -> Result<Header, LlamaError> {
    let mut file = File::open(path).map_err(|error| LlamaError::io(path, error))?;
    let file_bytes = file
        .metadata()
        .map_err(|error| LlamaError::io(path, error))?
        .len();
    let mut length = [0_u8; 8];
    if file_bytes < LENGTH_BYTES {
        return Err(SafetensorsError::TooShort {
            file_bytes,
        }
        .into());
    }
    file.read_exact(&mut length)
        .map_err(|error| LlamaError::io(path, error))?;
    let header_bytes = header_length(u64::from_le_bytes(length), file_bytes)?;
    let capacity = usize::try_from(header_bytes).map_err(|_| SafetensorsError::HeaderLength {
        header_bytes,
        limit_bytes: HEADER_BYTES_MAX,
        file_bytes,
    })?;
    let mut text = vec![0_u8; capacity];
    file.read_exact(&mut text)
        .map_err(|error| LlamaError::io(path, error))?;
    Ok(parse_header(&text, file_bytes)?)
}

fn header_length(
    header_bytes: u64,
    file_bytes: u64,
) -> Result<u64, SafetensorsError> {
    let available = file_bytes.saturating_sub(LENGTH_BYTES);
    if header_bytes == 0 || header_bytes > HEADER_BYTES_MAX || header_bytes > available {
        Err(SafetensorsError::HeaderLength {
            header_bytes,
            limit_bytes: HEADER_BYTES_MAX,
            file_bytes,
        })
    } else {
        Ok(header_bytes)
    }
}

pub fn parse_header(
    text: &[u8],
    file_bytes: u64,
) -> Result<Header, SafetensorsError> {
    let header_bytes = u64::try_from(text.len()).map_err(|_| SafetensorsError::HeaderLength {
        header_bytes: u64::MAX,
        limit_bytes: HEADER_BYTES_MAX,
        file_bytes,
    })?;
    let header_bytes = header_length(header_bytes, file_bytes)?;
    let data_start = LENGTH_BYTES.saturating_add(header_bytes);
    let data_bytes = file_bytes.saturating_sub(data_start);
    let entries: BTreeMap<String, Value> =
        serde_json::from_slice(text).map_err(SafetensorsError::Header)?;
    let (metadata, tensors) = entries.into_iter().try_fold(
        (BTreeMap::new(), Vec::new()),
        |(metadata, mut tensors), (name, value)| {
            if name == "__metadata__" {
                let metadata = serde_json::from_value::<BTreeMap<String, String>>(value)
                    .map_err(|_| SafetensorsError::Metadata)?;
                Ok::<_, SafetensorsError>((metadata, tensors))
            } else {
                tensors.push(tensor(name, value)?);
                Ok((metadata, tensors))
            }
        },
    )?;
    let tensors = contiguous(tensors, data_bytes)?;
    Ok(Header {
        header_bytes,
        data_start,
        data_bytes,
        metadata,
        tensors,
    })
}

fn tensor(
    name: String,
    value: Value,
) -> Result<TensorEntry, SafetensorsError> {
    let dto: TensorDto = serde_json::from_value(value).map_err(SafetensorsError::Header)?;
    let dtype = Dtype::parse(&dto.dtype).ok_or_else(|| SafetensorsError::Dtype {
        name: name.clone(),
        dtype: dto.dtype.clone(),
    })?;
    let [start, end] = dto.data_offsets;
    if end < start {
        return Err(SafetensorsError::InvalidOffsets {
            name,
            start,
            end,
        });
    }
    let expected_bytes = dto
        .shape
        .iter()
        .try_fold(dtype.size_bytes(), |total, &dim| total.checked_mul(dim))
        .ok_or_else(|| SafetensorsError::ShapeOverflow {
            name: name.clone(),
        })?;
    let len = end.saturating_sub(start);
    if expected_bytes != len {
        return Err(SafetensorsError::SizeMismatch {
            name,
            expected_bytes,
            actual_bytes: len,
        });
    }
    Ok(TensorEntry {
        name,
        dtype,
        shape: dto.shape,
        start,
        len,
    })
}

fn contiguous(
    mut tensors: Vec<TensorEntry>,
    data_bytes: u64,
) -> Result<Vec<TensorEntry>, SafetensorsError> {
    tensors.sort_by_key(|tensor| (tensor.start, tensor.len));
    let covered_bytes = tensors.iter().try_fold(0_u64, |expected_start, tensor| {
        if tensor.start != expected_start {
            return Err(SafetensorsError::Gap {
                name: tensor.name.clone(),
                expected_start,
                actual_start: tensor.start,
            });
        }
        tensor
            .start
            .checked_add(tensor.len)
            .ok_or_else(|| SafetensorsError::InvalidOffsets {
                name: tensor.name.clone(),
                start: tensor.start,
                end: u64::MAX,
            })
    })?;
    if covered_bytes == data_bytes {
        Ok(tensors)
    } else {
        Err(SafetensorsError::Coverage {
            covered_bytes,
            data_bytes,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use serde_json::{
        Value,
        json,
    };

    use super::{
        Dtype,
        parse_header,
    };
    use crate::error::SafetensorsError;
    use crate::test_support::set;

    pub(crate) fn file_bytes(
        header: &Value,
        data_bytes: u64,
    ) -> (Vec<u8>, u64) {
        let text = header.to_string().into_bytes();
        let total = u64::try_from(text.len())
            .ok()
            .and_then(|len| len.checked_add(8))
            .and_then(|len| len.checked_add(data_bytes))
            .unwrap_or(u64::MAX);
        (text, total)
    }

    fn parse(
        header: &Value,
        data_bytes: u64,
    ) -> Result<super::Header, SafetensorsError> {
        let (text, total) = file_bytes(header, data_bytes);
        parse_header(&text, total)
    }

    fn valid() -> Value {
        json!({
            "__metadata__": {"format": "mlx"},
            "b": {"dtype": "F16", "shape": [2, 3], "data_offsets": [16, 28]},
            "a": {"dtype": "U32", "shape": [4], "data_offsets": [0, 16]}
        })
    }

    #[test]
    fn valid_headers_are_sorted_by_offset() -> Result<(), SafetensorsError> {
        let header = parse(&valid(), 28)?;
        let names: Vec<&str> = header
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect();
        assert_eq!(names, ["a", "b"], "tensor order");
        assert_eq!(
            header.metadata.get("format").map(String::as_str),
            Some("mlx"),
            "metadata"
        );
        assert_eq!(
            header.tensors.first().map(|tensor| tensor.dtype),
            Some(Dtype::U32),
            "dtype"
        );
        assert_eq!(header.data_bytes, 28, "data bytes");
        Ok(())
    }

    #[test]
    fn malformed_headers_are_rejected() {
        let mut gap = valid();
        set(&mut gap, "/b/data_offsets", json!([20, 32]));
        let mut overlap = valid();
        set(&mut overlap, "/b/data_offsets", json!([8, 20]));
        let mut reversed = valid();
        set(&mut reversed, "/b/data_offsets", json!([28, 16]));
        let mut shape = valid();
        set(&mut shape, "/b/shape", json!([2, 4]));
        let mut dtype = valid();
        set(&mut dtype, "/b/dtype", json!("BF16"));
        let mut unknown = valid();
        set(&mut unknown, "/b/extra", json!(1));
        let mut overflow = valid();
        set(&mut overflow, "/b/shape", json!([u64::MAX, 4]));
        let mut metadata = valid();
        set(&mut metadata, "/__metadata__", json!({"format": 1}));
        let cases: [(&str, Value, u64); 9] = [
            ("gap", gap, 32),
            ("overlap", overlap, 28),
            ("reversed", reversed, 28),
            ("shape", shape, 28),
            ("dtype", dtype, 28),
            ("unknown field", unknown, 28),
            ("overflow", overflow, 28),
            ("metadata", metadata, 28),
            ("trailing data", valid(), 40),
        ];
        for (case, header, data_bytes) in cases {
            assert!(
                parse(&header, data_bytes).is_err(),
                "{case} must be rejected"
            );
        }
        assert!(
            matches!(
                parse_header(b"{}", 4),
                Err(SafetensorsError::HeaderLength { .. })
            ),
            "header longer than the file"
        );
        assert!(
            matches!(parse_header(b"[1]", 11), Err(SafetensorsError::Header(_))),
            "non-object header"
        );
    }
}
