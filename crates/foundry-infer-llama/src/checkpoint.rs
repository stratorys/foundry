use std::collections::BTreeMap;

use crate::config::{
    BITS,
    GROUP_SIZE,
    LlamaConfig,
};
use crate::error::CheckpointError;
use crate::safetensors::{
    Dtype,
    TensorEntry,
};

const PACK_BITS: u32 = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRef {
    pub name: String,
    pub start: u64,
    pub len: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizedTensor {
    pub rows: u32,
    pub cols: u32,
    pub weight: TensorRef,
    pub scales: TensorRef,
    pub biases: TensorRef,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerTensors {
    pub input_norm: TensorRef,
    pub q_proj: QuantizedTensor,
    pub k_proj: QuantizedTensor,
    pub v_proj: QuantizedTensor,
    pub o_proj: QuantizedTensor,
    pub post_attention_norm: TensorRef,
    pub gate_proj: QuantizedTensor,
    pub up_proj: QuantizedTensor,
    pub down_proj: QuantizedTensor,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub embed: QuantizedTensor,
    pub layers: Vec<LayerTensors>,
    pub norm: TensorRef,
}

impl Checkpoint {
    pub fn tensors(&self) -> impl Iterator<Item = &TensorRef> {
        fn quantized(tensor: &QuantizedTensor) -> std::array::IntoIter<&TensorRef, 3> {
            [&tensor.weight, &tensor.scales, &tensor.biases].into_iter()
        }
        quantized(&self.embed)
            .chain(self.layers.iter().flat_map(move |layer| {
                [&layer.input_norm, &layer.post_attention_norm]
                    .into_iter()
                    .chain(
                        [
                            &layer.q_proj,
                            &layer.k_proj,
                            &layer.v_proj,
                            &layer.o_proj,
                            &layer.gate_proj,
                            &layer.up_proj,
                            &layer.down_proj,
                        ]
                        .into_iter()
                        .flat_map(quantized),
                    )
            }))
            .chain([&self.norm])
    }

    pub fn from_entries(
        config: &LlamaConfig,
        entries: &[TensorEntry],
    ) -> Result<Self, CheckpointError> {
        let mut catalog = Catalog::new(entries)?;
        let hidden = config.hidden;
        let embed = catalog.quantized("model.embed_tokens", config.vocab, hidden)?;
        let layers = (0..config.layers)
            .map(|layer| {
                let prefix = format!("model.layers.{layer}");
                let attention = format!("{prefix}.self_attn");
                let mlp = format!("{prefix}.mlp");
                Ok(LayerTensors {
                    input_norm: catalog
                        .vector(&format!("{prefix}.input_layernorm.weight"), hidden)?,
                    q_proj: catalog.quantized(
                        &format!("{attention}.q_proj"),
                        config.query_dim(),
                        hidden,
                    )?,
                    k_proj: catalog.quantized(
                        &format!("{attention}.k_proj"),
                        config.kv_dim(),
                        hidden,
                    )?,
                    v_proj: catalog.quantized(
                        &format!("{attention}.v_proj"),
                        config.kv_dim(),
                        hidden,
                    )?,
                    o_proj: catalog.quantized(
                        &format!("{attention}.o_proj"),
                        hidden,
                        config.query_dim(),
                    )?,
                    post_attention_norm: catalog
                        .vector(&format!("{prefix}.post_attention_layernorm.weight"), hidden)?,
                    gate_proj: catalog.quantized(
                        &format!("{mlp}.gate_proj"),
                        config.intermediate,
                        hidden,
                    )?,
                    up_proj: catalog.quantized(
                        &format!("{mlp}.up_proj"),
                        config.intermediate,
                        hidden,
                    )?,
                    down_proj: catalog.quantized(
                        &format!("{mlp}.down_proj"),
                        hidden,
                        config.intermediate,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, CheckpointError>>()?;
        let norm = catalog.vector("model.norm.weight", hidden)?;
        catalog.finish()?;
        Ok(Self {
            embed,
            layers,
            norm,
        })
    }
}

struct Catalog<'entries> {
    remaining: BTreeMap<&'entries str, &'entries TensorEntry>,
}

impl<'entries> Catalog<'entries> {
    fn new(entries: &'entries [TensorEntry]) -> Result<Self, CheckpointError> {
        let remaining = entries.iter().try_fold(BTreeMap::new(), |mut map, entry| {
            if map.insert(entry.name.as_str(), entry).is_some() {
                Err(CheckpointError::Unexpected {
                    name: entry.name.clone(),
                })
            } else {
                Ok(map)
            }
        })?;
        Ok(Self {
            remaining,
        })
    }

    fn take(
        &mut self,
        name: &str,
        dtype: Dtype,
        shape: &[u64],
    ) -> Result<TensorRef, CheckpointError> {
        let entry = self
            .remaining
            .remove(name)
            .ok_or_else(|| CheckpointError::Missing {
                name: name.to_owned(),
            })?;
        if entry.dtype != dtype {
            return Err(CheckpointError::Dtype {
                name: name.to_owned(),
                expected: dtype.name(),
                actual: entry.dtype.name(),
            });
        }
        if entry.shape != shape {
            return Err(CheckpointError::Shape {
                name: name.to_owned(),
                expected: shape.to_vec(),
                actual: entry.shape.clone(),
            });
        }
        Ok(TensorRef {
            name: entry.name.clone(),
            start: entry.start,
            len: entry.len,
        })
    }

    fn vector(
        &mut self,
        name: &str,
        len: u32,
    ) -> Result<TensorRef, CheckpointError> {
        self.take(name, Dtype::F16, &[u64::from(len)])
    }

    fn quantized(
        &mut self,
        prefix: &str,
        rows: u32,
        cols: u32,
    ) -> Result<QuantizedTensor, CheckpointError> {
        let per_word = PACK_BITS
            .checked_div(BITS)
            .ok_or(CheckpointError::Overflow)?;
        let words = cols
            .checked_div(per_word)
            .ok_or(CheckpointError::Overflow)?;
        let groups = cols
            .checked_div(GROUP_SIZE)
            .ok_or(CheckpointError::Overflow)?;
        let rows_u64 = u64::from(rows);
        Ok(QuantizedTensor {
            rows,
            cols,
            weight: self.take(
                &format!("{prefix}.weight"),
                Dtype::U32,
                &[rows_u64, u64::from(words)],
            )?,
            scales: self.take(
                &format!("{prefix}.scales"),
                Dtype::F16,
                &[rows_u64, u64::from(groups)],
            )?,
            biases: self.take(
                &format!("{prefix}.biases"),
                Dtype::F16,
                &[rows_u64, u64::from(groups)],
            )?,
        })
    }

    fn finish(self) -> Result<(), CheckpointError> {
        self.remaining.into_keys().next().map_or(Ok(()), |name| {
            Err(CheckpointError::Unexpected {
                name: name.to_owned(),
            })
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use serde_json::{
        Map,
        Value,
        json,
    };

    use super::Checkpoint;
    use crate::config::LlamaConfig;
    use crate::config::tests::tiny;
    use crate::error::CheckpointError;
    use crate::safetensors::parse_header;
    use crate::safetensors::tests::file_bytes;

    pub(crate) fn tensor_specs(config: &LlamaConfig) -> Vec<(String, &'static str, Vec<u64>)> {
        let quantized = |name: String, rows: u32, cols: u32| {
            let rows = u64::from(rows);
            let cols = u64::from(cols);
            vec![
                (format!("{name}.weight"), "U32", vec![rows, cols / 8]),
                (format!("{name}.scales"), "F16", vec![rows, cols / 64]),
                (format!("{name}.biases"), "F16", vec![rows, cols / 64]),
            ]
        };
        let hidden = config.hidden;
        let vector = |name: String| vec![(name, "F16", vec![u64::from(hidden)])];
        let mut specs = quantized("model.embed_tokens".to_owned(), config.vocab, hidden);
        for layer in 0..config.layers {
            let prefix = format!("model.layers.{layer}");
            specs.extend(vector(format!("{prefix}.input_layernorm.weight")));
            specs.extend(vector(format!("{prefix}.post_attention_layernorm.weight")));
            for (name, rows, cols) in [
                ("self_attn.q_proj", config.query_dim(), hidden),
                ("self_attn.k_proj", config.kv_dim(), hidden),
                ("self_attn.v_proj", config.kv_dim(), hidden),
                ("self_attn.o_proj", hidden, config.query_dim()),
                ("mlp.gate_proj", config.intermediate, hidden),
                ("mlp.up_proj", config.intermediate, hidden),
                ("mlp.down_proj", hidden, config.intermediate),
            ] {
                specs.extend(quantized(format!("{prefix}.{name}"), rows, cols));
            }
        }
        specs.extend(vector("model.norm.weight".to_owned()));
        specs
    }

    pub(crate) fn header_for(specs: &[(String, &'static str, Vec<u64>)]) -> (Value, u64) {
        let mut offset = 0_u64;
        let mut map = Map::new();
        map.insert("__metadata__".to_owned(), json!({"format": "mlx"}));
        for (name, dtype, shape) in specs {
            let size = if *dtype == "F16" { 2 } else { 4 };
            let bytes = shape.iter().product::<u64>().saturating_mul(size);
            let end = offset.saturating_add(bytes);
            map.insert(
                name.clone(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}),
            );
            offset = end;
        }
        (Value::Object(map), offset)
    }

    fn checkpoint(
        specs: &[(String, &'static str, Vec<u64>)]
    ) -> Result<Checkpoint, Box<dyn std::error::Error>> {
        let config = LlamaConfig::parse(&tiny().to_string())?;
        let (header, data_bytes) = header_for(specs);
        let (text, total) = file_bytes(&header, data_bytes);
        let header = parse_header(&text, total)?;
        Ok(Checkpoint::from_entries(&config, &header.tensors)?)
    }

    fn checkpoint_error(specs: &[(String, &'static str, Vec<u64>)]) -> Option<CheckpointError> {
        checkpoint(specs)
            .err()
            .and_then(|error| error.downcast::<CheckpointError>().ok())
            .map(|error| *error)
    }

    #[test]
    fn complete_checkpoints_are_typed() -> Result<(), Box<dyn std::error::Error>> {
        let config = LlamaConfig::parse(&tiny().to_string())?;
        let checkpoint = checkpoint(&tensor_specs(&config))?;
        assert_eq!(checkpoint.layers.len(), 2, "layers");
        assert_eq!(checkpoint.embed.rows, 64, "vocabulary rows");
        assert_eq!(checkpoint.tensors().count(), 3 + 2 * 23 + 1, "tensor count");
        Ok(())
    }

    #[test]
    fn invalid_checkpoints_are_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let config = LlamaConfig::parse(&tiny().to_string())?;
        let specs = tensor_specs(&config);
        let without = |name: &str| -> Vec<_> {
            specs
                .iter()
                .filter(|(entry, ..)| entry != name)
                .cloned()
                .collect()
        };
        assert_eq!(
            checkpoint_error(&without("model.layers.1.mlp.up_proj.scales")),
            Some(CheckpointError::Missing {
                name: "model.layers.1.mlp.up_proj.scales".to_owned()
            }),
            "missing tensor"
        );
        let mut extra = specs.clone();
        extra.push(("lm_head.weight".to_owned(), "U32", vec![64, 16]));
        assert_eq!(
            checkpoint_error(&extra),
            Some(CheckpointError::Unexpected {
                name: "lm_head.weight".to_owned()
            }),
            "unexpected tensor"
        );
        let mut dtype = specs.clone();
        if let Some(entry) = dtype
            .iter_mut()
            .find(|(name, ..)| name == "model.norm.weight")
        {
            entry.1 = "F32";
        }
        assert!(
            matches!(
                checkpoint_error(&dtype),
                Some(CheckpointError::Dtype { .. })
            ),
            "wrong dtype"
        );
        let mut shape = specs.clone();
        if let Some(entry) = shape
            .iter_mut()
            .find(|(name, ..)| name == "model.layers.0.self_attn.k_proj.weight")
        {
            entry.2 = vec![64, 16];
        }
        assert!(
            matches!(
                checkpoint_error(&shape),
                Some(CheckpointError::Shape { .. })
            ),
            "wrong quantized shape"
        );
        Ok(())
    }
}
