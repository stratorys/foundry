use std::collections::{
    BTreeMap,
    BTreeSet,
};
use std::fs;
use std::path::Path;

use tracing::error;

use crate::eval::EvalError;
use crate::weights::{
    Shard,
    WeightView,
    Weights,
};

const INPUT_SUFFIX: &str = ".input";
const OUTPUT_SUFFIX: &str = ".output";
const MAX_ACTIVATIONS_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Activation<'weights> {
    pub name: &'weights str,
    pub input: WeightView<'weights>,
    pub output: WeightView<'weights>,
}

pub fn read_activations(path: &Path) -> Result<Weights<Vec<u8>>, EvalError> {
    let read_error = |error| {
        error!(
            message = "Reading the activations file failed.",
            path = %path.display(),
            %error,
        );
        EvalError::ActivationsRead
    };
    let nbytes = fs::metadata(path).map_err(read_error)?.len();
    if nbytes > MAX_ACTIVATIONS_BYTES {
        error!(
            message = "Activations file exceeds the maximum size.",
            path = %path.display(),
            nbytes,
            max_nbytes = MAX_ACTIVATIONS_BYTES,
        );
        return Err(EvalError::ActivationsTooLarge);
    }
    let bytes = fs::read(path).map_err(read_error)?;
    Weights::from_shards(
        None,
        vec![Shard {
            name: path.display().to_string(),
            bytes,
        }],
    )
    .map_err(|error| {
        error!(
            message = "Activations file is not a valid safetensors file.",
            path = %path.display(),
            %error,
        );
        EvalError::ActivationsInvalid
    })
}

pub fn activations<S: AsRef<[u8]>>(weights: &Weights<S>) -> Result<Vec<Activation<'_>>, EvalError> {
    let views: BTreeMap<&str, WeightView<'_>> =
        weights.tensors().map(|view| (view.name, view)).collect();
    let names = views
        .keys()
        .map(|&name| {
            name.strip_suffix(INPUT_SUFFIX)
                .or_else(|| name.strip_suffix(OUTPUT_SUFFIX))
                .ok_or_else(|| {
                    error!(
                        message = "Activation is neither an input nor an output.",
                        name
                    );
                    EvalError::UnpairedActivation
                })
        })
        .collect::<Result<BTreeSet<&str>, EvalError>>()?;
    names
        .into_iter()
        .map(|name| {
            let input = views.get(format!("{name}{INPUT_SUFFIX}").as_str());
            let output = views.get(format!("{name}{OUTPUT_SUFFIX}").as_str());
            let (Some(&input), Some(&output)) = (input, output) else {
                error!(
                    message = "Activation has no matching input or output.",
                    name,
                    has_input = input.is_some(),
                    has_output = output.is_some(),
                );
                return Err(EvalError::UnpairedActivation);
            };
            Ok(Activation {
                name,
                input,
                output,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::eval::{
        Activation,
        EvalError,
        activations,
    };
    use crate::nn::{
        DType,
        Shape,
    };
    use crate::weights::{
        Shard,
        WeightView,
        Weights,
    };

    #[test]
    fn activations_pair_inputs_and_outputs_by_module() {
        let header = serde_json::to_vec(&json!({
            "model.norm.input": { "dtype": "BF16", "shape": [2], "data_offsets": [0, 4] },
            "model.norm.output": { "dtype": "BF16", "shape": [2], "data_offsets": [4, 8] },
            "model.embed_tokens.input": { "dtype": "U32", "shape": [1], "data_offsets": [8, 12] },
            "model.embed_tokens.output": { "dtype": "F32", "shape": [1], "data_offsets": [12, 16] },
        }))
        .expect("the header serializes");
        let norm_input = [0, 1, 2, 3];
        let norm_output = [4, 5, 6, 7];
        let embed_input = [8, 9, 10, 11];
        let embed_output = [12, 13, 14, 15];
        let file: Vec<u8> = u64::try_from(header.len())
            .expect("the header length fits in u64")
            .to_le_bytes()
            .into_iter()
            .chain(header)
            .chain(norm_input)
            .chain(norm_output)
            .chain(embed_input)
            .chain(embed_output)
            .collect();
        let weights = Weights::from_shards(
            None,
            vec![Shard {
                name: "activations.safetensors".to_owned(),
                bytes: file,
            }],
        )
        .expect("the file is valid");
        let view = |name, dtype, dims: &[usize], bytes| WeightView {
            name,
            dtype,
            shape: Shape::try_from(dims).expect("the shape is valid"),
            bytes,
        };
        assert_eq!(
            activations(&weights),
            Ok(vec![
                Activation {
                    name: "model.embed_tokens",
                    input: view("model.embed_tokens.input", DType::U32, &[1], &embed_input),
                    output: view("model.embed_tokens.output", DType::F32, &[1], &embed_output),
                },
                Activation {
                    name: "model.norm",
                    input: view("model.norm.input", DType::BF16, &[2], &norm_input),
                    output: view("model.norm.output", DType::BF16, &[2], &norm_output),
                },
            ]),
            "two modules in name order"
        );
    }

    #[test]
    fn activations_reject_unpaired_or_unnamed_tensors() {
        let weights = |names: &[&str]| {
            let header = serde_json::to_vec(
                &names
                    .iter()
                    .zip([[0, 4], [4, 8]])
                    .map(|(name, offsets)| {
                        (
                            (*name).to_owned(),
                            json!({ "dtype": "F32", "shape": [1], "data_offsets": offsets }),
                        )
                    })
                    .collect::<serde_json::Map<String, serde_json::Value>>(),
            )
            .expect("the header serializes");
            let file: Vec<u8> = u64::try_from(header.len())
                .expect("the header length fits in u64")
                .to_le_bytes()
                .into_iter()
                .chain(header)
                .chain(names.iter().flat_map(|_| [0; 4]))
                .collect();
            Weights::from_shards(
                None,
                vec![Shard {
                    name: "activations.safetensors".to_owned(),
                    bytes: file,
                }],
            )
            .expect("the file is valid")
        };
        [
            vec!["model.norm.output"],
            vec!["model.norm.input"],
            vec!["model.norm.input", "model.norm.weight"],
        ]
        .into_iter()
        .for_each(|names| {
            assert_eq!(
                activations(&weights(&names)),
                Err(EvalError::UnpairedActivation),
                "tensors {names:?}"
            );
        });
    }
}
