use tracing::error;

use crate::core::{
    Backend,
    DType,
    Tensor,
};
use crate::models::llama::LlamaError;
use crate::weights::{
    WeightView,
    Weights,
};

const DTYPE: DType = DType::BF16;

pub fn load_tensor<B: Backend, S: AsRef<[u8]>>(
    backend: &mut B,
    weights: &Weights<S>,
    name: &str,
    dims_expected: &[usize],
) -> Result<Tensor<B>, LlamaError<B::Error>> {
    let view = checked_view(weights.tensor(name), name, dims_expected)?;
    Ok(Tensor::from_bytes(
        backend, view.bytes, view.dtype, view.shape,
    )?)
}

fn checked_view<'weights, E: std::error::Error + 'static>(
    view: Option<WeightView<'weights>>,
    name: &str,
    dims_expected: &[usize],
) -> Result<WeightView<'weights>, LlamaError<E>> {
    let view = view.ok_or_else(|| {
        error!(message = "Tensor is not in the weights.", name);
        LlamaError::TensorNotFound
    })?;
    if view.dtype != DTYPE {
        error!(
            message = "Tensor dtype is not the model dtype.",
            name,
            dtype = ?view.dtype,
            dtype_expected = ?DTYPE,
        );
        return Err(LlamaError::TensorDTypeMismatch);
    }
    if view.shape.dims() != dims_expected {
        error!(
            message = "Tensor shape does not match the config.",
            name,
            dims = ?view.shape.dims(),
            ?dims_expected,
        );
        return Err(LlamaError::TensorShapeMismatch);
    }
    Ok(view)
}

#[cfg(test)]
mod tests {
    use crate::core::{
        CoreError,
        DType,
        Shape,
    };
    use crate::models::llama::LlamaError;
    use crate::models::llama::weights::checked_view;
    use crate::weights::WeightView;

    #[test]
    fn checked_view_accepts_a_bf16_tensor_of_the_expected_shape() {
        let view = WeightView {
            name: "model.norm.weight",
            dtype: DType::BF16,
            shape: Shape::try_from([4].as_slice()).expect("the shape is valid"),
            bytes: &[0; 8],
        };
        assert_eq!(
            checked_view::<CoreError>(Some(view), "model.norm.weight", &[4]),
            Ok(view),
            "a bf16 tensor of 4 elements"
        );
    }

    #[test]
    fn checked_view_rejects_a_missing_tensor_a_wrong_dtype_or_a_wrong_shape() {
        let view = |dtype| WeightView {
            name: "model.norm.weight",
            dtype,
            shape: Shape::try_from([4].as_slice()).expect("the shape is valid"),
            bytes: &[0; 16],
        };
        assert_eq!(
            [
                checked_view::<CoreError>(None, "model.norm.weight", &[4]),
                checked_view::<CoreError>(Some(view(DType::F32)), "model.norm.weight", &[4]),
                checked_view::<CoreError>(Some(view(DType::BF16)), "model.norm.weight", &[8]),
            ],
            [
                Err(LlamaError::TensorNotFound),
                Err(LlamaError::TensorDTypeMismatch),
                Err(LlamaError::TensorShapeMismatch),
            ],
            "absent, then f32, then 4 elements where 8 are expected"
        );
    }
}
