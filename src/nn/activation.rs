use crate::core::{
    Backend,
    CoreError,
    DType,
    Shape,
    Tensor,
};

pub struct Silu<B: Backend> {
    one: Tensor<B>,
}

impl<B: Backend> Silu<B> {
    pub fn new(backend: &mut B) -> Result<Self, B::Error> {
        Ok(Self {
            one: Tensor::upload(
                backend,
                &1.0_f32.to_le_bytes(),
                DType::F32,
                Shape::try_from([1].as_slice())?,
            )?,
        })
    }

    pub fn forward(
        &self,
        backend: &mut B,
        x: &Tensor<B>,
    ) -> Result<Tensor<B>, B::Error> {
        let x_f32 = x.cast(backend, DType::F32)?;
        let denominator = x_f32.neg(backend)?.exp(backend)?.add(backend, &self.one)?;
        x_f32.div(backend, &denominator)?.cast(backend, x.dtype())
    }
}

pub fn softmax_last_axis<B: Backend>(
    backend: &mut B,
    x: &Tensor<B>,
) -> Result<Tensor<B>, B::Error> {
    let axis = x
        .shape()
        .rank()
        .checked_sub(1)
        .ok_or(CoreError::AxisOutOfRange {
            axis: 0,
            rank: 0,
        })?;
    let x_f32 = x.cast(backend, DType::F32)?;
    let row_max = x_f32.max(backend, axis)?;
    let exps = x_f32.sub(backend, &row_max)?.exp(backend)?;
    let row_sum = exps.sum(backend, axis)?;
    exps.div(backend, &row_sum)?.cast(backend, x.dtype())
}

#[cfg(test)]
mod tests {
    use crate::backend::cpu::CpuBackend;
    use crate::core::{
        DType,
        Shape,
        Tensor,
    };
    use crate::nn::{
        Silu,
        softmax_last_axis,
    };

    #[test]
    fn softmax_rows_match_hand_computed_values() {
        let mut backend = CpuBackend::new();
        let x = Tensor::upload(
            &mut backend,
            &[1.0_f32, 2.0, 3.0, 0.0, 0.0, 0.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .cast(&mut backend, DType::BF16)
        .expect("the cast succeeds");
        let output = softmax_last_axis(&mut backend, &x).expect("the softmax succeeds");
        assert_eq!(output.dtype(), DType::BF16, "output keeps the input dtype");
        assert_eq!(output.shape().dims(), &[2, 3], "output dims");
        let actual: Vec<f32> = output
            .cast(&mut backend, DType::F32)
            .expect("the cast succeeds")
            .download(&mut backend)
            .expect("the download succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        let third = 1.0_f32 / 3.0;
        let expected = [0.090_030_6_f32, 0.244_728_5, 0.665_241, third, third, third];
        assert_eq!(actual.len(), expected.len(), "element count");
        actual.iter().zip(expected).for_each(|(&actual, expected)| {
            assert!(
                (actual - expected).abs() <= 1e-2 * expected.abs(),
                "softmax: {actual} is not within 1e-2 of {expected}"
            );
        });
    }

    #[test]
    fn silu_matches_hand_computed_values() {
        let mut backend = CpuBackend::new();
        let silu = Silu::new(&mut backend).expect("the silu is built");
        let x = Tensor::upload(
            &mut backend,
            &[0.0_f32, 1.0, -1.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .cast(&mut backend, DType::BF16)
        .expect("the cast succeeds");
        let output = silu.forward(&mut backend, &x).expect("the silu succeeds");
        assert_eq!(output.dtype(), DType::BF16, "output keeps the input dtype");
        let actual: Vec<f32> = output
            .cast(&mut backend, DType::F32)
            .expect("the cast succeeds")
            .download(&mut backend)
            .expect("the download succeeds")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        let expected = [0.0_f32, 0.731_058_6, -0.268_941_4];
        assert_eq!(actual.len(), expected.len(), "element count");
        actual.iter().zip(expected).for_each(|(&actual, expected)| {
            assert!(
                (actual - expected).abs() <= 1e-2 * expected.abs(),
                "silu: {actual} is not within 1e-2 of {expected}"
            );
        });
    }
}
