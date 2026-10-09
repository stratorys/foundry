use crate::core::{
    Backend,
    CoreError,
    DType,
    Shape,
    Tensor,
};

pub struct RmsNorm<B: Backend> {
    weight: Tensor<B>,
    eps: Tensor<B>,
    dim_inv: Tensor<B>,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(
        backend: &mut B,
        weight: Tensor<B>,
        eps: f32,
    ) -> Result<Self, B::Error> {
        let dim = weight.shape().element_count();
        let dim_f32 =
            u16::try_from(dim)
                .map(f32::from)
                .map_err(|_| CoreError::DimensionTooLargeForF32 {
                    dim,
                    dim_max: u16::MAX.into(),
                })?;
        let scalar_shape = Shape::try_from([1].as_slice())?;
        Ok(Self {
            eps: Tensor::upload(backend, &eps.to_le_bytes(), DType::F32, scalar_shape)?,
            dim_inv: Tensor::upload(
                backend,
                &dim_f32.recip().to_le_bytes(),
                DType::F32,
                scalar_shape,
            )?,
            weight,
        })
    }

    pub fn forward(
        &self,
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
        let scale = x_f32
            .mul(backend, &x_f32)?
            .sum(backend, axis)?
            .mul(backend, &self.dim_inv)?
            .add(backend, &self.eps)?
            .sqrt(backend)?
            .recip(backend)?;
        x_f32
            .mul(backend, &scale)?
            .cast(backend, x.dtype())?
            .mul(backend, &self.weight)
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::cpu::{
        CpuBackend,
        CpuError,
    };
    use crate::core::{
        CoreError,
        DType,
        Shape,
        Tensor,
    };
    use crate::nn::RmsNorm;

    #[test]
    fn rms_norm_matches_hand_computed_values() {
        let mut backend = CpuBackend::new();
        let weight = Tensor::upload(
            &mut backend,
            &[1.0_f32, 2.0, 0.5, -1.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([4].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .cast(&mut backend, DType::BF16)
        .expect("the cast succeeds");
        let norm = RmsNorm::new(&mut backend, weight, 1e-5).expect("the norm is built");
        let x = Tensor::upload(
            &mut backend,
            &[1.0_f32, 2.0, 3.0, 4.0, -2.0, 0.0, 2.0, 4.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 4].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .cast(&mut backend, DType::BF16)
        .expect("the cast succeeds");
        let output = norm.forward(&mut backend, &x).expect("the norm succeeds");
        assert_eq!(output.dtype(), DType::BF16, "output keeps the input dtype");
        assert_eq!(output.shape().dims(), &[2, 4], "output dims");
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
        let expected = [
            0.365_148_f32,
            1.460_593,
            0.547_722,
            -1.460_593,
            -0.816_496,
            0.0,
            0.408_248,
            -1.632_992,
        ];
        assert_eq!(actual.len(), expected.len(), "element count");
        actual.iter().zip(expected).for_each(|(&actual, expected)| {
            assert!(
                (actual - expected).abs() <= 1e-2 * expected.abs(),
                "rms_norm: {actual} is not within 1e-2 of {expected}"
            );
        });
    }

    #[test]
    fn rms_norm_rejects_a_dimension_beyond_u16() {
        let mut backend = CpuBackend::new();
        let weight = Tensor::zeros(
            &mut backend,
            DType::BF16,
            Shape::try_from([65_536].as_slice()).expect("the shape is valid"),
        )
        .expect("the zeros succeed");
        assert!(
            matches!(
                RmsNorm::new(&mut backend, weight, 1e-5),
                Err(CpuError::Core(CoreError::DimensionTooLargeForF32 {
                    dim: 65_536,
                    dim_max: 65_535,
                }))
            ),
            "a dimension above u16::MAX is rejected"
        );
    }
}
