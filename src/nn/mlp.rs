use crate::core::{
    Backend,
    Tensor,
};
use crate::nn::{
    Linear,
    Silu,
};

pub struct SwigluMlp<B: Backend> {
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
    silu: Silu<B>,
}

impl<B: Backend> SwigluMlp<B> {
    pub fn new(
        backend: &mut B,
        gate: Linear<B>,
        up: Linear<B>,
        down: Linear<B>,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            gate,
            up,
            down,
            silu: Silu::new(backend)?,
        })
    }

    pub fn forward(
        &self,
        backend: &mut B,
        x: &Tensor<B>,
    ) -> Result<Tensor<B>, B::Error> {
        let gate = self.gate.forward(backend, x)?;
        let up = self.up.forward(backend, x)?;
        let hidden = self.silu.forward(backend, &gate)?.mul(backend, &up)?;
        self.down.forward(backend, &hidden)
    }
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
        Linear,
        SwigluMlp,
    };

    #[test]
    fn swiglu_mlp_with_identity_projections_gives_silu_of_x_times_x() {
        let mut backend = CpuBackend::new();
        let identity_bytes: Vec<u8> = [1.0_f32, 0.0, 0.0, 1.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let identity_shape = Shape::try_from([2, 2].as_slice()).expect("the shape is valid");
        let [gate, up, down] = [(); 3].map(|()| {
            Linear::new(
                Tensor::upload(&mut backend, &identity_bytes, DType::F32, identity_shape)
                    .expect("the upload succeeds")
                    .cast(&mut backend, DType::BF16)
                    .expect("the cast succeeds"),
            )
        });
        let mlp = SwigluMlp::new(&mut backend, gate, up, down).expect("the mlp is built");
        let x = Tensor::upload(
            &mut backend,
            &[1.0_f32, 0.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([1, 2].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .cast(&mut backend, DType::BF16)
        .expect("the cast succeeds");
        let output = mlp.forward(&mut backend, &x).expect("the mlp succeeds");
        assert_eq!(output.shape().dims(), &[1, 2], "output dims");
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
        let expected = [0.731_058_6_f32, 0.0];
        assert_eq!(actual.len(), expected.len(), "element count");
        actual.iter().zip(expected).for_each(|(&actual, expected)| {
            assert!(
                (actual - expected).abs() <= 1e-2 * expected.abs(),
                "swiglu mlp: {actual} is not within 1e-2 of {expected}"
            );
        });
    }
}
