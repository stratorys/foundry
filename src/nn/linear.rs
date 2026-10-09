use crate::core::{
    Backend,
    Tensor,
};

pub struct Linear<B: Backend> {
    weight: Tensor<B>,
}

impl<B: Backend> Linear<B> {
    pub fn new(weight: Tensor<B>) -> Self {
        Self {
            weight,
        }
    }

    pub fn forward(
        &self,
        backend: &mut B,
        x: &Tensor<B>,
    ) -> Result<Tensor<B>, B::Error> {
        x.matmul(backend, &self.weight.permute(&[1, 0])?)
    }
}

pub struct Embedding<B: Backend> {
    table: Tensor<B>,
}

impl<B: Backend> Embedding<B> {
    pub fn new(table: Tensor<B>) -> Self {
        Self {
            table,
        }
    }

    pub fn forward(
        &self,
        backend: &mut B,
        token_ids: &Tensor<B>,
    ) -> Result<Tensor<B>, B::Error> {
        self.table.gather(backend, token_ids)
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
        Embedding,
        Linear,
    };

    #[test]
    fn linear_multiplies_by_the_transposed_weight() {
        let mut backend = CpuBackend::new();
        let weight = Tensor::upload(
            &mut backend,
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .cast(&mut backend, DType::BF16)
        .expect("the cast succeeds");
        let linear = Linear::new(weight);
        let x = Tensor::upload(
            &mut backend,
            &[1.0_f32, 0.0, -1.0, 2.0, 1.0, 0.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([2, 3].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .cast(&mut backend, DType::BF16)
        .expect("the cast succeeds");
        let output = linear
            .forward(&mut backend, &x)
            .expect("the linear succeeds");
        assert_eq!(output.shape().dims(), &[2, 2], "output dims");
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
        assert_eq!(
            actual,
            vec![-2.0, -2.0, 4.0, 13.0],
            "x times the transposed weight"
        );
    }

    #[test]
    fn embedding_gathers_the_rows_of_the_token_ids() {
        let mut backend = CpuBackend::new();
        let table = Tensor::upload(
            &mut backend,
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::F32,
            Shape::try_from([3, 2].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds")
        .cast(&mut backend, DType::BF16)
        .expect("the cast succeeds");
        let embedding = Embedding::new(table);
        let token_ids = Tensor::upload(
            &mut backend,
            &[2_u32, 0]
                .iter()
                .flat_map(|token_id| token_id.to_le_bytes())
                .collect::<Vec<u8>>(),
            DType::U32,
            Shape::try_from([2].as_slice()).expect("the shape is valid"),
        )
        .expect("the upload succeeds");
        let output = embedding
            .forward(&mut backend, &token_ids)
            .expect("the embedding succeeds");
        assert_eq!(output.shape().dims(), &[2, 2], "output dims");
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
        assert_eq!(actual, vec![5.0, 6.0, 1.0, 2.0], "rows 2 then 0");
    }
}
