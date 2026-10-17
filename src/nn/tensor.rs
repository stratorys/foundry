use std::error::Error;
use std::rc::Rc;

use tracing::error;

use crate::nn::primitive::{
    BinaryOp,
    Op,
    ReduceOp,
    UnaryOp,
    output,
};
use crate::nn::{
    Backend,
    CoreError,
    DType,
    Layout,
    Shape,
    TensorError,
};

pub struct Operand<'layout, S> {
    storage: S,
    layout: &'layout Layout,
    dtype: DType,
}

impl<'layout, S> Operand<'layout, S> {
    fn new(
        storage: S,
        layout: &'layout Layout,
        dtype: DType,
    ) -> Self {
        Self {
            storage,
            layout,
            dtype,
        }
    }

    pub fn storage(&self) -> &S { &self.storage }

    pub fn storage_mut(&mut self) -> &mut S { &mut self.storage }

    pub fn layout(&self) -> &'layout Layout { self.layout }

    pub fn dtype(&self) -> DType { self.dtype }
}

pub struct Tensor<B: Backend> {
    storage: Rc<B::Storage>,
    layout: Layout,
    dtype: DType,
}

impl<B: Backend> Clone for Tensor<B> {
    fn clone(&self) -> Self { self.view(self.layout) }
}

impl<B: Backend> Tensor<B> {
    pub fn from_bytes(
        backend: &mut B,
        bytes: &[u8],
        dtype: DType,
        shape: Shape,
    ) -> Result<Self, TensorError<B::Error>> {
        let nbytes = shape.nbytes(dtype);
        if bytes.len() != nbytes {
            error!(
                message = "Byte length does not match the shape and dtype.",
                bytes = bytes.len(),
                bytes_expected = nbytes,
                ?dtype,
                dims = ?shape.dims(),
            );
            return Err(CoreError::ByteLengthMismatch.into());
        }
        let storage = backend
            .storage_from_bytes(bytes)
            .map_err(backend_failed("from_bytes"))?;
        Ok(Self::from_storage(storage, dtype, shape))
    }

    pub fn zeros(
        backend: &mut B,
        dtype: DType,
        shape: Shape,
    ) -> Result<Self, TensorError<B::Error>> {
        let storage = backend
            .zeros(shape.nbytes(dtype))
            .map_err(backend_failed("zeros"))?;
        Ok(Self::from_storage(storage, dtype, shape))
    }

    pub fn to_bytes(
        &self,
        backend: &mut B,
    ) -> Result<Vec<u8>, TensorError<B::Error>> {
        if !self.layout.is_contiguous() {
            error!(message = "Reading bytes requires a contiguous layout.", layout = ?self.layout);
            return Err(CoreError::ToBytesNonContiguous.into());
        }
        backend
            .storage_to_bytes(&self.operand())
            .map_err(backend_failed("to_bytes"))
    }

    pub fn dtype(&self) -> DType { self.dtype }

    pub fn layout(&self) -> &Layout { &self.layout }

    pub fn shape(&self) -> &Shape { self.layout.shape() }

    pub fn reshape(
        &self,
        shape: Shape,
    ) -> Result<Self, CoreError> {
        Ok(self.view(self.layout.reshape(shape)?))
    }

    pub fn permute(
        &self,
        dims: &[usize],
    ) -> Result<Self, CoreError> {
        Ok(self.view(self.layout.permute(dims)?))
    }

    pub fn narrow(
        &self,
        dim: usize,
        start: usize,
        length: usize,
    ) -> Result<Self, CoreError> {
        Ok(self.view(self.layout.narrow(dim, start, length)?))
    }

    pub fn expand(
        &self,
        shape: Shape,
    ) -> Result<Self, CoreError> {
        Ok(self.view(self.layout.expand(shape)?))
    }

    pub fn neg(
        &self,
        backend: &mut B,
    ) -> Result<Self, TensorError<B::Error>> {
        Self::apply(backend, Op::Unary(UnaryOp::Neg), [self])
    }

    pub fn exp(
        &self,
        backend: &mut B,
    ) -> Result<Self, TensorError<B::Error>> {
        Self::apply(backend, Op::Unary(UnaryOp::Exp), [self])
    }

    pub fn sqrt(
        &self,
        backend: &mut B,
    ) -> Result<Self, TensorError<B::Error>> {
        Self::apply(backend, Op::Unary(UnaryOp::Sqrt), [self])
    }

    pub fn reciprocal(
        &self,
        backend: &mut B,
    ) -> Result<Self, TensorError<B::Error>> {
        Self::apply(backend, Op::Unary(UnaryOp::Reciprocal), [self])
    }

    pub fn add(
        &self,
        backend: &mut B,
        rhs: &Self,
    ) -> Result<Self, TensorError<B::Error>> {
        self.binary(backend, BinaryOp::Add, rhs)
    }

    pub fn sub(
        &self,
        backend: &mut B,
        rhs: &Self,
    ) -> Result<Self, TensorError<B::Error>> {
        self.binary(backend, BinaryOp::Sub, rhs)
    }

    pub fn mul(
        &self,
        backend: &mut B,
        rhs: &Self,
    ) -> Result<Self, TensorError<B::Error>> {
        self.binary(backend, BinaryOp::Mul, rhs)
    }

    pub fn div(
        &self,
        backend: &mut B,
        rhs: &Self,
    ) -> Result<Self, TensorError<B::Error>> {
        self.binary(backend, BinaryOp::Div, rhs)
    }

    pub fn sum_keepdim(
        &self,
        backend: &mut B,
        dim: usize,
    ) -> Result<Self, TensorError<B::Error>> {
        let op = ReduceOp::Sum;
        Self::apply(
            backend,
            Op::Reduce {
                op,
                dim,
            },
            [self],
        )
    }

    pub fn amax_keepdim(
        &self,
        backend: &mut B,
        dim: usize,
    ) -> Result<Self, TensorError<B::Error>> {
        let op = ReduceOp::Amax;
        Self::apply(
            backend,
            Op::Reduce {
                op,
                dim,
            },
            [self],
        )
    }

    pub fn argmax_keepdim(
        &self,
        backend: &mut B,
        dim: usize,
    ) -> Result<Self, TensorError<B::Error>> {
        let op = ReduceOp::Argmax;
        Self::apply(
            backend,
            Op::Reduce {
                op,
                dim,
            },
            [self],
        )
    }

    pub fn matmul(
        &self,
        backend: &mut B,
        rhs: &Self,
    ) -> Result<Self, TensorError<B::Error>> {
        Self::apply(backend, Op::Matmul, [self, rhs])
    }

    pub fn contiguous(
        &self,
        backend: &mut B,
    ) -> Result<Self, TensorError<B::Error>> {
        if self.layout.is_contiguous() {
            return Ok(self.view(self.layout));
        }
        Self::apply(backend, Op::Copy, [self])
    }

    pub fn to_dtype(
        &self,
        backend: &mut B,
        dtype: DType,
    ) -> Result<Self, TensorError<B::Error>> {
        Self::apply(backend, Op::ToDtype(dtype), [self])
    }

    pub fn index_select(
        &self,
        backend: &mut B,
        dim: usize,
        indices: &Self,
    ) -> Result<Self, TensorError<B::Error>> {
        Self::apply(
            backend,
            Op::IndexSelect {
                dim,
            },
            [self, indices],
        )
    }

    pub fn cat(
        backend: &mut B,
        tensors: [&Self; 2],
        dim: usize,
    ) -> Result<Self, TensorError<B::Error>> {
        Self::apply(
            backend,
            Op::Cat {
                dim,
            },
            tensors,
        )
    }

    pub fn slice_scatter(
        &mut self,
        backend: &mut B,
        src: &Self,
        dim: usize,
        start: usize,
    ) -> Result<(), TensorError<B::Error>> {
        let op = Op::SliceScatter {
            dim,
            start,
        };
        output(
            op,
            &[(self.dtype, *self.shape()), (src.dtype, *src.shape())],
        )?;
        if !self.layout.is_contiguous() {
            error!(
                message = "Slice scatter requires a contiguous target layout.",
                layout = ?self.layout,
            );
            return Err(CoreError::SliceScatterNonContiguous.into());
        }
        let storage = Rc::get_mut(&mut self.storage).ok_or_else(|| {
            error!(message = "Slice scatter target shares its storage with another tensor.");
            CoreError::SharedStorage
        })?;
        backend
            .run_in_place(
                op,
                Operand::new(storage, &self.layout, self.dtype),
                &[src.operand()],
            )
            .map_err(backend_failed(op.name()))
    }

    fn binary(
        &self,
        backend: &mut B,
        op: BinaryOp,
        rhs: &Self,
    ) -> Result<Self, TensorError<B::Error>> {
        let shape = Shape::broadcast(self.shape(), rhs.shape())?;
        Self::apply(
            backend,
            Op::Binary(op),
            [&self.expand(shape)?, &rhs.expand(shape)?],
        )
    }

    fn apply<const N: usize>(
        backend: &mut B,
        op: Op,
        inputs: [&Self; N],
    ) -> Result<Self, TensorError<B::Error>> {
        let (shape, dtype) = output(op, &inputs.map(|input| (input.dtype, *input.shape())))?;
        let storage = backend
            .run(op, &inputs.map(Self::operand), &shape, dtype)
            .map_err(backend_failed(op.name()))?;
        Ok(Self::from_storage(storage, dtype, shape))
    }

    fn from_storage(
        storage: B::Storage,
        dtype: DType,
        shape: Shape,
    ) -> Self {
        Self {
            storage: Rc::new(storage),
            layout: Layout::contiguous(shape),
            dtype,
        }
    }

    fn view(
        &self,
        layout: Layout,
    ) -> Self {
        Self {
            storage: Rc::clone(&self.storage),
            layout,
            dtype: self.dtype,
        }
    }

    fn operand(&self) -> Operand<'_, &B::Storage> {
        Operand::new(&self.storage, &self.layout, self.dtype)
    }
}

fn backend_failed<E: Error + 'static>(operation: &'static str) -> impl FnOnce(E) -> TensorError<E> {
    move |source| TensorError::Backend {
        operation,
        source,
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use crate::nn::TensorError;
    use crate::nn::tensor::backend_failed;

    #[derive(Debug, thiserror::Error)]
    #[error("backend diagnostic: {detail}")]
    struct Failure {
        detail: String,
    }

    #[test]
    fn backend_conversion_preserves_operation_and_original_error() {
        let error = backend_failed("matmul")(Failure {
            detail: "allocation rejected".to_owned(),
        });
        assert!(
            matches!(
                &error,
                TensorError::Backend { operation: "matmul", source }
                    if source.detail == "allocation rejected"
            ),
            "the concrete error and operation are retained"
        );
        let source = error.source().expect("the backend failure has a source");
        let failure = source
            .downcast_ref::<Failure>()
            .expect("the original type is retained");
        assert_eq!(
            failure.detail, "allocation rejected",
            "the diagnostic is retained without tracing"
        );
        assert_eq!(
            error.to_string(),
            "Backend operation `matmul` failed.",
            "display identifies the operation"
        );
    }
}
