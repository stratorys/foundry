use std::error::Error;
use std::rc::Rc;

use tracing::error;

use crate::core::primitive::{
    BinaryOp,
    ReduceOp,
    UnaryOp,
    binary_rule,
    cast_rule,
    concat_rule,
    gather_rule,
    matmul_rule,
    reduce_rule,
    slice_update_rule,
    unary_rule,
};
use crate::core::{
    Backend,
    CoreError,
    DType,
    FloatDType,
    Layout,
    Shape,
    TensorError,
};

pub struct Operand<'storage, S, D = DType> {
    storage: &'storage S,
    layout: &'storage Layout,
    dtype: D,
}

impl<'storage, S, D: Copy> Operand<'storage, S, D> {
    fn new(
        storage: &'storage S,
        layout: &'storage Layout,
        dtype: D,
    ) -> Self {
        Self {
            storage,
            layout,
            dtype,
        }
    }

    pub fn storage(&self) -> &'storage S { self.storage }

    pub fn layout(&self) -> &'storage Layout { self.layout }

    pub fn dtype(&self) -> D { self.dtype }
}

pub struct OperandMut<'storage, S, D = DType> {
    storage: &'storage mut S,
    layout: &'storage Layout,
    dtype: D,
}

impl<'storage, S, D: Copy> OperandMut<'storage, S, D> {
    fn new(
        storage: &'storage mut S,
        layout: &'storage Layout,
        dtype: D,
    ) -> Self {
        Self {
            storage,
            layout,
            dtype,
        }
    }

    pub fn storage_mut(&mut self) -> &mut S { self.storage }

    pub fn layout(&self) -> &'storage Layout { self.layout }

    pub fn dtype(&self) -> D { self.dtype }
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
    pub fn upload(
        backend: &mut B,
        bytes: &[u8],
        dtype: DType,
        shape: Shape,
    ) -> Result<Self, TensorError<B::Error>> {
        let byte_len = shape.byte_len(dtype);
        if bytes.len() != byte_len {
            error!(
                message = "Byte length does not match the shape and dtype.",
                bytes = bytes.len(),
                bytes_expected = byte_len,
                ?dtype,
                dims = ?shape.dims(),
            );
            return Err(CoreError::ByteLengthMismatch.into());
        }
        let storage = backend.upload(bytes).map_err(backend_failed("upload"))?;
        Ok(Self::from_storage(storage, dtype, shape))
    }

    pub fn zeros(
        backend: &mut B,
        dtype: DType,
        shape: Shape,
    ) -> Result<Self, TensorError<B::Error>> {
        let storage = backend
            .zeros(shape.byte_len(dtype))
            .map_err(backend_failed("zeros"))?;
        Ok(Self::from_storage(storage, dtype, shape))
    }

    pub fn download(
        &self,
        backend: &mut B,
    ) -> Result<Vec<u8>, TensorError<B::Error>> {
        if !self.layout.is_contiguous() {
            error!(message = "Download requires a contiguous layout.", layout = ?self.layout);
            return Err(CoreError::DownloadNonContiguous.into());
        }
        backend
            .download(self.operand())
            .map_err(backend_failed("download"))
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
        axes: &[usize],
    ) -> Result<Self, CoreError> {
        Ok(self.view(self.layout.permute(axes)?))
    }

    pub fn narrow(
        &self,
        axis: usize,
        start: usize,
        len: usize,
    ) -> Result<Self, CoreError> {
        Ok(self.view(self.layout.narrow(axis, start, len)?))
    }

    pub fn broadcast_as(
        &self,
        shape: Shape,
    ) -> Result<Self, CoreError> {
        Ok(self.view(self.layout.broadcast_as(shape)?))
    }

    pub fn neg(
        &self,
        backend: &mut B,
    ) -> Result<Self, TensorError<B::Error>> {
        self.unary(backend, UnaryOp::Neg)
    }

    pub fn exp(
        &self,
        backend: &mut B,
    ) -> Result<Self, TensorError<B::Error>> {
        self.unary(backend, UnaryOp::Exp)
    }

    pub fn sqrt(
        &self,
        backend: &mut B,
    ) -> Result<Self, TensorError<B::Error>> {
        self.unary(backend, UnaryOp::Sqrt)
    }

    pub fn recip(
        &self,
        backend: &mut B,
    ) -> Result<Self, TensorError<B::Error>> {
        self.unary(backend, UnaryOp::Recip)
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

    pub fn sum(
        &self,
        backend: &mut B,
        axis: usize,
    ) -> Result<Self, TensorError<B::Error>> {
        self.reduce(backend, ReduceOp::Sum, axis)
    }

    pub fn max(
        &self,
        backend: &mut B,
        axis: usize,
    ) -> Result<Self, TensorError<B::Error>> {
        self.reduce(backend, ReduceOp::Max, axis)
    }

    pub fn argmax(
        &self,
        backend: &mut B,
        axis: usize,
    ) -> Result<Self, TensorError<B::Error>> {
        self.reduce(backend, ReduceOp::Argmax, axis)
    }

    pub fn matmul(
        &self,
        backend: &mut B,
        rhs: &Self,
    ) -> Result<Self, TensorError<B::Error>> {
        let spec = matmul_rule(self.dtype, &self.layout, rhs.dtype, &rhs.layout)?;
        let storage = backend
            .matmul(
                self.float_operand(spec.dtype()),
                rhs.float_operand(spec.dtype()),
                &spec,
            )
            .map_err(backend_failed("matmul"))?;
        Ok(Self::from_storage(
            storage,
            spec.dtype().into(),
            *spec.output(),
        ))
    }

    pub fn contiguous(
        &self,
        backend: &mut B,
    ) -> Result<Self, TensorError<B::Error>> {
        if self.layout.is_contiguous() {
            return Ok(self.view(self.layout));
        }
        let storage = backend
            .copy(self.operand())
            .map_err(backend_failed("copy"))?;
        Ok(Self::from_storage(storage, self.dtype, *self.shape()))
    }

    pub fn cast(
        &self,
        backend: &mut B,
        dtype: DType,
    ) -> Result<Self, TensorError<B::Error>> {
        let spec = cast_rule(self.dtype, dtype)?;
        let storage = backend
            .cast(self.float_operand(spec.from()), spec.to())
            .map_err(backend_failed("cast"))?;
        Ok(Self::from_storage(storage, spec.to().into(), *self.shape()))
    }

    pub fn gather(
        &self,
        backend: &mut B,
        indices: &Self,
    ) -> Result<Self, TensorError<B::Error>> {
        let spec = gather_rule(self.dtype, &self.layout, indices.dtype, indices.shape())?;
        let storage = backend
            .gather(self.float_operand(spec.dtype()), indices.operand(), &spec)
            .map_err(backend_failed("gather"))?;
        Ok(Self::from_storage(
            storage,
            spec.dtype().into(),
            *spec.output(),
        ))
    }

    pub fn concat(
        &self,
        backend: &mut B,
        rhs: &Self,
        axis: usize,
    ) -> Result<Self, TensorError<B::Error>> {
        let spec = concat_rule(self.dtype, self.shape(), rhs.dtype, rhs.shape(), axis)?;
        let storage = backend
            .concat(
                self.float_operand(spec.dtype()),
                rhs.float_operand(spec.dtype()),
                &spec,
            )
            .map_err(backend_failed("concat"))?;
        Ok(Self::from_storage(
            storage,
            spec.dtype().into(),
            *spec.output(),
        ))
    }

    pub fn slice_update(
        &mut self,
        backend: &mut B,
        update: &Self,
        axis: usize,
        start: usize,
    ) -> Result<(), TensorError<B::Error>> {
        let spec = slice_update_rule(
            self.dtype,
            &self.layout,
            update.dtype,
            update.shape(),
            axis,
            start,
        )?;
        if !self.layout.is_contiguous() {
            error!(
                message = "Slice update requires a contiguous target layout.",
                layout = ?self.layout,
            );
            return Err(CoreError::SliceUpdateNonContiguous.into());
        }
        let storage = Rc::get_mut(&mut self.storage).ok_or_else(|| {
            error!(message = "Slice update target shares its storage with another tensor.");
            CoreError::SharedStorage
        })?;
        backend
            .slice_update(
                OperandMut::new(storage, &self.layout, spec.dtype()),
                update.float_operand(spec.dtype()),
                &spec,
            )
            .map_err(backend_failed("slice_update"))
    }

    fn unary(
        &self,
        backend: &mut B,
        op: UnaryOp,
    ) -> Result<Self, TensorError<B::Error>> {
        let dtype = unary_rule(self.dtype)?;
        let storage = backend
            .unary(op, self.float_operand(dtype))
            .map_err(backend_failed("unary"))?;
        Ok(Self::from_storage(storage, dtype.into(), *self.shape()))
    }

    fn binary(
        &self,
        backend: &mut B,
        op: BinaryOp,
        rhs: &Self,
    ) -> Result<Self, TensorError<B::Error>> {
        let spec = binary_rule(self.dtype, self.shape(), rhs.dtype, rhs.shape())?;
        let lhs = self.broadcast_as(*spec.output())?;
        let rhs = rhs.broadcast_as(*spec.output())?;
        let storage = backend
            .binary(
                op,
                lhs.float_operand(spec.dtype()),
                rhs.float_operand(spec.dtype()),
            )
            .map_err(backend_failed("binary"))?;
        Ok(Self::from_storage(
            storage,
            spec.dtype().into(),
            *spec.output(),
        ))
    }

    fn reduce(
        &self,
        backend: &mut B,
        op: ReduceOp,
        axis: usize,
    ) -> Result<Self, TensorError<B::Error>> {
        let spec = reduce_rule(op, self.dtype, &self.layout, axis)?;
        let storage = backend
            .reduce(self.float_operand(spec.dtype()), &spec)
            .map_err(backend_failed("reduce"))?;
        Ok(Self::from_storage(
            storage,
            spec.dtype_output(),
            *spec.output(),
        ))
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

    fn operand(&self) -> Operand<'_, B::Storage> {
        Operand::new(&self.storage, &self.layout, self.dtype)
    }

    fn float_operand(
        &self,
        dtype: FloatDType,
    ) -> Operand<'_, B::Storage, FloatDType> {
        Operand::new(&self.storage, &self.layout, dtype)
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

    use crate::core::TensorError;
    use crate::core::tensor::backend_failed;

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
