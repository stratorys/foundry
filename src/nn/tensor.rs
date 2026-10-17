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
    DType,
    Device,
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

pub struct Tensor<D: Device> {
    storage: Rc<D::Storage>,
    layout: Layout,
    dtype: DType,
}

impl<D: Device> Clone for Tensor<D> {
    fn clone(&self) -> Self { self.view(self.layout) }
}

impl<D: Device> Tensor<D> {
    pub fn from_bytes(
        device: &mut D,
        bytes: &[u8],
        dtype: DType,
        shape: Shape,
    ) -> Result<Self, TensorError> {
        let nbytes = shape.nbytes(dtype);
        if bytes.len() != nbytes {
            error!(
                message = "Byte length does not match the shape and dtype.",
                bytes = bytes.len(),
                bytes_expected = nbytes,
                ?dtype,
                dims = ?shape.dims(),
            );
            return Err(TensorError::ByteLengthMismatch);
        }
        let storage = device
            .storage_from_bytes(bytes)
            .map_err(device_failed("from_bytes"))?;
        Ok(Self::from_storage(storage, dtype, shape))
    }

    pub fn zeros(
        device: &mut D,
        dtype: DType,
        shape: Shape,
    ) -> Result<Self, TensorError> {
        let storage = device
            .zeros(shape.nbytes(dtype))
            .map_err(device_failed("zeros"))?;
        Ok(Self::from_storage(storage, dtype, shape))
    }

    pub fn to_bytes(
        &self,
        device: &mut D,
    ) -> Result<Vec<u8>, TensorError> {
        if !self.layout.is_contiguous() {
            error!(message = "Reading bytes requires a contiguous layout.", layout = ?self.layout);
            return Err(TensorError::ToBytesNonContiguous);
        }
        device
            .storage_to_bytes(&self.operand())
            .map_err(device_failed("to_bytes"))
    }

    pub fn dtype(&self) -> DType { self.dtype }

    pub fn layout(&self) -> &Layout { &self.layout }

    pub fn shape(&self) -> &Shape { self.layout.shape() }

    pub fn reshape(
        &self,
        shape: Shape,
    ) -> Result<Self, TensorError> {
        Ok(self.view(self.layout.reshape(shape)?))
    }

    pub fn permute(
        &self,
        dims: &[usize],
    ) -> Result<Self, TensorError> {
        Ok(self.view(self.layout.permute(dims)?))
    }

    pub fn narrow(
        &self,
        dim: usize,
        start: usize,
        length: usize,
    ) -> Result<Self, TensorError> {
        Ok(self.view(self.layout.narrow(dim, start, length)?))
    }

    pub fn expand(
        &self,
        shape: Shape,
    ) -> Result<Self, TensorError> {
        Ok(self.view(self.layout.expand(shape)?))
    }

    pub fn neg(
        &self,
        device: &mut D,
    ) -> Result<Self, TensorError> {
        Self::apply(device, Op::Unary(UnaryOp::Neg), [self])
    }

    pub fn exp(
        &self,
        device: &mut D,
    ) -> Result<Self, TensorError> {
        Self::apply(device, Op::Unary(UnaryOp::Exp), [self])
    }

    pub fn sqrt(
        &self,
        device: &mut D,
    ) -> Result<Self, TensorError> {
        Self::apply(device, Op::Unary(UnaryOp::Sqrt), [self])
    }

    pub fn reciprocal(
        &self,
        device: &mut D,
    ) -> Result<Self, TensorError> {
        Self::apply(device, Op::Unary(UnaryOp::Reciprocal), [self])
    }

    pub fn add(
        &self,
        device: &mut D,
        rhs: &Self,
    ) -> Result<Self, TensorError> {
        self.binary(device, BinaryOp::Add, rhs)
    }

    pub fn sub(
        &self,
        device: &mut D,
        rhs: &Self,
    ) -> Result<Self, TensorError> {
        self.binary(device, BinaryOp::Sub, rhs)
    }

    pub fn mul(
        &self,
        device: &mut D,
        rhs: &Self,
    ) -> Result<Self, TensorError> {
        self.binary(device, BinaryOp::Mul, rhs)
    }

    pub fn div(
        &self,
        device: &mut D,
        rhs: &Self,
    ) -> Result<Self, TensorError> {
        self.binary(device, BinaryOp::Div, rhs)
    }

    pub fn sum_keepdim(
        &self,
        device: &mut D,
        dim: usize,
    ) -> Result<Self, TensorError> {
        let op = ReduceOp::Sum;
        Self::apply(
            device,
            Op::Reduce {
                op,
                dim,
            },
            [self],
        )
    }

    pub fn amax_keepdim(
        &self,
        device: &mut D,
        dim: usize,
    ) -> Result<Self, TensorError> {
        let op = ReduceOp::Amax;
        Self::apply(
            device,
            Op::Reduce {
                op,
                dim,
            },
            [self],
        )
    }

    pub fn argmax_keepdim(
        &self,
        device: &mut D,
        dim: usize,
    ) -> Result<Self, TensorError> {
        let op = ReduceOp::Argmax;
        Self::apply(
            device,
            Op::Reduce {
                op,
                dim,
            },
            [self],
        )
    }

    pub fn matmul(
        &self,
        device: &mut D,
        rhs: &Self,
    ) -> Result<Self, TensorError> {
        Self::apply(device, Op::Matmul, [self, rhs])
    }

    pub fn contiguous(
        &self,
        device: &mut D,
    ) -> Result<Self, TensorError> {
        if self.layout.is_contiguous() {
            return Ok(self.view(self.layout));
        }
        Self::apply(device, Op::Copy, [self])
    }

    pub fn to_dtype(
        &self,
        device: &mut D,
        dtype: DType,
    ) -> Result<Self, TensorError> {
        Self::apply(device, Op::ToDtype(dtype), [self])
    }

    pub fn index_select(
        &self,
        device: &mut D,
        dim: usize,
        indices: &Self,
    ) -> Result<Self, TensorError> {
        Self::apply(
            device,
            Op::IndexSelect {
                dim,
            },
            [self, indices],
        )
    }

    pub fn cat(
        device: &mut D,
        tensors: [&Self; 2],
        dim: usize,
    ) -> Result<Self, TensorError> {
        Self::apply(
            device,
            Op::Cat {
                dim,
            },
            tensors,
        )
    }

    pub fn slice_scatter(
        &mut self,
        device: &mut D,
        src: &Self,
        dim: usize,
        start: usize,
    ) -> Result<(), TensorError> {
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
            return Err(TensorError::SliceScatterNonContiguous);
        }
        let storage = Rc::get_mut(&mut self.storage).ok_or_else(|| {
            error!(message = "Slice scatter target shares its storage with another tensor.");
            TensorError::SharedStorage
        })?;
        device
            .run_in_place(
                op,
                Operand::new(storage, &self.layout, self.dtype),
                &[src.operand()],
            )
            .map_err(device_failed(op.name()))
    }

    fn binary(
        &self,
        device: &mut D,
        op: BinaryOp,
        rhs: &Self,
    ) -> Result<Self, TensorError> {
        let shape = Shape::broadcast(self.shape(), rhs.shape())?;
        Self::apply(
            device,
            Op::Binary(op),
            [&self.expand(shape)?, &rhs.expand(shape)?],
        )
    }

    fn apply<const N: usize>(
        device: &mut D,
        op: Op,
        inputs: [&Self; N],
    ) -> Result<Self, TensorError> {
        let (shape, dtype) = output(op, &inputs.map(|input| (input.dtype, *input.shape())))?;
        let storage = device
            .run(op, &inputs.map(Self::operand), &shape, dtype)
            .map_err(device_failed(op.name()))?;
        Ok(Self::from_storage(storage, dtype, shape))
    }

    fn from_storage(
        storage: D::Storage,
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

    fn operand(&self) -> Operand<'_, &D::Storage> {
        Operand::new(&self.storage, &self.layout, self.dtype)
    }
}

fn device_failed<E: Error>(operation: &'static str) -> impl FnOnce(E) -> TensorError {
    move |error| {
        error!(message = "Device operation failed.", operation, %error);
        TensorError::Device
    }
}

#[cfg(test)]
mod tests {
    use std::fmt;

    use crate::nn::TensorError;
    use crate::nn::tensor::device_failed;

    #[test]
    fn device_failure_becomes_the_device_variant() {
        assert_eq!(
            device_failed("matmul")(fmt::Error),
            TensorError::Device,
            "the device cause is logged, not returned"
        );
    }
}
