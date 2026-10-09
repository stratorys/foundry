use std::rc::Rc;

use crate::core::primitive::{
    BinaryOp,
    ReduceOp,
    UnaryOp,
    binary_rule,
    cast_rule,
    concat_rule,
    copy_rule,
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
    Layout,
    Operand,
    OperandMut,
    Shape,
};

pub struct Tensor<B: Backend> {
    storage: Rc<B::Storage>,
    layout: Layout,
    dtype: DType,
}

impl<B: Backend> Tensor<B> {
    pub fn upload(
        backend: &mut B,
        bytes: &[u8],
        dtype: DType,
        shape: Shape,
    ) -> Result<Self, B::Error> {
        let storage = backend.upload(bytes, dtype, &shape)?;
        Ok(Self::from_storage(storage, dtype, shape)?)
    }

    pub fn zeros(
        backend: &mut B,
        dtype: DType,
        shape: Shape,
    ) -> Result<Self, B::Error> {
        let storage = backend.zeros(dtype, &shape)?;
        Ok(Self::from_storage(storage, dtype, shape)?)
    }

    pub fn download(
        &self,
        backend: &mut B,
    ) -> Result<Vec<u8>, B::Error> {
        backend.download(self.operand())
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
    ) -> Result<Self, B::Error> {
        self.unary(backend, UnaryOp::Neg)
    }

    pub fn exp(
        &self,
        backend: &mut B,
    ) -> Result<Self, B::Error> {
        self.unary(backend, UnaryOp::Exp)
    }

    pub fn sqrt(
        &self,
        backend: &mut B,
    ) -> Result<Self, B::Error> {
        self.unary(backend, UnaryOp::Sqrt)
    }

    pub fn recip(
        &self,
        backend: &mut B,
    ) -> Result<Self, B::Error> {
        self.unary(backend, UnaryOp::Recip)
    }

    pub fn add(
        &self,
        backend: &mut B,
        rhs: &Self,
    ) -> Result<Self, B::Error> {
        self.binary(backend, BinaryOp::Add, rhs)
    }

    pub fn sub(
        &self,
        backend: &mut B,
        rhs: &Self,
    ) -> Result<Self, B::Error> {
        self.binary(backend, BinaryOp::Sub, rhs)
    }

    pub fn mul(
        &self,
        backend: &mut B,
        rhs: &Self,
    ) -> Result<Self, B::Error> {
        self.binary(backend, BinaryOp::Mul, rhs)
    }

    pub fn div(
        &self,
        backend: &mut B,
        rhs: &Self,
    ) -> Result<Self, B::Error> {
        self.binary(backend, BinaryOp::Div, rhs)
    }

    pub fn sum(
        &self,
        backend: &mut B,
        axis: usize,
    ) -> Result<Self, B::Error> {
        self.reduce(backend, ReduceOp::Sum, axis)
    }

    pub fn max(
        &self,
        backend: &mut B,
        axis: usize,
    ) -> Result<Self, B::Error> {
        self.reduce(backend, ReduceOp::Max, axis)
    }

    pub fn argmax(
        &self,
        backend: &mut B,
        axis: usize,
    ) -> Result<Self, B::Error> {
        self.reduce(backend, ReduceOp::Argmax, axis)
    }

    pub fn matmul(
        &self,
        backend: &mut B,
        rhs: &Self,
    ) -> Result<Self, B::Error> {
        let (dtype, shape) = matmul_rule(self.dtype, self.shape(), rhs.dtype, rhs.shape())?;
        let storage = backend.matmul(self.operand(), rhs.operand())?;
        Ok(Self::from_storage(storage, dtype, shape)?)
    }

    pub fn contiguous(
        &self,
        backend: &mut B,
    ) -> Result<Self, B::Error> {
        if self.layout.is_contiguous() {
            return Ok(self.view(self.layout));
        }
        let (dtype, shape) = copy_rule(self.dtype, self.shape())?;
        let storage = backend.copy(self.operand())?;
        Ok(Self::from_storage(storage, dtype, shape)?)
    }

    pub fn cast(
        &self,
        backend: &mut B,
        dtype: DType,
    ) -> Result<Self, B::Error> {
        let (dtype_output, shape) = cast_rule(self.dtype, dtype, self.shape())?;
        let storage = backend.cast(self.operand(), dtype)?;
        Ok(Self::from_storage(storage, dtype_output, shape)?)
    }

    pub fn gather(
        &self,
        backend: &mut B,
        indices: &Self,
    ) -> Result<Self, B::Error> {
        let (dtype, shape) = gather_rule(self.dtype, self.shape(), indices.dtype, indices.shape())?;
        let storage = backend.gather(self.operand(), indices.operand())?;
        Ok(Self::from_storage(storage, dtype, shape)?)
    }

    pub fn concat(
        &self,
        backend: &mut B,
        rhs: &Self,
        axis: usize,
    ) -> Result<Self, B::Error> {
        let (dtype, shape) = concat_rule(self.dtype, self.shape(), rhs.dtype, rhs.shape(), axis)?;
        let storage = backend.concat(self.operand(), rhs.operand(), axis)?;
        Ok(Self::from_storage(storage, dtype, shape)?)
    }

    pub fn slice_update(
        &mut self,
        backend: &mut B,
        update: &Self,
        axis: usize,
        start: usize,
    ) -> Result<(), B::Error> {
        slice_update_rule(
            self.dtype,
            self.shape(),
            update.dtype,
            update.shape(),
            axis,
            start,
        )?;
        if !self.layout.is_contiguous() {
            return Err(CoreError::SliceUpdateNonContiguous.into());
        }
        let storage = Rc::get_mut(&mut self.storage).ok_or(CoreError::SharedStorage)?;
        backend.slice_update(
            OperandMut::new(storage, &self.layout, self.dtype),
            update.operand(),
            axis,
            start,
        )
    }

    fn unary(
        &self,
        backend: &mut B,
        op: UnaryOp,
    ) -> Result<Self, B::Error> {
        let (dtype, shape) = unary_rule(self.dtype, self.shape())?;
        let storage = backend.unary(op, self.operand())?;
        Ok(Self::from_storage(storage, dtype, shape)?)
    }

    fn binary(
        &self,
        backend: &mut B,
        op: BinaryOp,
        rhs: &Self,
    ) -> Result<Self, B::Error> {
        let (dtype, shape) = binary_rule(self.dtype, self.shape(), rhs.dtype, rhs.shape())?;
        let (lhs, rhs) = (self.broadcast_as(shape)?, rhs.broadcast_as(shape)?);
        let storage = backend.binary(op, lhs.operand(), rhs.operand())?;
        Ok(Self::from_storage(storage, dtype, shape)?)
    }

    fn reduce(
        &self,
        backend: &mut B,
        op: ReduceOp,
        axis: usize,
    ) -> Result<Self, B::Error> {
        let (dtype, shape) = reduce_rule(op, self.dtype, self.shape(), axis)?;
        let storage = backend.reduce(op, self.operand(), axis)?;
        Ok(Self::from_storage(storage, dtype, shape)?)
    }

    fn from_storage(
        storage: B::Storage,
        dtype: DType,
        shape: Shape,
    ) -> Result<Self, CoreError> {
        Ok(Self {
            storage: Rc::new(storage),
            layout: Layout::contiguous(shape)?,
            dtype,
        })
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
}

#[cfg(test)]
mod tests {
    use crate::core::{
        Layout,
        Shape,
    };

    fn contiguous(dims: &[usize]) -> Layout {
        Layout::contiguous(Shape::try_from(dims).expect("valid shape")).expect("valid layout")
    }

    #[test]
    fn narrow_on_leading_axis_needs_no_copy() {
        let layout = contiguous(&[8, 16, 128])
            .narrow(0, 2, 3)
            .expect("in bounds");
        assert!(layout.is_contiguous(), "leading narrow stays contiguous");
        assert_eq!(layout.offset(), 2 * 16 * 128, "offset moved by two rows");
    }

    #[test]
    fn narrow_after_permute_needs_a_copy() {
        let layout = contiguous(&[16, 8, 128])
            .permute(&[1, 0, 2])
            .expect("valid permutation")
            .narrow(1, 0, 4)
            .expect("in bounds");
        assert_eq!(layout.shape().dims(), &[8, 4, 128], "narrowed dims");
        assert!(!layout.is_contiguous(), "permuted narrow is not contiguous");
    }
}
