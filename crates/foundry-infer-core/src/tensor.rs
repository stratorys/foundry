use crate::dtype::DType;
use crate::error::CoreError;
use crate::memory::{
    Alignment,
    ByteSize,
};
use crate::shape::Shape;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorDesc {
    dtype: DType,
    shape: Shape,
}

impl TensorDesc {
    pub fn new(
        dtype: DType,
        shape: Shape,
    ) -> Self {
        Self {
            dtype,
            shape,
        }
    }

    pub fn dtype(&self) -> DType { self.dtype }

    pub fn shape(&self) -> &Shape { &self.shape }

    pub fn element_count(&self) -> Result<u64, CoreError> { self.shape.element_count() }

    pub fn byte_size(&self) -> Result<ByteSize, CoreError> {
        ByteSize::from_bytes(self.element_count()?).checked_mul(self.dtype.size_in_bytes())
    }

    pub fn aligned_byte_size(
        &self,
        alignment: Alignment,
    ) -> Result<ByteSize, CoreError> {
        self.byte_size()?.align_up(alignment)
    }
}

#[cfg(test)]
mod tests {
    use super::TensorDesc;
    use crate::dtype::DType;
    use crate::error::CoreError;
    use crate::memory::{
        Alignment,
        ByteSize,
        MemoryBudget,
        MemorySpace,
    };
    use crate::shape::Shape;

    #[test]
    fn byte_size_is_elements_times_element_width() {
        let tensor = TensorDesc::new(DType::BF16, Shape::new([3, 5]));
        assert_eq!(tensor.dtype(), DType::BF16, "dtype is kept");
        assert_eq!(tensor.shape(), &Shape::new([3, 5]), "shape is kept");
        assert_eq!(tensor.element_count(), Ok(15), "elements of [3, 5]");
        assert_eq!(
            tensor.byte_size(),
            Ok(ByteSize::from_bytes(30)),
            "15 BF16 elements"
        );
    }

    #[test]
    fn aligned_byte_size_rounds_up() -> Result<(), CoreError> {
        let tensor = TensorDesc::new(DType::F32, Shape::new([3]));
        assert_eq!(
            tensor.aligned_byte_size(Alignment::new(256)?),
            Ok(ByteSize::from_bytes(256)),
            "12 bytes padded to 256"
        );
        Ok(())
    }

    #[test]
    fn an_overflowing_byte_size_is_an_error() {
        let tensor = TensorDesc::new(DType::F32, Shape::new([u64::MAX / 2]));
        assert_eq!(
            tensor.byte_size(),
            Err(CoreError::Overflow),
            "u64::MAX / 2 F32 elements overflow"
        );
    }

    #[test]
    fn describes_more_than_20_gib_of_weights_without_allocating_them() -> Result<(), CoreError> {
        let projection = TensorDesc::new(DType::BF16, Shape::new([8192, 28672]));
        let mut layers = (0..48).flat_map(|_| [&projection, &projection, &projection]);
        let total = layers.try_fold(ByteSize::from_bytes(0), |total, tensor| {
            total.checked_add(tensor.byte_size()?)
        })?;

        assert_eq!(
            total,
            ByteSize::from_bytes(48 * 3 * 8192 * 28672 * 2),
            "total weight bytes"
        );
        assert!(
            total > ByteSize::from_gib(20)?,
            "the workload exceeds 20 GiB"
        );
        let budget = MemoryBudget::new(MemorySpace::Device, ByteSize::from_gib(7)?);
        assert!(
            budget.check(total).is_err(),
            "the workload does not fit the device budget"
        );
        assert!(
            budget.check(projection.byte_size()?).is_ok(),
            "one projection fits the device budget"
        );
        Ok(())
    }
}
