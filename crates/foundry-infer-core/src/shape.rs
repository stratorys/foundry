use crate::error::CoreError;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<u64>,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<u64>>) -> Self {
        Self {
            dims: dims.into(),
        }
    }

    pub fn scalar() -> Self {
        Self {
            dims: Vec::new(),
        }
    }

    pub fn rank(&self) -> usize { self.dims.len() }

    pub fn dims(&self) -> &[u64] { &self.dims }

    pub fn element_count(&self) -> Result<u64, CoreError> {
        if self.dims.contains(&0) {
            return Ok(0);
        }
        self.dims.iter().try_fold(1_u64, |count, &dim| {
            count.checked_mul(dim).ok_or(CoreError::Overflow)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Shape;
    use crate::error::CoreError;

    #[test]
    fn a_scalar_has_one_element() {
        let shape = Shape::scalar();
        assert_eq!(shape.rank(), 0, "a scalar has rank 0");
        assert_eq!(shape.element_count(), Ok(1), "a scalar has one element");
    }

    #[test]
    fn counts_the_product_of_the_dimensions() {
        let shape = Shape::new([2, 3, 4]);
        assert_eq!(shape.rank(), 3, "rank of [2, 3, 4]");
        assert_eq!(shape.dims(), &[2, 3, 4], "dimensions are kept in order");
        assert_eq!(shape.element_count(), Ok(24), "elements of [2, 3, 4]");
    }

    #[test]
    fn a_zero_dimension_has_no_elements() {
        let shapes = [
            Shape::new([8, 0, 8]),
            Shape::new([0]),
            Shape::new([u64::MAX, 2, 0]),
            Shape::new([0, u64::MAX, u64::MAX]),
        ];
        for shape in shapes {
            assert_eq!(
                shape.element_count(),
                Ok(0),
                "{:?} has a zero dimension",
                shape.dims()
            );
        }
    }

    #[test]
    fn an_overflowing_product_is_an_error() {
        assert_eq!(
            Shape::new([u64::MAX, 2]).element_count(),
            Err(CoreError::Overflow),
            "u64::MAX × 2 overflows"
        );
    }
}
