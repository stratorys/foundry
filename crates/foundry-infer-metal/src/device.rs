use foundry_infer_core::ByteSize;

use crate::error::MetalError;

pub(crate) fn allocation_length(
    bytes: ByteSize,
    limit: usize,
) -> Result<usize, MetalError> {
    if bytes.bytes() == 0 {
        return Err(MetalError::EmptyAllocation);
    }
    let too_large = || MetalError::AllocationTooLarge {
        requested: bytes,
        limit: byte_size(limit),
    };
    let length = usize::try_from(bytes.bytes()).map_err(|_| too_large())?;
    if length > limit {
        Err(too_large())
    } else {
        Ok(length)
    }
}

pub(crate) fn byte_size(length: usize) -> ByteSize {
    ByteSize::from_bytes(u64::try_from(length).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use foundry_infer_core::ByteSize;

    use super::allocation_length;
    use crate::error::MetalError;

    #[test]
    fn allocation_lengths_are_checked_against_the_limit() {
        let cases = [
            (0, 64, Err(MetalError::EmptyAllocation)),
            (1, 64, Ok(1)),
            (64, 64, Ok(64)),
            (
                65,
                64,
                Err(MetalError::AllocationTooLarge {
                    requested: ByteSize::from_bytes(65),
                    limit: ByteSize::from_bytes(64),
                }),
            ),
            (
                u64::MAX,
                usize::MAX,
                if usize::BITS < u64::BITS {
                    Err(MetalError::AllocationTooLarge {
                        requested: ByteSize::from_bytes(u64::MAX),
                        limit: super::byte_size(usize::MAX),
                    })
                } else {
                    Ok(usize::MAX)
                },
            ),
        ];
        for (bytes, limit, expected) in cases {
            assert_eq!(
                allocation_length(ByteSize::from_bytes(bytes), limit),
                expected,
                "{bytes} bytes against a {limit} byte limit"
            );
        }
    }
}
