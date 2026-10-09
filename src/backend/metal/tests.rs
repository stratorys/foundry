use crate::backend::metal::{
    MetalBackend,
    MetalError,
};
use crate::core::{
    DType,
    Shape,
    Tensor,
};

fn backend() -> MetalBackend { MetalBackend::new().expect("a Metal device is available") }

fn shape(dims: &[usize]) -> Shape { Shape::try_from(dims).expect("the shape is valid") }

fn round_trip(
    backend: &mut MetalBackend,
    bytes: &[u8],
    dtype: DType,
    dims: &[usize],
) -> Vec<u8> {
    Tensor::upload(backend, bytes, dtype, shape(dims))
        .expect("the upload succeeds")
        .download(backend)
        .expect("the download succeeds")
}

#[test]
fn f32_round_trip_returns_the_same_bytes() {
    let bytes: Vec<u8> = [1.0_f32, -2.5, 1.5e-3, f32::MIN_POSITIVE, -0.0, f32::MAX]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let downloaded = round_trip(&mut backend(), &bytes, DType::F32, &[2, 3]);
    assert_eq!(downloaded, bytes, "f32 bytes survive the round trip");
}

#[test]
fn f16_round_trip_returns_the_same_bytes() {
    let bytes: Vec<u8> = [0x3C00_u16, 0xC100, 0x4248, 0x0001, 0x8000, 0x7BFF]
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect();
    let downloaded = round_trip(&mut backend(), &bytes, DType::F16, &[3, 2]);
    assert_eq!(downloaded, bytes, "f16 bytes survive the round trip");
}

#[test]
fn bf16_round_trip_returns_the_same_bytes() {
    let bytes: Vec<u8> = [0x3F80_u16, 0xC020, 0x4049, 0x0001, 0x8000, 0x7F7F]
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect();
    let downloaded = round_trip(&mut backend(), &bytes, DType::BF16, &[6]);
    assert_eq!(downloaded, bytes, "bf16 bytes survive the round trip");
}

#[test]
fn u32_round_trip_returns_the_same_bytes() {
    let bytes: Vec<u8> = [0_u32, 1, 128_000, u32::MAX]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let downloaded = round_trip(&mut backend(), &bytes, DType::U32, &[2, 1, 2]);
    assert_eq!(downloaded, bytes, "u32 bytes survive the round trip");
}

#[test]
fn empty_round_trip_returns_no_bytes() {
    let downloaded = round_trip(&mut backend(), &[], DType::F32, &[0, 3]);
    assert!(downloaded.is_empty(), "an empty tensor downloads no bytes");
}

#[test]
fn zeros_download_only_zero_bytes() {
    let mut backend = backend();
    [DType::F32, DType::F16, DType::BF16, DType::U32]
        .into_iter()
        .for_each(|dtype| {
            let downloaded = Tensor::zeros(&mut backend, dtype, shape(&[2, 3]))
                .expect("zeros succeeds")
                .download(&mut backend)
                .expect("the download succeeds");
            assert_eq!(
                downloaded.len(),
                6 * dtype.size_bytes(),
                "zeros of {dtype:?} has the shape's byte length"
            );
            assert!(
                downloaded.iter().all(|&byte| byte == 0),
                "zeros of {dtype:?} downloads only zero bytes"
            );
        });
}

#[test]
fn zeros_after_upload_keeps_both_values() {
    let mut backend = backend();
    let bytes: Vec<u8> = [7_u32, 8, 9]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let uploaded =
        Tensor::upload(&mut backend, &bytes, DType::U32, shape(&[3])).expect("the upload succeeds");
    let zeros = Tensor::zeros(&mut backend, DType::U32, shape(&[3])).expect("zeros succeeds");
    assert_eq!(
        zeros.download(&mut backend).expect("the download succeeds"),
        vec![0; 12],
        "zeros downloads zero bytes"
    );
    assert_eq!(
        uploaded
            .download(&mut backend)
            .expect("the download succeeds"),
        bytes,
        "the earlier upload is unchanged"
    );
}

#[test]
fn upload_with_wrong_byte_length_fails() {
    let result = Tensor::upload(&mut backend(), &[0; 10], DType::F32, shape(&[2, 2]));
    assert!(
        matches!(
            result,
            Err(MetalError::ByteLengthMismatch {
                bytes: 10,
                bytes_expected: 16
            })
        ),
        "a byte length that does not match shape × dtype is rejected"
    );
}

#[test]
fn narrow_on_leading_axis_downloads_the_sub_range() {
    let mut backend = backend();
    let bytes: Vec<u8> = (0_u32..6).flat_map(u32::to_le_bytes).collect();
    let narrowed = Tensor::upload(&mut backend, &bytes, DType::U32, shape(&[3, 2]))
        .expect("the upload succeeds")
        .narrow(0, 1, 2)
        .expect("the narrow is valid");
    let expected: Vec<u8> = (2_u32..6).flat_map(u32::to_le_bytes).collect();
    assert_eq!(
        narrowed
            .download(&mut backend)
            .expect("the download succeeds"),
        expected,
        "the narrowed rows are downloaded"
    );
}

#[test]
fn permuted_download_is_rejected() {
    let mut backend = backend();
    let bytes: Vec<u8> = (0_u32..6).flat_map(u32::to_le_bytes).collect();
    let permuted = Tensor::upload(&mut backend, &bytes, DType::U32, shape(&[3, 2]))
        .expect("the upload succeeds")
        .permute(&[1, 0])
        .expect("the permutation is valid");
    assert!(
        matches!(
            permuted.download(&mut backend),
            Err(MetalError::DownloadNonContiguous)
        ),
        "a non-contiguous layout cannot be downloaded"
    );
}
