#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CpuError {
    #[error("Element index overflows usize.")]
    IndexOverflow,

    #[error("Element is outside its storage.")]
    IndexOutOfStorage,
}
