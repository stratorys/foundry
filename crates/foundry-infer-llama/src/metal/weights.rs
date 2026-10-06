use std::collections::BTreeMap;
use std::fs::File;
use std::io::{
    Read,
    Seek,
    SeekFrom,
};
use std::path::Path;

use objc2_metal::MTLBuffer;

use crate::checkpoint::{
    Checkpoint,
    QuantizedTensor,
    TensorRef,
};
use crate::error::LlamaError;
use crate::metal::context::{
    Buffer,
    Context,
    allocated_size,
};
use crate::metal::error::GpuError;
use crate::metal::kernels::{
    QuantView,
    View,
};
use crate::safetensors::Header;

const ALIGNMENT: usize = 256;

#[derive(Clone, Copy, Debug)]
pub(crate) struct QuantOffsets {
    weight: usize,
    scales: usize,
    biases: usize,
    rows: u32,
    cols: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerOffsets {
    pub(crate) input_norm: usize,
    pub(crate) q_proj: QuantOffsets,
    pub(crate) k_proj: QuantOffsets,
    pub(crate) v_proj: QuantOffsets,
    pub(crate) o_proj: QuantOffsets,
    pub(crate) post_attention_norm: usize,
    pub(crate) gate_proj: QuantOffsets,
    pub(crate) up_proj: QuantOffsets,
    pub(crate) down_proj: QuantOffsets,
}

pub(crate) struct ResidentWeights {
    buffer: Buffer,
    pub(crate) embed: QuantOffsets,
    pub(crate) layers: Vec<LayerOffsets>,
    pub(crate) norm: usize,
    pub(crate) tensor_bytes: u64,
}

struct Placements {
    offsets: BTreeMap<String, usize>,
}

impl Placements {
    fn get(
        &self,
        tensor: &TensorRef,
    ) -> Result<usize, GpuError> {
        self.offsets
            .get(&tensor.name)
            .copied()
            .ok_or(GpuError::OutOfBounds {
                what: "weight placement",
            })
    }

    fn quantized(
        &self,
        tensor: &QuantizedTensor,
    ) -> Result<QuantOffsets, GpuError> {
        Ok(QuantOffsets {
            weight: self.get(&tensor.weight)?,
            scales: self.get(&tensor.scales)?,
            biases: self.get(&tensor.biases)?,
            rows: tensor.rows,
            cols: tensor.cols,
        })
    }
}

fn overflow() -> GpuError {
    GpuError::Overflow {
        what: "resident weight layout",
    }
}

fn layout(checkpoint: &Checkpoint) -> Result<(Placements, usize), GpuError> {
    let (offsets, total) = checkpoint.tensors().try_fold(
        (BTreeMap::new(), 0_usize),
        |(mut offsets, cursor), tensor| {
            let start = cursor
                .checked_next_multiple_of(ALIGNMENT)
                .ok_or_else(overflow)?;
            let len = usize::try_from(tensor.len).map_err(|_| overflow())?;
            let end = start.checked_add(len).ok_or_else(overflow)?;
            offsets.insert(tensor.name.clone(), start);
            Ok::<_, GpuError>((offsets, end))
        },
    )?;
    Ok((
        Placements {
            offsets,
        },
        total,
    ))
}

impl ResidentWeights {
    pub(crate) fn load(
        context: &Context,
        path: &Path,
        header: &Header,
        checkpoint: &Checkpoint,
    ) -> Result<Self, LlamaError> {
        let (placements, total) = layout(checkpoint)?;
        let buffer = context.shared(total)?;
        let mut file = File::open(path).map_err(|error| LlamaError::io(path, error))?;
        let mut ordered: Vec<&TensorRef> = checkpoint.tensors().collect();
        ordered.sort_by_key(|tensor| tensor.start);
        let base = buffer.contents().cast::<u8>().as_ptr();
        let capacity = buffer.length();
        let tensor_bytes = ordered.iter().try_fold(0_u64, |sum, tensor| {
            let offset = placements.get(tensor)?;
            let len = usize::try_from(tensor.len).map_err(|_| overflow())?;
            if offset.checked_add(len).is_none_or(|end| end > capacity) {
                return Err(LlamaError::from(overflow()));
            }
            let position = header
                .data_start
                .checked_add(tensor.start)
                .ok_or_else(overflow)?;
            file.seek(SeekFrom::Start(position))
                .map_err(|error| LlamaError::io(path, error))?;
            // SAFETY: the buffer was just allocated with shared storage and
            // `capacity` bytes, `offset + len <= capacity` was checked above,
            // placements of distinct tensors never overlap, and no GPU work
            // references the buffer yet, so this is the only live reference
            // to the range.
            let destination = unsafe { std::slice::from_raw_parts_mut(base.add(offset), len) };
            file.read_exact(destination)
                .map_err(|error| LlamaError::io(path, error))?;
            sum.checked_add(tensor.len).ok_or_else(|| overflow().into())
        })?;
        let layers = checkpoint
            .layers
            .iter()
            .map(|layer| {
                Ok(LayerOffsets {
                    input_norm: placements.get(&layer.input_norm)?,
                    q_proj: placements.quantized(&layer.q_proj)?,
                    k_proj: placements.quantized(&layer.k_proj)?,
                    v_proj: placements.quantized(&layer.v_proj)?,
                    o_proj: placements.quantized(&layer.o_proj)?,
                    post_attention_norm: placements.get(&layer.post_attention_norm)?,
                    gate_proj: placements.quantized(&layer.gate_proj)?,
                    up_proj: placements.quantized(&layer.up_proj)?,
                    down_proj: placements.quantized(&layer.down_proj)?,
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok(Self {
            embed: placements.quantized(&checkpoint.embed)?,
            norm: placements.get(&checkpoint.norm)?,
            layers,
            buffer,
            tensor_bytes,
        })
    }

    pub(crate) fn view(
        &self,
        offset: usize,
    ) -> View<'_> {
        View::at(&self.buffer, offset)
    }

    pub(crate) fn quantized(
        &self,
        offsets: &QuantOffsets,
    ) -> QuantView<'_> {
        QuantView {
            weight: self.view(offsets.weight),
            scales: self.view(offsets.scales),
            biases: self.view(offsets.biases),
            rows: offsets.rows,
            cols: offsets.cols,
        }
    }

    pub(crate) fn allocated_bytes(&self) -> usize { allocated_size(&self.buffer) }
}
