use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer,
    MTLCommandBuffer,
    MTLCommandEncoder,
    MTLComputeCommandEncoder,
    MTLComputePipelineState,
    MTLDevice,
    MTLLibrary,
    MTLSize,
};

use crate::metal::error::GpuError;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

const QMV_ROWS_PER_GROUP: u32 = 8;
const QMV_THREADS: usize = 64;
const QMM_TILE: u32 = 64;
const QMM_THREADS: usize = 128;
const NORM_THREADS: usize = 512;
const ATTENTION_THREADS: usize = 256;
const ARGMAX_THREADS: usize = 1024;
const ELEMENTWISE_THREADS: usize = 256;
const ROPE_PAIRS: usize = 64;

pub(crate) struct Kernels {
    embed: Pipeline,
    rms_norm: Pipeline,
    qmv: Pipeline,
    qmv_residual: Pipeline,
    qmm: Pipeline,
    qmm_residual: Pipeline,
    rope_store: Pipeline,
    attention: Pipeline,
    swiglu: Pipeline,
    argmax: Pipeline,
}

fn pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    kernel: &'static str,
    threads: usize,
) -> Result<Pipeline, GpuError> {
    let function = library
        .newFunctionWithName(&NSString::from_str(kernel))
        .ok_or(GpuError::MissingKernel(kernel))?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| GpuError::PipelineCreation {
            kernel,
            message: error.localizedDescription().to_string(),
        })?;
    let limit = pipeline.maxTotalThreadsPerThreadgroup();
    if threads > limit {
        return Err(GpuError::ThreadgroupTooLarge {
            kernel,
            required: threads,
            limit,
        });
    }
    Ok(pipeline)
}

impl Kernels {
    pub(crate) fn compile(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, GpuError> {
        let source = NSString::from_str(include_str!("kernels.metal"));
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|error| {
                GpuError::ShaderCompilation(error.localizedDescription().to_string())
            })?;
        let build = |kernel, threads| pipeline(device, &library, kernel, threads);
        Ok(Self {
            embed: build("embed_gather", ELEMENTWISE_THREADS)?,
            rms_norm: build("rms_norm", NORM_THREADS)?,
            qmv: build("qmv", QMV_THREADS)?,
            qmv_residual: build("qmv_residual", QMV_THREADS)?,
            qmm: build("qmm", QMM_THREADS)?,
            qmm_residual: build("qmm_residual", QMM_THREADS)?,
            rope_store: build("rope_store", ROPE_PAIRS)?,
            attention: build("attention", ATTENTION_THREADS)?,
            swiglu: build("swiglu", ELEMENTWISE_THREADS)?,
            argmax: build("argmax", ARGMAX_THREADS)?,
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct View<'buffer> {
    pub(crate) buffer: &'buffer ProtocolObject<dyn MTLBuffer>,
    pub(crate) offset: usize,
}

impl<'buffer> View<'buffer> {
    pub(crate) fn new(buffer: &'buffer ProtocolObject<dyn MTLBuffer>) -> Self {
        Self {
            buffer,
            offset: 0,
        }
    }

    pub(crate) fn at(
        buffer: &'buffer ProtocolObject<dyn MTLBuffer>,
        offset: usize,
    ) -> Self {
        Self {
            buffer,
            offset,
        }
    }

    pub(crate) fn element<T>(
        self,
        index: usize,
    ) -> Result<Self, GpuError> {
        let offset = index
            .checked_mul(size_of::<T>())
            .and_then(|bytes| bytes.checked_add(self.offset))
            .ok_or(GpuError::Overflow {
                what: "buffer offset",
            })?;
        Ok(Self::at(self.buffer, offset))
    }
}

#[derive(Clone, Copy)]
pub(crate) struct QuantView<'buffer> {
    pub(crate) weight: View<'buffer>,
    pub(crate) scales: View<'buffer>,
    pub(crate) biases: View<'buffer>,
    pub(crate) rows: u32,
    pub(crate) cols: u32,
}

#[repr(C)]
struct EmbedParams {
    rows: u32,
    hidden: u32,
}

#[repr(C)]
struct NormParams {
    dim: u32,
    eps: f32,
}

#[repr(C)]
struct MatmulParams {
    rows: u32,
    in_features: u32,
    out_features: u32,
}

#[repr(C)]
pub(crate) struct RopeParams {
    pub(crate) rows: u32,
    pub(crate) offset: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) capacity: u32,
}

#[repr(C)]
pub(crate) struct AttentionParams {
    pub(crate) rows: u32,
    pub(crate) offset: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) capacity: u32,
    pub(crate) scale: f32,
}

#[repr(C)]
struct ElementwiseParams {
    count: u32,
}

#[repr(C)]
struct ArgmaxParams {
    count: u32,
    slot: u32,
}

pub(crate) struct KvCache<'buffer> {
    pub(crate) keys: View<'buffer>,
    pub(crate) values: View<'buffer>,
}

fn size(
    width: usize,
    height: usize,
    depth: usize,
) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}

fn count(value: u32) -> Result<usize, GpuError> {
    usize::try_from(value).map_err(|_| GpuError::Overflow {
        what: "dispatch size",
    })
}

fn unsupported(what: &'static str) -> GpuError {
    GpuError::OutOfBounds {
        what,
    }
}

pub(crate) struct Encoder<'kernels> {
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    kernels: &'kernels Kernels,
}

impl<'kernels> Encoder<'kernels> {
    pub(crate) fn new(
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        kernels: &'kernels Kernels,
    ) -> Result<Self, GpuError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(GpuError::EncoderCreation)?;
        Ok(Self {
            encoder,
            kernels,
        })
    }

    pub(crate) fn end(self) { self.encoder.endEncoding(); }

    fn bind<P>(
        &self,
        pipeline: &Pipeline,
        views: &[View<'_>],
        params: &P,
    ) {
        self.encoder.setComputePipelineState(pipeline);
        let mut index = 0_usize;
        for view in views {
            // SAFETY: every view references a live buffer owned by the model
            // or the session, which outlive the command buffer, and the
            // kernels index those buffers within the dimensions validated
            // by the caller.
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(view.buffer), view.offset, index)
            };
            index = index.saturating_add(1);
        }
        // SAFETY: `params` is a live `#[repr(C)]` value whose layout matches
        // the kernel's constant struct, and Metal copies its bytes before
        // `setBytes` returns.
        unsafe {
            self.encoder.setBytes_length_atIndex(
                NonNull::from(params).cast(),
                size_of::<P>(),
                index,
            )
        };
    }

    fn groups(
        &self,
        groups: MTLSize,
        threads: MTLSize,
    ) {
        self.encoder
            .dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
    }

    fn threads(
        &self,
        grid: MTLSize,
        threads: MTLSize,
    ) {
        self.encoder
            .dispatchThreads_threadsPerThreadgroup(grid, threads);
    }

    pub(crate) fn embed(
        &self,
        ids: View<'_>,
        table: &QuantView<'_>,
        out: View<'_>,
        rows: u32,
    ) -> Result<(), GpuError> {
        let params = EmbedParams {
            rows,
            hidden: table.cols,
        };
        self.bind(
            &self.kernels.embed,
            &[ids, table.weight, table.scales, table.biases, out],
            &params,
        );
        let words = count(table.cols)?.checked_div(8).unwrap_or(0);
        self.threads(
            size(words, count(rows)?, 1),
            size(words.min(ELEMENTWISE_THREADS), 1, 1),
        );
        Ok(())
    }

    pub(crate) fn rms_norm(
        &self,
        x: View<'_>,
        weight: View<'_>,
        out: View<'_>,
        rows: u32,
        dim: u32,
        eps: f32,
    ) -> Result<(), GpuError> {
        let params = NormParams {
            dim,
            eps,
        };
        self.bind(&self.kernels.rms_norm, &[x, weight, out], &params);
        self.groups(size(count(rows)?, 1, 1), size(NORM_THREADS, 1, 1));
        Ok(())
    }

    pub(crate) fn matmul(
        &self,
        x: View<'_>,
        weights: &QuantView<'_>,
        out: View<'_>,
        rows: u32,
        residual: bool,
    ) -> Result<(), GpuError> {
        let params = MatmulParams {
            rows,
            in_features: weights.cols,
            out_features: weights.rows,
        };
        let views = [x, weights.weight, weights.scales, weights.biases, out];
        if rows == 1 {
            if weights.cols.checked_rem(32) != Some(0)
                || weights.rows.checked_rem(QMV_ROWS_PER_GROUP) != Some(0)
            {
                return Err(unsupported("qmv dimensions"));
            }
            let pipeline = if residual {
                &self.kernels.qmv_residual
            } else {
                &self.kernels.qmv
            };
            self.bind(pipeline, &views, &params);
            let groups = count(weights.rows.div_ceil(QMV_ROWS_PER_GROUP))?;
            self.groups(size(groups, count(rows)?, 1), size(QMV_THREADS, 1, 1));
        } else {
            if weights.cols.checked_rem(QMM_TILE) != Some(0)
                || weights.rows.checked_rem(QMM_TILE) != Some(0)
            {
                return Err(unsupported("qmm dimensions"));
            }
            let pipeline = if residual {
                &self.kernels.qmm_residual
            } else {
                &self.kernels.qmm
            };
            self.bind(pipeline, &views, &params);
            self.groups(
                size(
                    count(weights.rows.div_ceil(QMM_TILE))?,
                    count(rows.div_ceil(QMM_TILE))?,
                    1,
                ),
                size(QMM_THREADS, 1, 1),
            );
        }
        Ok(())
    }

    pub(crate) fn rope_store(
        &self,
        qkv: [View<'_>; 3],
        cache: &KvCache<'_>,
        frequencies: View<'_>,
        params: &RopeParams,
    ) -> Result<(), GpuError> {
        let [q, k, v] = qkv;
        self.bind(
            &self.kernels.rope_store,
            &[q, k, v, cache.keys, cache.values, frequencies],
            params,
        );
        let heads = params
            .kv_heads
            .checked_mul(2)
            .and_then(|kv| kv.checked_add(params.heads))
            .ok_or(GpuError::Overflow {
                what: "rope heads",
            })?;
        self.threads(
            size(ROPE_PAIRS, count(heads)?, count(params.rows)?),
            size(ROPE_PAIRS, 1, 1),
        );
        Ok(())
    }

    pub(crate) fn attention(
        &self,
        q: View<'_>,
        cache: &KvCache<'_>,
        out: View<'_>,
        params: &AttentionParams,
    ) -> Result<(), GpuError> {
        self.bind(
            &self.kernels.attention,
            &[q, cache.keys, cache.values, out],
            params,
        );
        self.groups(
            size(count(params.heads)?, count(params.rows)?, 1),
            size(ATTENTION_THREADS, 1, 1),
        );
        Ok(())
    }

    pub(crate) fn swiglu(
        &self,
        gate: View<'_>,
        up: View<'_>,
        elements: u32,
    ) -> Result<(), GpuError> {
        let params = ElementwiseParams {
            count: elements,
        };
        self.bind(&self.kernels.swiglu, &[gate, up], &params);
        self.threads(
            size(count(elements)?, 1, 1),
            size(ELEMENTWISE_THREADS, 1, 1),
        );
        Ok(())
    }

    pub(crate) fn argmax(
        &self,
        logits: View<'_>,
        outputs: View<'_>,
        next: View<'_>,
        vocab: u32,
        slot: u32,
    ) {
        let params = ArgmaxParams {
            count: vocab,
            slot,
        };
        self.bind(&self.kernels.argmax, &[logits, outputs, next], &params);
        self.groups(size(1, 1, 1), size(ARGMAX_THREADS, 1, 1));
    }
}
