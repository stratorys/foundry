use half::f16;
use objc2_metal::MTLCommandBuffer;

use crate::config::LlamaConfig;
use crate::error::{
    LlamaError,
    RequestError,
};
use crate::metal::context::{
    Buffer,
    CommandBuffer,
    Context,
    allocated_size,
    read,
    wait,
    write,
};
use crate::metal::error::GpuError;
use crate::metal::kernels::{
    AttentionParams,
    Encoder,
    KvCache,
    RopeParams,
    View,
};
use crate::metal::model::LlamaModel;

pub const CONTEXT_CAPACITY: usize = 640;
pub const PREFILL_CAPACITY: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Embedding,
    AttentionNorm(u32),
    Query(u32),
    Key(u32),
    Value(u32),
    Attention(u32),
    AttentionResidual(u32),
    MlpNorm(u32),
    Gate(u32),
    Up(u32),
    Swiglu(u32),
    Output(u32),
    FinalNorm,
    Logits,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionMemory {
    pub kv_cache_bytes: usize,
    pub activation_bytes: usize,
}

struct Dims {
    hidden: usize,
    query: usize,
    kv: usize,
    intermediate: usize,
    vocab: usize,
    kv_layer_halves: usize,
}

impl Dims {
    fn new(config: &LlamaConfig) -> Result<Self, GpuError> {
        let size = |value: u32| {
            usize::try_from(value).map_err(|_| GpuError::Overflow {
                what: "model dimension",
            })
        };
        let kv = size(config.kv_dim())?;
        Ok(Self {
            hidden: size(config.hidden)?,
            query: size(config.query_dim())?,
            kv,
            intermediate: size(config.intermediate)?,
            vocab: size(config.vocab)?,
            kv_layer_halves: kv.checked_mul(CONTEXT_CAPACITY).ok_or(GpuError::Overflow {
                what: "KV cache size",
            })?,
        })
    }
}

struct Activations {
    ids: Buffer,
    next: Buffer,
    outputs: Buffer,
    x: Buffer,
    normed: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attention: Buffer,
    gate: Buffer,
    up: Buffer,
    logits: Buffer,
}

impl Activations {
    fn new(
        context: &Context,
        dims: &Dims,
    ) -> Result<Self, GpuError> {
        let rows = |width: usize| {
            width
                .checked_mul(PREFILL_CAPACITY)
                .ok_or(GpuError::Overflow {
                    what: "activation size",
                })
        };
        Ok(Self {
            ids: context.shared_elements::<u32>(PREFILL_CAPACITY)?,
            next: context.shared_elements::<u32>(1)?,
            outputs: context.shared_elements::<u32>(CONTEXT_CAPACITY)?,
            x: context.shared_elements::<f16>(rows(dims.hidden)?)?,
            normed: context.shared_elements::<f16>(rows(dims.hidden)?)?,
            q: context.shared_elements::<f16>(rows(dims.query)?)?,
            k: context.shared_elements::<f16>(rows(dims.kv)?)?,
            v: context.shared_elements::<f16>(rows(dims.kv)?)?,
            attention: context.shared_elements::<f16>(rows(dims.query)?)?,
            gate: context.shared_elements::<f16>(rows(dims.intermediate)?)?,
            up: context.shared_elements::<f16>(rows(dims.intermediate)?)?,
            logits: context.shared_elements::<f16>(dims.vocab)?,
        })
    }

    fn allocated_bytes(&self) -> usize {
        [
            &self.ids,
            &self.next,
            &self.outputs,
            &self.x,
            &self.normed,
            &self.q,
            &self.k,
            &self.v,
            &self.attention,
            &self.gate,
            &self.up,
            &self.logits,
        ]
        .into_iter()
        .map(|buffer| allocated_size(buffer))
        .fold(0, usize::saturating_add)
    }
}

struct Pass<'context> {
    #[cfg(feature = "probe")]
    context: &'context Context,
    command_buffer: CommandBuffer,
    encoder: Encoder<'context>,
}

impl<'context> Pass<'context> {
    fn new(context: &'context Context) -> Result<Self, GpuError> {
        let command_buffer = context.command_buffer()?;
        let encoder = Encoder::new(&command_buffer, &context.kernels)?;
        Ok(Self {
            #[cfg(feature = "probe")]
            context,
            command_buffer,
            encoder,
        })
    }

    fn commit(self) -> CommandBuffer {
        self.encoder.end();
        self.command_buffer.commit();
        self.command_buffer
    }

    #[cfg(feature = "probe")]
    fn flush(self) -> Result<Self, GpuError> {
        let context = self.context;
        wait(&self.commit())?;
        Self::new(context)
    }
}

#[derive(Clone, Copy)]
enum Source {
    Prompt,
    Previous,
}

#[derive(Clone, Copy)]
struct Step {
    rows: u32,
    offset: u32,
    slot: u32,
}

type Observer<'observer> =
    dyn FnMut(Pass<'_>, Stage, Step) -> Result<Pass<'_>, LlamaError> + 'observer;

pub struct Session<'model> {
    model: &'model LlamaModel,
    dims: Dims,
    kv_cache: Buffer,
    activations: Activations,
    position: usize,
    prepared: Option<usize>,
}

fn to_u32(
    value: usize,
    what: &'static str,
) -> Result<u32, GpuError> {
    u32::try_from(value).map_err(|_| GpuError::Overflow {
        what,
    })
}

impl<'model> Session<'model> {
    pub(crate) fn new(model: &'model LlamaModel) -> Result<Self, LlamaError> {
        let context = model.context();
        let dims = Dims::new(model.config())?;
        let layers = usize::try_from(model.config().layers).map_err(|_| GpuError::Overflow {
            what: "layer count",
        })?;
        let kv_halves = dims
            .kv_layer_halves
            .checked_mul(2)
            .and_then(|halves| halves.checked_mul(layers))
            .ok_or(GpuError::Overflow {
                what: "KV cache size",
            })?;
        let kv_cache = context.shared_elements::<f16>(kv_halves)?;
        let activations = Activations::new(context, &dims)?;
        Ok(Self {
            model,
            dims,
            kv_cache,
            activations,
            position: 0,
            prepared: None,
        })
    }

    pub fn position(&self) -> usize { self.position }

    pub fn memory(&self) -> SessionMemory {
        SessionMemory {
            kv_cache_bytes: allocated_size(&self.kv_cache),
            activation_bytes: self.activations.allocated_bytes(),
        }
    }

    fn validate(
        &self,
        tokens: &[u32],
        extra: usize,
    ) -> Result<(), RequestError> {
        if tokens.is_empty() {
            return Err(RequestError::EmptyPrompt);
        }
        if tokens.len() > PREFILL_CAPACITY {
            return Err(RequestError::PromptTooLong {
                tokens: tokens.len(),
                limit: PREFILL_CAPACITY,
            });
        }
        let requested = tokens.len().saturating_add(extra);
        if self
            .position
            .checked_add(requested)
            .is_none_or(|end| end > CONTEXT_CAPACITY)
        {
            return Err(RequestError::CapacityExceeded {
                position: self.position,
                requested,
                capacity: CONTEXT_CAPACITY,
            });
        }
        let vocab = self.model.config().vocab;
        tokens
            .iter()
            .enumerate()
            .find(|(_, token)| **token >= vocab)
            .map_or(Ok(()), |(index, &token)| {
                Err(RequestError::TokenOutOfRange {
                    index,
                    token,
                    vocab,
                })
            })
    }

    pub fn prepare(
        &mut self,
        prompt: &[u32],
    ) -> Result<(), LlamaError> {
        if self.position != 0 {
            return Err(RequestError::SessionNotFresh {
                position: self.position,
            }
            .into());
        }
        self.validate(prompt, 0)?;
        write(&self.activations.ids, 0, prompt)?;
        self.prepared = Some(prompt.len());
        Ok(())
    }

    pub fn generate(
        &mut self,
        count: usize,
        mut on_token: impl FnMut(usize, u32),
    ) -> Result<Vec<u32>, LlamaError> {
        let prompt = self.prepared.take().ok_or(RequestError::NotPrepared)?;
        if count == 0 {
            return Err(RequestError::NoTokensRequested.into());
        }
        let requested = prompt.saturating_add(count);
        if requested > CONTEXT_CAPACITY {
            return Err(RequestError::CapacityExceeded {
                position: self.position,
                requested,
                capacity: CONTEXT_CAPACITY,
            }
            .into());
        }
        let mut pending = self.submit(Source::Prompt, prompt, 0)?;
        let mut tokens = Vec::with_capacity(count);
        for index in 1..count {
            let submitted = self.submit(Source::Previous, 1, index)?;
            tokens.push(self.collect(&pending, index.saturating_sub(1), &mut on_token)?);
            pending = submitted;
        }
        tokens.push(self.collect(&pending, count.saturating_sub(1), &mut on_token)?);
        Ok(tokens)
    }

    fn submit(
        &mut self,
        source: Source,
        rows: usize,
        slot: usize,
    ) -> Result<CommandBuffer, LlamaError> {
        let step = Step {
            rows: to_u32(rows, "row count")?,
            offset: to_u32(self.position, "position")?,
            slot: to_u32(slot, "output slot")?,
        };
        let ids = match source {
            Source::Prompt => View::new(&self.activations.ids),
            Source::Previous => View::new(&self.activations.next),
        };
        let pass = Pass::new(self.model.context())?;
        let pass = self.forward(pass, ids, step, &mut |pass, _, _| Ok(pass))?;
        self.position = self.position.saturating_add(rows);
        Ok(pass.commit())
    }

    fn collect(
        &self,
        command_buffer: &CommandBuffer,
        slot: usize,
        on_token: &mut impl FnMut(usize, u32),
    ) -> Result<u32, LlamaError> {
        wait(command_buffer)?;
        let token = read::<u32>(&self.activations.outputs, slot, 1)?
            .first()
            .copied()
            .ok_or(GpuError::OutOfBounds {
                what: "output token",
            })?;
        on_token(slot, token);
        Ok(token)
    }

    fn kv(
        &self,
        layer: usize,
    ) -> Result<KvCache<'_>, GpuError> {
        let base = layer
            .checked_mul(2)
            .and_then(|index| index.checked_mul(self.dims.kv_layer_halves))
            .ok_or(GpuError::Overflow {
                what: "KV cache offset",
            })?;
        let cache = View::new(&self.kv_cache);
        Ok(KvCache {
            keys: cache.element::<f16>(base)?,
            values: cache.element::<f16>(base.saturating_add(self.dims.kv_layer_halves))?,
        })
    }

    fn forward<'pass>(
        &self,
        mut pass: Pass<'pass>,
        ids: View<'_>,
        step: Step,
        observe: &mut Observer<'_>,
    ) -> Result<Pass<'pass>, LlamaError> {
        let model = self.model;
        let config = model.config();
        let weights = model.weights();
        let buffers = &self.activations;
        let x = View::new(&buffers.x);
        let normed = View::new(&buffers.normed);
        let q = View::new(&buffers.q);
        let k = View::new(&buffers.k);
        let v = View::new(&buffers.v);
        let attention = View::new(&buffers.attention);
        let gate = View::new(&buffers.gate);
        let up = View::new(&buffers.up);
        let rows = step.rows;
        pass.encoder
            .embed(ids, &weights.quantized(&weights.embed), x, rows)?;
        pass = observe(pass, Stage::Embedding, step)?;
        let scale = 1.0 / f32::from(u16::try_from(config.head_dim).unwrap_or(u16::MAX)).sqrt();
        for (index, layer) in weights.layers.iter().enumerate() {
            let number = to_u32(index, "layer index")?;
            let cache = self.kv(index)?;
            pass.encoder.rms_norm(
                x,
                weights.view(layer.input_norm),
                normed,
                rows,
                config.hidden,
                config.rms_eps,
            )?;
            pass = observe(pass, Stage::AttentionNorm(number), step)?;
            pass.encoder
                .matmul(normed, &weights.quantized(&layer.q_proj), q, rows, false)?;
            pass.encoder
                .matmul(normed, &weights.quantized(&layer.k_proj), k, rows, false)?;
            pass.encoder
                .matmul(normed, &weights.quantized(&layer.v_proj), v, rows, false)?;
            pass.encoder.rope_store(
                [q, k, v],
                &cache,
                View::new(model.frequencies()),
                &RopeParams {
                    rows,
                    offset: step.offset,
                    heads: config.heads,
                    kv_heads: config.kv_heads,
                    capacity: to_u32(CONTEXT_CAPACITY, "capacity")?,
                },
            )?;
            pass = observe(pass, Stage::Query(number), step)?;
            pass = observe(pass, Stage::Key(number), step)?;
            pass = observe(pass, Stage::Value(number), step)?;
            pass.encoder.attention(
                q,
                &cache,
                attention,
                &AttentionParams {
                    rows,
                    offset: step.offset,
                    heads: config.heads,
                    kv_heads: config.kv_heads,
                    capacity: to_u32(CONTEXT_CAPACITY, "capacity")?,
                    scale,
                },
            )?;
            pass = observe(pass, Stage::Attention(number), step)?;
            pass.encoder
                .matmul(attention, &weights.quantized(&layer.o_proj), x, rows, true)?;
            pass = observe(pass, Stage::AttentionResidual(number), step)?;
            pass.encoder.rms_norm(
                x,
                weights.view(layer.post_attention_norm),
                normed,
                rows,
                config.hidden,
                config.rms_eps,
            )?;
            pass = observe(pass, Stage::MlpNorm(number), step)?;
            pass.encoder.matmul(
                normed,
                &weights.quantized(&layer.gate_proj),
                gate,
                rows,
                false,
            )?;
            pass.encoder
                .matmul(normed, &weights.quantized(&layer.up_proj), up, rows, false)?;
            pass = observe(pass, Stage::Gate(number), step)?;
            pass = observe(pass, Stage::Up(number), step)?;
            let elements = rows
                .checked_mul(config.intermediate)
                .ok_or(GpuError::Overflow {
                    what: "swiglu size",
                })?;
            pass.encoder.swiglu(gate, up, elements)?;
            pass = observe(pass, Stage::Swiglu(number), step)?;
            pass.encoder
                .matmul(gate, &weights.quantized(&layer.down_proj), x, rows, true)?;
            pass = observe(pass, Stage::Output(number), step)?;
        }
        let last = usize::try_from(rows.saturating_sub(1))
            .ok()
            .and_then(|row| row.checked_mul(self.dims.hidden))
            .ok_or(GpuError::Overflow {
                what: "last row offset",
            })?;
        pass.encoder.rms_norm(
            x.element::<f16>(last)?,
            weights.view(weights.norm),
            normed,
            1,
            config.hidden,
            config.rms_eps,
        )?;
        pass = observe(pass, Stage::FinalNorm, step)?;
        let logits = View::new(&buffers.logits);
        pass.encoder
            .matmul(normed, &weights.quantized(&weights.embed), logits, 1, false)?;
        pass.encoder.argmax(
            logits,
            View::new(&buffers.outputs),
            View::new(&buffers.next),
            config.vocab,
            step.slot,
        );
        observe(pass, Stage::Logits, step)
    }
}

#[cfg(feature = "probe")]
impl Session<'_> {
    pub fn forward_probed(
        &mut self,
        tokens: &[u32],
        observer: &mut dyn FnMut(Stage, &[f16]),
    ) -> Result<u32, LlamaError> {
        self.validate(tokens, 0)?;
        write(&self.activations.ids, 0, tokens)?;
        let step = Step {
            rows: to_u32(tokens.len(), "row count")?,
            offset: to_u32(self.position, "position")?,
            slot: 0,
        };
        let pass = Pass::new(self.model.context())?;
        let pass = self.forward(
            pass,
            View::new(&self.activations.ids),
            step,
            &mut |pass, stage, step| {
                let pass = pass.flush()?;
                let values = self.stage_values(stage, step)?;
                observer(stage, &values);
                Ok(pass)
            },
        )?;
        wait(&pass.commit())?;
        self.position = self.position.saturating_add(tokens.len());
        read::<u32>(&self.activations.outputs, 0, 1)?
            .first()
            .copied()
            .ok_or_else(|| {
                GpuError::OutOfBounds {
                    what: "output token",
                }
                .into()
            })
    }

    fn stage_values(
        &self,
        stage: Stage,
        step: Step,
    ) -> Result<Vec<f16>, GpuError> {
        let rows = usize::try_from(step.rows).map_err(|_| GpuError::Overflow {
            what: "row count",
        })?;
        let span = |width: usize| {
            width.checked_mul(rows).ok_or(GpuError::Overflow {
                what: "probe size",
            })
        };
        let buffers = &self.activations;
        match stage {
            Stage::Embedding
            | Stage::AttentionNorm(_)
            | Stage::AttentionResidual(_)
            | Stage::MlpNorm(_)
            | Stage::Output(_) => {
                let buffer = if matches!(stage, Stage::AttentionNorm(_) | Stage::MlpNorm(_)) {
                    &buffers.normed
                } else {
                    &buffers.x
                };
                read(buffer, 0, span(self.dims.hidden)?)
            }
            Stage::Query(_) => read(&buffers.q, 0, span(self.dims.query)?),
            Stage::Attention(_) => read(&buffers.attention, 0, span(self.dims.query)?),
            Stage::Gate(_) | Stage::Swiglu(_) => {
                read(&buffers.gate, 0, span(self.dims.intermediate)?)
            }
            Stage::Up(_) => read(&buffers.up, 0, span(self.dims.intermediate)?),
            Stage::Key(layer) | Stage::Value(layer) => self.cache_rows(stage, layer, step),
            Stage::FinalNorm => read(&buffers.normed, 0, self.dims.hidden),
            Stage::Logits => read(&buffers.logits, 0, self.dims.vocab),
        }
    }

    fn cache_rows(
        &self,
        stage: Stage,
        layer: u32,
        step: Step,
    ) -> Result<Vec<f16>, GpuError> {
        let config = self.model.config();
        let head_dim = usize::try_from(config.head_dim).map_err(|_| GpuError::Overflow {
            what: "head dim",
        })?;
        let kv_heads = usize::try_from(config.kv_heads).map_err(|_| GpuError::Overflow {
            what: "kv heads",
        })?;
        let layer = usize::try_from(layer).map_err(|_| GpuError::Overflow {
            what: "layer",
        })?;
        let rows = usize::try_from(step.rows).map_err(|_| GpuError::Overflow {
            what: "rows",
        })?;
        let offset = usize::try_from(step.offset).map_err(|_| GpuError::Overflow {
            what: "offset",
        })?;
        let cache = self.kv(layer)?;
        let view = if matches!(stage, Stage::Key(_)) {
            cache.keys
        } else {
            cache.values
        };
        let base = view.offset.checked_div(size_of::<f16>()).unwrap_or(0);
        let heads: Vec<Vec<f16>> = (0..kv_heads)
            .map(|head| {
                let start = head
                    .checked_mul(CONTEXT_CAPACITY)
                    .and_then(|position| position.checked_add(offset))
                    .and_then(|position| position.checked_mul(head_dim))
                    .and_then(|position| position.checked_add(base))
                    .ok_or(GpuError::Overflow {
                        what: "cache probe",
                    })?;
                read::<f16>(&self.kv_cache, start, rows.saturating_mul(head_dim))
            })
            .collect::<Result<_, _>>()?;
        Ok((0..rows)
            .flat_map(|row| {
                heads.iter().flat_map(move |head| {
                    head.iter()
                        .skip(row.saturating_mul(head_dim))
                        .take(head_dim)
                        .copied()
                })
            })
            .collect())
    }
}
