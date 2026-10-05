use std::cell::RefCell;
use std::collections::BTreeMap;
use std::iter;
use std::ptr::NonNull;
use std::rc::Rc;
use std::time::Duration;

use foundry_infer_core::{
    Alignment,
    ByteSize,
    DeviceCapabilities,
    OpId,
};
use foundry_infer_runtime::{
    Backend,
    CommandContext,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder,
    MTLBuffer,
    MTLCommandBuffer,
    MTLCommandEncoder,
    MTLCommandQueue,
    MTLComputeCommandEncoder,
    MTLComputePipelineState,
    MTLCreateSystemDefaultDevice,
    MTLDevice,
    MTLGPUFamily,
    MTLResourceOptions,
    MTLSize,
};

use crate::clock::HostClock;
use crate::device::{
    allocation_length,
    byte_size,
};
use crate::diagnostics::{
    AllocationCategory,
    Boundary,
    CommandLabel,
    DiagnosticLimits,
    GpuLog,
    GpuWork,
    Ledger,
    MetalDiagnostics,
    Submission,
};
use crate::error::MetalError;
use crate::pipeline::{
    self,
    CHUNK,
    Pipeline,
};
use crate::submission::{
    Buffer,
    CommandBuffer,
    Device,
    Event,
    Submissions,
    completion,
};
use crate::tracked::{
    SharedLedger,
    TrackedAllocation,
    TrackedBuffer,
};

type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

const RESULT: usize = size_of::<u32>();

pub struct MetalBuffer {
    buffer: TrackedBuffer,
    payload: usize,
}

pub struct MetalStaging {
    buffer: TrackedBuffer,
}

pub struct MetalStream {
    queue: Queue,
    waits: Vec<Event>,
}

pub struct MetalEvent {
    event: Event,
    command_buffer: CommandBuffer,
}

struct Checksum {
    buffer: TrackedBuffer,
    command_buffer: CommandBuffer,
    count: usize,
}

struct Pending {
    command_buffer: CommandBuffer,
    event: Event,
    waits: Vec<Event>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    Copy,
    Compute,
}

pub struct MetalBackend {
    device: Device,
    pipeline: Pipeline,
    capabilities: DeviceCapabilities,
    clock: HostClock,
    checksums: BTreeMap<OpId, Checksum>,
    submissions: Submissions,
    ledger: Option<SharedLedger>,
    label: CommandLabel,
    #[cfg(test)]
    fault: Option<(Fault, usize)>,
}

impl MetalBackend {
    pub fn new() -> Result<Self, MetalError> { Self::from_device(MTLCreateSystemDefaultDevice()) }

    fn from_device(device: Option<Device>) -> Result<Self, MetalError> {
        let device = device.ok_or(MetalError::DeviceUnavailable)?;
        let unsupported = |requirement| MetalError::UnsupportedDevice {
            name: device.name().to_string(),
            requirement,
        };
        if !device.supportsFamily(MTLGPUFamily::Apple4)
            && !device.supportsFamily(MTLGPUFamily::Mac2)
        {
            return Err(unsupported("non-uniform threadgroup dispatch"));
        }
        let align = device
            .heapBufferSizeAndAlignWithLength_options(1, MTLResourceOptions::StorageModePrivate)
            .align;
        let alignment = u64::try_from(align)
            .ok()
            .and_then(|bytes| Alignment::new(bytes).ok())
            .ok_or_else(|| unsupported("power-of-two buffer alignment"))?;
        let capabilities = DeviceCapabilities {
            device_memory: ByteSize::from_bytes(device.recommendedMaxWorkingSetSize()),
            unified_memory: device.hasUnifiedMemory(),
            async_host_to_device: true,
            concurrent_copy_compute: false,
            alignment,
        };
        let pipeline = pipeline::compile(&device)?;
        let clock = HostClock::new()?;
        Ok(Self {
            device,
            pipeline,
            capabilities,
            clock,
            checksums: BTreeMap::new(),
            submissions: Submissions::default(),
            ledger: None,
            label: CommandLabel::default(),
            #[cfg(test)]
            fault: None,
        })
    }

    pub fn device_name(&self) -> String { self.device.name().to_string() }

    pub fn working_set_recommended(&self) -> ByteSize { self.capabilities.device_memory }

    pub fn buffer_length_max(&self) -> ByteSize { byte_size(self.device.maxBufferLength()) }

    pub fn host_clock(&self) -> HostClock { self.clock }

    pub fn clear_checksums(&mut self) { self.checksums.clear(); }

    pub fn checksums(
        &self,
        op: OpId,
    ) -> Result<Vec<u32>, MetalError> {
        let checksum = self.checksums.get(&op).ok_or(MetalError::UnknownOp(op))?;
        completion(&checksum.command_buffer)?;
        let pointer = checksum.buffer.native().contents().cast::<u32>();
        // SAFETY: `synthetic_compute` allocated this shared buffer with room
        // for `count` u32 results, `completion` proved the GPU finished
        // writing it, and `checksum` keeps the buffer alive for the
        // lifetime of the slice.
        let values = unsafe { std::slice::from_raw_parts(pointer.as_ptr(), checksum.count) };
        Ok(values.to_vec())
    }

    pub fn begin_diagnostics(
        &mut self,
        limits: DiagnosticLimits,
    ) -> Result<(), MetalError> {
        if self.ledger.is_some() || self.submissions.recording() {
            return Err(MetalError::DiagnosticsActive);
        }
        let mut ledger = Ledger::new(self.clock, self.device.clone(), limits)?;
        let log = GpuLog::new(limits)?;
        ledger.snapshot(Boundary::ExecutionStart);
        self.ledger = Some(Rc::new(RefCell::new(ledger)));
        self.submissions.record(log);
        self.label = CommandLabel::default();
        Ok(())
    }

    pub fn snapshot(
        &mut self,
        boundary: Boundary,
    ) {
        if let Some(Ok(mut ledger)) = self.ledger.as_ref().map(|ledger| ledger.try_borrow_mut()) {
            ledger.snapshot(boundary);
        }
    }

    pub fn take_diagnostics(&mut self) -> Option<MetalDiagnostics> {
        let ledger = self.ledger.take()?;
        let (log, pending) = self.submissions.take_log()?;
        self.label = CommandLabel::default();
        let mut ledger = ledger.try_borrow_mut().ok()?;
        Some(MetalDiagnostics::extract(&mut ledger, log, pending))
    }

    fn buffer(
        &self,
        bytes: ByteSize,
        options: MTLResourceOptions,
    ) -> Result<Buffer, MetalError> {
        let length = allocation_length(bytes, self.device.maxBufferLength())?;
        self.device
            .newBufferWithLength_options(length, options)
            .ok_or(MetalError::AllocationFailed {
                bytes,
            })
    }

    fn track(
        &self,
        buffer: Buffer,
        category: AllocationCategory,
        requested: usize,
    ) -> TrackedBuffer {
        TrackedAllocation::new(
            buffer,
            self.ledger.as_ref(),
            category,
            byte_size(requested).bytes(),
            self.label,
        )
    }

    fn shared(
        &self,
        contents: &[u8],
        category: AllocationCategory,
    ) -> Result<TrackedBuffer, MetalError> {
        let buffer = self.buffer(
            byte_size(contents.len()),
            MTLResourceOptions::StorageModeShared,
        )?;
        // SAFETY: the buffer was just allocated with at least `contents.len()`
        // bytes of shared storage, so the destination is valid, and a
        // fresh allocation cannot overlap `contents`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                contents.as_ptr(),
                buffer.contents().cast::<u8>().as_ptr(),
                contents.len(),
            )
        };
        Ok(self.track(buffer, category, contents.len()))
    }

    fn begin(
        &mut self,
        stream: &MetalStream,
    ) -> Result<Pending, MetalError> {
        self.submissions.retire();
        let event = self.device.newEvent().ok_or(MetalError::EventCreation)?;
        let command_buffer = stream
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;
        let waits = stream.waits.clone();
        for wait in &waits {
            command_buffer.encodeWaitForEvent_value(wait, 1);
        }
        Ok(Pending {
            command_buffer,
            event,
            waits,
        })
    }

    fn submit(
        &mut self,
        stream: &mut MetalStream,
        pending: Pending,
        buffers: Vec<TrackedBuffer>,
        pipeline: Option<Pipeline>,
    ) -> MetalEvent {
        let Pending {
            command_buffer,
            event,
            mut waits,
        } = pending;
        let work = if pipeline.is_some() {
            GpuWork::Compute
        } else {
            GpuWork::Copy
        };
        command_buffer.encodeSignalEvent_value(&event, 1);
        let submission = self.submissions.recording().then(|| Submission {
            label: self.label,
            work,
            submitted: self.clock.now(),
        });
        command_buffer.commit();
        stream.waits.clear();
        waits.push(event.clone());
        self.submissions
            .register(command_buffer.clone(), buffers, waits, pipeline, submission);
        MetalEvent {
            event,
            command_buffer,
        }
    }

    #[cfg(test)]
    fn in_flight_len(&self) -> usize { self.submissions.in_flight_len() }

    #[cfg(test)]
    fn released(&self) -> &[objc2_metal::MTLCommandBufferStatus] { self.submissions.released() }

    #[cfg(test)]
    fn read_device(
        &self,
        buffer: &MetalBuffer,
    ) -> Result<Vec<u8>, MetalError> {
        let length = buffer.payload;
        let shared = self.buffer(byte_size(length), MTLResourceOptions::StorageModeShared)?;
        let queue = self
            .device
            .newCommandQueue()
            .ok_or(MetalError::QueueCreation)?;
        let command_buffer = queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;
        let encoder = command_buffer
            .blitCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;
        // SAFETY: `length` is the payload last copied into `buffer`, which fits
        // in both the device buffer and the `shared` buffer allocated
        // with exactly `length` bytes.
        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                buffer.buffer.native(),
                0,
                &shared,
                0,
                length,
            )
        };
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        completion(&command_buffer)?;
        // SAFETY: `shared` holds `length` bytes and `waitUntilCompleted`
        // returned, so the GPU no longer writes to it.
        let bytes =
            unsafe { std::slice::from_raw_parts(shared.contents().cast::<u8>().as_ptr(), length) };
        Ok(bytes.to_vec())
    }

    #[cfg(test)]
    fn fail_after(
        &mut self,
        fault: Fault,
        calls: usize,
    ) {
        self.fault = Some((fault, calls));
    }

    #[cfg(test)]
    fn inject(
        &mut self,
        fault: Fault,
    ) -> Result<(), MetalError> {
        match self.fault {
            Some((planned, 0)) if planned == fault => {
                self.fault = None;
                Err(MetalError::InjectedFault)
            }
            Some((planned, calls)) if planned == fault => {
                self.fault = Some((planned, calls.saturating_sub(1)));
                Ok(())
            }
            Some(_) | None => Ok(()),
        }
    }

    #[cfg(not(test))]
    fn inject(
        &mut self,
        _fault: Fault,
    ) -> Result<(), MetalError> {
        Ok(())
    }
}

impl Backend for MetalBackend {
    type DeviceBuffer = MetalBuffer;
    type Error = MetalError;
    type Event = MetalEvent;
    type HostBuffer = MetalStaging;
    type Stream = MetalStream;

    fn capabilities(&self) -> DeviceCapabilities { self.capabilities }

    fn allocate_device(
        &mut self,
        bytes: ByteSize,
    ) -> Result<MetalBuffer, MetalError> {
        let buffer = self.buffer(bytes, MTLResourceOptions::StorageModePrivate)?;
        let requested = buffer.length();
        Ok(MetalBuffer {
            buffer: self.track(buffer, AllocationCategory::PrivateSlot, requested),
            payload: 0,
        })
    }

    fn allocate_host(
        &mut self,
        contents: &[u8],
    ) -> Result<MetalStaging, MetalError> {
        Ok(MetalStaging {
            buffer: self.shared(contents, AllocationCategory::Staging)?,
        })
    }

    fn create_stream(&mut self) -> Result<MetalStream, MetalError> {
        Ok(MetalStream {
            queue: self
                .device
                .newCommandQueue()
                .ok_or(MetalError::QueueCreation)?,
            waits: Vec::new(),
        })
    }

    fn copy_to_device(
        &mut self,
        stream: &mut MetalStream,
        source: &MetalStaging,
        destination: &mut MetalBuffer,
    ) -> Result<MetalEvent, MetalError> {
        let length = source.buffer.native().length();
        if length > destination.buffer.native().length() {
            return Err(MetalError::CopyTooLarge {
                source: byte_size(length),
                destination: byte_size(destination.buffer.native().length()),
            });
        }
        let pending = self.begin(stream)?;
        let encoder = pending
            .command_buffer
            .blitCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;
        // SAFETY: `length` is the source buffer length and was checked above
        // not to exceed the destination length, so both ranges are in
        // bounds.
        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                source.buffer.native(),
                0,
                destination.buffer.native(),
                0,
                length,
            )
        };
        encoder.endEncoding();
        let buffers = vec![Rc::clone(&source.buffer), Rc::clone(&destination.buffer)];
        let event = self.submit(stream, pending, buffers, None);
        destination.payload = length;
        self.inject(Fault::Copy)?;
        Ok(event)
    }

    fn synthetic_compute(
        &mut self,
        stream: &mut MetalStream,
        op: OpId,
        weights: &[&MetalBuffer],
        _duration_hint: Option<Duration>,
    ) -> Result<MetalEvent, MetalError> {
        let lengths = weights
            .iter()
            .map(|weight| {
                let length = weight.payload;
                u32::try_from(length).map_err(|_| MetalError::PayloadTooLarge {
                    bytes: byte_size(length),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let too_many = || MetalError::TooManyWeights {
            count: weights.len(),
        };
        let offsets = (0..weights.len())
            .map(|index| index.checked_mul(RESULT).ok_or_else(too_many))
            .collect::<Result<Vec<_>, _>>()?;
        let result_len = weights
            .len()
            .max(1)
            .checked_mul(RESULT)
            .ok_or_else(too_many)?;
        let result = self.shared(&vec![0; result_len], AllocationCategory::ChecksumResult)?;
        let pending = self.begin(stream)?;
        let encoder = pending
            .command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;
        encoder.setComputePipelineState(&self.pipeline);
        let width = self.pipeline.maxTotalThreadsPerThreadgroup();
        for ((weight, length), offset) in weights.iter().zip(&lengths).zip(offsets) {
            let chunks = weight.payload.div_ceil(CHUNK);
            if chunks == 0 {
                continue;
            }
            // SAFETY: the kernel reads at most `length` bytes of the weight
            // buffer, `offset` addresses one u32 inside `result`,
            // and Metal copies the `length` bytes before `setBytes`
            // returns.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(weight.buffer.native()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(result.native()), offset, 1);
                encoder.setBytes_length_atIndex(NonNull::from(length).cast(), size_of::<u32>(), 2)
            };
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: chunks,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: width.min(chunks),
                    height: 1,
                    depth: 1,
                },
            );
        }
        encoder.endEncoding();
        let buffers = weights
            .iter()
            .map(|weight| Rc::clone(&weight.buffer))
            .chain(iter::once(Rc::clone(&result)))
            .collect();
        let event = self.submit(stream, pending, buffers, Some(self.pipeline.clone()));
        self.checksums.insert(
            op,
            Checksum {
                buffer: result,
                command_buffer: event.command_buffer.clone(),
                count: weights.len(),
            },
        );
        self.inject(Fault::Compute)?;
        Ok(event)
    }

    fn wait_stream(
        &mut self,
        stream: &mut MetalStream,
        event: &MetalEvent,
    ) -> Result<(), MetalError> {
        stream.waits.push(event.event.clone());
        Ok(())
    }

    fn wait_host(
        &mut self,
        event: &MetalEvent,
    ) -> Result<(), MetalError> {
        event.command_buffer.waitUntilCompleted();
        let outcome = completion(&event.command_buffer);
        self.submissions.retire();
        outcome?;
        self.submissions.take_fault().map_or(Ok(()), Err)
    }

    fn drain(&mut self) -> Result<(), MetalError> { self.submissions.drain() }

    fn annotate(
        &mut self,
        context: Option<&CommandContext<'_>>,
    ) {
        if self.submissions.recording() {
            self.label = context.map(CommandLabel::from).unwrap_or_default();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{
        BTreeMap,
        BTreeSet,
    };
    use std::env;
    use std::error::Error;
    use std::num::NonZeroU32;
    use std::time::Duration;

    use foundry_infer_core::{
        ByteSize,
        Graph,
        Id,
        MemoryBudget,
        MemorySpace,
        Op,
        OpId,
        TensorId,
    };
    use foundry_infer_plan::{
        synthetic_chain_graph,
        synthetic_chain_plan,
    };
    use foundry_infer_runtime::{
        Backend,
        BackendOperation,
        ExecutionFailure,
        RuntimeError,
        execute,
    };
    use objc2_metal::{
        MTLBuffer,
        MTLCommandBuffer,
        MTLCommandBufferStatus,
        MTLDevice,
    };

    use super::{
        Fault,
        MetalBackend,
        MetalBuffer,
        MetalEvent,
        MetalStream,
    };
    use crate::diagnostics::{
        AllocationCategory,
        AllocationChange,
        Boundary,
        DiagnosticLimits,
        GpuTiming,
        GpuWork,
        MetalDiagnostics,
    };
    use crate::error::MetalError;
    use crate::pipeline::reference_checksum;

    type TestResult = Result<(), Box<dyn Error>>;

    const LAYERS: u32 = 5;
    const WEIGHT: u64 = 1000;
    const SLOT: u64 = 4096;

    #[expect(
        clippy::print_stderr,
        reason = "the test reports whether GPU behavior was verified or skipped"
    )]
    fn gpu() -> Result<Option<MetalBackend>, Box<dyn Error>> {
        match MetalBackend::new() {
            Ok(backend) => {
                eprintln!("metal: running on {}", backend.device_name());
                Ok(Some(backend))
            }
            Err(
                error @ (MetalError::DeviceUnavailable
                | MetalError::UnsupportedDevice {
                    ..
                }),
            ) if env::var_os("FOUNDRY_REQUIRE_METAL").is_none() => {
                eprintln!("metal: SKIPPED, GPU behavior not verified: {error}");
                Ok(None)
            }
            Err(error) => Err(error.into()),
        }
    }

    fn pattern(
        layer: u32,
        length: usize,
    ) -> Vec<u8> {
        let seed = usize::try_from(layer).unwrap_or(usize::MAX);
        (0..length)
            .map(|index| {
                let value = seed
                    .wrapping_mul(37)
                    .wrapping_add(index.wrapping_mul(11))
                    .wrapping_add(5);
                u8::try_from(value % 256).unwrap_or_default()
            })
            .collect()
    }

    fn stage(
        backend: &mut MetalBackend,
        stream: &mut MetalStream,
        payload: &[u8],
        slot: u64,
    ) -> Result<(MetalBuffer, MetalEvent), Box<dyn Error>> {
        let mut destination = backend.allocate_device(ByteSize::from_bytes(slot))?;
        let staging = backend.allocate_host(payload)?;
        let event = backend.copy_to_device(stream, &staging, &mut destination)?;
        Ok((destination, event))
    }

    fn all_completed(backend: &MetalBackend) -> bool {
        !backend.released().is_empty()
            && backend
                .released()
                .iter()
                .all(|&status| status == MTLCommandBufferStatus::Completed)
    }

    struct Workload {
        graph: Graph,
        payloads: BTreeMap<TensorId, Vec<u8>>,
    }

    impl Workload {
        fn new() -> Result<Self, Box<dyn Error>> {
            let graph = synthetic_chain_graph(
                LAYERS,
                ByteSize::from_bytes(WEIGHT),
                Some(Duration::from_millis(1)),
            )?;
            let length = usize::try_from(WEIGHT)?;
            let payloads = (0_u32..)
                .zip(graph.weights())
                .map(|(layer, weight)| (weight.tensor, pattern(layer, length)))
                .collect();
            Ok(Self {
                graph,
                payloads,
            })
        }

        fn run(
            &self,
            backend: &mut MetalBackend,
            slots: u32,
        ) -> Result<Result<(), RuntimeError<MetalError>>, Box<dyn Error>> {
            let slots = NonZeroU32::new(slots).ok_or("zero slots")?;
            let plan = synthetic_chain_plan(&self.graph, slots, backend.capabilities().alignment)?;
            let weights = self
                .payloads
                .iter()
                .map(|(&tensor, payload)| (tensor, payload.as_slice()))
                .collect();
            let budget = MemoryBudget::new(MemorySpace::Device, ByteSize::from_mib(1)?);
            Ok(execute(backend, &self.graph, &plan, budget, &weights))
        }

        fn diagnosed(
            &self,
            backend: &mut MetalBackend,
            slots: u32,
        ) -> Result<MetalDiagnostics, Box<dyn Error>> {
            let plan = synthetic_chain_plan(
                &self.graph,
                NonZeroU32::new(slots).ok_or("zero slots")?,
                backend.capabilities().alignment,
            )?;
            backend.begin_diagnostics(DiagnosticLimits::for_plan(&plan)?)?;
            self.run(backend, slots)??;
            backend.drain()?;
            backend.snapshot(Boundary::AfterDrain);
            for (op, checksum) in self.expected()? {
                assert_eq!(backend.checksums(op)?, vec![checksum], "op {}", op.index());
            }
            backend.clear_checksums();
            backend.snapshot(Boundary::AfterChecksumRelease);
            Ok(backend
                .take_diagnostics()
                .ok_or("diagnostics were recorded")?)
        }

        fn expected(&self) -> Result<BTreeMap<OpId, u32>, Box<dyn Error>> {
            self.graph
                .ops()
                .map(|(op, desc)| match desc {
                    Op::SyntheticCompute {
                        inputs, ..
                    } => {
                        let tensor = inputs.get(1).ok_or("synthetic op without weight")?;
                        let payload = self.payloads.get(tensor).ok_or("missing payload")?;
                        Ok((op, reference_checksum(payload)))
                    }
                    Op::MatMul {
                        ..
                    } => Err("unexpected matmul".into()),
                })
                .collect()
        }
    }

    #[test]
    fn a_missing_device_is_a_clear_error() {
        let error = MetalBackend::from_device(None).err();
        assert_eq!(
            error,
            Some(MetalError::DeviceUnavailable),
            "no device must be rejected"
        );
        assert_eq!(
            error.map(|error| error.to_string()),
            Some("no Metal device is available".to_owned()),
            "the error must explain itself"
        );
    }

    #[test]
    fn capabilities_describe_the_device() -> TestResult {
        let Some(backend) = gpu()? else { return Ok(()) };
        let capabilities = backend.capabilities();
        assert_eq!(
            capabilities.unified_memory,
            backend.device.hasUnifiedMemory(),
            "unified memory comes from the device"
        );
        assert_eq!(
            capabilities.device_memory,
            backend.working_set_recommended(),
            "device memory is the recommended working set"
        );
        assert!(
            !capabilities.concurrent_copy_compute,
            "overlap is not claimed"
        );
        assert_eq!(
            backend.buffer_length_max().bytes(),
            u64::try_from(backend.device.maxBufferLength())?,
            "the buffer length limit comes from the device"
        );
        Ok(())
    }

    #[test]
    fn copies_land_in_private_memory() -> TestResult {
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        let mut stream = backend.create_stream()?;
        let payload = pattern(3, 1000);
        let (destination, event) = stage(&mut backend, &mut stream, &payload, SLOT)?;
        backend.wait_host(&event)?;
        assert_eq!(
            destination.buffer.native().length(),
            4096,
            "the slot keeps its size"
        );
        assert_eq!(
            backend.read_device(&destination)?,
            payload,
            "the copy must be exact"
        );
        Ok(())
    }

    #[test]
    fn computation_consumes_copied_bytes() -> TestResult {
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        let mut copy = backend.create_stream()?;
        let mut compute = backend.create_stream()?;
        let payloads = [pattern(0, 1), pattern(1, 1000), pattern(2, 100_003)];
        let mut slots = Vec::new();
        for payload in &payloads {
            let (slot, event) = stage(&mut backend, &mut copy, payload, 128 * 1024)?;
            backend.wait_stream(&mut compute, &event)?;
            slots.push(slot);
        }
        let weights: Vec<&MetalBuffer> = slots.iter().collect();
        let op = OpId::from_index(0);
        let event = backend.synthetic_compute(&mut compute, op, &weights, None)?;
        backend.wait_host(&event)?;
        let expected: Vec<u32> = payloads
            .iter()
            .map(|payload| reference_checksum(payload))
            .collect();
        assert_eq!(
            backend.checksums(op)?,
            expected,
            "each weight has its checksum"
        );
        Ok(())
    }

    #[test]
    fn sequential_double_and_triple_buffered_plans_compute_every_layer() -> TestResult {
        let workload = Workload::new()?;
        let expected = workload.expected()?;
        let distinct: BTreeSet<u32> = expected.values().copied().collect();
        assert_eq!(
            distinct.len(),
            expected.len(),
            "layers must be distinguishable"
        );
        for slots in 1..=3 {
            let Some(mut backend) = gpu()? else {
                return Ok(());
            };
            workload.run(&mut backend, slots)??;
            for (&op, &checksum) in &expected {
                assert_eq!(
                    backend.checksums(op)?,
                    vec![checksum],
                    "op {} with {slots} slots",
                    op.index()
                )
            }
            backend.drain()?;
            assert_eq!(
                backend.in_flight_len(),
                0,
                "{slots} slots leave nothing in flight"
            );
            assert!(
                all_completed(&backend),
                "{slots} slots release completed work only"
            );
        }
        Ok(())
    }

    #[test]
    fn repeated_runs_leave_no_residue() -> TestResult {
        let workload = Workload::new()?;
        let expected = workload.expected()?;
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        for round in 0..3 {
            for slots in 1..=3 {
                workload.run(&mut backend, slots)??;
                backend.drain()?;
                assert_eq!(
                    backend.in_flight_len(),
                    0,
                    "round {round} with {slots} slots leaves nothing in flight"
                );
                for (&op, &checksum) in &expected {
                    assert_eq!(
                        backend.checksums(op)?,
                        vec![checksum],
                        "op {} in round {round} with {slots} slots",
                        op.index()
                    );
                }
                backend.clear_checksums();
                assert!(
                    backend.checksums.is_empty(),
                    "round {round} with {slots} slots keeps no checksum results"
                );
            }
        }
        assert!(
            all_completed(&backend),
            "repeated runs release completed work only"
        );
        Ok(())
    }

    #[test]
    fn staging_dropped_after_submission_stays_alive() -> TestResult {
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        let mut copy = backend.create_stream()?;
        let mut compute = backend.create_stream()?;
        let payload = pattern(4, 50_000);
        let mut slot = backend.allocate_device(ByteSize::from_bytes(64 * 1024))?;
        let staging = backend.allocate_host(&payload)?;
        let copied = backend.copy_to_device(&mut copy, &staging, &mut slot)?;
        drop(staging);
        backend.wait_stream(&mut compute, &copied)?;
        let op = OpId::from_index(7);
        let event = backend.synthetic_compute(&mut compute, op, &[&slot], None)?;
        backend.wait_host(&event)?;
        assert_eq!(
            backend.checksums(op)?,
            vec![reference_checksum(&payload)],
            "staged bytes survive"
        );
        Ok(())
    }

    #[test]
    fn events_dropped_in_flight_are_drained() -> TestResult {
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        let mut copy = backend.create_stream()?;
        let mut compute = backend.create_stream()?;
        let payload = pattern(5, 20_000);
        let (slot, copied) = stage(&mut backend, &mut copy, &payload, 32 * 1024)?;
        backend.wait_stream(&mut compute, &copied)?;
        drop(copied);
        let op = OpId::from_index(1);
        drop(backend.synthetic_compute(&mut compute, op, &[&slot], None)?);
        backend.drain()?;
        assert_eq!(backend.in_flight_len(), 0, "drain completes everything");
        assert!(all_completed(&backend), "only completed work is released");
        assert_eq!(
            backend.checksums(op)?,
            vec![reference_checksum(&payload)],
            "the result is intact"
        );
        Ok(())
    }

    #[test]
    fn host_waits_report_completion() -> TestResult {
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        let mut stream = backend.create_stream()?;
        let (_slot, event) = stage(&mut backend, &mut stream, &pattern(6, 256), SLOT)?;
        backend.wait_host(&event)?;
        assert_eq!(
            event.command_buffer.status(),
            MTLCommandBufferStatus::Completed,
            "a host wait establishes completion"
        );
        Ok(())
    }

    #[test]
    fn drain_completes_work_on_every_queue() -> TestResult {
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        let mut expected_checksums = Vec::new();
        let mut held = Vec::new();
        for layer in 0..3 {
            let mut stream = backend.create_stream()?;
            let payload = pattern(layer, 10_000 + usize::try_from(layer)?);
            let (slot, _copied) = stage(&mut backend, &mut stream, &payload, 16 * 1024)?;
            let op = OpId::from_index(layer);
            backend.synthetic_compute(&mut stream, op, &[&slot], None)?;
            expected_checksums.push((op, reference_checksum(&payload)));
            held.push((stream, slot));
        }
        backend.drain()?;
        assert_eq!(backend.in_flight_len(), 0, "drain waits for every queue");
        assert_eq!(
            backend.released().len(),
            6,
            "every submission is released once"
        );
        assert!(all_completed(&backend), "only completed work is released");
        for (op, checksum) in expected_checksums {
            assert_eq!(backend.checksums(op)?, vec![checksum], "op {}", op.index());
        }
        Ok(())
    }

    #[test]
    fn invalid_sizes_are_rejected_before_submission() -> TestResult {
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        let mut stream = backend.create_stream()?;
        let staging = backend.allocate_host(&pattern(0, 2048))?;
        let mut slot = backend.allocate_device(ByteSize::from_bytes(1024))?;
        assert_eq!(
            backend
                .copy_to_device(&mut stream, &staging, &mut slot)
                .err(),
            Some(MetalError::CopyTooLarge {
                source: ByteSize::from_bytes(2048),
                destination: ByteSize::from_bytes(1024),
            }),
            "an oversized copy is rejected"
        );
        assert_eq!(backend.in_flight_len(), 0, "nothing is submitted");
        assert_eq!(
            backend.allocate_device(ByteSize::from_bytes(0)).err(),
            Some(MetalError::EmptyAllocation),
            "empty device buffers are rejected"
        );
        assert_eq!(
            backend.allocate_host(&[]).err(),
            Some(MetalError::EmptyAllocation),
            "empty staging buffers are rejected"
        );
        let limit = u64::try_from(backend.device.maxBufferLength())?;
        assert_eq!(
            backend
                .allocate_device(ByteSize::from_bytes(limit + 1))
                .err(),
            Some(MetalError::AllocationTooLarge {
                requested: ByteSize::from_bytes(limit + 1),
                limit: ByteSize::from_bytes(limit),
            }),
            "allocations beyond the device limit are rejected"
        );
        Ok(())
    }

    #[test]
    fn failures_after_submission_are_drained_before_release() -> TestResult {
        let workload = Workload::new()?;
        let expected = workload.expected()?;
        for (fault, calls) in [(Fault::Copy, 1), (Fault::Compute, 1)] {
            let Some(mut backend) = gpu()? else {
                return Ok(());
            };
            backend.fail_after(fault, calls);
            let Err(RuntimeError::Execution {
                failure:
                    ExecutionFailure::Backend {
                        operation,
                        source,
                    },
                cleanup,
            }) = workload.run(&mut backend, 3)?
            else {
                return Err(format!("{fault:?} fault did not fail the run").into());
            };
            assert_eq!(
                source,
                MetalError::InjectedFault,
                "{fault:?} reports the fault"
            );
            assert_eq!(cleanup, None, "{fault:?} drains cleanly");
            assert_eq!(
                backend.in_flight_len(),
                0,
                "{fault:?} leaves nothing in flight"
            );
            assert!(
                all_completed(&backend),
                "{fault:?} releases completed work only"
            );
            match (fault, operation) {
                (
                    Fault::Copy,
                    BackendOperation::Copy {
                        ..
                    },
                ) => {}
                (
                    Fault::Compute,
                    BackendOperation::Launch {
                        op, ..
                    },
                ) => {
                    let checksum = expected.get(&op).copied().ok_or("unknown op")?;
                    assert_eq!(
                        backend.checksums(op)?,
                        vec![checksum],
                        "the faulted launch still completed"
                    );
                }
                (fault, operation) => {
                    return Err(format!("{fault:?} failed at {operation:?}").into());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn diagnostics_time_every_completed_command_buffer() -> TestResult {
        let workload = Workload::new()?;
        for slots in 1..=3 {
            let Some(mut backend) = gpu()? else {
                return Ok(());
            };
            let diagnostics = workload.diagnosed(&mut backend, slots)?;
            let after = backend.host_clock().now();
            assert!(
                diagnostics.gpu_timestamps_valid(),
                "{slots} slots: every interval has valid timestamps"
            );
            let copies: Vec<_> = diagnostics
                .gpu
                .iter()
                .filter(|interval| interval.work == GpuWork::Copy)
                .collect();
            let launches: Vec<_> = diagnostics
                .gpu
                .iter()
                .filter(|interval| interval.work == GpuWork::Compute)
                .collect();
            let layers = usize::try_from(LAYERS)?;
            assert_eq!(copies.len(), layers, "{slots} slots: one copy per layer");
            assert_eq!(
                launches.len(),
                layers,
                "{slots} slots: one launch per layer"
            );
            for interval in &diagnostics.gpu {
                let GpuTiming::Valid {
                    start,
                    end,
                } = interval.timing
                else {
                    return Err("invalid timing".into());
                };
                assert!(
                    interval.submitted <= start && start <= end && end <= after,
                    "{slots} slots: GPU time follows CPU submission on the host clock"
                );
                assert!(interval.completed, "{slots} slots: the buffer completed");
                assert!(
                    interval.label.index.is_some() && interval.label.stream.is_some(),
                    "{slots} slots: the command and stream are named"
                );
            }
            assert!(
                copies.iter().all(|interval| interval.label.tensor.is_some()
                    && interval.label.slot.is_some()
                    && interval.label.bytes == Some(ByteSize::from_bytes(WEIGHT))),
                "{slots} slots: copies name their tensor, slot and bytes"
            );
            assert!(
                launches.iter().all(|interval| interval.label.op.is_some()
                    && interval.label.slot.is_some()
                    && interval.work.kernel() == Some("checksum")),
                "{slots} slots: launches name their op, slot and kernel"
            );
        }
        Ok(())
    }

    #[test]
    fn diagnostics_account_every_allocation_once() -> TestResult {
        let workload = Workload::new()?;
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        let diagnostics = workload.diagnosed(&mut backend, 2)?;
        let layers = u64::from(LAYERS);
        for (category, expected) in [
            (AllocationCategory::Staging, layers),
            (AllocationCategory::PrivateSlot, 2),
            (AllocationCategory::ChecksumResult, layers),
        ] {
            let usage = diagnostics
                .categories
                .get(&category)
                .ok_or("missing category")?;
            assert_eq!(usage.allocations, expected, "{category:?} allocations");
            assert_eq!(usage.frees, expected, "{category:?} frees");
            assert_eq!(usage.live_requested, 0, "{category:?} is released");
            assert!(
                usage.peak_allocated >= usage.peak_requested && usage.peak_requested > 0,
                "{category:?} allocated sizes cover requests"
            );
        }
        let boundaries: Vec<Boundary> = diagnostics
            .snapshots
            .iter()
            .map(|snapshot| snapshot.boundary)
            .collect();
        assert_eq!(
            boundaries,
            [
                Boundary::ExecutionStart,
                Boundary::AfterDrain,
                Boundary::AfterChecksumRelease
            ],
            "every boundary is sampled"
        );
        let after_drain = diagnostics.snapshots.get(1).ok_or("no drain snapshot")?;
        assert_eq!(
            after_drain
                .categories
                .get(&AllocationCategory::ChecksumResult)
                .map(|usage| usage.live_requested),
            Some(layers * 4),
            "checksum results stay live until they are cleared"
        );
        assert_eq!(diagnostics.live_allocations, 0, "nothing stays live");
        assert_eq!(
            diagnostics.allocations.len(),
            usize::try_from(2 * (2 + layers * 2))?,
            "every allocation and free is an event"
        );
        assert!(
            diagnostics.device_sampled_max >= diagnostics.tracked.peak_allocated,
            "the sampled device maximum covers tracked buffers"
        );
        assert!(
            diagnostics.storage_bytes > 0,
            "diagnostic storage is counted"
        );
        Ok(())
    }

    #[test]
    fn retained_buffers_are_counted_once_until_their_work_completes() -> TestResult {
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        let workload = Workload::new()?;
        let plan = synthetic_chain_plan(
            &workload.graph,
            NonZeroU32::new(1).ok_or("zero slots")?,
            backend.capabilities().alignment,
        )?;
        backend.begin_diagnostics(DiagnosticLimits::for_plan(&plan)?)?;
        let mut stream = backend.create_stream()?;
        let (slot, event) = stage(&mut backend, &mut stream, &pattern(9, 4000), SLOT)?;
        let clone = std::rc::Rc::clone(&slot.buffer);
        drop(slot);
        drop(clone);
        drop(event);
        let live = |backend: &MetalBackend| {
            backend
                .ledger
                .as_ref()
                .and_then(|ledger| ledger.borrow().live_usage(AllocationCategory::PrivateSlot))
                .unwrap_or_default()
        };
        assert_eq!(
            live(&backend),
            (1, SLOT),
            "the dropped slot is retained by its in-flight copy"
        );
        backend.drain()?;
        assert_eq!(live(&backend), (0, 0), "draining releases the slot");
        let diagnostics = backend.take_diagnostics().ok_or("no diagnostics")?;
        let slots: Vec<AllocationChange> = diagnostics
            .allocations
            .iter()
            .filter(|event| event.category == AllocationCategory::PrivateSlot)
            .map(|event| event.change)
            .collect();
        assert_eq!(
            slots,
            [AllocationChange::Allocated, AllocationChange::Freed],
            "the slot is allocated and freed once"
        );
        Ok(())
    }

    #[test]
    fn repeated_diagnosed_executions_accumulate_nothing() -> TestResult {
        let workload = Workload::new()?;
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        for round in 0..3 {
            for slots in 1..=3 {
                let diagnostics = workload.diagnosed(&mut backend, slots)?;
                let end = diagnostics.snapshots.last().ok_or("no final snapshot")?;
                assert_eq!(
                    (end.live_allocations, end.tracked.live_allocated),
                    (0, 0),
                    "round {round} with {slots} slots ends with no tracked buffer"
                );
                assert_eq!(backend.in_flight_len(), 0, "nothing is in flight");
            }
        }
        assert!(
            backend.take_diagnostics().is_none(),
            "diagnostics end with their execution"
        );
        Ok(())
    }

    #[test]
    fn diagnostics_cannot_overlap() -> TestResult {
        let Some(mut backend) = gpu()? else {
            return Ok(());
        };
        let workload = Workload::new()?;
        let plan = synthetic_chain_plan(
            &workload.graph,
            NonZeroU32::new(1).ok_or("zero slots")?,
            backend.capabilities().alignment,
        )?;
        let limits = DiagnosticLimits::for_plan(&plan)?;
        backend.begin_diagnostics(limits)?;
        assert_eq!(
            backend.begin_diagnostics(limits).err(),
            Some(MetalError::DiagnosticsActive),
            "a second recording is rejected"
        );
        assert!(backend.take_diagnostics().is_some(), "the first one ends");
        backend.begin_diagnostics(limits)?;
        Ok(())
    }

    #[test]
    fn limits_follow_the_plan() -> TestResult {
        let workload = Workload::new()?;
        let alignment = foundry_infer_core::Alignment::new(256)?;
        for slots in 1..=3 {
            let plan = synthetic_chain_plan(
                &workload.graph,
                NonZeroU32::new(slots).ok_or("zero slots")?,
                alignment,
            )?;
            let limits = DiagnosticLimits::for_plan(&plan)?;
            let layers = usize::try_from(LAYERS)?;
            assert_eq!(
                limits.gpu_intervals(),
                2 * layers,
                "{slots} slots: submissions"
            );
            assert_eq!(
                limits.allocation_events(),
                2 * (usize::try_from(slots)? + 2 * layers),
                "{slots} slots: allocation events"
            );
        }
        Ok(())
    }
}
