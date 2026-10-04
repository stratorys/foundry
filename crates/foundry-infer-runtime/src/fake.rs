use std::cell::{
    Cell,
    RefCell,
};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::time::Duration;

use foundry_infer_core::{
    Alignment,
    ByteSize,
    CoreError,
    DeviceCapabilities,
    OpId,
};

use crate::backend::Backend;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Operation {
    AllocateDevice,
    AllocateHost,
    CreateStream,
    Copy,
    Compute,
    StreamWait,
    HostWait,
    Drain,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Entry {
    AllocatedDevice {
        resource: u32,
        bytes: u64,
    },
    Staged {
        resource: u32,
        contents: Vec<u8>,
    },
    CreatedStream {
        stream: u32,
    },
    SubmittedCopy {
        work: usize,
        stream: u32,
        source: u32,
        destination: u32,
    },
    SubmittedCompute {
        work: usize,
        stream: u32,
        op: OpId,
        weights: Vec<u32>,
        duration_hint: Option<Duration>,
    },
    StreamWait {
        stream: u32,
        work: usize,
    },
    HostWait {
        work: usize,
    },
    Computed {
        op: OpId,
        weights: Vec<Vec<u8>>,
    },
    Completed {
        work: usize,
    },
    HandleDropped {
        resource: u32,
    },
    Freed {
        resource: u32,
    },
    Drained,
    Failed(Operation),
}

#[derive(Debug, PartialEq, Eq)]
pub struct FakeError(pub Operation);

impl fmt::Display for FakeError {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(formatter, "fake {:?} failure", self.0)
    }
}

impl Error for FakeError {}

#[derive(Default)]
struct Shared {
    log: RefCell<Vec<Entry>>,
    live_device: Cell<u64>,
}

impl Shared {
    fn push(
        &self,
        entry: Entry,
    ) {
        self.log.borrow_mut().push(entry);
    }
}

struct Resource {
    id: u32,
    device: bool,
    capacity: u64,
    contents: RefCell<Vec<u8>>,
    shared: Rc<Shared>,
}

impl Drop for Resource {
    fn drop(&mut self) {
        if self.device {
            let live = self.shared.live_device.get();
            self.shared
                .live_device
                .set(live.saturating_sub(self.capacity));
        }
        self.shared.push(Entry::Freed {
            resource: self.id,
        });
    }
}

pub struct FakeBuffer(Rc<Resource>);

impl Drop for FakeBuffer {
    fn drop(&mut self) {
        self.0.shared.push(Entry::HandleDropped {
            resource: self.0.id,
        });
    }
}

pub struct FakeStream(u32);

pub struct FakeEvent(usize);

enum Action {
    Copy {
        source: Rc<Resource>,
        destination: Rc<Resource>,
    },
    Compute {
        op: OpId,
        weights: Vec<Rc<Resource>>,
    },
}

struct Work {
    dependencies: Vec<usize>,
    action: Option<Action>,
}

pub struct FakeBackend {
    shared: Rc<Shared>,
    device_memory: u64,
    alignment: Alignment,
    resources: u32,
    streams: u32,
    work: Vec<Work>,
    tails: BTreeMap<u32, usize>,
    waits: BTreeMap<u32, Vec<usize>>,
    calls: BTreeMap<Operation, usize>,
    failures: BTreeMap<Operation, usize>,
    late_failures: BTreeMap<Operation, usize>,
}

impl FakeBackend {
    pub fn new(device_memory: u64) -> Result<Self, CoreError> {
        Ok(Self {
            shared: Rc::default(),
            device_memory,
            alignment: Alignment::new(256)?,
            resources: 0,
            streams: 0,
            work: Vec::new(),
            tails: BTreeMap::new(),
            waits: BTreeMap::new(),
            calls: BTreeMap::new(),
            failures: BTreeMap::new(),
            late_failures: BTreeMap::new(),
        })
    }

    pub fn fail_on(
        mut self,
        operation: Operation,
        call: usize,
    ) -> Self {
        self.failures.insert(operation, call);
        self
    }

    pub fn fail_after_submit(
        mut self,
        operation: Operation,
        call: usize,
    ) -> Self {
        self.late_failures.insert(operation, call);
        self
    }

    pub fn log(&self) -> Vec<Entry> { self.shared.log.borrow().clone() }

    pub fn live_device(&self) -> u64 { self.shared.live_device.get() }

    pub fn pending(&self) -> usize {
        self.work
            .iter()
            .filter(|work| work.action.is_some())
            .count()
    }

    fn call(
        &mut self,
        operation: Operation,
    ) -> Result<usize, FakeError> {
        let count = self.calls.entry(operation).or_insert(0);
        let call = *count;
        *count = count.saturating_add(1);
        if self.failures.get(&operation) == Some(&call) {
            self.shared.push(Entry::Failed(operation));
            return Err(FakeError(operation));
        }
        Ok(call)
    }

    fn after_submit<T>(
        &self,
        operation: Operation,
        call: usize,
        value: T,
    ) -> Result<T, FakeError> {
        if self.late_failures.get(&operation) == Some(&call) {
            self.shared.push(Entry::Failed(operation));
            return Err(FakeError(operation));
        }
        Ok(value)
    }

    fn resource(
        &mut self,
        device: bool,
        capacity: u64,
        contents: Vec<u8>,
    ) -> Rc<Resource> {
        let id = self.resources;
        self.resources = self.resources.saturating_add(1);
        Rc::new(Resource {
            id,
            device,
            capacity,
            contents: RefCell::new(contents),
            shared: Rc::clone(&self.shared),
        })
    }

    fn submit(
        &mut self,
        stream: u32,
        action: Action,
    ) -> usize {
        let mut dependencies = self.waits.remove(&stream).unwrap_or_default();
        dependencies.extend(self.tails.get(&stream));
        let work = self.work.len();
        self.work.push(Work {
            dependencies,
            action: Some(action),
        });
        self.tails.insert(stream, work);
        work
    }

    fn complete(
        &mut self,
        work: usize,
    ) {
        let Some(entry) = self.work.get_mut(work) else {
            return;
        };
        let Some(action) = entry.action.take() else {
            return;
        };
        let dependencies = entry.dependencies.clone();
        for dependency in dependencies {
            self.complete(dependency);
        }
        match &action {
            Action::Copy {
                source,
                destination,
            } => {
                let source = source.contents.borrow();
                let mut destination = destination.contents.borrow_mut();
                for (target, byte) in destination.iter_mut().zip(source.iter()) {
                    *target = *byte;
                }
            }
            Action::Compute {
                op,
                weights,
            } => self.shared.push(Entry::Computed {
                op: *op,
                weights: weights
                    .iter()
                    .map(|weight| weight.contents.borrow().clone())
                    .collect(),
            }),
        }
        self.shared.push(Entry::Completed {
            work,
        });
        drop(action);
    }
}

impl Backend for FakeBackend {
    type DeviceBuffer = FakeBuffer;
    type HostBuffer = FakeBuffer;
    type Stream = FakeStream;
    type Event = FakeEvent;
    type Error = FakeError;

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            device_memory: ByteSize::from_bytes(self.device_memory),
            unified_memory: false,
            async_host_to_device: true,
            concurrent_copy_compute: true,
            alignment: self.alignment,
        }
    }

    fn allocate_device(
        &mut self,
        bytes: ByteSize,
    ) -> Result<FakeBuffer, FakeError> {
        self.call(Operation::AllocateDevice)?;
        let live = self.live_device().saturating_add(bytes.bytes());
        if live > self.device_memory {
            self.shared.push(Entry::Failed(Operation::AllocateDevice));
            return Err(FakeError(Operation::AllocateDevice));
        }
        let size =
            usize::try_from(bytes.bytes()).map_err(|_| FakeError(Operation::AllocateDevice))?;
        let resource = self.resource(true, bytes.bytes(), vec![0; size]);
        self.shared.live_device.set(live);
        self.shared.push(Entry::AllocatedDevice {
            resource: resource.id,
            bytes: bytes.bytes(),
        });
        Ok(FakeBuffer(resource))
    }

    fn allocate_host(
        &mut self,
        contents: &[u8],
    ) -> Result<FakeBuffer, FakeError> {
        self.call(Operation::AllocateHost)?;
        let capacity = u64::try_from(contents.len()).unwrap_or(u64::MAX);
        let resource = self.resource(false, capacity, contents.to_vec());
        self.shared.push(Entry::Staged {
            resource: resource.id,
            contents: contents.to_vec(),
        });
        Ok(FakeBuffer(resource))
    }

    fn create_stream(&mut self) -> Result<FakeStream, FakeError> {
        self.call(Operation::CreateStream)?;
        let stream = self.streams;
        self.streams = self.streams.saturating_add(1);
        self.shared.push(Entry::CreatedStream {
            stream,
        });
        Ok(FakeStream(stream))
    }

    fn copy_to_device(
        &mut self,
        stream: &FakeStream,
        source: &FakeBuffer,
        destination: &FakeBuffer,
    ) -> Result<FakeEvent, FakeError> {
        let call = self.call(Operation::Copy)?;
        let work = self.submit(
            stream.0,
            Action::Copy {
                source: Rc::clone(&source.0),
                destination: Rc::clone(&destination.0),
            },
        );
        self.shared.push(Entry::SubmittedCopy {
            work,
            stream: stream.0,
            source: source.0.id,
            destination: destination.0.id,
        });
        self.after_submit(Operation::Copy, call, FakeEvent(work))
    }

    fn synthetic_compute(
        &mut self,
        stream: &FakeStream,
        op: OpId,
        weights: &[&FakeBuffer],
        duration_hint: Option<Duration>,
    ) -> Result<FakeEvent, FakeError> {
        let call = self.call(Operation::Compute)?;
        let work = self.submit(
            stream.0,
            Action::Compute {
                op,
                weights: weights.iter().map(|weight| Rc::clone(&weight.0)).collect(),
            },
        );
        self.shared.push(Entry::SubmittedCompute {
            work,
            stream: stream.0,
            op,
            weights: weights.iter().map(|weight| weight.0.id).collect(),
            duration_hint,
        });
        self.after_submit(Operation::Compute, call, FakeEvent(work))
    }

    fn wait_stream(
        &mut self,
        stream: &FakeStream,
        event: &FakeEvent,
    ) -> Result<(), FakeError> {
        let call = self.call(Operation::StreamWait)?;
        self.waits.entry(stream.0).or_default().push(event.0);
        self.shared.push(Entry::StreamWait {
            stream: stream.0,
            work: event.0,
        });
        self.after_submit(Operation::StreamWait, call, ())
    }

    fn wait_host(
        &mut self,
        event: &FakeEvent,
    ) -> Result<(), FakeError> {
        self.call(Operation::HostWait)?;
        self.shared.push(Entry::HostWait {
            work: event.0,
        });
        self.complete(event.0);
        Ok(())
    }

    fn drain(&mut self) -> Result<(), FakeError> {
        self.call(Operation::Drain)?;
        for work in 0..self.work.len() {
            self.complete(work);
        }
        self.shared.push(Entry::Drained);
        Ok(())
    }
}
