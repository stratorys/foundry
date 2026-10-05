use std::cell::RefCell;
use std::rc::Rc;

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAllocation,
    MTLBuffer,
};

use crate::diagnostics::{
    AllocationCategory,
    CommandLabel,
    Ledger,
};
use crate::submission::Buffer;

pub(crate) type SharedLedger = Rc<RefCell<Ledger>>;

pub(crate) type TrackedBuffer = Rc<TrackedAllocation>;

pub(crate) struct TrackedAllocation {
    buffer: Buffer,
    entry: Option<(u64, SharedLedger)>,
}

impl TrackedAllocation {
    pub(crate) fn new(
        buffer: Buffer,
        ledger: Option<&SharedLedger>,
        category: AllocationCategory,
        requested: u64,
        label: CommandLabel,
    ) -> TrackedBuffer {
        let entry = ledger.and_then(|ledger| {
            let allocated =
                u64::try_from(MTLAllocation::allocatedSize(&*buffer)).unwrap_or(u64::MAX);
            let id = ledger
                .try_borrow_mut()
                .ok()?
                .allocate(category, requested, allocated, label);
            Some((id, Rc::clone(ledger)))
        });
        Rc::new(Self {
            buffer,
            entry,
        })
    }

    pub(crate) fn native(&self) -> &ProtocolObject<dyn MTLBuffer> { &self.buffer }
}

impl Drop for TrackedAllocation {
    fn drop(&mut self) {
        if let Some((id, ledger)) = &self.entry {
            if let Ok(mut ledger) = ledger.try_borrow_mut() {
                ledger.free(*id);
            }
        }
    }
}
