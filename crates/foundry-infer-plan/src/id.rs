#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferSlot(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId(u32);

impl BufferSlot {
    pub const fn new(index: u32) -> Self { Self(index) }

    pub const fn index(self) -> u32 { self.0 }
}

impl EventId {
    pub const fn new(index: u32) -> Self { Self(index) }

    pub const fn index(self) -> u32 { self.0 }
}

impl StreamId {
    pub const fn new(index: u32) -> Self { Self(index) }

    pub const fn index(self) -> u32 { self.0 }
}
