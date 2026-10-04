use std::collections::BTreeMap;

use crate::id::{
    EventId,
    StreamId,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Access {
    pub stream: StreamId,
    pub seq: u64,
    pub event: EventId,
}

#[derive(Clone, Debug, Default)]
pub struct Clock(BTreeMap<StreamId, u64>);

impl Clock {
    pub fn join(
        &mut self,
        other: &Self,
    ) {
        for (&stream, &seq) in &other.0 {
            let entry = self.0.entry(stream).or_insert(0);
            *entry = (*entry).max(seq);
        }
    }

    pub fn covers(
        &self,
        access: &Access,
    ) -> bool {
        self.0
            .get(&access.stream)
            .is_some_and(|&seq| seq >= access.seq)
    }
}

#[derive(Default)]
struct Stream {
    submitted: u64,
    clock: Clock,
}

#[derive(Default)]
pub struct Order {
    streams: BTreeMap<StreamId, Stream>,
    events: BTreeMap<EventId, Clock>,
    host: Clock,
}

impl Order {
    pub fn ready(
        &self,
        stream: StreamId,
    ) -> Clock {
        let mut clock = self.host.clone();
        if let Some(state) = self.streams.get(&stream) {
            clock.join(&state.clock);
        }
        clock
    }

    pub fn host(&self) -> &Clock { &self.host }

    pub fn submit(
        &mut self,
        stream: StreamId,
        event: EventId,
    ) -> Access {
        let mut clock = self.ready(stream);
        let state = self.streams.entry(stream).or_default();
        state.submitted = state.submitted.saturating_add(1);
        let access = Access {
            stream,
            seq: state.submitted,
            event,
        };
        clock.0.insert(stream, access.seq);
        state.clock = clock.clone();
        self.events.insert(event, clock);
        access
    }

    pub fn wait_stream(
        &mut self,
        stream: StreamId,
        event: EventId,
    ) {
        if let Some(clock) = self.events.get(&event) {
            self.streams.entry(stream).or_default().clock.join(clock);
        }
    }

    pub fn wait_host(
        &mut self,
        event: EventId,
    ) {
        if let Some(clock) = self.events.get(&event) {
            self.host.join(clock);
        }
    }
}
