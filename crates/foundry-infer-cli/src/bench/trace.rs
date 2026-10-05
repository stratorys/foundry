use std::collections::BTreeMap;

use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Value;

use crate::bench::stats::lossy;

pub(crate) const PROCESS: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Track {
    CpuRuntime,
    CpuWaits,
    CpuHarness,
    GpuCopy,
    GpuCompute,
    Memory,
}

impl Track {
    pub(crate) const ALL: [Self; 6] = [
        Self::CpuRuntime,
        Self::CpuWaits,
        Self::CpuHarness,
        Self::GpuCopy,
        Self::GpuCompute,
        Self::Memory,
    ];

    pub(crate) const fn id(self) -> u32 {
        match self {
            Self::CpuRuntime => 1,
            Self::CpuWaits => 2,
            Self::CpuHarness => 3,
            Self::GpuCopy => 10,
            Self::GpuCompute => 11,
            Self::Memory => 20,
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::CpuRuntime => "CPU runtime: commands, staging, submission",
            Self::CpuWaits => "CPU host waits and drains",
            Self::CpuHarness => "CPU benchmark harness: timed region, verification",
            Self::GpuCopy => "GPU copy command buffers",
            Self::GpuCompute => "GPU checksum compute command buffers",
            Self::Memory => "Allocations: lifetimes and counters",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct TraceEvent {
    pub(crate) name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) cat: Option<String>,
    pub(crate) ph: String,
    pub(crate) ts: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) dur: Option<f64>,
    pub(crate) pid: u32,
    pub(crate) tid: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) s: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) args: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct TraceDocument {
    #[serde(rename = "traceEvents")]
    pub(crate) trace_events: Vec<TraceEvent>,
    #[serde(rename = "displayTimeUnit")]
    pub(crate) display_time_unit: String,
    #[serde(rename = "otherData")]
    pub(crate) other_data: BTreeMap<String, Value>,
}

pub(crate) fn lane_id(
    track: Track,
    lane: usize,
) -> Option<u32> {
    track
        .id()
        .checked_mul(100)?
        .checked_add(u32::try_from(lane).ok()?)
}

pub(crate) fn separate_overlaps(
    events: &mut [TraceEvent],
    track: Track,
) -> Vec<TraceEvent> {
    let mut order: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, event)| event.ph == "X" && event.tid == track.id())
        .map(|(index, _)| index)
        .collect();
    order.sort_by(|first, second| {
        let start = |index: &usize| events.get(*index).map_or(0.0, |event| event.ts);
        start(first).total_cmp(&start(second))
    });
    let mut lanes: Vec<f64> = Vec::new();
    for index in order {
        let Some(event) = events.get_mut(index) else {
            continue;
        };
        let end = event.ts + event.dur.unwrap_or_default();
        let lane = match lanes.iter().position(|&busy| busy <= event.ts) {
            Some(lane) => lane,
            None => {
                lanes.push(end);
                lanes.len().saturating_sub(1)
            }
        };
        if let Some(busy) = lanes.get_mut(lane) {
            *busy = end;
        }
        if lane > 0 {
            event.tid = lane_id(track, lane).unwrap_or(event.tid);
            event
                .args
                .insert("overlap_lane".to_owned(), Value::from(lane));
        }
    }
    (1..lanes.len())
        .filter_map(|lane| {
            let tid = lane_id(track, lane)?;
            let named = |name: &str, value: Value| TraceEvent {
                tid,
                args: BTreeMap::from([(name.to_owned(), value)]),
                ..event(
                    if name == "name" {
                        "thread_name"
                    } else {
                        "thread_sort_index"
                    },
                    "M",
                    track,
                    0.0,
                )
            };
            Some([
                named(
                    "name",
                    Value::from(format!("{} (overlap lane {lane})", track.name())),
                ),
                named("sort_index", Value::from(tid)),
            ])
        })
        .flatten()
        .collect()
}

pub(crate) fn micros(nanos: u64) -> f64 { lossy(u128::from(nanos)) / 1000.0 }

fn event(
    name: &str,
    ph: &str,
    track: Track,
    ts: f64,
) -> TraceEvent {
    TraceEvent {
        name: name.to_owned(),
        cat: None,
        ph: ph.to_owned(),
        ts,
        dur: None,
        pid: PROCESS,
        tid: track.id(),
        id: None,
        s: None,
        args: BTreeMap::new(),
    }
}

pub(crate) fn metadata() -> Vec<TraceEvent> {
    let process = TraceEvent {
        args: BTreeMap::from([("name".to_owned(), Value::from("foundry bench"))]),
        ..event("process_name", "M", Track::CpuRuntime, 0.0)
    };
    let threads = Track::ALL.into_iter().flat_map(|track| {
        [
            TraceEvent {
                args: BTreeMap::from([("name".to_owned(), Value::from(track.name()))]),
                ..event("thread_name", "M", track, 0.0)
            },
            TraceEvent {
                args: BTreeMap::from([("sort_index".to_owned(), Value::from(track.id()))]),
                ..event("thread_sort_index", "M", track, 0.0)
            },
        ]
    });
    std::iter::once(process).chain(threads).collect()
}

pub(crate) fn span(
    name: &str,
    category: &str,
    track: Track,
    start_ns: u64,
    end_ns: u64,
    args: BTreeMap<String, Value>,
) -> TraceEvent {
    TraceEvent {
        cat: Some(category.to_owned()),
        dur: Some(micros(end_ns.saturating_sub(start_ns))),
        args,
        ..event(name, "X", track, micros(start_ns))
    }
}

pub(crate) fn instant(
    name: &str,
    category: &str,
    track: Track,
    at_ns: u64,
    args: BTreeMap<String, Value>,
) -> TraceEvent {
    TraceEvent {
        cat: Some(category.to_owned()),
        s: Some("t".to_owned()),
        args,
        ..event(name, "i", track, micros(at_ns))
    }
}

pub(crate) fn counter(
    name: &str,
    at_ns: u64,
    values: BTreeMap<String, Value>,
) -> TraceEvent {
    TraceEvent {
        cat: Some("memory".to_owned()),
        args: values,
        ..event(name, "C", Track::Memory, micros(at_ns))
    }
}

pub(crate) fn lifetime(
    name: &str,
    begin: bool,
    id: String,
    at_ns: u64,
    args: BTreeMap<String, Value>,
) -> TraceEvent {
    TraceEvent {
        cat: Some("allocation".to_owned()),
        id: Some(id),
        args,
        ..event(
            name,
            if begin { "b" } else { "e" },
            Track::Memory,
            micros(at_ns),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        Track,
        counter,
        lane_id,
        metadata,
        micros,
        separate_overlaps,
        span,
    };

    #[test]
    fn microseconds_keep_nanosecond_precision() {
        assert!((micros(1_500) - 1.5).abs() < 1e-12, "1500 ns is 1.5 us");
        assert!(
            (micros(3_600_000_000_001) - 3_600_000_000.001).abs() < 1e-6,
            "an hour keeps its nanoseconds"
        );
    }

    #[test]
    fn every_track_is_named() {
        let names: Vec<(u32, String)> = metadata()
            .into_iter()
            .filter(|event| event.name == "thread_name")
            .map(|event| {
                (
                    event.tid,
                    event
                        .args
                        .get("name")
                        .and_then(|name| name.as_str())
                        .unwrap_or_default()
                        .to_owned(),
                )
            })
            .collect();
        assert_eq!(names.len(), Track::ALL.len(), "one name per track");
        let tids: std::collections::BTreeSet<u32> = names.iter().map(|(tid, _)| *tid).collect();
        assert_eq!(tids.len(), Track::ALL.len(), "track ids are distinct");
    }

    #[test]
    fn spans_and_counters_serialize_as_chrome_events() -> Result<(), Box<dyn std::error::Error>> {
        let value = serde_json::to_value(span(
            "copy",
            "gpu",
            Track::GpuCopy,
            2_000,
            5_000,
            BTreeMap::new(),
        ))?;
        assert_eq!(
            value,
            serde_json::json!({
                "name": "copy", "cat": "gpu", "ph": "X", "ts": 2.0, "dur": 3.0,
                "pid": 1, "tid": 10
            }),
            "a complete event"
        );
        let value = serde_json::to_value(counter(
            "bytes",
            1_000,
            BTreeMap::from([("staging".to_owned(), serde_json::Value::from(4))]),
        ))?;
        assert_eq!(
            value.get("ph").and_then(|ph| ph.as_str()),
            Some("C"),
            "a counter event"
        );
        Ok(())
    }

    #[test]
    fn overlapping_spans_move_to_named_lanes() {
        let mut events = vec![
            span("a", "gpu", Track::GpuCompute, 0, 10_000, BTreeMap::new()),
            span(
                "b",
                "gpu",
                Track::GpuCompute,
                8_000,
                12_000,
                BTreeMap::new(),
            ),
            span(
                "c",
                "gpu",
                Track::GpuCompute,
                12_000,
                14_000,
                BTreeMap::new(),
            ),
            span("d", "gpu", Track::GpuCopy, 9_000, 9_500, BTreeMap::new()),
        ];
        let lanes = separate_overlaps(&mut events, Track::GpuCompute);
        let tids: Vec<u32> = events.iter().map(|event| event.tid).collect();
        let lane = lane_id(Track::GpuCompute, 1).unwrap_or_default();
        assert_eq!(
            tids,
            [
                Track::GpuCompute.id(),
                lane,
                Track::GpuCompute.id(),
                Track::GpuCopy.id()
            ],
            "only the overlapping span moves, and other tracks are untouched"
        );
        assert_eq!(
            (
                events.get(1).map(|event| event.ts),
                events.get(1).and_then(|event| event.dur)
            ),
            (Some(8.0), Some(4.0)),
            "timestamps are unchanged"
        );
        assert!(
            lanes.iter().any(|event| event.tid == lane
                && event.name == "thread_name"
                && event
                    .args
                    .get("name")
                    .and_then(|name| name.as_str())
                    .is_some_and(|name| name.contains("overlap lane 1"))),
            "the extra lane is named"
        );
    }
}
