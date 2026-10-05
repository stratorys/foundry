use std::collections::BTreeMap;
use std::error::Error;
use std::num::{
    NonZeroU32,
    NonZeroU64,
};

use foundry_infer_core::{
    Alignment,
    ByteSize,
    Graph,
    TensorId,
};
use foundry_infer_plan::{
    Command,
    ExecutionPlan,
};

pub(crate) const BUFFER_COUNTS: [u32; 3] = [1, 2, 3];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Sizes {
    pub(crate) layers: u32,
    pub(crate) weight: ByteSize,
    pub(crate) weight_len: usize,
    pub(crate) total: ByteSize,
}

impl Sizes {
    pub(crate) fn new(
        layers: NonZeroU32,
        weight_mib: NonZeroU64,
    ) -> Result<Self, String> {
        let weight = ByteSize::from_mib(weight_mib.get())
            .map_err(|_| format!("a weight of {weight_mib} MiB overflows a 64-bit byte count"))?;
        let total = weight.checked_mul(u64::from(layers.get())).map_err(|_| {
            format!("{layers} layers of {weight_mib} MiB overflow a 64-bit byte count")
        })?;
        let addressable = |bytes: ByteSize, what: &str| {
            usize::try_from(bytes.bytes())
                .map_err(|_| format!("the {what} of {bytes} exceeds the host address space"))
        };
        let weight_len = addressable(weight, "layer weight")?;
        addressable(total, "host payload")?;
        Ok(Self {
            layers: layers.get(),
            weight,
            weight_len,
            total,
        })
    }
}

pub(crate) fn aligned_slot(
    weight: ByteSize,
    alignment: Alignment,
) -> Result<ByteSize, String> {
    weight
        .align_up(alignment)
        .map_err(|_| format!("aligning a {weight} weight slot overflows"))
}

pub(crate) fn check_buffer_limit(
    slot: ByteSize,
    limit: ByteSize,
) -> Result<(), String> {
    if slot > limit {
        Err(format!(
            "an aligned weight slot of {slot} exceeds the {limit} Metal buffer length limit"
        ))
    } else {
        Ok(())
    }
}

pub(crate) fn slot_budget(
    weight: ByteSize,
    alignment: Alignment,
) -> Result<ByteSize, String> {
    let slots = BUFFER_COUNTS.into_iter().max().unwrap_or_default();
    aligned_slot(weight, alignment)?
        .checked_mul(u64::from(slots))
        .map_err(|_| format!("the slot budget for {slots} aligned {weight} slots overflows"))
}

pub(crate) fn round_order(round: u32) -> [u32; 3] {
    let mut order = BUFFER_COUNTS;
    order.rotate_left(usize::try_from(round % 3).unwrap_or_default());
    order
}

pub(crate) fn planned_peak(plan: &ExecutionPlan) -> Result<ByteSize, Box<dyn Error>> {
    let mut live = BTreeMap::new();
    let mut current = ByteSize::default();
    let mut peak = current;
    for command in plan.commands() {
        match command {
            Command::Allocate {
                slot,
                bytes,
            } => {
                live.insert(*slot, *bytes);
                current = current.checked_add(*bytes)?;
                peak = peak.max(current);
            }
            Command::Release {
                slot,
            } => {
                let freed = live.remove(slot).unwrap_or_default();
                current = ByteSize::from_bytes(current.bytes().saturating_sub(freed.bytes()));
            }
            Command::Prefetch {
                ..
            }
            | Command::Wait {
                ..
            }
            | Command::Launch {
                ..
            } => {}
        }
    }
    Ok(peak)
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

pub(crate) fn payload(
    layer: u32,
    length: usize,
) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(|_| {
        format!(
            "cannot allocate {} of host payload for layer {layer}",
            ByteSize::from_bytes(u64::try_from(length).unwrap_or(u64::MAX))
        )
    })?;
    bytes.resize(length, 0);
    let mut state = u64::from(layer).wrapping_mul(0xD1B5_4A32_D192_ED03);
    for chunk in bytes.chunks_mut(size_of::<u64>()) {
        for (byte, value) in chunk.iter_mut().zip(splitmix(&mut state).to_le_bytes()) {
            *byte = value;
        }
    }
    Ok(bytes)
}

pub(crate) fn payloads(
    graph: &Graph,
    length: usize,
) -> Result<BTreeMap<TensorId, Vec<u8>>, String> {
    (0_u32..)
        .zip(graph.weights())
        .map(|(layer, weight)| Ok((weight.tensor, payload(layer, length)?)))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::error::Error;
    use std::num::{
        NonZeroU32,
        NonZeroU64,
    };

    use foundry_infer_core::{
        Alignment,
        ByteSize,
        MemoryBudget,
        MemorySpace,
    };
    use foundry_infer_plan::{
        synthetic_chain_graph,
        synthetic_chain_plan,
    };

    use super::{
        Sizes,
        aligned_slot,
        check_buffer_limit,
        payload,
        payloads,
        planned_peak,
        round_order,
        slot_budget,
    };

    type TestResult = Result<(), Box<dyn Error>>;

    fn sizes(
        layers: u32,
        weight_mib: u64,
    ) -> Result<Result<Sizes, String>, Box<dyn Error>> {
        Ok(Sizes::new(
            NonZeroU32::new(layers).ok_or("zero layers")?,
            NonZeroU64::new(weight_mib).ok_or("zero weight")?,
        ))
    }

    #[test]
    fn sizes_are_derived_from_the_arguments() -> TestResult {
        assert_eq!(
            sizes(48, 64)??,
            Sizes {
                layers: 48,
                weight: ByteSize::from_mib(64)?,
                weight_len: 64 << 20,
                total: ByteSize::from_gib(3)?,
            },
            "the default workload"
        );
        Ok(())
    }

    #[test]
    fn size_overflow_is_reported() -> TestResult {
        let error = sizes(1, u64::MAX)?.err().unwrap_or_default();
        assert!(
            error.contains("overflows"),
            "an oversized weight is rejected, got {error:?}"
        );
        let error = sizes(u32::MAX, 1 << 40)?.err().unwrap_or_default();
        assert!(
            error.contains("overflow"),
            "an oversized payload is rejected, got {error:?}"
        );
        Ok(())
    }

    #[test]
    fn slots_are_aligned_and_checked_against_the_buffer_limit() -> TestResult {
        let alignment = Alignment::new(256)?;
        assert_eq!(
            aligned_slot(ByteSize::from_bytes(1000), alignment)?,
            ByteSize::from_bytes(1024),
            "a slot rounds up to the alignment"
        );
        let error = aligned_slot(ByteSize::from_bytes(u64::MAX - 1), alignment)
            .err()
            .unwrap_or_default();
        assert!(
            error.contains("overflows"),
            "an overflowing slot is rejected, got {error:?}"
        );
        let limit = ByteSize::from_bytes(1024);
        assert_eq!(
            check_buffer_limit(limit, limit),
            Ok(()),
            "a slot at the limit fits"
        );
        let error = check_buffer_limit(ByteSize::from_bytes(1025), limit)
            .err()
            .unwrap_or_default();
        assert!(
            error.contains("exceeds the 1.0 KiB Metal buffer length limit"),
            "a slot beyond the limit is rejected, got {error:?}"
        );
        Ok(())
    }

    #[test]
    fn the_slot_budget_holds_three_aligned_slots() -> TestResult {
        assert_eq!(
            slot_budget(ByteSize::from_bytes(1000), Alignment::new(256)?)?,
            ByteSize::from_bytes(3 * 1024),
            "slots are aligned before multiplying"
        );
        assert!(
            slot_budget(ByteSize::from_bytes(u64::MAX - 1), Alignment::new(256)?).is_err(),
            "an overflowing budget is rejected"
        );
        Ok(())
    }

    #[test]
    fn payloads_are_deterministic_distinct_and_exact() -> TestResult {
        for length in [1, 7, 8, 1001] {
            let first = payload(0, length)?;
            assert_eq!(first.len(), length, "{length} bytes are generated");
            assert_eq!(first, payload(0, length)?, "{length} bytes repeat");
        }
        let graph = synthetic_chain_graph(6, ByteSize::from_bytes(4096), None)?;
        let generated = payloads(&graph, 4096)?;
        assert_eq!(generated.len(), 6, "every layer has a payload");
        assert_eq!(generated, payloads(&graph, 4096)?, "payloads repeat");
        let distinct: BTreeSet<&Vec<u8>> = generated.values().collect();
        assert_eq!(distinct.len(), 6, "layers have distinct contents");
        for weight in graph.weights() {
            assert_eq!(
                generated.get(&weight.tensor).map(Vec::len),
                Some(4096),
                "each payload matches its weight"
            );
        }
        Ok(())
    }

    #[test]
    fn planned_peaks_count_resident_slots() -> TestResult {
        let alignment = Alignment::new(256)?;
        let weight = ByteSize::from_bytes(1000);
        let budget = MemoryBudget::new(MemorySpace::Device, slot_budget(weight, alignment)?);
        for (layers, expected) in [(5, [1, 2, 3]), (2, [1, 2, 2])] {
            let graph = synthetic_chain_graph(layers, weight, None)?;
            for (slots, resident) in [1, 2, 3].into_iter().zip(expected) {
                let plan = synthetic_chain_plan(
                    &graph,
                    NonZeroU32::new(slots).ok_or("zero slots")?,
                    alignment,
                )?;
                plan.validate(&graph, budget)?;
                assert_eq!(
                    planned_peak(&plan)?,
                    ByteSize::from_bytes(resident * 1024),
                    "{layers} layers with {slots} slots"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn rounds_rotate_the_configuration_order() {
        assert_eq!(round_order(0), [1, 2, 3], "first round");
        assert_eq!(round_order(1), [2, 3, 1], "second round");
        assert_eq!(round_order(2), [3, 1, 2], "third round");
        assert_eq!(round_order(3), [1, 2, 3], "the rotation repeats");
    }
}
