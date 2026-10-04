use foundry_infer_core::{
    ByteSize,
    Id,
    OpId,
    TensorId,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::error::CodecError;
use crate::id::{
    BufferSlot,
    EventId,
    StreamId,
};
use crate::plan::{
    Command,
    ExecutionPlan,
    WaitTarget,
    WeightBinding,
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanRepr {
    commands: Vec<CommandRepr>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum CommandRepr {
    Allocate {
        slot: u32,
        bytes: u64,
    },
    Prefetch {
        tensor: u32,
        slot: u32,
        stream: u32,
        event: u32,
    },
    Wait {
        target: TargetRepr,
        event: u32,
    },
    Launch {
        op: u32,
        stream: u32,
        weight_slots: Vec<BindingRepr>,
        event: u32,
    },
    Release {
        slot: u32,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum TargetRepr {
    Host,
    Stream(u32),
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingRepr {
    tensor: u32,
    slot: u32,
}

impl From<&Command> for CommandRepr {
    fn from(command: &Command) -> Self {
        match command {
            Command::Allocate {
                slot,
                bytes,
            } => Self::Allocate {
                slot: slot.index(),
                bytes: bytes.bytes(),
            },
            Command::Prefetch {
                tensor,
                slot,
                stream,
                event,
            } => Self::Prefetch {
                tensor: tensor.index(),
                slot: slot.index(),
                stream: stream.index(),
                event: event.index(),
            },
            Command::Wait {
                target,
                event,
            } => Self::Wait {
                target: match target {
                    WaitTarget::Host => TargetRepr::Host,
                    WaitTarget::Stream(stream) => TargetRepr::Stream(stream.index()),
                },
                event: event.index(),
            },
            Command::Launch {
                op,
                stream,
                weight_slots,
                event,
            } => Self::Launch {
                op: op.index(),
                stream: stream.index(),
                weight_slots: weight_slots
                    .iter()
                    .map(|binding| BindingRepr {
                        tensor: binding.tensor.index(),
                        slot: binding.slot.index(),
                    })
                    .collect(),
                event: event.index(),
            },
            Command::Release {
                slot,
            } => Self::Release {
                slot: slot.index(),
            },
        }
    }
}

impl From<CommandRepr> for Command {
    fn from(repr: CommandRepr) -> Self {
        match repr {
            CommandRepr::Allocate {
                slot,
                bytes,
            } => Self::Allocate {
                slot: BufferSlot::new(slot),
                bytes: ByteSize::from_bytes(bytes),
            },
            CommandRepr::Prefetch {
                tensor,
                slot,
                stream,
                event,
            } => Self::Prefetch {
                tensor: TensorId::from_index(tensor),
                slot: BufferSlot::new(slot),
                stream: StreamId::new(stream),
                event: EventId::new(event),
            },
            CommandRepr::Wait {
                target,
                event,
            } => Self::Wait {
                target: match target {
                    TargetRepr::Host => WaitTarget::Host,
                    TargetRepr::Stream(stream) => WaitTarget::Stream(StreamId::new(stream)),
                },
                event: EventId::new(event),
            },
            CommandRepr::Launch {
                op,
                stream,
                weight_slots,
                event,
            } => Self::Launch {
                op: OpId::from_index(op),
                stream: StreamId::new(stream),
                weight_slots: weight_slots
                    .into_iter()
                    .map(|binding| WeightBinding {
                        tensor: TensorId::from_index(binding.tensor),
                        slot: BufferSlot::new(binding.slot),
                    })
                    .collect(),
                event: EventId::new(event),
            },
            CommandRepr::Release {
                slot,
            } => Self::Release {
                slot: BufferSlot::new(slot),
            },
        }
    }
}

impl ExecutionPlan {
    pub fn to_json(&self) -> Result<String, CodecError> {
        let repr = PlanRepr {
            commands: self.commands().iter().map(CommandRepr::from).collect(),
        };
        serde_json::to_string_pretty(&repr).map_err(CodecError::new)
    }

    pub fn from_json(json: &str) -> Result<Self, CodecError> {
        let repr = serde_json::from_str::<PlanRepr>(json).map_err(CodecError::new)?;
        Ok(Self::new(
            repr.commands.into_iter().map(Command::from).collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use foundry_infer_core::Alignment;

    use crate::error::PlanError;
    use crate::fixtures::{
        TestResult,
        budget,
        chain,
        one_layer,
        plan,
        without,
    };
    use crate::plan::ExecutionPlan;
    use crate::synthetic::synthetic_chain_plan;

    #[test]
    fn a_plan_round_trips_through_json() -> TestResult {
        let graph = chain(5)?;
        let slots = NonZeroU32::new(3).ok_or("three slots")?;
        let original = synthetic_chain_plan(&graph, slots, Alignment::new(256)?)?;
        let json = original.to_json()?;
        let decoded = ExecutionPlan::from_json(&json)?;
        assert_eq!(decoded, original, "the decoded plan equals the original");
        assert_eq!(decoded.to_json()?, json, "encoding is stable");
        assert_eq!(decoded.validate(&graph, budget()), Ok(()), "it validates");
        Ok(())
    }

    #[test]
    fn the_encoding_uses_integer_indices() -> TestResult {
        let json = plan(one_layer()).to_json()?;
        let compact = json.split_whitespace().collect::<String>();
        let expected = concat!(
            r#"{"commands":["#,
            r#"{"kind":"allocate","slot":0,"bytes":1024},"#,
            r#"{"kind":"prefetch","tensor":1,"slot":0,"stream":0,"event":0},"#,
            r#"{"kind":"wait","target":{"stream":1},"event":0},"#,
            r#"{"kind":"launch","op":0,"stream":1,"weight_slots":[{"tensor":1,"slot":0}],"event":1},"#,
            r#"{"kind":"wait","target":"host","event":1},"#,
            r#"{"kind":"release","slot":0}"#,
            r#"]}"#
        );
        assert_eq!(compact, expected, "the wire format");
        Ok(())
    }

    #[test]
    fn malformed_input_is_rejected() {
        let inputs = [
            "",
            "{",
            "[]",
            r#"{"commands":[{"kind":"allocate","slot":0}]}"#,
            r#"{"commands":[{"kind":"compile","slot":0}]}"#,
            r#"{"commands":[{"kind":"release","slot":0,"extra":1}]}"#,
            r#"{"commands":[{"kind":"release","slot":-1}]}"#,
            r#"{"commands":[{"kind":"release","slot":4294967296}]}"#,
            r#"{"commands":[{"kind":"allocate","slot":0,"bytes":18446744073709551616}]}"#,
            r#"{"commands":[{"kind":"wait","target":"device","event":0}]}"#,
            r#"{"commands":[],"graph":{}}"#,
        ];
        for input in inputs {
            assert!(
                ExecutionPlan::from_json(input).is_err(),
                "{input:?} is rejected"
            );
        }
    }

    #[test]
    fn decoding_does_not_validate() -> TestResult {
        let invalid = plan(without(one_layer(), 2)).to_json()?;
        let decoded = ExecutionPlan::from_json(&invalid)?;
        assert!(
            matches!(
                decoded.validate(&chain(1)?, budget()),
                Err(PlanError::TransferNotReady { .. })
            ),
            "the decoded plan still fails validation"
        );
        Ok(())
    }
}
