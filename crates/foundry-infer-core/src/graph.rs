use std::time::Duration;

use crate::error::GraphError;
use crate::id::{
    Id,
    IdSpace,
    OpId,
    TensorId,
};
use crate::memory::ByteSize;
use crate::tensor::TensorDesc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightSource {
    HostMemory,
    MappedFile { offset: u64, len: ByteSize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightDesc {
    pub tensor: TensorId,
    pub bytes: ByteSize,
    pub source: WeightSource,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    MatMul {
        input: TensorId,
        weight: TensorId,
        output: TensorId,
    },
    SyntheticCompute {
        inputs: Vec<TensorId>,
        output: TensorId,
        duration_hint: Option<Duration>,
    },
}

impl Op {
    pub fn output(&self) -> TensorId {
        match self {
            Self::MatMul {
                output, ..
            }
            | Self::SyntheticCompute {
                output, ..
            } => *output,
        }
    }
}

#[derive(Clone, Copy)]
enum State {
    Uninitialized,
    Input,
    Weight,
    Output(OpId),
}

#[derive(Debug)]
pub struct Graph {
    tensor_ids: IdSpace<TensorId>,
    op_ids: IdSpace<OpId>,
    tensors: Vec<(TensorId, TensorDesc)>,
    inputs: Vec<TensorId>,
    weights: Vec<WeightDesc>,
    ops: Vec<(OpId, Op)>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            tensor_ids: IdSpace::new(),
            op_ids: IdSpace::new(),
            tensors: Vec::new(),
            inputs: Vec::new(),
            weights: Vec::new(),
            ops: Vec::new(),
        }
    }

    pub fn add_tensor(
        &mut self,
        desc: TensorDesc,
    ) -> Result<TensorId, GraphError> {
        let id = self.tensor_ids.allocate()?;
        self.tensors.push((id, desc));
        Ok(id)
    }

    pub fn mark_input(
        &mut self,
        tensor: TensorId,
    ) -> Result<(), GraphError> {
        self.check_known(tensor)?;
        if self.inputs.contains(&tensor) {
            return Err(GraphError::DuplicateInput(tensor));
        }
        if self.is_weight(tensor) {
            return Err(GraphError::InputWeightConflict(tensor));
        }
        self.inputs.push(tensor);
        Ok(())
    }

    pub fn add_weight(
        &mut self,
        weight: WeightDesc,
    ) -> Result<(), GraphError> {
        self.check_known(weight.tensor)?;
        if self.is_weight(weight.tensor) {
            return Err(GraphError::DuplicateWeight(weight.tensor));
        }
        if self.inputs.contains(&weight.tensor) {
            return Err(GraphError::InputWeightConflict(weight.tensor));
        }
        self.weights.push(weight);
        Ok(())
    }

    pub fn add_op(
        &mut self,
        op: Op,
    ) -> Result<OpId, GraphError> {
        let id = self.op_ids.allocate()?;
        self.ops.push((id, op));
        Ok(id)
    }

    pub fn tensors(&self) -> impl ExactSizeIterator<Item = (TensorId, &TensorDesc)> {
        self.tensors.iter().map(|(id, desc)| (*id, desc))
    }

    pub fn tensor(
        &self,
        id: TensorId,
    ) -> Option<&TensorDesc> {
        position(id)
            .and_then(|index| self.tensors.get(index))
            .map(|(_, desc)| desc)
    }

    pub fn inputs(&self) -> &[TensorId] { &self.inputs }

    pub fn weights(&self) -> &[WeightDesc] { &self.weights }

    pub fn ops(&self) -> impl ExactSizeIterator<Item = (OpId, &Op)> {
        self.ops.iter().map(|(id, op)| (*id, op))
    }

    pub fn validate(&self) -> Result<(), GraphError> {
        let sizes = self
            .tensors
            .iter()
            .map(|(tensor, desc)| {
                desc.byte_size().map_err(|source| GraphError::TensorSize {
                    tensor: *tensor,
                    source,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut states = vec![State::Uninitialized; self.tensors.len()];

        for &tensor in &self.inputs {
            let state = slot(&mut states, tensor).ok_or(GraphError::UnknownTensor(tensor))?;
            match *state {
                State::Uninitialized => *state = State::Input,
                State::Input => return Err(GraphError::DuplicateInput(tensor)),
                State::Weight | State::Output(_) => {
                    return Err(GraphError::InputWeightConflict(tensor));
                }
            }
        }

        for weight in &self.weights {
            let tensor = weight.tensor;
            let expected = position(tensor)
                .and_then(|index| sizes.get(index))
                .copied()
                .ok_or(GraphError::UnknownTensor(tensor))?;
            let state = slot(&mut states, tensor).ok_or(GraphError::UnknownTensor(tensor))?;
            match *state {
                State::Uninitialized => {}
                State::Input => return Err(GraphError::InputWeightConflict(tensor)),
                State::Weight | State::Output(_) => {
                    return Err(GraphError::DuplicateWeight(tensor));
                }
            }
            if weight.bytes != expected {
                return Err(GraphError::WeightSizeMismatch {
                    tensor,
                    expected,
                    actual: weight.bytes,
                });
            }
            match weight.source {
                WeightSource::HostMemory => {}
                WeightSource::MappedFile {
                    offset,
                    len,
                } => {
                    if len != weight.bytes {
                        return Err(GraphError::MappedLenMismatch {
                            tensor,
                            len,
                            bytes: weight.bytes,
                        });
                    }
                    if offset.checked_add(len.bytes()).is_none() {
                        return Err(GraphError::MappedRangeOverflow {
                            tensor,
                            offset,
                            len,
                        });
                    }
                }
            }
            *state = State::Weight;
        }

        for (op_id, op) in &self.ops {
            let op_id = *op_id;
            let pair;
            let reads: &[TensorId] = match op {
                Op::MatMul {
                    input,
                    weight,
                    ..
                } => {
                    pair = [*input, *weight];
                    &pair
                }
                Op::SyntheticCompute {
                    inputs, ..
                } => inputs,
            };
            for &tensor in reads {
                let state = slot(&mut states, tensor).ok_or(GraphError::UnknownOperand {
                    op: op_id,
                    tensor,
                })?;
                if matches!(state, State::Uninitialized) {
                    return Err(GraphError::UninitializedRead {
                        op: op_id,
                        tensor,
                    });
                }
            }
            let tensor = op.output();
            let state = slot(&mut states, tensor).ok_or(GraphError::UnknownOperand {
                op: op_id,
                tensor,
            })?;
            match *state {
                State::Uninitialized => *state = State::Output(op_id),
                State::Input => {
                    return Err(GraphError::OverwritesInput {
                        op: op_id,
                        tensor,
                    });
                }
                State::Weight => {
                    return Err(GraphError::OverwritesWeight {
                        op: op_id,
                        tensor,
                    });
                }
                State::Output(first) => {
                    return Err(GraphError::MultipleProducers {
                        tensor,
                        first,
                        second: op_id,
                    });
                }
            }
        }
        Ok(())
    }

    fn check_known(
        &self,
        tensor: TensorId,
    ) -> Result<(), GraphError> {
        self.tensor(tensor)
            .map(|_| ())
            .ok_or(GraphError::UnknownTensor(tensor))
    }

    fn is_weight(
        &self,
        tensor: TensorId,
    ) -> bool {
        self.weights.iter().any(|weight| weight.tensor == tensor)
    }
}

fn position(id: TensorId) -> Option<usize> { usize::try_from(id.index()).ok() }

fn slot(
    states: &mut [State],
    tensor: TensorId,
) -> Option<&mut State> {
    position(tensor).and_then(|index| states.get_mut(index))
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::time::Duration;

    use super::{
        Graph,
        Op,
        WeightDesc,
        WeightSource,
    };
    use crate::dtype::DType;
    use crate::error::{
        CoreError,
        GraphError,
    };
    use crate::id::{
        Id,
        TensorId,
    };
    use crate::memory::ByteSize;
    use crate::shape::Shape;
    use crate::tensor::TensorDesc;

    fn activation(graph: &mut Graph) -> Result<TensorId, GraphError> {
        graph.add_tensor(TensorDesc::new(DType::F32, Shape::new([1, 64])))
    }

    fn input(graph: &mut Graph) -> Result<TensorId, GraphError> {
        let tensor = activation(graph)?;
        graph.mark_input(tensor)?;
        Ok(tensor)
    }

    fn host_weight(
        tensor: TensorId,
        bytes: u64,
    ) -> WeightDesc {
        WeightDesc {
            tensor,
            bytes: ByteSize::from_bytes(bytes),
            source: WeightSource::HostMemory,
        }
    }

    fn weight(graph: &mut Graph) -> Result<TensorId, GraphError> {
        let tensor = graph.add_tensor(TensorDesc::new(DType::U8, Shape::new([256])))?;
        graph.add_weight(host_weight(tensor, 256))?;
        Ok(tensor)
    }

    fn compute(
        inputs: impl Into<Vec<TensorId>>,
        output: TensorId,
    ) -> Op {
        Op::SyntheticCompute {
            inputs: inputs.into(),
            output,
            duration_hint: None,
        }
    }

    fn foreign_tensor() -> Result<TensorId, GraphError> {
        let mut other = Graph::new();
        for _ in 0..8 {
            activation(&mut other)?;
        }
        activation(&mut other)
    }

    #[test]
    fn an_empty_graph_is_valid() {
        let graph = Graph::new();
        assert_eq!(graph.tensors().len(), 0, "no tensors");
        assert!(graph.inputs().is_empty(), "no inputs");
        assert!(graph.weights().is_empty(), "no weights");
        assert_eq!(graph.ops().len(), 0, "no ops");
        assert_eq!(graph.validate(), Ok(()), "an empty graph validates");
    }

    #[test]
    fn inputs_and_weights_are_available_to_the_first_op() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        let x = input(&mut graph)?;
        let w = weight(&mut graph)?;
        let y = activation(&mut graph)?;
        graph.add_op(compute([x, w], y))?;
        assert_eq!(graph.inputs(), &[x], "the input is registered");
        assert_eq!(
            graph.weights(),
            &[host_weight(w, 256)],
            "the weight is registered"
        );
        assert_eq!(graph.validate(), Ok(()), "the first op reads both");
        Ok(())
    }

    #[test]
    fn a_chain_validates_in_insertion_order() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        let mut previous = input(&mut graph)?;
        let mut expected = Vec::new();
        for _ in 0..4 {
            let w = weight(&mut graph)?;
            let output = activation(&mut graph)?;
            let op = compute([previous, w], output);
            let id = graph.add_op(op.clone())?;
            expected.push((id, op));
            previous = output;
        }
        let ops = graph
            .ops()
            .map(|(id, op)| (id, op.clone()))
            .collect::<Vec<_>>();
        assert_eq!(ops, expected, "ops are kept in insertion order");
        let indices = ops.iter().map(|(id, _)| id.index()).collect::<Vec<_>>();
        assert_eq!(indices, [0, 1, 2, 3], "op ids follow insertion order");
        assert_eq!(graph.validate(), Ok(()), "the chain validates");
        Ok(())
    }

    #[test]
    fn tensors_are_inspectable_by_id() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        let desc = TensorDesc::new(DType::BF16, Shape::new([4, 4]));
        let id = graph.add_tensor(desc.clone())?;
        assert_eq!(graph.tensor(id), Some(&desc), "lookup by id");
        assert_eq!(
            graph.tensors().collect::<Vec<_>>(),
            [(id, &desc)],
            "iteration yields ids and descriptions"
        );
        Ok(())
    }

    #[test]
    fn unused_tensors_are_allowed() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        activation(&mut graph)?;
        graph.add_tensor(TensorDesc::new(DType::U8, Shape::new([16])))?;
        assert_eq!(graph.validate(), Ok(()), "unused tensors validate");
        Ok(())
    }

    #[test]
    fn unknown_tensors_are_rejected() -> Result<(), GraphError> {
        let unknown = foreign_tensor()?;
        let mut graph = Graph::new();
        let x = input(&mut graph)?;
        let y = activation(&mut graph)?;
        assert_eq!(
            graph.mark_input(unknown),
            Err(GraphError::UnknownTensor(unknown)),
            "an unknown input"
        );
        assert_eq!(
            graph.add_weight(host_weight(unknown, 256)),
            Err(GraphError::UnknownTensor(unknown)),
            "an unknown weight"
        );

        let mut reads_unknown = Graph::new();
        let x2 = input(&mut reads_unknown)?;
        let y2 = activation(&mut reads_unknown)?;
        let op = reads_unknown.add_op(compute([x2, unknown], y2))?;
        assert_eq!(
            reads_unknown.validate(),
            Err(GraphError::UnknownOperand {
                op,
                tensor: unknown
            }),
            "an unknown operand"
        );

        let op = graph.add_op(compute([x], unknown))?;
        assert_eq!(
            graph.validate(),
            Err(GraphError::UnknownOperand {
                op,
                tensor: unknown
            }),
            "an unknown output"
        );
        assert!(graph.tensor(y).is_some(), "known tensors stay known");
        Ok(())
    }

    #[test]
    fn a_duplicate_input_is_rejected() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        let x = input(&mut graph)?;
        assert_eq!(
            graph.mark_input(x),
            Err(GraphError::DuplicateInput(x)),
            "the second registration fails"
        );
        assert_eq!(graph.inputs(), &[x], "the input is registered once");
        Ok(())
    }

    #[test]
    fn reading_an_uninitialized_tensor_is_rejected() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        let x = activation(&mut graph)?;
        let y = activation(&mut graph)?;
        let op = graph.add_op(compute([x], y))?;
        assert_eq!(
            graph.validate(),
            Err(GraphError::UninitializedRead {
                op,
                tensor: x
            }),
            "x is neither an input, a weight, nor produced"
        );
        Ok(())
    }

    #[test]
    fn reading_a_later_output_is_rejected() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        let x = input(&mut graph)?;
        let y = activation(&mut graph)?;
        let z = activation(&mut graph)?;
        let first = graph.add_op(compute([z], y))?;
        graph.add_op(compute([x], z))?;
        assert_eq!(
            graph.validate(),
            Err(GraphError::UninitializedRead {
                op: first,
                tensor: z
            }),
            "z is produced after it is read"
        );
        Ok(())
    }

    #[test]
    fn reading_the_own_output_is_rejected() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        let x = input(&mut graph)?;
        let y = activation(&mut graph)?;
        let op = graph.add_op(compute([x, y], y))?;
        assert_eq!(
            graph.validate(),
            Err(GraphError::UninitializedRead {
                op,
                tensor: y
            }),
            "an op cannot read what it produces"
        );
        Ok(())
    }

    #[test]
    fn a_tensor_has_one_producer() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        let x = input(&mut graph)?;
        let y = activation(&mut graph)?;
        let first = graph.add_op(compute([x], y))?;
        let second = graph.add_op(compute([x], y))?;
        assert_eq!(
            graph.validate(),
            Err(GraphError::MultipleProducers {
                tensor: y,
                first,
                second
            }),
            "y is written twice"
        );
        Ok(())
    }

    #[test]
    fn inputs_and_weights_cannot_be_overwritten() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        let x = input(&mut graph)?;
        let op = graph.add_op(compute([x], x))?;
        assert_eq!(
            graph.validate(),
            Err(GraphError::OverwritesInput {
                op,
                tensor: x
            }),
            "an input is overwritten"
        );

        let mut graph = Graph::new();
        let x = input(&mut graph)?;
        let w = weight(&mut graph)?;
        let op = graph.add_op(compute([x], w))?;
        assert_eq!(
            graph.validate(),
            Err(GraphError::OverwritesWeight {
                op,
                tensor: w
            }),
            "a weight is overwritten"
        );
        Ok(())
    }

    #[test]
    fn a_tensor_has_at_most_one_weight() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        let w = weight(&mut graph)?;
        assert_eq!(
            graph.add_weight(host_weight(w, 256)),
            Err(GraphError::DuplicateWeight(w)),
            "the second weight fails"
        );
        assert_eq!(graph.weights().len(), 1, "one weight is registered");
        Ok(())
    }

    #[test]
    fn a_tensor_cannot_be_both_input_and_weight() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        let x = input(&mut graph)?;
        assert_eq!(
            graph.add_weight(host_weight(x, 256)),
            Err(GraphError::InputWeightConflict(x)),
            "an input cannot become a weight"
        );
        let w = weight(&mut graph)?;
        assert_eq!(
            graph.mark_input(w),
            Err(GraphError::InputWeightConflict(w)),
            "a weight cannot become an input"
        );
        assert_eq!(graph.validate(), Ok(()), "the graph stays valid");
        Ok(())
    }

    #[test]
    fn a_weight_matches_its_tensor_size() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        let w = graph.add_tensor(TensorDesc::new(DType::F16, Shape::new([128])))?;
        graph.add_weight(host_weight(w, 128))?;
        assert_eq!(
            graph.validate(),
            Err(GraphError::WeightSizeMismatch {
                tensor: w,
                expected: ByteSize::from_bytes(256),
                actual: ByteSize::from_bytes(128),
            }),
            "128 F16 elements hold 256 bytes"
        );
        Ok(())
    }

    #[test]
    fn a_mapped_range_matches_its_weight() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        let w = graph.add_tensor(TensorDesc::new(DType::U8, Shape::new([256])))?;
        graph.add_weight(WeightDesc {
            tensor: w,
            bytes: ByteSize::from_bytes(256),
            source: WeightSource::MappedFile {
                offset: 4096,
                len: ByteSize::from_bytes(512),
            },
        })?;
        assert_eq!(
            graph.validate(),
            Err(GraphError::MappedLenMismatch {
                tensor: w,
                len: ByteSize::from_bytes(512),
                bytes: ByteSize::from_bytes(256),
            }),
            "the mapped length differs from the weight"
        );
        Ok(())
    }

    #[test]
    fn a_mapped_range_cannot_overflow() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        let w = graph.add_tensor(TensorDesc::new(DType::U8, Shape::new([256])))?;
        let len = ByteSize::from_bytes(256);
        graph.add_weight(WeightDesc {
            tensor: w,
            bytes: len,
            source: WeightSource::MappedFile {
                offset: u64::MAX,
                len,
            },
        })?;
        assert_eq!(
            graph.validate(),
            Err(GraphError::MappedRangeOverflow {
                tensor: w,
                offset: u64::MAX,
                len
            }),
            "offset + len overflows"
        );
        Ok(())
    }

    #[test]
    fn a_tensor_size_must_be_computable() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        let tensor = graph.add_tensor(TensorDesc::new(DType::F32, Shape::new([u64::MAX / 2])))?;
        let error = graph.validate();
        assert_eq!(
            error,
            Err(GraphError::TensorSize {
                tensor,
                source: CoreError::Overflow
            }),
            "u64::MAX / 2 F32 elements overflow"
        );
        let source = error
            .err()
            .as_ref()
            .and_then(Error::source)
            .map(ToString::to_string);
        assert_eq!(
            source,
            Some(CoreError::Overflow.to_string()),
            "the core error is the source"
        );
        Ok(())
    }

    #[test]
    fn a_matmul_is_described_and_validated() -> Result<(), GraphError> {
        let mut graph = Graph::new();
        let x = input(&mut graph)?;
        let w = graph.add_tensor(TensorDesc::new(DType::BF16, Shape::new([64, 32])))?;
        let y = graph.add_tensor(TensorDesc::new(DType::F32, Shape::new([1, 32])))?;
        let matmul = Op::MatMul {
            input: x,
            weight: w,
            output: y,
        };
        let op = graph.add_op(matmul.clone())?;
        assert_eq!(
            graph.ops().collect::<Vec<_>>(),
            [(op, &matmul)],
            "the matmul is kept as described"
        );
        assert_eq!(matmul.output(), y, "the matmul writes y");
        assert_eq!(
            graph.validate(),
            Err(GraphError::UninitializedRead {
                op,
                tensor: w
            }),
            "the weight is not described yet"
        );
        graph.add_weight(host_weight(w, 64 * 32 * 2))?;
        assert_eq!(graph.validate(), Ok(()), "the matmul validates");
        Ok(())
    }

    #[test]
    fn failed_mutations_leave_the_graph_unchanged() -> Result<(), GraphError> {
        let unknown = foreign_tensor()?;
        let mut graph = Graph::new();
        let x = input(&mut graph)?;
        let w = weight(&mut graph)?;
        let y = activation(&mut graph)?;
        graph.add_op(compute([x, w], y))?;

        let snapshot = |graph: &Graph| {
            (
                graph
                    .tensors()
                    .map(|(id, desc)| (id, desc.clone()))
                    .collect::<Vec<_>>(),
                graph.inputs().to_vec(),
                graph.weights().to_vec(),
                graph
                    .ops()
                    .map(|(id, op)| (id, op.clone()))
                    .collect::<Vec<_>>(),
            )
        };
        let before = snapshot(&graph);
        let failures = [
            graph.mark_input(unknown),
            graph.mark_input(x),
            graph.mark_input(w),
            graph.add_weight(host_weight(unknown, 256)),
            graph.add_weight(host_weight(w, 256)),
            graph.add_weight(host_weight(x, 256)),
        ];
        assert!(
            failures.iter().all(Result::is_err),
            "every mutation fails: {failures:?}"
        );
        assert_eq!(snapshot(&graph), before, "the graph is unchanged");
        assert_eq!(
            activation(&mut graph)?.index(),
            3,
            "the next tensor id follows the last registered one"
        );
        assert_eq!(
            graph.add_op(compute([y], y))?.index(),
            1,
            "the next op id follows the last appended one"
        );
        Ok(())
    }

    #[test]
    fn a_core_error_is_wrapped() {
        let error = GraphError::from(CoreError::IdsExhausted);
        assert_eq!(
            error,
            GraphError::Core(CoreError::IdsExhausted),
            "conversion wraps the core error"
        );
        assert_eq!(
            Error::source(&error).map(ToString::to_string),
            Some(CoreError::IdsExhausted.to_string()),
            "the core error is the source"
        );
    }

    #[test]
    fn describes_more_than_20_gib_of_weights_as_metadata() -> Result<(), GraphError> {
        let layer_bytes = ByteSize::from_mib(450)?;
        let mut graph = Graph::new();
        let mut previous = input(&mut graph)?;
        let mut offset = ByteSize::from_bytes(0);

        for _ in 0..48 {
            let tensor = graph.add_tensor(TensorDesc::new(
                DType::U8,
                Shape::new([layer_bytes.bytes()]),
            ))?;
            graph.add_weight(WeightDesc {
                tensor,
                bytes: layer_bytes,
                source: WeightSource::MappedFile {
                    offset: offset.bytes(),
                    len: layer_bytes,
                },
            })?;
            offset = offset.checked_add(layer_bytes)?;
            let output = activation(&mut graph)?;
            graph.add_op(Op::SyntheticCompute {
                inputs: vec![previous, tensor],
                output,
                duration_hint: Some(Duration::from_millis(60)),
            })?;
            previous = output;
        }

        let total = graph
            .weights()
            .iter()
            .try_fold(ByteSize::from_bytes(0), |total, weight| {
                total.checked_add(weight.bytes)
            })?;
        assert_eq!(graph.weights().len(), 48, "one weight per layer");
        assert_eq!(graph.ops().len(), 48, "one op per layer");
        assert_eq!(total, layer_bytes.checked_mul(48)?, "total weight bytes");
        assert!(
            total > ByteSize::from_gib(20)?,
            "{total} of weights exceed 20 GiB"
        );
        assert_eq!(graph.validate(), Ok(()), "the workload validates");
        Ok(())
    }
}
