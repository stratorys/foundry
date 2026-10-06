use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::{
    Value,
    json,
};

use crate::bench::report::{
    BuildProvenance,
    Environment,
    RepositoryProvenance,
};

pub(crate) const SCHEMA_VERSION: u32 = 1;
pub(crate) const PROTOCOL_ID: &str = "token-ids-512x128-v1";
pub(crate) const WARMUP_EXECUTIONS: u32 = 2;
pub(crate) const MEASURED_EXECUTIONS: u32 = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Status {
    Running,
    Succeeded,
    Failed,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Failure {
    pub(crate) stage: String,
    pub(crate) message: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Command {
    pub(crate) argv: Vec<String>,
    pub(crate) working_directory: Option<String>,
    pub(crate) snapshot: String,
    pub(crate) manifest: String,
    pub(crate) report: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Verification {
    pub(crate) command: Vec<String>,
    pub(crate) returncode: Option<i32>,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Checkpoint {
    pub(crate) repository: String,
    pub(crate) revision: String,
    pub(crate) manifest_sha256: String,
    pub(crate) weights_file_bytes: Option<u64>,
    pub(crate) weights_file_sha256_manifest: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Input {
    pub(crate) token_count: usize,
    pub(crate) sha256: String,
    pub(crate) token_ids: Vec<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Loading {
    pub(crate) device: String,
    pub(crate) load_ns: u64,
    pub(crate) definition: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Execution {
    pub(crate) phase: &'static str,
    pub(crate) index: u32,
    pub(crate) generated_count: usize,
    pub(crate) token_ids: Vec<u32>,
    pub(crate) elapsed_ns: u64,
    pub(crate) token_available_ns: Vec<u64>,
    pub(crate) first_token_ns: Option<u64>,
    pub(crate) decode_tokens_per_s: Option<f64>,
    pub(crate) device_allocated_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct MetricSummary {
    pub(crate) unit: &'static str,
    pub(crate) count: usize,
    pub(crate) median: f64,
    pub(crate) mean: f64,
    pub(crate) min: f64,
    pub(crate) max: f64,
    pub(crate) stdev_sample: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Mismatch {
    pub(crate) phase: &'static str,
    pub(crate) index: u32,
    pub(crate) first_divergent_position: usize,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Consistency {
    pub(crate) consistent: Option<bool>,
    pub(crate) compared_executions: usize,
    pub(crate) mismatches: Vec<Mismatch>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct Memory {
    pub(crate) definitions: BTreeMap<&'static str, &'static str>,
    pub(crate) weight_tensor_bytes: Option<u64>,
    pub(crate) weight_buffer_allocated_bytes: Option<u64>,
    pub(crate) rope_buffer_allocated_bytes: Option<u64>,
    pub(crate) kv_cache_allocated_bytes: Option<u64>,
    pub(crate) activation_allocated_bytes: Option<u64>,
    pub(crate) device_allocated_bytes_after_load: Option<u64>,
    pub(crate) device_allocated_bytes_after_session: Option<u64>,
    pub(crate) device_allocated_bytes_max_after_execution: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Foundry {
    pub(crate) build: Option<BuildProvenance>,
    pub(crate) repository: RepositoryProvenance,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Report {
    pub(crate) schema_version: u32,
    pub(crate) engine: &'static str,
    pub(crate) status: Status,
    pub(crate) failure: Option<Failure>,
    pub(crate) started_utc: String,
    pub(crate) finished_utc: Option<String>,
    pub(crate) command: Command,
    pub(crate) foundry: Foundry,
    pub(crate) environment: Option<Environment>,
    pub(crate) protocol: Value,
    pub(crate) timing_boundaries: Value,
    pub(crate) metrics: Value,
    pub(crate) checkpoint: Option<Checkpoint>,
    pub(crate) verification: Option<Verification>,
    pub(crate) input: Option<Input>,
    pub(crate) loading: Option<Loading>,
    pub(crate) memory: Memory,
    pub(crate) executions: Vec<Execution>,
    pub(crate) statistics: Option<BTreeMap<&'static str, MetricSummary>>,
    pub(crate) consistency: Option<Consistency>,
}

pub(crate) fn protocol() -> Value {
    json!({
        "id": PROTOCOL_ID,
        "scenario": "synthetic-token prefill and fixed-length greedy decode, token IDs only",
        "batch_size": 1,
        "input_tokens": 512,
        "output_tokens": 128,
        "warmup_executions": WARMUP_EXECUTIONS,
        "measured_executions": MEASURED_EXECUTIONS,
        "input": "tools/mlx-baseline/inputs/random-512-seed0.json embedded at build time, SHA-256 checked",
        "sampler": "greedy: argmax over float16 log-probabilities (logits - logsumexp), first index on ties",
        "generation_api": "foundry_infer_llama Session::prepare then Session::generate",
        "eos_handling": "ignored: generation runs for exactly 128 tokens",
        "kv_cache": "fresh Session per execution: float16 KV cache with 640 positions per layer",
        "prefill": "one 512-row forward pass",
        "prompt_cache_reuse": false,
        "model_loading": "LlamaModel::load: manifest and header validation, then all weights read once into one resident shared Metal buffer",
        "synchronization": "an empty command buffer is committed and awaited before starting and before stopping each execution timer",
    })
}

pub(crate) fn timing_boundaries() -> Value {
    json!({
        "clock": "std::time::Instant (monotonic)",
        "start": "after the session is allocated, the prompt is written to its token buffer and the queue is synchronized",
        "token_available": "when the host has read the token ID after waiting for the command buffer that produced it",
        "stop": "after the 128th token is available and a final queue synchronization returns",
        "included": ["prefill", "decode", "token retrieval to the host", "final GPU synchronization"],
        "excluded": [
            "manifest verification",
            "model loading and kernel compilation",
            "input preparation",
            "session and KV cache allocation",
            "report writing",
        ],
    })
}

pub(crate) fn metrics() -> Value {
    json!({
        "elapsed_ns": {"unit": "ns", "definition": "Timer start to timer stop, as defined by the timing boundaries."},
        "token_available_ns": {"unit": "ns", "definition": "For each generated token, the time from the timer start until its ID is available on the host."},
        "first_token_ns": {"unit": "ns", "definition": "token_available_ns[0]."},
        "decode_tokens_per_s": {"unit": "tokens/s", "definition": "127 / (token_available_ns[127] - token_available_ns[0]) in seconds."},
        "device_allocated_bytes": {"unit": "bytes", "definition": "MTLDevice.currentAllocatedSize sampled after the execution completes."},
    })
}

pub(crate) fn memory_definitions() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        (
            "weight_tensor_bytes",
            "Sum of the safetensors byte ranges loaded into the resident weight buffer.",
        ),
        (
            "weight_buffer_allocated_bytes",
            "MTLAllocation.allocatedSize of the single resident weight buffer (tensors placed at \
             256-byte aligned offsets).",
        ),
        (
            "rope_buffer_allocated_bytes",
            "MTLAllocation.allocatedSize of the float32 RoPE frequency buffer.",
        ),
        (
            "kv_cache_allocated_bytes",
            "MTLAllocation.allocatedSize of one session's float16 KV cache (28 layers x 2 x 8 \
             heads x 640 positions x 128).",
        ),
        (
            "activation_allocated_bytes",
            "Sum of MTLAllocation.allocatedSize over one session's activation, token and logits \
             buffers.",
        ),
        (
            "device_allocated_bytes_*",
            "MTLDevice.currentAllocatedSize: all Metal allocations of this process on the device. \
             Not process memory, and not comparable with the MLX allocator's peak memory.",
        ),
    ])
}
