# Reference fixtures and benchmarks

Generate reference outputs with transformers and MLX, check Foundry against them, then measure
Foundry's speed and memory use. All three engines run on GPU with the same bf16 model weights.

## Requirements

- macOS on Apple Silicon.
- Rust nightly and [uv](https://docs.astral.sh/uv/).
- `meta-llama/Llama-3.2-3B-Instruct` already in the Hugging Face cache. These commands do not
  download model weights. Each Python script pins its dependencies.

Run all commands from the repository root, in the same terminal, one at a time.
Continue only when the previous command succeeds.

## 1. Generate the references

The scripts use the eight prompts in `prompts.txt`, generate up to 32 tokens per prompt, and
record the top 5 token probabilities at each step. Their output files are called fixtures.
Transformers selects CUDA when available, otherwise MPS. MLX requires a Metal GPU. Both scripts
fail if their GPU backend is unavailable; inference never falls back to the CPU. Host transfers
are used to read results and write fixture files.

```sh
export HF_HUB_OFFLINE=1

uv run --script tools/reference/transformers_fixtures.py --model meta-llama/Llama-3.2-3B-Instruct --max-tokens 32 --top-k 5 --out /tmp/llama-3.2-3B-transformers

uv run --script tools/reference/mlx_fixtures.py --model meta-llama/Llama-3.2-3B-Instruct --max-tokens 32 --top-k 5 --out /tmp/llama-3.2-3B-mlx
```

Fixtures stay in `/tmp`. Generate them again if they are removed or the cached model changes.

## 2. Build and choose a results directory

```sh
cargo +nightly build --release --locked

export TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export OUT="benchmarks/${TIMESTAMP}-llama-3.2-3b-metal"
mkdir -p benchmarks
mkdir "$OUT"
echo "$OUT"
```

The UTC timestamp is generated once for both reports. For example, the directory may be
`benchmarks/20261017T094813Z-llama-3.2-3b-metal/`. Stop if `mkdir` reports that it already exists.
For another run, export a new timestamp and output directory.

## 3. Evaluate, then benchmark

Keep the same `OUT` value for these two commands. Run the benchmark only if the evaluation
succeeds. Keep the machine on power and avoid other heavy workloads while measuring.

```sh
./target/release/foundry eval --model meta-llama/Llama-3.2-3B-Instruct --reference /tmp/llama-3.2-3B-transformers --reference /tmp/llama-3.2-3B-mlx --out "$OUT"

./target/release/foundry bench --model meta-llama/Llama-3.2-3B-Instruct --prompt-tokens 512 --new-tokens 128 --timing-scope generation --depth 0 --batch-size 1 --seed 0 --warmup-runs 1 --warmup-ms 0 --runs 10 --out "$OUT"
```

The benchmark uses random token IDs with a fixed seed: 512 prompt tokens, exactly 128 new tokens (127 decode steps),
batch size 1 and no initial cache depth. It ignores EOS, runs one warmup, then measures ten runs.

Results are written to `$OUT`:

| File | Contents |
|---|---|
| `eval.json` | Module and generation comparisons against transformers and MLX. |
| `report.json` | Configuration, token IDs, raw timings, common metrics, peak Metal memory and source commit. |

## Common metrics

All measurements in `report.json` come from Foundry. One `metrics` object contains the
statistics and throughputs, alongside configuration, input IDs, generated IDs and ordered raw
timings. There are no per-engine groups or compatibility formats.

Each latency/throughput summary contains `count`, `mean`, sample `stdev` and `variance`, `min`, `max` and interpolated
`p10`, `p25`, `p50`, `p75`, `p90`, `p99`. Throughputs are summarized across per-run rates, rather
than computed as the inverse of the mean latency. A metric without applicable samples is
`null`; a zero duration never produces an infinite rate. Ten runs give an initial dispersion,
not a reliable tail-latency estimate.

For each run let **P** be prompt tokens across the batch, **D** be the number of decode steps
times batch size, **O** be the actual number of output IDs across the batch, **Tp** be the prompt
step duration, **Td** be the sum of all decode step durations, **T = Tp + Td**, and **G** be
full generation wall time. Rates below use seconds; stored latency values use nanoseconds.

| Field under `metrics` | Definition |
|---|---|
| `prompt_step_ns` | Prompt forward or host step, according to `config.timing_scope`. |
| `decode_step_ns` | Secondary distribution of all decode step durations, mixing contexts. |
| `decode_by_position` | Step and ITL summaries across repetitions at each identical position and context. |
| `inter_token_ns` | Distribution of host availability intervals after the first token; generation scope only. |
| `tpot_ns` | Distribution of per-run `(generation_ns - first_token_ns) / (N - 1)`; generation scope and N > 1 only. |
| `step_sum_ns` | Per-run sum of the prompt and every decode step. |
| `prompt_tokens_per_s` | P / Tp; unavailable without a prompt. |
| `decode_tokens_per_s` | D / Td; unavailable without decode steps. |
| `first_token_ns` | Decoder creation through the first token available on the host; generation scope only. |
| `generation_ns` | Decoder creation through the final host token, excluding teardown; generation scope only. |
| `output_tokens_per_s` | O / G; unavailable in forward scope or without output tokens. |
| `total_tokens_per_s` | (P + O) / G, counting input and output tokens; generation scope only. |
| `processed_tokens_per_s` | (P + D) / T, counting input tokens passed through timed forwards. |
| `decode_dispatch_count` | Dispatch count per decode step; profiling only. |
| `decode_allocation_count` | Buffer allocation count per decode step; profiling only. |
| `decode_allocated_bytes` | Sum of requested buffer bytes per decode step; profiling only. |
| `decode_gpu_busy_ns` | GPU command-buffer interval union per decode step, in nanoseconds; profiling only. |
| `decode_gpu_idle_ns` | Per-step `step_ns - gpu_busy_ns`; profiling only. |
| `decode_kernel_gpu_ns` | Map of kernel keys to per-step GPU time summaries, in nanoseconds; `kernels` mode only. |

Choose `--timing-scope generation` (default) or `--timing-scope forward`. Run them in separate
invocations and output directories so synchronization used to isolate forwards does not
perturb the generation pass. Both retain host wall times, not GPU-only times.

- **Generation:** step timers include CPU input preparation, allocations, encoding, GPU work,
  greedy selection and token readback. `first_token_ns` and `generation_ns` additionally
  include decoder/KV initialization, any initial cache depth and orchestration. Input drawing,
  result-array preparation, model loading, report writing and decoder teardown are excluded.
  Pending device work is synchronized before the decoder-creation clock starts.
- **Forward:** each timer starts before input tensor preparation and ends after logits are
  ready, with explicit synchronization. Cache/depth setup is completed before the first timer;
  selection and readback happen outside the timer to supply the next input. Host first-token
  and full-generation metrics are `null`, as are output/total wall-time throughputs.

`--new-tokens N` means exactly N output tokens **per sequence**, including the first token
from prefill. With a prompt, N = 1 executes no decode steps; N = 128 executes 127. N = 0
processes only the prompt through ready logits, without selection or generated IDs. Without a
prompt, a random start input is supplied and N decode steps produce N outputs. Prompt and
output counts cannot both be zero. `--gen-tokens` is removed; there is no compatibility alias.
The KV capacity is depth + prompt + actual decode steps.

All decode steps are retained. Depth is excluded from step-throughput numerators and step
sums, but included in full-generation latencies. Batch size multiplies token counts, not
elapsed durations. Nanoseconds are storage units, not accuracy guarantees.

### Memory scope

- `runs[].peak_memory_bytes`: peak observed for that run, reset after pending device work is
  synchronized and before decoder/KV creation. Includes resident weights, KV cache and
  temporaries, regardless of timing scope. Loading and previous warmup/run peaks are excluded;
  allocations still resident at the start remain included.
- `peak_memory_bytes`: maximum of the measured run peaks, excluding warmup runs.
- `session_peak_memory_bytes`: cumulative peak since device creation, including model loading,
  warmup and every measured run.
- `peak_memory_source`: `metal_current_allocated_size` for both peak scopes. Memory fields are
  omitted when tracking is unavailable.

Metal samples `currentAllocatedSize` after buffer allocations and at run reset. Both counters
subtract the same baseline taken at device creation; reset does not subtract resident weights.
These are observed Metal allocation peaks, not process resident memory, separate measurements
of weights/KV/temporaries, or an exact continuous hardware peak. No extra synchronization is
inserted inside timed steps to measure memory.

### Device profiling

Use `--profile steps` to record dispatches, allocations and GPU time while keeping the normal
command-buffer batching and in-flight limit. Recording counters adds CPU overhead. Use
`--profile kernels` to also attribute GPU time to individual kernel keys: it runs one dispatch
per command buffer and waits for completion before submitting the next. Its wall times and
throughputs are perturbed and are unsuitable for normal performance comparisons.

For a small diagnostic run, use separate output directories under the `OUT` selected above:

```sh
./target/release/foundry bench --model meta-llama/Llama-3.2-3B-Instruct --prompt-tokens 4 --new-tokens 3 --depth 7 --timing-scope generation --profile steps --warmup-runs 0 --runs 1 --out "$OUT/profile-steps"

./target/release/foundry bench --model meta-llama/Llama-3.2-3B-Instruct --prompt-tokens 4 --new-tokens 3 --depth 7 --timing-scope generation --profile kernels --warmup-runs 0 --runs 1 --out "$OUT/profile-kernels"
```

Use `--timing-scope forward` and new output directories to inspect forwards alone. Profiles
follow the same boundary as each step timer: generation includes greedy argmax and token
readback; forward excludes both and discards their counters before the next step. Model
loading, cache initialization and initial depth processing are excluded from step profiles.
With `--new-tokens 0`, the prompt profile covers processing through ready logits, without
selection.

The selected mode is stored in `config.profile`. Raw profiles appear in
`runs[].prompt_profile` and `runs[].decode[].profile`:

| Profile field | Meaning |
|---|---|
| `kernels[key].dispatch_count` | Number of dispatches for this key during the step. |
| `kernels[key].gpu_ns` | Sum of GPU command-buffer durations for this key, in nanoseconds; present only in `kernels` mode. |
| `allocation_count` | Number of buffer allocations during the step. |
| `allocated_bytes` | Sum of requested buffer sizes, in bytes, including buffers freed during the step; distinct from peak or resident memory. |
| `gpu_busy_ns` | Duration of the union of completed command-buffer GPU intervals, in nanoseconds; overlapping intervals count once. |

Kernel keys name operations such as `binary_add` or `reduce_argmax`. Matmul keys include
batch product and matrix dimensions, for example `matmul b1 m1 k3072 n8192`. Operand copies
are counted and timed separately.

The `metrics.decode_*` profiling summaries pool decode steps from measured runs, excluding
the prompt and warmups. Each `decode_kernel_gpu_ns` entry summarizes the total time for its
key in each decode step that ran it; missing keys contribute no sample. `decode_gpu_idle_ns`
summarizes the time outside the recorded command-buffer intervals. It is not a hardware
utilization counter or a separate measurement of CPU encoding, allocation or waiting time.
Command-buffer durations include all time between Metal's `GPUStartTime` and `GPUEndTime`.

Without profiling, raw profile fields are omitted, scalar profiling summaries are `null`,
and `decode_kernel_gpu_ns` is an empty map. In `steps` mode that map is also empty.

### Decode samples and context

Each run records a `decode` array. Every sample contains `position` (zero-based decode index),
`context_tokens` (cached tokens per sequence **before** the one-token input), `step_ns` and
`inter_token_ns` (ITL). With depth C and prompt length P, the first decode has context C + P;
without a prompt it has context C. Following contexts increment by one. Each step attends over
its cached context plus its new input token. Prompt processing is recorded separately in
`prompt_step_ns`.

ITL is the difference between consecutive **host token completion events**, reusing the
existing clock readings. It includes orchestration between steps, unlike `step_ns`. Without a
prompt, the first decode produces the first output and has no predecessor: its ITL is `null`.
With a prompt, the first decode ITL starts at the prompt output's completion. In forward scope
all ITLs and TPOT are `null`, because host generation is outside that measurement boundary.
With one output token, there are no intervals and TPOT is `null`.

For example, outputs available at 100, 250 and 500 ms have ITLs of 150 and 250 ms and TPOT of
200 ms. TPOT is per sequence and is not divided by batch size. A synchronous batch has one
completion event per step; latency sample counts are not multiplied by its number of sequences.

`metrics.decode_by_position` groups samples by **both** position and context. Each entry has
`position`, `context_tokens`, a `step_ns` summary and an optional `inter_token_ns` summary. For
ten repetitions, each position normally has ten samples, not ten times the decode length.
Prefer these summaries for context-dependent comparisons. Global decode/ITL percentiles mix
contexts and do not make successive steps statistically independent. `metrics.tpot_ns` has
one sample per measured run with at least two outputs, rather than one sample per interval.

### Comparing engines

Use the same model snapshot (including any conversion), token IDs, batch, cache depth/capacity,
weight/KV types, generation count, EOS policy and warmup. Record engine versions and full
commands. Matching metric names alone do not establish equal measurement boundaries.

| Engine / reference | Common quantities to compare | Boundary to check |
|---|---|---|
| [ZML](https://github.com/zml/zml/tree/master/examples) | Loading/execution latency and generation throughput from the chosen example. | Compilation, weight loading and execution are separate phases; identify what the selected example times. |
| [llama.cpp `llama-bench`](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) | Prompt/decode or combined processed-token throughput, run latencies, dispersion. | `llama-bench` excludes sampling; Foundry host steps include it. Use forward scope for comparison; verify the remaining input-preparation and synchronization boundaries. |
| [MLX-LM benchmark](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py) | Prompt throughput, generation throughput, peak memory. | Check the first-token numerator, final forward, synchronization and memory allocator scope in the pinned version. |
| [Candle Llama example](https://github.com/huggingface/candle/blob/main/candle-examples/examples/llama/main.rs) | Generation throughput and output counts. | Check whether prefill/first output and decoding or printing are included. |
| [vLLM benchmarks](https://github.com/vllm-project/vllm/tree/main/vllm/benchmarks) | Local latency and input/output throughput; first-token latency, TPOT and ITL. | Use a local benchmark for local inference comparisons. Client/server measurements include additional work. |
| [SGLang benchmarks](https://docs.sglang.ai/developer_guide/bench_serving) | Latency and input/output throughput; first-token latency, TPOT and ITL. | Separate engine execution from streaming-client latency and concurrent serving throughput. |

GPU command-buffer times are available through device profiling above. Loading/compilation
time, process RSS and serving metrics are not currently instrumented in this report. For external
metrics with a different numerator or interval, use the retained raw samples and explicit
formula; do not label a Foundry calculation as an actual run of that engine.

## Record the environment

For later comparisons, also record the machine, toolchain and fixture checksums:

```sh
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/environment.txt"
git rev-parse HEAD >> "$OUT/environment.txt"
rustc +nightly --version >> "$OUT/environment.txt"
sw_vers >> "$OUT/environment.txt"
sysctl hw.model hw.memsize machdep.cpu.brand_string >> "$OUT/environment.txt"
shasum -a 256 /tmp/llama-3.2-3B-transformers/* /tmp/llama-3.2-3B-mlx/* > "$OUT/reference-sha256.txt"
```

Use a clean source tree and the same machine, build settings, model snapshot and reference engine
versions when comparing runs. The recorded Llama benchmark uses snapshot
`0cb88a4f764b7a12671c53f0838cd831a0843b95`. Timings will still vary between runs.

## Reference options and files

| Script option | Default | Purpose |
|---|---|---|
| `--model ID` | required | Cached Hugging Face model. Use the same model in Foundry. |
| `--out DIR` | required | Fixture output directory. |
| `--prompt TEXT` | eight prompts from `prompts.txt` | Custom text prompt; repeatable. |
| `--prompt-ids 1,2,3` | none | Custom token IDs; repeatable. |
| `--max-tokens N` | 32 | Maximum generated tokens per prompt. |
| `--top-k K` | 5 | Number of token log probabilities recorded per step. |
| `--device DEVICE` | `auto` | GPU backend: `auto`, `cuda`, or `mps`; transformers only. |

Custom prompts replace the default prompts. For a smaller model, use
`HuggingFaceTB/SmolLM2-135M-Instruct` with separate fixture directories under `/tmp`.

| Fixture | Contents |
|---|---|
| `generation.json` | Prompt IDs, generated IDs, top-k log probabilities and engine versions. |
| `last_logits.safetensors` | Last prompt token's logits, used for first-step metrics. |
| `activations.safetensors` | Embedding and layer-0 module inputs and outputs; transformers only. |

Evaluation requires activations from at least one reference. An activations file must hold at
least one module, and each module must match a Foundry submodule. A checked module passes when
at least 99.9% of its elements are within tolerance, with no NaN or Inf. Generation passes when
token IDs match or, at the first difference, each token is in the other engine's top-k.
First-step metrics and the transformers/MLX comparison are informational. Any module or
generation failure makes `foundry eval` exit with an error.
