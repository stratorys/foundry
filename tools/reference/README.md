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

./target/release/foundry bench --model meta-llama/Llama-3.2-3B-Instruct --prompt-tokens 512 --gen-tokens 128 --depth 0 --batch-size 1 --seed 0 --warmup-runs 1 --warmup-ms 0 --runs 10 --out "$OUT"
```

The benchmark uses random token IDs with a fixed seed: 512 prompt tokens, 128 decode steps,
batch size 1 and no initial cache depth. It ignores EOS, runs one warmup, then measures ten runs.

Results are written to `$OUT`:

| File | Contents |
|---|---|
| `eval.json` | Module and generation comparisons against transformers and MLX. |
| `report.json` | Configuration, token IDs, raw timings, statistics, peak Metal memory and source commit. |

All speed measurements are from Foundry. Groups named `mlx`, `llama_cpp`, or other engines
present those measurements using each engine's metric conventions.

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

Evaluation requires activations from at least one reference. A checked module passes when at
least 99.9% of its elements are within tolerance, with no NaN or Inf. Generation passes when
token IDs match or, at the first difference, each token is in the other engine's top-k.
First-step metrics and the transformers/MLX comparison are informational. Any module or
generation failure makes `foundry eval` exit with an error.
