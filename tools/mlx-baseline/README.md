# mlx-baseline

Reproducible MLX reference measurement for `mlx-community/Llama-3.2-3B-Instruct-4bit@7f0dc925e0d0afb0322d96f9255cfddf2ba5636e`.
One scenario: 512 synthetic input tokens → 128 generated tokens, batch size 1.
Everything runs through `uv` with the locked environment in this directory.

## Commands (fish)

Tests:

```fish
uv run --project tools/mlx-baseline --locked \
  python -m unittest discover \
  -s tools/mlx-baseline/tests -p 'test_*.py'
```

Pilot run:

```fish
set snapshot "$HOME/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-4bit/snapshots/7f0dc925e0d0afb0322d96f9255cfddf2ba5636e"
set archive ".private-data/benchmarks/mlx-pilot-"(date -u +%Y%m%dT%H%M%SZ)"-$fish_pid"

uv run --project tools/mlx-baseline --locked \
  tools/mlx-baseline/bench.py \
  --snapshot "$snapshot" \
  --manifest .private-data/models/llama-3.2-3b-instruct-4bit/manifest.json \
  --output-dir "$archive"
```

Token-ID protocol (`token-ids-512x128-v1`, shared with `foundry infer-bench`), comparison, and private numerical fixtures:

```fish
uv run --project tools/mlx-baseline --locked tools/mlx-baseline/bench.py --protocol token-ids \
  --snapshot "$snapshot" --manifest .private-data/models/llama-3.2-3b-instruct-4bit/manifest.json --output-dir "$archive"
uv run --project tools/mlx-baseline --locked tools/mlx-baseline/compare.py \
  --mlx "$archive/report.json" --foundry "$foundry_report" --output-dir "$comparison"
uv run --project tools/mlx-baseline --locked tools/mlx-baseline/fixtures.py \
  --snapshot "$snapshot" --manifest .private-data/models/llama-3.2-3b-instruct-4bit/manifest.json --output-dir "$fixtures"
```

`--protocol token-ids` uses `generate_step`, which yields token IDs without detokenization and does not stop at EOS. The input comes from `inputs/random-512-seed0.json`, which must equal the generator output. The timer starts after `mx.synchronize()`. Each token's availability time (`token_available_ns`) is recorded when its ID reaches the host, and the timer stops after a final `mx.synchronize()`. The derived metrics are `first_token_ns = token_available_ns[0]` and `decode_tokens_per_s = 127 / (t[127] − t[0])`. The default protocol, `stream-generate-v1`, is unchanged. Fixtures are derived from the weights and stay outside version control.

`--output-dir` must not exist. Exit status: `0` succeeded, `1` run failed (a report is still written), `2` invalid arguments (nothing is written).

## Protocol

Before anything is loaded, the harness checks the manifest's repository and revision. It then runs `scripts/verify-model-manifest.py` with the current interpreter and aborts on any mismatch.

- Input: `random.Random(0)` produces `[128000]`, followed by 511 draws of `randrange(128000)`. It is generated once, and the IDs go directly to MLX with no chat template and no tokenization.
- 2 warmup executions, then 10 measured executions.
- Every execution uses greedy argmax (temperature 0), resets `mx.random.seed(0)`, and gets a fresh unquantized KV cache.
- There is no prompt-cache reuse, no speculative decoding, no conversion, and no dtype override.
- EOS is ignored through a per-run `TokenizerWrapper` with an empty EOS set. Tokenizer files and the loaded wrapper are left unchanged.
- Success requires exactly 128 generated tokens in every execution. Greedy outputs are compared across executions, and any mismatch is reported in `consistency`.

## Timing boundaries

The timer is `perf_counter_ns`, started after `mx.synchronize()` and stopped after the generator ends and a final `mx.synchronize()`.

- Included: `mlx_lm.stream_generate` (prefill and decode), plus the normal generator and detokenization overhead.
- Excluded: verification, model loading, input generation, metadata collection, KV cache construction, and report writing.

## Metrics

| Field | Unit | Definition |
|---|---|---|
| `elapsed_ns` | ns | Harness wall time for one execution. |
| `first_token_ns` | ns | Timer start to the first yielded token. Includes harness and generator overhead; not isolated GPU prefill time. |
| `prompt_tps` | tokens/s | mlx-lm: prompt tokens / time until the first yielded token. |
| `generation_tps` | tokens/s | mlx-lm: generated tokens (including the first) / time since the first token. |
| `peak_memory_bytes` | bytes | `mx.get_peak_memory()` after `mx.reset_peak_memory()` at the start of each execution. |
| `peak_memory_gb` | GB (10⁹ bytes) | Same counter as reported by mlx-lm. |

The memory figures are MLX allocator statistics. They include resident weights and do not measure process memory.

Statistics (median, mean, min, max, sample standard deviation) cover measured executions only and appear only on success.

## Archive

The output directory holds `report.json` (`schema_version: 1`) and copies of `pyproject.toml`, `uv.lock`, `.python-version`, and the model's `CONFIGURATION.md`.

`report.json` contains:

- Status and failure context
- The command arguments
- The manifest copy and the verifier output
- Environment: dependency versions, Python, machine, RAM, GPU, and macOS
- Source hashes
- The protocol, input IDs, timing boundaries, and metric definitions
- Every execution in chronological order, with its generated token IDs
- The consistency result and the statistics
