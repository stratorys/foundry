# Reference fixtures

These scripts run a Hugging Face model with transformers and with MLX. Their outputs are the
reference that Foundry's model tests compare against. The tests check that Foundry computes the
same thing as both engines on the same bf16 weights.

## Requirements

- [uv](https://docs.astral.sh/uv/). Each script pins its own dependencies.
- The model must already be in the Hugging Face cache. The scripts never download weights.

## Commands

Run from the repository root:

```sh
HF_HUB_OFFLINE=1 uv run --script tools/reference/transformers_fixtures.py --model HuggingFaceTB/SmolLM2-135M-Instruct
HF_HUB_OFFLINE=1 uv run --script tools/reference/mlx_fixtures.py --model HuggingFaceTB/SmolLM2-135M-Instruct
HF_HUB_OFFLINE=1 uv run --script tools/reference/transformers_fixtures.py --model meta-llama/Llama-3.2-3B-Instruct
HF_HUB_OFFLINE=1 uv run --script tools/reference/mlx_fixtures.py --model meta-llama/Llama-3.2-3B-Instruct
```

By default, the scripts use the eight prompts in `prompts.txt`, taken from vLLM's
`tests/prompts/example.txt`. Options:

| Option | Default | Purpose |
|---|---|---|
| `--prompt TEXT` | the prompts in `prompts.txt` | Text prompt, tokenized with the model tokenizer. Repeatable. |
| `--prompt-ids 1,2,3` | none | Raw prompt token IDs, such as the random prompt in a `foundry bench` report. Repeatable. |
| `--max-tokens N` | 32 | Maximum number of greedy tokens per prompt, as in vLLM's model tests. |
| `--top-k K` | 5 | Number of top logprobs recorded at each step, as in vLLM's model tests. |
| `--device DEVICE` | `cpu` | Torch device. transformers only. |
| `--out DIR` | `.private-data/fixtures/<owner>--<name>/<engine>/` | Output directory. |

Passing `--prompt` or `--prompt-ids` replaces the default prompts. To keep them, pass them again
with `--prompt`.

## Outputs

Fixtures are never committed.

| File | Content | Used to check |
|---|---|---|
| `generation.json` | For each prompt: the prompt IDs, the greedy output IDs (generation stops after an EOS ID from `config.json`), and the top-k logprobs at each step. It also records the engine versions, dtype and device. | Generation, using vLLM's rule: equal IDs, or each token is in the other engine's top-k at the step where they diverge. |
| `last_logits.safetensors` | f32 logits of the last prompt token, one tensor per prompt (`prompt.<index>`). | Logits, using llama.cpp's rule: top 10, max absolute difference, NMSE < 1e-2. |
| `activations.safetensors` | transformers only. Input and output of the embedding and of each layer 0 submodule, for the first prompt. | Each layer 0 module, as ZML checks it. |
