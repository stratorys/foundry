# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub==1.30.0",
#     "mlx==0.32.0",
#     "mlx-lm==0.31.3",
#     "numpy==2.3.5",
#     "safetensors==0.8.0",
# ]
# ///

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from safetensors.numpy import save_file

ENGINE = "mlx"
PROMPTS_PATH = Path(__file__).with_name("prompts.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Write greedy generation and last-token logits of a Hugging Face model run with MLX.")
    parser.add_argument("--model", required=True, help="Hugging Face repository id, as owner/name.")
    parser.add_argument("--prompt", action="append", default=[], help="Text prompt, tokenized with the model tokenizer. Repeatable.")
    parser.add_argument("--prompt-ids", action="append", default=[], help="Comma-separated prompt token IDs. Repeatable.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    return parser.parse_args()


def eos_token_ids(snapshot):
    eos = json.loads((snapshot / "config.json").read_text())["eos_token_id"]
    return eos if isinstance(eos, list) else [eos]


def prompts(args, tokenizer):
    texts = args.prompt or ([] if args.prompt_ids else PROMPTS_PATH.read_text().splitlines())
    from_text = [{"text": text, "prompt_ids": tokenizer.encode(text)} for text in texts]
    from_ids = [{"text": None, "prompt_ids": [int(token) for token in ids.split(",")]} for ids in args.prompt_ids]
    return from_text + from_ids


def top_logprobs(logits, top_k):
    logprobs = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32))
    ids = mx.argsort(-logprobs)[:top_k]
    return [{"id": int(token), "logprob": float(logprobs[token])} for token in ids.tolist()]


def generate(model, prompt_ids, args, eos):
    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt_ids]), cache=cache)[0, -1]
    first_logits = np.array(logits.astype(mx.float32))
    output_ids = []
    logprobs = []
    for _ in range(args.max_tokens):
        token = int(mx.argmax(logits))
        output_ids.append(token)
        logprobs.append(top_logprobs(logits, args.top_k))
        if token in eos:
            break
        logits = model(mx.array([[token]]), cache=cache)[0, -1]
    return output_ids, logprobs, first_logits


def main():
    args = parse_args()
    snapshot = Path(snapshot_download(args.model, local_files_only=True))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load(str(snapshot))
    eos = eos_token_ids(snapshot)

    records = []
    last_logits = {}
    for index, prompt in enumerate(prompts(args, tokenizer)):
        output_ids, logprobs, first_logits = generate(model, prompt["prompt_ids"], args, eos)
        records.append({**prompt, "output_ids": output_ids, "top_logprobs": logprobs})
        last_logits[f"prompt.{index}"] = first_logits

    report = {
        "model": args.model,
        "engine": ENGINE,
        "versions": {package: version(package) for package in ["mlx", "mlx-lm"]},
        "dtype": "bfloat16",
        "device": str(mx.default_device()),
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
        "eos_token_ids": eos,
        "prompts": records,
    }
    (out / "generation.json").write_text(json.dumps(report, indent=2) + "\n")
    save_file(last_logits, str(out / "last_logits.safetensors"))
    print(f"Wrote {len(records)} prompts to {out}.")


if __name__ == "__main__":
    main()
