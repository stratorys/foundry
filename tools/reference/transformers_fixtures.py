# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub==1.30.0",
#     "numpy==2.3.5",
#     "safetensors==0.8.0",
#     "torch==2.13.0",
#     "transformers==5.12.1",
# ]
# ///

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path

# Unsupported MPS operations must fail rather than run on the CPU.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ENGINE = "transformers"
PROMPTS_PATH = Path(__file__).with_name("prompts.txt")
LAYER_0_MODULES = [
    "model.embed_tokens",
    "model.layers.0.input_layernorm",
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.post_attention_layernorm",
    "model.layers.0.mlp",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Write greedy generation, last-token logits and layer 0 activations of a Hugging Face model.")
    parser.add_argument("--model", required=True, help="Hugging Face repository id, as owner/name.")
    parser.add_argument("--prompt", action="append", default=[], help="Text prompt, tokenized with the model tokenizer. Repeatable.")
    parser.add_argument("--prompt-ids", action="append", default=[], help="Comma-separated prompt token IDs. Repeatable.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cuda", "mps"], default="auto", help="GPU backend; auto selects CUDA, then MPS. No CPU fallback.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    return parser.parse_args()


def select_device(requested):
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    if requested in ("auto", "mps") and torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError(f"No available GPU for --device {requested}; a CUDA or MPS GPU is required.")


def eos_token_ids(snapshot):
    eos = json.loads((snapshot / "config.json").read_text())["eos_token_id"]
    return eos if isinstance(eos, list) else [eos]


def prompts(args, tokenizer):
    texts = args.prompt or ([] if args.prompt_ids else PROMPTS_PATH.read_text().splitlines())
    from_text = [{"text": text, "prompt_ids": tokenizer(text)["input_ids"]} for text in texts]
    from_ids = [{"text": None, "prompt_ids": [int(token) for token in ids.split(",")]} for ids in args.prompt_ids]
    return from_text + from_ids


def top_logprobs(logits, top_k):
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    values, ids = torch.topk(logprobs, top_k)
    return [{"id": int(token), "logprob": float(value)} for token, value in zip(ids, values)]


def generate(model, prompt_ids, args, eos):
    input_ids = torch.tensor([prompt_ids], device=args.device)
    output = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        do_sample=False,
        max_new_tokens=args.max_tokens,
        eos_token_id=eos,
        pad_token_id=eos[0],
        output_logits=True,
        return_dict_in_generate=True,
    )
    output_ids = output.sequences[0, len(prompt_ids):].tolist()
    steps = [logits[0] for logits in output.logits][: len(output_ids)]
    return output_ids, [top_logprobs(logits, args.top_k) for logits in steps], steps[0].float().cpu()


def layer_0_activations(model, prompt_ids, device):
    modules = dict(model.named_modules())
    activations = {}

    def record(name):
        def hook(_module, inputs, output):
            input = inputs[0].detach().cpu()
            activations[f"{name}.input"] = (input.to(torch.uint32) if input.dtype == torch.int64 else input).clone()
            activations[f"{name}.output"] = output.detach().cpu().clone()

        return hook

    handles = [modules[name].register_forward_hook(record(name)) for name in LAYER_0_MODULES]
    with torch.no_grad():
        model(torch.tensor([prompt_ids], device=device))
    for handle in handles:
        handle.remove()
    return activations


def main():
    args = parse_args()
    args.device = select_device(args.device)
    print(f"Selected GPU: {args.device}.", flush=True)
    snapshot = Path(snapshot_download(args.model, local_files_only=True))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16).to(args.device).eval()
    eos = eos_token_ids(snapshot)

    records = []
    last_logits = {}
    for index, prompt in enumerate(prompts(args, tokenizer)):
        output_ids, logprobs, first_logits = generate(model, prompt["prompt_ids"], args, eos)
        records.append({**prompt, "output_ids": output_ids, "top_logprobs": logprobs})
        last_logits[f"prompt.{index}"] = first_logits.contiguous()

    report = {
        "model": args.model,
        "engine": ENGINE,
        "versions": {package: version(package) for package in ["torch", "transformers"]},
        "dtype": "bfloat16",
        "device": args.device,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
        "eos_token_ids": eos,
        "prompts": records,
    }
    (out / "generation.json").write_text(json.dumps(report, indent=2) + "\n")
    save_file(last_logits, out / "last_logits.safetensors")
    save_file(layer_0_activations(model, records[0]["prompt_ids"], args.device), out / "activations.safetensors")
    print(f"Wrote {len(records)} prompts to {out}.")


if __name__ == "__main__":
    main()
