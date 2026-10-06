import argparse
import json
import math
import sys
from pathlib import Path

import bench

DETAILED_LAYERS = (0, 27)
DECODE_PROBE_STEP = 0
FIXTURE_NAME = "fixtures.safetensors"
SUMMARY_NAME = "noise-floor.json"


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Write private MLX numerical fixtures for Foundry Llama validation."
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


class Capture:
    def __init__(self, mx, layers):
        self.mx = mx
        self.layers = layers
        self.values = {}

    def wanted(self, name):
        if not name.startswith("layers."):
            return True
        _, layer, probe = name.split(".", 2)
        return int(layer) in self.layers or probe in ("attention_residual", "output")

    def __call__(self, name, value):
        if self.wanted(name):
            self.values[name] = value.reshape(-1, value.shape[-1]) if value.ndim > 2 else value
            self.mx.eval(self.values[name])


def heads_last(value):
    batch, heads, length, dim = value.shape
    return value.transpose(0, 2, 1, 3).reshape(length, heads * dim)


def mlx_forward(mx, model, tokens, cache, capture):
    from mlx_lm.models.activations import swiglu
    from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention

    inner = model.model
    h = inner.embed_tokens(tokens)
    capture("embedding", h)
    mask = create_attention_mask(h, cache[0])
    for index, (layer, layer_cache) in enumerate(zip(inner.layers, cache)):
        attention = layer.self_attn
        x = layer.input_layernorm(h)
        capture(f"layers.{index}.attention_norm", x)
        batch, length, _ = x.shape
        q = attention.q_proj(x).reshape(batch, length, attention.n_heads, -1).transpose(0, 2, 1, 3)
        k = attention.k_proj(x).reshape(batch, length, attention.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = attention.v_proj(x).reshape(batch, length, attention.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = attention.rope(q, offset=layer_cache.offset)
        k = attention.rope(k, offset=layer_cache.offset)
        capture(f"layers.{index}.query", heads_last(q))
        capture(f"layers.{index}.key", heads_last(k))
        capture(f"layers.{index}.value", heads_last(v))
        keys, values = layer_cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(q, keys, values, cache=layer_cache, scale=attention.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        capture(f"layers.{index}.attention", out)
        h = h + attention.o_proj(out)
        capture(f"layers.{index}.attention_residual", h)
        x = layer.post_attention_layernorm(h)
        capture(f"layers.{index}.mlp_norm", x)
        gate = layer.mlp.gate_proj(x)
        up = layer.mlp.up_proj(x)
        capture(f"layers.{index}.gate", gate)
        capture(f"layers.{index}.up", up)
        activated = swiglu(gate, up)
        capture(f"layers.{index}.swiglu", activated)
        h = h + layer.mlp.down_proj(activated)
        capture(f"layers.{index}.output", h)
        mx.eval(h, keys, values)
    normed = inner.norm(h)
    capture("final_norm", normed[:, -1:, :])
    logits = inner.embed_tokens.as_linear(normed)[:, -1:, :]
    capture("logits", logits)
    return logits


class Reference:
    def __init__(self, mx, model):
        self.mx = mx
        self.model = model
        self.keys = [None] * len(model.model.layers)
        self.values = [None] * len(model.model.layers)
        self.embedding = self.dense(model.model.embed_tokens)

    def dense(self, module):
        mx = self.mx
        weight = mx.dequantize(
            module.weight,
            module.scales.astype(mx.float32),
            module.biases.astype(mx.float32),
            group_size=module.group_size,
            bits=module.bits,
        )
        mx.eval(weight)
        return weight

    def linear(self, module, x):
        return x @ self.dense(module).T

    def norm(self, module, x):
        return self.mx.fast.rms_norm(x, module.weight.astype(self.mx.float32), module.eps)

    def forward(self, tokens, offset, capture):
        mx = self.mx
        inner = self.model.model
        h = self.embedding[tokens]
        capture("embedding", h)
        length = h.shape[1]
        for index, layer in enumerate(inner.layers):
            attention = layer.self_attn
            x = self.norm(layer.input_layernorm, h)
            capture(f"layers.{index}.attention_norm", x)
            q = self.linear(attention.q_proj, x).reshape(1, length, attention.n_heads, -1).transpose(0, 2, 1, 3)
            k = self.linear(attention.k_proj, x).reshape(1, length, attention.n_kv_heads, -1).transpose(0, 2, 1, 3)
            v = self.linear(attention.v_proj, x).reshape(1, length, attention.n_kv_heads, -1).transpose(0, 2, 1, 3)
            q = attention.rope(q, offset=offset)
            k = attention.rope(k, offset=offset)
            capture(f"layers.{index}.query", heads_last(q))
            capture(f"layers.{index}.key", heads_last(k))
            capture(f"layers.{index}.value", heads_last(v))
            keys = k if self.keys[index] is None else mx.concatenate([self.keys[index], k], axis=2)
            values = v if self.values[index] is None else mx.concatenate([self.values[index], v], axis=2)
            self.keys[index], self.values[index] = keys, values
            mask = "causal" if length > 1 else None
            out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=attention.scale, mask=mask)
            out = out.transpose(0, 2, 1, 3).reshape(1, length, -1)
            capture(f"layers.{index}.attention", out)
            h = h + self.linear(attention.o_proj, out)
            capture(f"layers.{index}.attention_residual", h)
            x = self.norm(layer.post_attention_layernorm, h)
            capture(f"layers.{index}.mlp_norm", x)
            gate = self.linear(layer.mlp.gate_proj, x)
            up = self.linear(layer.mlp.up_proj, x)
            capture(f"layers.{index}.gate", gate)
            capture(f"layers.{index}.up", up)
            activated = gate * mx.sigmoid(gate) * up
            capture(f"layers.{index}.swiglu", activated)
            h = h + self.linear(layer.mlp.down_proj, activated)
            capture(f"layers.{index}.output", h)
            mx.eval(h, keys, values)
        normed = self.norm(inner.norm, h[:, -1:, :])
        capture("final_norm", normed)
        logits = normed @ self.embedding.T
        capture("logits", logits)
        return logits


def relative_l2(np, actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), sys.float_info.min))


def main(argv=None):
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    output_dir = Path(arguments.output_dir)
    if output_dir.exists():
        print(f"error: {output_dir} already exists", file=sys.stderr)
        return 2
    bench.load_manifest(arguments.manifest)
    verification = bench.run_verifier(arguments.manifest, arguments.snapshot)
    if verification["returncode"] != 0:
        print(verification["stdout"], verification["stderr"], file=sys.stderr)
        return 1

    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    input_ids = bench.load_input_ids()
    model, _ = load(arguments.snapshot)
    prompt = mx.array(input_ids, dtype=mx.uint32)
    continuation = [
        token
        for (token, _), _ in zip(
            generate_step(
                prompt,
                model,
                max_tokens=bench.OUTPUT_TOKENS,
                sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
                prompt_cache=make_prompt_cache(model),
                prefill_step_size=bench.PREFILL_STEP_SIZE,
            ),
            range(bench.OUTPUT_TOKENS),
        )
    ]

    tensors = {
        "tokens.input": mx.array(input_ids, dtype=mx.uint32),
        "tokens.continuation": mx.array(continuation, dtype=mx.uint32),
    }
    layers = set(DETAILED_LAYERS)
    mlx_cache = make_prompt_cache(model)
    reference = Reference(mx, model)
    mlx_logits = []
    reference_logits = []
    model_logits = model(prompt[None], cache=make_prompt_cache(model))[:, -1:, :]
    steps = [input_ids] + [[token] for token in continuation[:-1]]
    offset = 0
    for step, tokens in enumerate(steps):
        phase = "prefill" if step == 0 else f"decode{step - 1}"
        probes = step == 0 or step - 1 == DECODE_PROBE_STEP
        mlx_capture = Capture(mx, layers)
        reference_capture = Capture(mx, layers)
        batch = mx.array(tokens, dtype=mx.uint32)[None]
        logits = mlx_forward(mx, model, batch, mlx_cache, mlx_capture)
        exact = reference.forward(batch, offset, reference_capture)
        if step == 0 and not mx.array_equal(logits, model_logits).item():
            print("error: the instrumented MLX forward differs from the model", file=sys.stderr)
            return 1
        mlx_logits.append(logits.reshape(-1))
        reference_logits.append(exact.reshape(-1))
        if probes:
            for name, value in mlx_capture.values.items():
                if name != "logits":
                    tensors[f"mlx.{phase}.{name}"] = value
            for name, value in reference_capture.values.items():
                if name != "logits":
                    tensors[f"reference.{phase}.{name}"] = value
        offset += len(tokens)
        mx.eval(mlx_logits[-1], reference_logits[-1])
    tensors["mlx.logits"] = mx.stack(mlx_logits)
    tensors["reference.logits"] = mx.stack(reference_logits)
    mx.eval(tensors)

    mlx_tokens = [int(token) for token in np.array(mx.argmax(tensors["mlx.logits"], axis=-1))]
    argmax_mismatches = [position for position, (a, b) in enumerate(zip(mlx_tokens, continuation)) if a != b]
    floors = {}
    for key in sorted(tensors):
        if key.startswith("mlx.") and key != "mlx.logits":
            name = key[len("mlx."):]
            floors[name] = relative_l2(np, np.array(tensors[key].astype(mx.float32)), np.array(tensors[f"reference.{name}"]))
    position_floors = [
        relative_l2(np, np.array(m.astype(mx.float32)), np.array(r))
        for m, r in zip(tensors["mlx.logits"], tensors["reference.logits"])
    ]
    floors["logits"] = max(position_floors)
    summary = {
        "repository": bench.REPOSITORY,
        "revision": bench.REVISION,
        "input_sha256": bench.input_sha256(input_ids),
        "continuation": continuation,
        "teacher_forced_mlx_argmax_mismatches": argmax_mismatches,
        "relative_l2_mlx_vs_f32_reference": floors,
        "logits_relative_l2_per_position": position_floors,
        "logits_max_abs_error_per_position": [
            float(np.max(np.abs(np.array(m.astype(mx.float32)) - np.array(r))))
            for m, r in zip(tensors["mlx.logits"], tensors["reference.logits"])
        ],
    }
    output_dir.mkdir(parents=True)
    mx.save_safetensors(str(output_dir / FIXTURE_NAME), tensors, metadata={"format": "foundry-fixtures"})
    (output_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    worst = max(floors.items(), key=lambda item: item[1])
    print(f"wrote {len(tensors)} tensors to {output_dir}; largest MLX noise floor {worst[0]} = {worst[1]:.3e}")
    if argmax_mismatches:
        print(f"note: teacher-forced MLX logits argmax differs from MLX greedy at {argmax_mismatches}")
    if not math.isfinite(worst[1]):
        print("error: MLX fixtures are not self-consistent", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
