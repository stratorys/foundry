import argparse
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCHEMA_VERSION = 1
REPOSITORY = "mlx-community/Llama-3.2-3B-Instruct-4bit"
REVISION = "7f0dc925e0d0afb0322d96f9255cfddf2ba5636e"
BOS_ID = 128000
DRAWN_ID_BOUND = 128000
BATCH_SIZE = 1
INPUT_TOKENS = 512
OUTPUT_TOKENS = 128
WARMUP_EXECUTIONS = 2
MEASURED_EXECUTIONS = 10
SEED = 0
PREFILL_STEP_SIZE = 2048
SUBPROCESS_TIMEOUT_S = 30
VERIFIER_TIMEOUT_S = 600

TOOL_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOL_DIR.parent.parent
HARNESS_PATH = Path(__file__).resolve()
VERIFIER_PATH = REPO_DIR / "scripts" / "verify-model-manifest.py"
ARCHIVED_TOOL_FILES = ("pyproject.toml", "uv.lock", ".python-version")
CONFIGURATION_NAME = "CONFIGURATION.md"
REPORT_NAME = "report.json"

PROTOCOL = {
    "scenario": "synthetic-token prefill and fixed-length greedy decode",
    "batch_size": BATCH_SIZE,
    "input_tokens": INPUT_TOKENS,
    "output_tokens": OUTPUT_TOKENS,
    "warmup_executions": WARMUP_EXECUTIONS,
    "measured_executions": MEASURED_EXECUTIONS,
    "input_generator": {
        "algorithm": "Python random.Random (Mersenne Twister)",
        "seed": SEED,
        "layout": f"[{BOS_ID}] followed by {INPUT_TOKENS - 1} draws of randrange({DRAWN_ID_BOUND})",
        "generated": "once per harness run; every execution reuses the same array",
    },
    "mlx_seed": {"value": SEED, "reset": "mx.random.seed before each execution"},
    "sampler": "explicit greedy: mx.argmax(logprobs, axis=-1)",
    "temperature": 0.0,
    "top_p": None,
    "top_k": None,
    "min_p": None,
    "logits_processors": None,
    "generation_api": "mlx_lm.generate.stream_generate with an mx.array prompt",
    "prompt_encoding": "token IDs passed directly; no chat template, no tokenization",
    "kv_cache": "fresh mlx_lm.models.cache.make_prompt_cache(model) per execution (unquantized KVCache per layer)",
    "kv_bits": None,
    "max_kv_size": None,
    "prefill_step_size": PREFILL_STEP_SIZE,
    "prompt_cache_reuse": False,
    "draft_model": None,
    "speculative_decoding": False,
    "eos_handling": "ignored: a per-run TokenizerWrapper with eos_token_ids=[] wraps the loaded tokenizer; the loaded wrapper and tokenizer files are not modified",
    "model_loading": "mlx_lm.load(snapshot) with default arguments",
    "weight_conversion": None,
    "activation_dtype_override": None,
    "synchronization": "mx.synchronize() before starting and before stopping each execution timer; lazy evaluation inside generation is unchanged",
}

TIMING_BOUNDARIES = {
    "clock": "time.perf_counter_ns",
    "included": [
        "mlx_lm stream_generate generation (prefill and decode)",
        "normal generator and streaming detokenization overhead",
        "final mx.synchronize()",
    ],
    "excluded": [
        "manifest verification",
        "model loading",
        "input generation",
        "metadata collection",
        "KV cache construction, seeding, peak-memory reset and the initial mx.synchronize()",
        "report writing and archiving",
    ],
}

METRICS = {
    "elapsed_ns": {
        "unit": "ns",
        "source": "harness",
        "definition": "Wall time from the timer start (after mx.synchronize) to the timer stop (after the generator is exhausted and mx.synchronize returns).",
    },
    "first_token_ns": {
        "unit": "ns",
        "source": "harness",
        "definition": "Wall time from the timer start to the first token yielded by stream_generate. Includes harness and generator overhead and the first sample; it is not isolated GPU prefill time.",
    },
    "prompt_tps": {
        "unit": "tokens/s",
        "source": "mlx_lm stream_generate GenerationResponse.prompt_tps",
        "definition": "Prompt token count divided by the time from generator start to the first yielded token (time.perf_counter inside stream_generate).",
    },
    "generation_tps": {
        "unit": "tokens/s",
        "source": "mlx_lm stream_generate GenerationResponse.generation_tps (final response)",
        "definition": "Generated token count (including the first token) divided by the time elapsed since the first token was yielded, as computed by mlx_lm.",
    },
    "peak_memory_bytes": {
        "unit": "bytes",
        "source": "mx.get_peak_memory() after the final synchronize",
        "definition": "Peak MLX allocator active memory since mx.reset_peak_memory() at the start of the execution. Includes resident model weights; MLX allocation statistics, not process memory.",
    },
    "peak_memory_gb": {
        "unit": "GB (10^9 bytes)",
        "source": "mlx_lm stream_generate GenerationResponse.peak_memory (final response)",
        "definition": "mx.get_peak_memory() / 1e9 as reported by mlx_lm, subject to the same per-execution reset. MLX allocation statistics, not process memory.",
    },
}

SUMMARIZED_METRICS = ("elapsed_ns", "first_token_ns", "prompt_tps", "generation_tps", "peak_memory_bytes", "peak_memory_gb")
INTEGER_METRICS = ("elapsed_ns", "first_token_ns", "peak_memory_bytes")


class HarnessError(Exception):
    pass


def utc_now():
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def sha256_file(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def make_input_ids():
    generator = random.Random(SEED)
    return [BOS_ID] + [generator.randrange(DRAWN_ID_BOUND) for _ in range(INPUT_TOKENS - 1)]


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description=f"MLX baseline: {INPUT_TOKENS} input tokens, {OUTPUT_TOKENS} generated tokens, batch {BATCH_SIZE}."
    )
    parser.add_argument("--snapshot", required=True, help="pinned Hugging Face snapshot directory")
    parser.add_argument("--manifest", required=True, help="manifest.json describing the snapshot")
    parser.add_argument("--output-dir", required=True, help="archive directory to create; must not exist")
    return parser.parse_args(argv)


def argument_problem(arguments):
    output_dir = Path(arguments.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        return f"output directory {output_dir} already exists"
    if not Path(arguments.snapshot).is_dir():
        return f"snapshot {arguments.snapshot} is not a directory"
    if not Path(arguments.manifest).is_file():
        return f"manifest {arguments.manifest} is not a file"
    return None


def run_verifier(manifest, snapshot):
    command = [sys.executable, str(VERIFIER_PATH), "--manifest", str(manifest), "--snapshot", str(snapshot)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=VERIFIER_TIMEOUT_S, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def load_manifest(path):
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    model = manifest.get("model") if isinstance(manifest, dict) else None
    if not isinstance(model, dict):
        raise HarnessError(f"manifest {path} has no model section")
    if model.get("repository") != REPOSITORY or model.get("revision") != REVISION:
        raise HarnessError(
            f"manifest {path} pins {model.get('repository')!r}@{model.get('revision')!r}, harness requires {REPOSITORY}@{REVISION}"
        )
    return manifest


def command_output(command):
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_S, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        return {"command": command, "error": str(error)}
    return {"command": command, "output": completed.stdout.strip()}


def sysctl_value(name):
    result = command_output(["sysctl", "-n", name])
    return result.get("output")


def parse_int(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def collect_environment(device):
    return {
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(), "executable": sys.executable},
        "dependencies": {
            distribution.metadata["Name"]: distribution.version
            for distribution in sorted(importlib.metadata.distributions(), key=lambda item: item.metadata["Name"].lower())
        },
        "machine": {
            "architecture": platform.machine(),
            "model": sysctl_value("hw.model"),
            "cpu": sysctl_value("machdep.cpu.brand_string"),
            "ram_bytes": parse_int(sysctl_value("hw.memsize")),
        },
        "gpu": device,
        "macos": {
            "product_version": command_output(["sw_vers", "-productVersion"]).get("output"),
            "build_version": command_output(["sw_vers", "-buildVersion"]).get("output"),
            "kernel": platform.release(),
        },
        "uv": command_output(["uv", "--version"]).get("output"),
    }


def archive_files(manifest_path, output_dir):
    sources = [TOOL_DIR / name for name in ARCHIVED_TOOL_FILES] + [Path(manifest_path).parent / CONFIGURATION_NAME]
    archived = []
    for source in sources:
        if not source.is_file():
            raise HarnessError(f"archive source {source} is missing")
        target = output_dir / source.name
        shutil.copyfile(source, target)
        archived.append({"name": source.name, "source": str(source), "sha256": sha256_file(target)})
    return archived


def summarize(executions):
    measured = [execution for execution in executions if execution["phase"] == "measured"]
    if len(measured) < 2:
        raise HarnessError(f"statistics require at least 2 measured executions, have {len(measured)}")
    return {
        metric: summarize_metric(metric, [execution[metric] for execution in measured])
        for metric in SUMMARIZED_METRICS
    }


def summarize_metric(metric, values):
    summary = {
        "unit": METRICS[metric]["unit"],
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev_sample": statistics.stdev(values),
    }
    if metric in INTEGER_METRICS:
        summary.update({key: round(summary[key]) for key in ("median", "mean", "stdev_sample")})
    return summary


def check_consistency(executions):
    if not executions:
        return {"consistent": None, "reference": None, "mismatches": []}
    reference = executions[0]
    mismatches = [
        {
            "phase": execution["phase"],
            "index": execution["index"],
            "first_divergent_position": first_divergence(reference["token_ids"], execution["token_ids"]),
        }
        for execution in executions[1:]
        if execution["token_ids"] != reference["token_ids"]
    ]
    return {
        "consistent": not mismatches,
        "reference": {"phase": reference["phase"], "index": reference["index"]},
        "compared_executions": len(executions),
        "mismatches": mismatches,
    }


def first_divergence(left, right):
    return next((position for position, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def execution_schedule():
    return [("warmup", index) for index in range(WARMUP_EXECUTIONS)] + [
        ("measured", index) for index in range(MEASURED_EXECUTIONS)
    ]


def make_record(phase, index, result, ignored_eos_ids):
    token_ids = list(result["token_ids"])
    return {
        "phase": phase,
        "index": index,
        "generated_count": len(token_ids),
        "token_ids": token_ids,
        "ignored_eos_count": sum(1 for token in token_ids if token in ignored_eos_ids),
        "elapsed_ns": int(result["elapsed_ns"]),
        "first_token_ns": None if result["first_token_ns"] is None else int(result["first_token_ns"]),
        "prompt_tps": result["prompt_tps"],
        "generation_tps": result["generation_tps"],
        "peak_memory_bytes": int(result["peak_memory_bytes"]),
        "peak_memory_gb": result["peak_memory_gb"],
        "finish_reason": result["finish_reason"],
    }


class MlxBackend:
    def __init__(self):
        self.mx = importlib.import_module("mlx.core")
        self.load_model = importlib.import_module("mlx_lm.utils").load
        self.stream_generate = importlib.import_module("mlx_lm.generate").stream_generate
        self.make_prompt_cache = importlib.import_module("mlx_lm.models.cache").make_prompt_cache
        self.tokenizer_wrapper = importlib.import_module("mlx_lm.tokenizer_utils").TokenizerWrapper
        self.tree_flatten = importlib.import_module("mlx.utils").tree_flatten
        self.model = None
        self.tokenizer = None

    def describe(self):
        info = self.mx.device_info()
        return {
            "default_device": str(self.mx.default_device()),
            "metal_available": self.mx.metal.is_available(),
            "device_info": {key: value if isinstance(value, (bool, int, float, str)) else str(value) for key, value in info.items()},
        }

    def load(self, snapshot):
        model, loaded = self.load_model(str(snapshot))
        self.model = model
        self.tokenizer = self.tokenizer_wrapper(
            loaded._tokenizer,
            detokenizer_class=loaded._detokenizer_class,
            eos_token_ids=[],
        )
        return {
            "loaded_eos_token_ids": sorted(loaded.eos_token_ids),
            "run_eos_token_ids": sorted(self.tokenizer.eos_token_ids),
            "detokenizer_class": loaded._detokenizer_class.__name__,
            "parameter_dtypes": sorted({str(value.dtype) for _, value in self.tree_flatten(model.parameters())}),
        }

    def greedy(self, logprobs):
        return self.mx.argmax(logprobs, axis=-1)

    def execute(self, input_ids, max_tokens):
        mx = self.mx
        prompt = mx.array(input_ids, dtype=mx.uint32)
        prompt_cache = self.make_prompt_cache(self.model)
        mx.eval(prompt)
        mx.random.seed(SEED)
        mx.synchronize()
        mx.reset_peak_memory()
        token_ids = []
        first_token_ns = None
        final = None
        start_ns = time.perf_counter_ns()
        for response in self.stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=self.greedy,
            prompt_cache=prompt_cache,
            prefill_step_size=PREFILL_STEP_SIZE,
            kv_bits=None,
            max_kv_size=None,
        ):
            if first_token_ns is None:
                first_token_ns = time.perf_counter_ns() - start_ns
            token_ids.append(response.token)
            final = response
        mx.synchronize()
        elapsed_ns = time.perf_counter_ns() - start_ns
        return {
            "token_ids": token_ids,
            "elapsed_ns": elapsed_ns,
            "first_token_ns": first_token_ns,
            "prompt_tps": None if final is None else final.prompt_tps,
            "generation_tps": None if final is None else final.generation_tps,
            "peak_memory_bytes": mx.get_peak_memory(),
            "peak_memory_gb": None if final is None else final.peak_memory,
            "finish_reason": None if final is None else final.finish_reason,
        }


class Run:
    def __init__(self, arguments, argv, backend_factory, verify):
        self.arguments = arguments
        self.backend_factory = backend_factory
        self.verify = verify
        self.output_dir = Path(arguments.output_dir)
        self.stage = "start"
        self.report = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "failure": None,
            "started_utc": utc_now(),
            "finished_utc": None,
            "command": {
                "argv": list(argv),
                "executable": sys.executable,
                "harness": str(HARNESS_PATH),
                "working_directory": str(Path.cwd()),
                "snapshot": str(Path(arguments.snapshot).resolve()),
                "manifest": str(Path(arguments.manifest).resolve()),
                "output_dir": str(self.output_dir.resolve()),
            },
            "model": {"repository": REPOSITORY, "revision": REVISION},
            "sources": {
                "harness": {"path": str(HARNESS_PATH), "sha256": sha256_file(HARNESS_PATH)},
                "verifier": {"path": str(VERIFIER_PATH), "sha256": sha256_file(VERIFIER_PATH)},
            },
            "protocol": PROTOCOL,
            "timing_boundaries": TIMING_BOUNDARIES,
            "metrics": METRICS,
            "executions": [],
        }

    def execute(self):
        report = self.report
        self.stage = "identity"
        manifest = load_manifest(self.arguments.manifest)
        report["manifest"] = {"sha256": sha256_file(self.arguments.manifest), "content": manifest}

        self.stage = "verification"
        verification = self.verify(self.arguments.manifest, self.arguments.snapshot)
        report["verification"] = verification
        if verification["returncode"] != 0:
            raise HarnessError(f"manifest verifier exited with status {verification['returncode']}")

        self.stage = "backend"
        backend = self.backend_factory()

        self.stage = "metadata"
        report["environment"] = collect_environment(backend.describe())

        self.stage = "archive"
        report["archived_files"] = archive_files(self.arguments.manifest, self.output_dir)

        self.stage = "input"
        input_ids = make_input_ids()
        report["input"] = {"token_count": len(input_ids), "sha256": hashlib.sha256(json.dumps(input_ids).encode()).hexdigest(), "token_ids": input_ids}

        self.stage = "load"
        report["loading"] = backend.load(self.arguments.snapshot)
        ignored_eos_ids = set(report["loading"].get("loaded_eos_token_ids", []))

        for phase, index in execution_schedule():
            self.stage = f"{phase}[{index}]"
            record = make_record(phase, index, backend.execute(input_ids, OUTPUT_TOKENS), ignored_eos_ids)
            report["executions"].append(record)
            if record["generated_count"] != OUTPUT_TOKENS:
                raise HarnessError(
                    f"{phase} execution {index} generated {record['generated_count']} tokens, expected {OUTPUT_TOKENS}"
                )

        self.stage = "statistics"
        report["statistics"] = summarize(report["executions"])
        report["status"] = "succeeded"

    def run(self):
        try:
            self.execute()
        except (Exception, KeyboardInterrupt) as error:
            self.report["status"] = "failed"
            self.report["failure"] = {"stage": self.stage, "type": type(error).__name__, "message": str(error)}
            self.report.pop("statistics", None)
        self.report["consistency"] = check_consistency(self.report["executions"])
        self.report["finished_utc"] = utc_now()
        with open(self.output_dir / REPORT_NAME, "x", encoding="utf-8") as handle:
            json.dump(self.report, handle, indent=2)
            handle.write("\n")
        return 0 if self.report["status"] == "succeeded" else 1


def print_summary(report, output_dir):
    if report["status"] != "succeeded":
        failure = report["failure"]
        print(f"FAILED at {failure['stage']}: {failure['type']}: {failure['message']}", file=sys.stderr)
        print(f"report: {output_dir / REPORT_NAME}", file=sys.stderr)
        return
    elapsed = report["statistics"]["elapsed_ns"]
    generation = report["statistics"]["generation_tps"]
    prompt = report["statistics"]["prompt_tps"]
    print(
        f"OK: {len(report['executions'])} executions ({WARMUP_EXECUTIONS} warmup), "
        f"elapsed median {elapsed['median'] / 1e6:.3f} ms, "
        f"prompt median {prompt['median']:.1f} tok/s, generation median {generation['median']:.1f} tok/s, "
        f"consistent outputs: {report['consistency']['consistent']}"
    )
    print(f"report: {output_dir / REPORT_NAME}")


def main(argv=None, backend_factory=MlxBackend, verify=run_verifier):
    argv = sys.argv[1:] if argv is None else argv
    arguments = parse_arguments(argv)
    problem = argument_problem(arguments)
    if problem is not None:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    output_dir = Path(arguments.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"error: cannot create output directory {output_dir}: {error}", file=sys.stderr)
        return 2
    run = Run(arguments, argv, backend_factory, verify)
    status = run.run()
    print_summary(run.report, output_dir)
    return status


if __name__ == "__main__":
    sys.exit(main())
