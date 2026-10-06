import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

PROTOCOL_ID = "token-ids-512x128-v1"
OUTPUT_TOKENS = 128
COMPARISON_NAME = "comparison.json"
SUMMARY_NAME = "SUMMARY.md"
METRICS = {
    "elapsed_ns": "Timer start (after GPU synchronization) to timer stop (after the 128th token and a final synchronization).",
    "first_token_ns": "token_available_ns[0]: timer start until the first token ID is on the host.",
    "decode_tokens_per_s": "127 / (token_available_ns[127] - token_available_ns[0]).",
}


class ComparisonError(Exception):
    pass


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_divergence(left, right):
    return next((index for index, (a, b) in enumerate(zip(left, right)) if a != b), None if left == right else min(len(left), len(right)))


def derived(execution):
    times = execution["token_available_ns"]
    if len(times) != OUTPUT_TOKENS or len(execution["token_ids"]) != OUTPUT_TOKENS:
        raise ComparisonError(f"{execution['phase']}[{execution['index']}] does not hold {OUTPUT_TOKENS} timed tokens")
    span = times[-1] - times[0]
    if span <= 0:
        raise ComparisonError(f"{execution['phase']}[{execution['index']}] has no decode interval")
    return {
        "elapsed_ns": execution["elapsed_ns"],
        "first_token_ns": times[0],
        "decode_tokens_per_s": (OUTPUT_TOKENS - 1) / (span / 1e9),
    }


def summarize(values):
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev_sample": statistics.stdev(values),
    }


def engine_summary(report):
    measured = [execution for execution in report["executions"] if execution["phase"] == "measured"]
    if len(measured) < 2:
        raise ComparisonError("at least two measured executions are required")
    rows = [derived(execution) for execution in measured]
    return {metric: summarize([row[metric] for row in rows]) for metric in METRICS}


def identity(report, engine):
    if engine == "mlx":
        model = report["model"]
        return {
            "repository": model["repository"],
            "revision": model["revision"],
            "manifest_sha256": report["manifest"]["sha256"],
            "machine_model": report["environment"]["machine"]["model"],
            "cpu": report["environment"]["machine"]["cpu"],
        }
    checkpoint = report["checkpoint"]
    environment = report["environment"]
    return {
        "repository": checkpoint["repository"],
        "revision": checkpoint["revision"],
        "manifest_sha256": checkpoint["manifest_sha256"],
        "machine_model": environment["machine_model"]["value"],
        "cpu": environment["cpu"]["value"],
    }


def check(mlx, foundry):
    problems = []
    for name, report in (("mlx", mlx), ("foundry", foundry)):
        if report.get("status") != "succeeded":
            problems.append(f"{name} report status is {report.get('status')}")
        if report.get("protocol", {}).get("id") != PROTOCOL_ID:
            problems.append(f"{name} report protocol is {report.get('protocol', {}).get('id')}, expected {PROTOCOL_ID}")
        if not report.get("consistency", {}).get("consistent"):
            problems.append(f"{name} report is not consistent across executions")
    if problems:
        raise ComparisonError("; ".join(problems))
    if mlx["input"]["sha256"] != foundry["input"]["sha256"] or mlx["input"]["token_ids"] != foundry["input"]["token_ids"]:
        raise ComparisonError("the engines used different inputs")
    left, right = identity(mlx, "mlx"), identity(foundry, "foundry")
    different = sorted(key for key in left if left[key] != right[key])
    if different:
        raise ComparisonError(f"checkpoint or machine identity differs: {different}")
    return left


def compare(mlx_path, foundry_path):
    mlx = json.loads(Path(mlx_path).read_text(encoding="utf-8"))
    foundry = json.loads(Path(foundry_path).read_text(encoding="utf-8"))
    shared = check(mlx, foundry)
    summaries = {"mlx": engine_summary(mlx), "foundry": engine_summary(foundry)}
    mlx_tokens = mlx["executions"][0]["token_ids"]
    foundry_tokens = foundry["executions"][0]["token_ids"]
    medians = {metric: {engine: summaries[engine][metric]["median"] for engine in summaries} for metric in METRICS}
    return {
        "protocol": PROTOCOL_ID,
        "identity": shared,
        "input_sha256": mlx["input"]["sha256"],
        "sources": {
            "mlx": {"path": str(Path(mlx_path).resolve()), "sha256": sha256_file(mlx_path)},
            "foundry": {"path": str(Path(foundry_path).resolve()), "sha256": sha256_file(foundry_path)},
        },
        "metric_definitions": METRICS,
        "metrics": summaries,
        "foundry_over_mlx_median": {
            metric: medians[metric]["foundry"] / medians[metric]["mlx"] for metric in METRICS
        },
        "greedy": {
            "identical": mlx_tokens == foundry_tokens,
            "first_divergence": first_divergence(mlx_tokens, foundry_tokens),
        },
        "memory": {
            "note": "The counters measure different things and are not compared.",
            "mlx": {
                "peak_memory_bytes_median": statistics.median(
                    execution["peak_memory_bytes"] for execution in mlx["executions"] if execution["phase"] == "measured"
                ),
                "definition": mlx["metrics"]["peak_memory_bytes"]["definition"],
            },
            "foundry": foundry["memory"],
        },
    }


def markdown(result):
    rows = [
        ("Total elapsed (ms)", "elapsed_ns", 1e-6),
        ("Time to first token (ms)", "first_token_ns", 1e-6),
        ("Decode throughput (tok/s)", "decode_tokens_per_s", 1.0),
    ]
    lines = [
        f"# MLX vs Foundry: {result['protocol']}",
        "",
        f"Checkpoint `{result['identity']['repository']}@{result['identity']['revision']}`, "
        f"{result['identity']['machine_model']} ({result['identity']['cpu']}), input SHA-256 `{result['input_sha256'][:12]}…`.",
        "Medians over 10 measured executions after 2 warmups; ± is the sample standard deviation.",
        "",
        "| Metric | MLX | Foundry | Foundry / MLX |",
        "|---|---|---|---|",
    ]
    for label, metric, scale in rows:
        mlx = result["metrics"]["mlx"][metric]
        foundry = result["metrics"]["foundry"][metric]
        lines.append(
            f"| {label} | {mlx['median'] * scale:.1f} ± {mlx['stdev_sample'] * scale:.1f} "
            f"| {foundry['median'] * scale:.1f} ± {foundry['stdev_sample'] * scale:.1f} "
            f"| {result['foundry_over_mlx_median'][metric]:.3f} |"
        )
    greedy = result["greedy"]
    lines += [
        "",
        f"Greedy outputs identical: {greedy['identical']} (first divergence: {greedy['first_divergence']}).",
        "",
        "Metric definitions:",
        *[f"- `{metric}`: {definition}" for metric, definition in result["metric_definitions"].items()],
        "",
        "Memory (not comparable; listed separately):",
        f"- MLX `peak_memory_bytes` median {result['memory']['mlx']['peak_memory_bytes_median']}: {result['memory']['mlx']['definition']}",
    ]
    foundry_memory = result["memory"]["foundry"]
    for key, value in foundry_memory.items():
        if key != "definitions":
            lines.append(f"- Foundry `{key}`: {value}")
    lines += [f"  - `{key}`: {value}" for key, value in foundry_memory.get("definitions", {}).items()]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare MLX and Foundry token-ids-512x128-v1 reports.")
    parser.add_argument("--mlx", required=True)
    parser.add_argument("--foundry", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    output_dir = Path(arguments.output_dir)
    if output_dir.exists():
        print(f"error: {output_dir} already exists", file=sys.stderr)
        return 2
    try:
        result = compare(arguments.mlx, arguments.foundry)
    except (ComparisonError, KeyError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True)
    (output_dir / COMPARISON_NAME).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = markdown(result)
    (output_dir / SUMMARY_NAME).write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
