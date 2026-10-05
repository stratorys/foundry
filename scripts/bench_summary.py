# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
import argparse
import json
import math
import sys

MASK = (1 << 64) - 1
GOLDEN_GAMMA = 0x9E3779B97F4A7C15
MIX_FIRST = 0xBF58476D1CE4E5B9
MIX_SECOND = 0x94D049BB133111EB
REJECTIONS_MAX = 128
PROBABILITY_LOWER = 0.025
PROBABILITY_UPPER = 0.975
GIB = 1073741824.0
TOLERANCE = 1e-9


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK

    def next(self):
        self.state = (self.state + GOLDEN_GAMMA) & MASK
        first = ((self.state ^ (self.state >> 30)) * MIX_FIRST) & MASK
        second = ((first ^ (first >> 27)) * MIX_SECOND) & MASK
        return second ^ (second >> 31)

    def below(self, bound):
        if bound <= 0:
            return None
        threshold = ((1 << 64) - bound) % bound
        for _ in range(REJECTIONS_MAX):
            value = self.next()
            if value >= threshold:
                return value % bound
        return None


def median_sorted(values):
    count = len(values)
    if count == 0:
        return None
    upper = values[count // 2]
    if count % 2 == 1:
        return upper
    lower = values[count // 2 - 1]
    return lower + (upper - lower) // 2


def mean(values):
    if not values:
        return None
    return float(sum(values)) / float(len(values))


def stddev(values):
    if len(values) < 2:
        return None
    center = mean(values)
    squares = 0.0
    for value in values:
        deviation = float(value) - center
        squares += deviation * deviation
    return math.sqrt(squares / float(len(values) - 1))


def percentile(sorted_values, probability):
    if not sorted_values:
        return None
    position = float(len(sorted_values) - 1) * probability
    lower = math.floor(position)
    fraction = position - lower
    index = int(lower)
    below = sorted_values[index]
    above = sorted_values[index + 1] if index + 1 < len(sorted_values) else below
    return below + fraction * (above - below)


def interval(replicates):
    ordered = sorted(replicates)
    return [percentile(ordered, PROBABILITY_LOWER), percentile(ordered, PROBABILITY_UPPER)]


def speedup(baseline, elapsed):
    if baseline is None or elapsed is None or baseline <= 0 or elapsed <= 0:
        return None
    return float(baseline) / float(elapsed)


def throughput(payload_bytes, elapsed):
    if payload_bytes is None or elapsed is None or elapsed <= 0:
        return None
    return float(payload_bytes) / GIB / (float(elapsed) / 1e9)


def marginal(samples, resamples, generator):
    if len(samples) < 2:
        return None, None
    medians = []
    means = []
    for _ in range(resamples):
        scratch = []
        for _ in samples:
            index = generator.below(len(samples))
            if index is not None:
                scratch.append(samples[index])
        means.append(mean(scratch))
        scratch.sort()
        medians.append(float(median_sorted(scratch)))
    return interval(medians), interval(means)


def paired(baseline, other, resamples, generator):
    if len(baseline) != len(other) or len(baseline) < 2:
        return None
    rounds = len(baseline)
    replicates = []
    for _ in range(resamples):
        first = []
        second = []
        for _ in range(rounds):
            index = generator.below(rounds)
            if index is None:
                return None
            first.append(baseline[index])
            second.append(other[index])
        first.sort()
        second.sort()
        ratio = speedup(median_sorted(first), median_sorted(second))
        if ratio is None:
            return None
        replicates.append(ratio)
    return interval(replicates)


def value(field):
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return field


def measured(report):
    executions = report.get("executions", [])
    protocol = report.get("protocol") or {}
    order = protocol.get("configurations") or sorted({e["buffers"] for e in executions})
    groups = []
    for buffers in order:
        samples = [
            e["elapsed_ns"]
            for e in executions
            if e["buffers"] == buffers
            and e["phase"] == "measured"
            and e["outcome"] == "completed"
            and e["checksums"] == "verified"
            and e.get("elapsed_ns") is not None
        ]
        groups.append((buffers, samples))
    return groups


def recalculate(report, resamples, seed):
    groups = measured(report)
    workload = value(report.get("workload")) or {}
    payload_bytes = workload.get("payload_bytes")
    generator = SplitMix64(seed)
    rows = []
    for buffers, samples in groups:
        ordered = sorted(samples)
        median_ci, mean_ci = marginal(samples, resamples, generator)
        rows.append(
            {
                "buffers": buffers,
                "samples": len(samples),
                "median_ns": median_sorted(ordered),
                "min_ns": ordered[0] if ordered else None,
                "max_ns": ordered[-1] if ordered else None,
                "mean_ns": mean(samples),
                "standard_deviation_ns": stddev(samples),
                "median_ci_ns": median_ci,
                "mean_ci_ns": mean_ci,
                "throughput_gib_s": throughput(payload_bytes, median_sorted(ordered)),
                "speedup": None,
                "speedup_ci": None,
            }
        )
    if groups:
        baseline_samples = groups[0][1]
        baseline_median = rows[0]["median_ns"]
        for row in rows:
            row["speedup"] = speedup(baseline_median, row["median_ns"])
        for row, (_, samples) in list(zip(rows, groups))[1:]:
            row["speedup_ci"] = paired(baseline_samples, samples, resamples, generator)
    return rows


def close(expected, actual):
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(close(e, a) for e, a in zip(expected, actual))
    return abs(expected - actual) <= TOLERANCE * max(abs(expected), abs(actual), 1.0)


def reported(row, key):
    field = value(row.get(key))
    if isinstance(field, dict) and "lower" in field:
        return [field["lower"], field["upper"]]
    return field


def compare(rows, statistics):
    mismatches = []
    for actual in rows:
        expected = next(
            (row for row in statistics["configurations"] if row["buffers"] == actual["buffers"]),
            None,
        )
        if expected is None:
            mismatches.append(f"{actual['buffers']} buffers are missing from the report")
            continue
        for key, recalculated in actual.items():
            if key == "buffers":
                continue
            if not close(reported(expected, key), recalculated):
                mismatches.append(
                    f"{actual['buffers']} buffers: {key} reported {reported(expected, key)!r}, "
                    f"recalculated {recalculated!r}"
                )
    return mismatches


def milliseconds(nanos):
    return "n/a" if nanos is None else f"{nanos / 1e6:.2f}"


def bounds(pair, scale=1e6):
    return "n/a" if pair is None else f"[{pair[0] / scale:.2f}, {pair[1] / scale:.2f}]"


def number(field):
    return "n/a" if field is None else f"{field:.2f}"


def print_table(report, rows, resamples, seed):
    run = report.get("run", {})
    environment = report.get("environment", {})
    workload = value(report.get("workload")) or {}
    print(f"run: {run.get('id', 'n/a')} ({run.get('status', 'n/a')}, {run.get('instrumentation', 'n/a')})")
    print(f"gpu: {value(environment.get('gpu'))}  os: {value(environment.get('os_version'))}")
    print(
        f"workload: {workload.get('layers')} layers x {workload.get('weight_bytes')} bytes, "
        f"bootstrap {resamples} resamples, seed {seed}"
    )
    print(
        f"{'buffers':>7}  {'n':>3}  {'median ms':>10}  {'mean ms':>10}  {'sd ms':>8}  "
        f"{'median 95% CI ms':>20}  {'GiB/s':>7}  {'speedup':>7}  {'speedup 95% CI':>16}"
    )
    for row in rows:
        print(
            f"{row['buffers']:>7}  {row['samples']:>3}  {milliseconds(row['median_ns']):>10}  "
            f"{milliseconds(row['mean_ns']):>10}  {milliseconds(row['standard_deviation_ns']):>8}  "
            f"{bounds(row['median_ci_ns']):>20}  {number(row['throughput_gib_s']):>7}  "
            f"{number(row['speedup']):>7}  {bounds(row['speedup_ci'], 1.0):>16}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate foundry bench statistics from a JSON report without a GPU."
    )
    parser.add_argument("report", help="report written by foundry bench --report")
    parser.add_argument("--json", action="store_true", help="print the recalculation as JSON")
    arguments = parser.parse_args()
    try:
        with open(arguments.report, encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, ValueError) as error:
        print(f"error: cannot read {arguments.report}: {error}", file=sys.stderr)
        return 2
    if report.get("schema_version") != 1:
        print(f"error: unsupported schema_version {report.get('schema_version')!r}", file=sys.stderr)
        return 2
    statistics = value(report.get("statistics"))
    parameters = (report.get("invocation") or {}).get("parameters") or {}
    method = (statistics or {}).get("method") or {}
    resamples = method.get("resamples", parameters.get("bootstrap_resamples"))
    seed = method.get("seed", parameters.get("seed"))
    if resamples is None or seed is None:
        print("error: the report names no bootstrap resamples or seed", file=sys.stderr)
        return 2
    rows = recalculate(report, resamples, seed)
    if arguments.json:
        print(json.dumps({"resamples": resamples, "seed": seed, "configurations": rows}, indent=2))
    else:
        print_table(report, rows, resamples, seed)
    if statistics is None:
        reason = (report.get("statistics") or {}).get("reason", "no statistics were reported")
        print(f"not compared: {reason}", file=sys.stderr)
        return 0
    mismatches = compare(rows, statistics)
    for mismatch in mismatches:
        print(f"mismatch: {mismatch}", file=sys.stderr)
    if mismatches:
        return 1
    print(f"match: every statistic agrees within a relative tolerance of {TOLERANCE:g}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
