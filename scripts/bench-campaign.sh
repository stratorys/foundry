#!/usr/bin/env bash
set -uo pipefail

LAYERS=48
WEIGHTS_MIB=(16 64 128)
UNTRACED_RUNS=3
WARMUP=2
ITERATIONS=30
RESAMPLES=10000
SEED=0

if [ "$#" -ne 1 ]; then
    echo "usage: $0 ARCHIVE_DIR" >&2
    exit 2
fi

root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
archive=$1
if [ -e "$archive" ]; then
    echo "error: $archive already exists; choose a new archive directory" >&2
    exit 1
fi
mkdir -p "$(dirname "$archive")" && mkdir "$archive" || exit 1
archive=$(cd "$archive" && pwd)
mkdir "$archive/untraced" "$archive/traced" "$archive/summaries" "$archive/logs" || exit 1
cd "$root" || exit 1

commands="$archive/commands.txt"
outcomes="$archive/outcomes.txt"
provenance="$archive/provenance.txt"

record() {
    printf '%s\n' "$*" >>"$commands"
}

capture() {
    printf '$ %s\n' "$*" >>"$provenance"
    "$@" >>"$provenance" 2>&1 || printf '(exit %s)\n' "$?" >>"$provenance"
    printf '\n' >>"$provenance"
}

{
    printf 'started_at_utc: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'script: %s\n' "$0"
    printf 'invocation: %q %q\n' "$0" "$1"
    printf 'working_directory: %s\n' "$root"
    printf 'archive: %s\n' "$archive"
    printf 'layers: %s\nweights_mib: %s\nuntraced_runs: %s\nwarmup: %s\niterations: %s\nresamples: %s\nseed: %s\n' \
        "$LAYERS" "${WEIGHTS_MIB[*]}" "$UNTRACED_RUNS" "$WARMUP" "$ITERATIONS" "$RESAMPLES" "$SEED"
} >"$archive/invocation.txt"

capture git rev-parse HEAD
capture git status --porcelain=v1 --untracked-files=normal
capture rustc -vV
capture cargo -V
capture uv --version
capture sw_vers
capture uname -a
capture /usr/sbin/sysctl -n hw.model machdep.cpu.brand_string hw.memsize

target_dir=${CARGO_TARGET_DIR:-$root/target}
binary="$target_dir/release/foundry"
record cargo build --release -p foundry-infer-cli --features metal
if ! cargo build --release -p foundry-infer-cli --features metal >"$archive/logs/build.log" 2>&1; then
    echo "build failed: exit $?" >>"$outcomes"
    echo "error: the release build failed; see $archive/logs/build.log" >&2
    exit 1
fi
capture shasum -a 256 "$binary"

failures=0

bench() {
    local name=$1
    shift
    local command=("$binary" bench --backend metal --layers "$LAYERS" --warmup "$WARMUP"
        --iterations "$ITERATIONS" --bootstrap-resamples "$RESAMPLES" --seed "$SEED" "$@")
    record "${command[*]}"
    "${command[@]}" >"$archive/logs/$name.stdout" 2>"$archive/logs/$name.stderr"
    local status=$?
    printf '%s exit %s\n' "$name" "$status" >>"$outcomes"
    if [ "$status" -ne 0 ]; then
        failures=$((failures + 1))
    fi
}

for weight in "${WEIGHTS_MIB[@]}"; do
    for run in $(seq 1 "$UNTRACED_RUNS"); do
        name="w${weight}-r${run}"
        bench "untraced-$name" --weight-mib "$weight" --report "$archive/untraced/$name.json"
    done
    bench "traced-w${weight}" --weight-mib "$weight" \
        --report "$archive/traced/w${weight}.json" --trace "$archive/traced/w${weight}.trace.json"
done

for report in "$archive"/untraced/*.json "$archive"/traced/w*.json; do
    case "$report" in
        *.trace.json) continue ;;
    esac
    [ -e "$report" ] || continue
    name="$(basename "$(dirname "$report")")-$(basename "$report" .json)"
    record uv run --quiet --script scripts/bench_summary.py "$report"
    uv run --quiet --script scripts/bench_summary.py "$report" >"$archive/summaries/$name.txt" 2>&1
    status=$?
    uv run --quiet --script scripts/bench_summary.py --json "$report" >"$archive/summaries/$name.json" 2>/dev/null
    printf 'summary %s exit %s\n' "$name" "$status" >>"$outcomes"
    if [ "$status" -ne 0 ]; then
        failures=$((failures + 1))
    fi
done

printf 'finished_at_utc: %s\nfailures: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$failures" >>"$archive/invocation.txt"
cat "$outcomes"
if [ "$failures" -ne 0 ]; then
    echo "error: $failures step(s) failed; every outcome is kept in $archive" >&2
    exit 1
fi
echo "campaign archived in $archive"
