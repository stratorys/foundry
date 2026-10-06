import contextlib
import importlib.util
import io
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

BENCH_PATH = Path(__file__).resolve().parent.parent / "bench.py"


def load_bench():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("mlx_baseline_bench", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_bench()


def passing_verifier(manifest, snapshot):
    return {"command": ["fake-verifier"], "returncode": 0, "stdout": "OK\n", "stderr": ""}


class FakeBackend:
    def __init__(self, counts=None):
        self.counts = counts or {}
        self.calls = 0
        self.loaded = False
        self.inputs = []

    def describe(self):
        return {"device_info": {"device_name": "fake"}}

    def load(self, snapshot):
        self.loaded = True
        return {"loaded_eos_token_ids": [128001, 128008, 128009], "run_eos_token_ids": []}

    def execute(self, input_ids, max_tokens):
        call = self.calls
        self.calls += 1
        self.inputs.append(list(input_ids))
        count = self.counts.get(call, max_tokens)
        return {
            "token_ids": [128009] + [7] * (count - 1) if count else [],
            "elapsed_ns": 1_000_000 * (call + 1),
            "first_token_ns": 100_000 * (call + 1),
            "prompt_tps": 1000.0 + call,
            "generation_tps": 50.0 + call,
            "peak_memory_bytes": 2_000_000_000 + call,
            "peak_memory_gb": (2_000_000_000 + call) / 1e9,
            "finish_reason": "length",
        }


    def execute_token_ids(self, input_ids, max_tokens):
        call = self.calls
        self.calls += 1
        self.inputs.append(list(input_ids))
        count = self.counts.get(call, max_tokens)
        return {
            "token_ids": [128009] + [7] * (count - 1) if count else [],
            "elapsed_ns": 2_000_000 * (call + 1),
            "token_available_ns": [100_000 * (call + 1) + 10_000 * position for position in range(count)],
            "peak_memory_bytes": 2_000_000_000 + call,
        }


class Workspace:
    def __init__(self, root, revision=bench.REVISION, repository=bench.REPOSITORY, manifest=None):
        self.root = Path(root)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        model_dir = self.root / "model"
        model_dir.mkdir()
        self.manifest = model_dir / "manifest.json"
        content = manifest if manifest is not None else {"schema_version": 1, "model": {"repository": repository, "revision": revision}}
        self.manifest.write_text(json.dumps(content))
        (model_dir / "CONFIGURATION.md").write_text("# configuration\n")
        self.output = self.root / "archive"

    def argv(self, protocol=None):
        argv = ["--snapshot", str(self.snapshot), "--manifest", str(self.manifest), "--output-dir", str(self.output)]
        return argv if protocol is None else argv + ["--protocol", protocol]

    def report(self):
        return json.loads((self.output / bench.REPORT_NAME).read_text())


def run_main(workspace, backend, verify=passing_verifier, protocol=None):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return bench.main(workspace.argv(protocol), backend_factory=lambda: backend, verify=verify)


def measured_record(index, elapsed_ns, phase="measured"):
    return bench.make_record(
        phase,
        index,
        {
            "token_ids": [1] * bench.OUTPUT_TOKENS,
            "elapsed_ns": elapsed_ns,
            "first_token_ns": elapsed_ns // 10,
            "prompt_tps": float(elapsed_ns),
            "generation_tps": float(elapsed_ns) / 2,
            "peak_memory_bytes": elapsed_ns * 3,
            "peak_memory_gb": elapsed_ns * 3 / 1e9,
            "finish_reason": "length",
        },
        set(),
    )


class InputTests(unittest.TestCase):
    def test_input_is_deterministic_with_one_initial_bos(self):
        first = bench.make_input_ids()
        self.assertEqual(first, bench.make_input_ids())
        self.assertEqual(len(first), 512)
        self.assertEqual(first[0], 128000)
        self.assertEqual(first.count(128000), 1)
        self.assertTrue(all(0 <= token < 128000 for token in first[1:]))

    def test_input_matches_reference_generator(self):
        import random

        generator = random.Random(0)
        self.assertEqual(bench.make_input_ids()[1:], [generator.randrange(128000) for _ in range(511)])


    def test_committed_input_file_is_the_generator_output(self):
        input_ids = bench.load_input_ids()
        self.assertEqual(input_ids, bench.make_input_ids())
        self.assertEqual(bench.input_sha256(input_ids), "9286ad09d58db9a18214d35afb5d92d2071e1fa9f8f1ccc63d2911401886dc49")

    def test_modified_input_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "input.json"
            path.write_text(json.dumps([128000] * 512))
            with self.assertRaises(bench.HarnessError):
                bench.load_input_ids(path)


class TokenIdsProtocolTests(unittest.TestCase):
    def test_default_protocol_is_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(root)
            backend = FakeBackend()
            self.assertEqual(run_main(workspace, backend), 0)
            report = workspace.report()
            self.assertEqual(report["protocol"]["id"], "stream-generate-v1")
            self.assertIn("generation_tps", report["statistics"])
            self.assertNotIn("token_available_ns", report["executions"][0])

    def test_token_ids_protocol_records_token_times(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(root)
            backend = FakeBackend()
            self.assertEqual(run_main(workspace, backend, protocol="token-ids"), 0)
            report = workspace.report()
            self.assertEqual(report["protocol"]["id"], "token-ids-512x128-v1")
            self.assertEqual(report["timing_boundaries"], bench.TOKEN_IDS_TIMING_BOUNDARIES)
            self.assertEqual(report["input"]["sha256"], bench.input_sha256(bench.make_input_ids()))
            first = report["executions"][0]
            self.assertEqual(len(first["token_available_ns"]), bench.OUTPUT_TOKENS)
            self.assertEqual(first["first_token_ns"], 100_000)
            self.assertAlmostEqual(first["decode_tokens_per_s"], 127 / (127 * 10_000 / 1e9))
            self.assertEqual(set(report["statistics"]), set(bench.TOKEN_IDS_SUMMARIZED_METRICS))
            self.assertTrue(report["consistency"]["consistent"])
            self.assertEqual(backend.inputs[0], bench.make_input_ids())

    def test_token_ids_protocol_requires_full_length(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(root)
            backend = FakeBackend(counts={3: 127})
            self.assertEqual(run_main(workspace, backend, protocol="token-ids"), 1)
            self.assertEqual(workspace.report()["failure"]["stage"], "measured[1]")

    def test_decode_throughput_needs_two_distinct_times(self):
        self.assertIsNone(bench.decode_tokens_per_s([5]))
        self.assertIsNone(bench.decode_tokens_per_s([5, 5]))
        self.assertAlmostEqual(bench.decode_tokens_per_s([0, 500_000_000, 1_000_000_000]), 2.0)


class StatisticsTests(unittest.TestCase):
    def test_warmup_is_excluded(self):
        measured = [10, 30, 20, 40, 50, 60, 70, 80, 90, 100]
        executions = [measured_record(0, 10**9, "warmup"), measured_record(1, 1, "warmup")] + [
            measured_record(index, value) for index, value in enumerate(measured)
        ]
        summary = bench.summarize(executions)
        elapsed = summary["elapsed_ns"]
        self.assertEqual(elapsed["count"], 10)
        self.assertEqual(elapsed["min"], 10)
        self.assertEqual(elapsed["max"], 100)
        self.assertEqual(elapsed["median"], 55)
        self.assertEqual(elapsed["mean"], 55)
        self.assertEqual(elapsed["stdev_sample"], round(statistics.stdev(measured)))
        self.assertAlmostEqual(summary["prompt_tps"]["stdev_sample"], statistics.stdev(measured))
        self.assertEqual(elapsed["unit"], "ns")

    def test_statistics_require_measured_executions(self):
        with self.assertRaises(bench.HarnessError):
            bench.summarize([measured_record(0, 10, "warmup"), measured_record(1, 20, "warmup")])

    def test_consistency_reports_mismatch(self):
        executions = [measured_record(0, 10, "warmup"), measured_record(0, 20)]
        executions[1]["token_ids"][5] = 2
        consistency = bench.check_consistency(executions)
        self.assertFalse(consistency["consistent"])
        self.assertEqual(consistency["mismatches"], [{"phase": "measured", "index": 0, "first_divergent_position": 5}])


class OutputDirectoryTests(unittest.TestCase):
    def test_existing_output_is_rejected_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(root)
            workspace.output.mkdir()
            marker = workspace.output / "keep.txt"
            marker.write_text("previous")
            backend = FakeBackend()
            self.assertEqual(run_main(workspace, backend), 2)
            self.assertEqual(marker.read_text(), "previous")
            self.assertEqual(sorted(path.name for path in workspace.output.iterdir()), ["keep.txt"])
            self.assertFalse(backend.loaded)

    def test_missing_snapshot_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(root)
            workspace.snapshot.rmdir()
            self.assertEqual(run_main(workspace, FakeBackend()), 2)
            self.assertFalse(workspace.output.exists())


class ManifestTests(unittest.TestCase):
    def test_verifier_failure_aborts_before_loading(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(root, manifest={"schema_version": 99, "model": {"repository": bench.REPOSITORY, "revision": bench.REVISION}})
            backend = FakeBackend()
            self.assertEqual(run_main(workspace, backend, verify=bench.run_verifier), 1)
            report = workspace.report()
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failure"]["stage"], "verification")
            self.assertNotEqual(report["verification"]["returncode"], 0)
            self.assertIn("schema_version", report["verification"]["stderr"])
            self.assertFalse(backend.loaded)
            self.assertNotIn("statistics", report)

    def test_wrong_model_revision_fails_before_verification(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(root, revision="0" * 40)
            backend = FakeBackend()
            calls = []
            status = run_main(workspace, backend, verify=lambda *arguments: calls.append(arguments) or passing_verifier(*arguments))
            self.assertEqual(status, 1)
            report = workspace.report()
            self.assertEqual(report["failure"]["stage"], "identity")
            self.assertEqual(calls, [])
            self.assertFalse(backend.loaded)

    def test_wrong_repository_fails(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(root, repository="someone/else")
            backend = FakeBackend()
            self.assertEqual(run_main(workspace, backend), 1)
            self.assertEqual(workspace.report()["failure"]["stage"], "identity")
            self.assertFalse(backend.loaded)


class GenerationTests(unittest.TestCase):
    def test_incomplete_generation_keeps_completed_executions(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(root)
            backend = FakeBackend(counts={5: 100})
            self.assertEqual(run_main(workspace, backend), 1)
            report = workspace.report()
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failure"]["stage"], "measured[3]")
            self.assertIn("100 tokens", report["failure"]["message"])
            self.assertEqual(len(report["executions"]), 6)
            self.assertEqual(report["executions"][-1]["generated_count"], 100)
            self.assertNotIn("statistics", report)
            self.assertEqual(backend.calls, 6)

    def test_backend_exception_is_reported(self):
        class Exploding(FakeBackend):
            def execute(self, input_ids, max_tokens):
                if self.calls == 1:
                    raise RuntimeError("device lost")
                return super().execute(input_ids, max_tokens)

        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(root)
            self.assertEqual(run_main(workspace, Exploding()), 1)
            report = workspace.report()
            self.assertEqual(report["failure"], {"stage": "warmup[1]", "type": "RuntimeError", "message": "device lost"})
            self.assertEqual(len(report["executions"]), 1)


class ReportTests(unittest.TestCase):
    def run_success(self, root):
        workspace = Workspace(root)
        backend = FakeBackend()
        self.assertEqual(run_main(workspace, backend), 0)
        return workspace, backend

    def test_successful_report_serialization(self):
        with tempfile.TemporaryDirectory() as root:
            workspace, backend = self.run_success(root)
            report = workspace.report()
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["status"], "succeeded")
            self.assertIsNone(report["failure"])
            self.assertTrue(report["started_utc"].endswith("Z"))
            self.assertEqual(report["command"]["argv"], workspace.argv())
            self.assertEqual(report["manifest"]["content"]["model"]["revision"], bench.REVISION)
            self.assertEqual(report["input"]["token_ids"], bench.make_input_ids())
            self.assertTrue(all(inputs == bench.make_input_ids() for inputs in backend.inputs))
            self.assertEqual([(item["phase"], item["index"]) for item in report["executions"]], bench.execution_schedule())
            self.assertTrue(all(item["generated_count"] == 128 for item in report["executions"]))
            self.assertTrue(all(item["ignored_eos_count"] == 1 for item in report["executions"]))
            self.assertTrue(report["consistency"]["consistent"])
            self.assertEqual(report["statistics"]["elapsed_ns"]["min"], 3_000_000)
            self.assertEqual(report["protocol"]["temperature"], 0.0)
            self.assertEqual(report["protocol"]["warmup_executions"], 2)
            self.assertEqual(report["protocol"]["measured_executions"], 10)
            self.assertIn("model loading", report["timing_boundaries"]["excluded"])
            self.assertEqual(len(report["sources"]["harness"]["sha256"]), 64)

    def test_metric_units_are_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            workspace, _ = self.run_success(root)
            report = workspace.report()
            for name, metric in report["metrics"].items():
                self.assertTrue(metric["unit"], name)
                self.assertTrue(metric["definition"], name)
            self.assertEqual(report["metrics"]["elapsed_ns"]["unit"], "ns")
            self.assertIn("not isolated GPU prefill", report["metrics"]["first_token_ns"]["definition"])
            self.assertIn("not process memory", report["metrics"]["peak_memory_bytes"]["definition"])
            for name, summary in report["statistics"].items():
                self.assertEqual(summary["unit"], report["metrics"][name]["unit"])
            for execution in report["executions"]:
                for name in bench.INTEGER_METRICS:
                    self.assertIsInstance(execution[name], int)
            for name in bench.INTEGER_METRICS:
                for key in ("median", "mean", "min", "max", "stdev_sample"):
                    self.assertIsInstance(report["statistics"][name][key], int)

    def test_archive_contains_configuration_files(self):
        with tempfile.TemporaryDirectory() as root:
            workspace, _ = self.run_success(root)
            names = sorted(path.name for path in workspace.output.iterdir())
            self.assertEqual(names, sorted(["pyproject.toml", "uv.lock", ".python-version", "CONFIGURATION.md", "report.json"]))

    def test_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(root)
            workspace.output.mkdir()
            run = bench.Run(bench.parse_arguments(workspace.argv()), workspace.argv(), FakeBackend, passing_verifier)
            (workspace.output / bench.REPORT_NAME).write_text("previous")
            with self.assertRaises(FileExistsError):
                run.run()
            self.assertEqual((workspace.output / bench.REPORT_NAME).read_text(), "previous")


if __name__ == "__main__":
    unittest.main()
