import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

COMPARE_PATH = Path(__file__).resolve().parent.parent / "compare.py"


def load_compare():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("mlx_baseline_compare", COMPARE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load_compare()

INPUT = {"sha256": "abc", "token_ids": [128000, 1, 2]}


def executions(step_ns, first_ns, tokens):
    return [
        {
            "phase": phase,
            "index": index,
            "token_ids": list(tokens),
            "elapsed_ns": first_ns + 127 * step_ns + index,
            "token_available_ns": [first_ns + position * step_ns for position in range(128)],
            "peak_memory_bytes": 10,
        }
        for phase, count in (("warmup", 2), ("measured", 10))
        for index in range(count)
    ]


def mlx_report(tokens=(5,) * 128):
    return {
        "status": "succeeded",
        "protocol": {"id": compare.PROTOCOL_ID},
        "consistency": {"consistent": True},
        "input": INPUT,
        "model": {"repository": "repo", "revision": "rev"},
        "manifest": {"sha256": "m"},
        "environment": {"machine": {"model": "Mac", "cpu": "M4"}},
        "metrics": {"peak_memory_bytes": {"definition": "mlx allocator peak"}},
        "executions": executions(10_000_000, 500_000_000, tokens),
    }


def foundry_report(tokens=(5,) * 128):
    return {
        "status": "succeeded",
        "protocol": {"id": compare.PROTOCOL_ID},
        "consistency": {"consistent": True},
        "input": INPUT,
        "checkpoint": {"repository": "repo", "revision": "rev", "manifest_sha256": "m"},
        "environment": {"machine_model": {"value": "Mac"}, "cpu": {"value": "M4"}},
        "memory": {"definitions": {"x": "y"}, "weight_tensor_bytes": 1},
        "executions": executions(20_000_000, 1_000_000_000, tokens),
    }


def write(directory, name, value):
    path = Path(directory) / name
    path.write_text(json.dumps(value))
    return path


class CompareTests(unittest.TestCase):
    def test_common_metrics_are_derived_from_raw_times(self):
        with tempfile.TemporaryDirectory() as root:
            result = compare.compare(write(root, "m.json", mlx_report()), write(root, "f.json", foundry_report()))
        self.assertAlmostEqual(result["metrics"]["mlx"]["decode_tokens_per_s"]["median"], 100.0)
        self.assertAlmostEqual(result["metrics"]["foundry"]["decode_tokens_per_s"]["median"], 50.0)
        self.assertEqual(result["metrics"]["foundry"]["first_token_ns"]["median"], 1_000_000_000)
        self.assertAlmostEqual(result["foundry_over_mlx_median"]["first_token_ns"], 2.0)
        self.assertTrue(result["greedy"]["identical"])
        self.assertIsNone(result["greedy"]["first_divergence"])
        self.assertEqual(result["metrics"]["mlx"]["elapsed_ns"]["count"], 10)

    def test_divergence_is_reported(self):
        tokens = [5] * 128
        tokens[17] = 6
        with tempfile.TemporaryDirectory() as root:
            result = compare.compare(write(root, "m.json", mlx_report()), write(root, "f.json", foundry_report(tokens)))
        self.assertEqual(result["greedy"]["first_divergence"], 17)

    def test_mismatched_runs_are_rejected(self):
        cases = []
        other_protocol = foundry_report()
        other_protocol["protocol"]["id"] = "stream-generate-v1"
        cases.append(other_protocol)
        other_input = foundry_report()
        other_input["input"] = {"sha256": "def", "token_ids": [1]}
        cases.append(other_input)
        other_revision = foundry_report()
        other_revision["checkpoint"]["revision"] = "main"
        cases.append(other_revision)
        other_machine = foundry_report()
        other_machine["environment"]["machine_model"]["value"] = "Mac2"
        cases.append(other_machine)
        failed = foundry_report()
        failed["status"] = "failed"
        cases.append(failed)
        for case in cases:
            with tempfile.TemporaryDirectory() as root, self.assertRaises(compare.ComparisonError):
                compare.compare(write(root, "m.json", mlx_report()), write(root, "f.json", case))

    def test_main_writes_both_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "out"
            argv = ["--mlx", str(write(root, "m.json", mlx_report())), "--foundry", str(write(root, "f.json", foundry_report())), "--output-dir", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compare.main(argv), 0)
            self.assertIn("Foundry / MLX", (output / compare.SUMMARY_NAME).read_text())
            self.assertTrue((output / compare.COMPARISON_NAME).is_file())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(compare.main(argv), 2)


if __name__ == "__main__":
    unittest.main()
