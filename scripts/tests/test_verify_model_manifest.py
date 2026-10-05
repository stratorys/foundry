# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

VERIFIER_PATH = Path(__file__).resolve().parent.parent / "verify-model-manifest.py"
REVISION = "0123456789abcdef0123456789abcdef01234567"
TEMPLATE = "{{- bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}"


def load_verifier():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("verify_model_manifest", VERIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = load_verifier()


def safetensors_bytes(tensors, metadata=None, header_override=None, length_override=None):
    header = {"__metadata__": metadata or {"format": "mlx"}}
    payload = b""
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    encoded = header_override if header_override is not None else json.dumps(header).encode("utf-8")
    length = length_override if length_override is not None else len(encoded)
    return struct.pack("<Q", length) + encoded + payload


def quantized_tensors():
    rows = 2
    columns = 64
    return [
        ("layer.proj.weight", "U32", [rows, columns // 8], bytes(range(rows * columns // 2))),
        ("layer.proj.scales", "F16", [rows, 1], b"\x00\x3c" * rows),
        ("layer.proj.biases", "F16", [rows, 1], b"\x00\x00" * rows),
        ("layer.norm.weight", "F16", [4], b"\x00\x3c" * 4),
    ]


def snapshot_contents():
    tensors = quantized_tensors()
    weights = safetensors_bytes(tensors)
    data_bytes = sum(len(data) for *_, data in tensors)
    config = {"model_type": "llama", "hidden_size": 4, "quantization": {"bits": 4, "group_size": 64}}
    tokenizer = {"added_tokens": [{"id": 10, "content": "<|bos|>"}, {"id": 11, "content": "<|eot|>"}]}
    tokenizer_config = {"bos_token": "<|bos|>", "eos_token": "<|eot|>", "chat_template": TEMPLATE}
    index = {
        "metadata": {"total_size": data_bytes},
        "weight_map": {name: "model.safetensors" for name, *_ in tensors},
    }
    return {
        "config.json": json.dumps(config).encode("utf-8"),
        "model.safetensors": weights,
        "model.safetensors.index.json": json.dumps(index).encode("utf-8"),
        "tokenizer.json": json.dumps(tokenizer).encode("utf-8"),
        "tokenizer_config.json": json.dumps(tokenizer_config).encode("utf-8"),
    }


def build_manifest(contents):
    header = verifier.parse_json(contents["model.safetensors"][8 : 8 + struct.unpack("<Q", contents["model.safetensors"][:8])[0]].decode())
    metadata = header.pop("__metadata__")
    header_bytes = struct.unpack("<Q", contents["model.safetensors"][:8])[0]
    return {
        "schema_version": 1,
        "model": {"repository": "synthetic/model", "revision": REVISION},
        "loading": {"conversion": "none"},
        "inspection": {},
        "files": [
            {"path": path, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for path, data in sorted(contents.items())
        ],
        "architecture": {"model_type": "llama", "hidden_size": 4},
        "safetensors_index": {"path": "model.safetensors.index.json"},
        "safetensors": [
            {
                "path": "model.safetensors",
                "header_bytes": header_bytes,
                "data_bytes": len(contents["model.safetensors"]) - 8 - header_bytes,
                "metadata": metadata,
                "tensors": [
                    {"name": name, "dtype": info["dtype"], "shape": info["shape"], "data_offsets": info["data_offsets"], "data_bytes": info["data_offsets"][1] - info["data_offsets"][0]}
                    for name, info in header.items()
                ],
            }
        ],
        "quantization": {
            "bits": 4,
            "group_size": 64,
            "pack_dtype": "U32",
            "scales_dtype": "F16",
            "biases_dtype": "F16",
            "quantized": [
                {
                    "logical": "layer.proj",
                    "logical_shape": [2, 64],
                    "weight": "layer.proj.weight",
                    "scales": "layer.proj.scales",
                    "biases": "layer.proj.biases",
                }
            ],
            "unquantized": ["layer.norm.weight"],
        },
        "tokenizer": {
            "special_tokens": [{"token": "<|bos|>", "id": 10}, {"token": "<|eot|>", "id": 11}],
            "bos": {"token": "<|bos|>"},
            "eos": {"token": "<|eot|>"},
            "chat_template": {"sha256": hashlib.sha256(TEMPLATE.encode("utf-8")).hexdigest()},
        },
    }


class VerifierTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.snapshot = self.root / "snapshots" / REVISION
        self.snapshot.mkdir(parents=True)
        self.contents = snapshot_contents()
        for path, data in self.contents.items():
            (self.snapshot / path).write_bytes(data)
        self.manifest = build_manifest(self.contents)

    def tearDown(self):
        self.temporary.cleanup()

    def run_verifier(self, manifest=None, snapshot=None):
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest if manifest is not None else self.manifest), encoding="utf-8")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = verifier.main(["--manifest", str(manifest_path), "--snapshot", str(snapshot or self.snapshot)])
        return status, stdout.getvalue() + stderr.getvalue()

    def replace_file(self, path, data, manifest_too=False):
        (self.snapshot / path).write_bytes(data)
        if manifest_too:
            self.contents[path] = data
            self.manifest = build_manifest(self.contents)

    def assert_mismatch(self, fragment, manifest=None):
        status, output = self.run_verifier(manifest)
        self.assertEqual(status, 1, output)
        self.assertIn(fragment, output)

    def assert_invalid_manifest(self, fragment, manifest):
        status, output = self.run_verifier(manifest)
        self.assertEqual(status, 2, output)
        self.assertIn(fragment, output)

    def test_valid_snapshot_passes(self):
        status, output = self.run_verifier()
        self.assertEqual(status, 0, output)
        self.assertIn("5 files hashed, 4 tensors verified", output)

    def test_huggingface_symlinks_to_blobs_pass(self):
        blobs = self.root / "blobs"
        blobs.mkdir()
        for path in self.contents:
            blob = blobs / hashlib.sha256(self.contents[path]).hexdigest()
            shutil.move(self.snapshot / path, blob)
            os.symlink(os.path.relpath(blob, self.snapshot), self.snapshot / path)
        status, output = self.run_verifier()
        self.assertEqual(status, 0, output)

    def test_modified_byte_with_same_size_fails(self):
        data = bytearray(self.contents["model.safetensors"])
        data[-1] ^= 0xFF
        self.replace_file("model.safetensors", bytes(data))
        self.assert_mismatch("model.safetensors: sha256")

    def test_size_change_fails(self):
        self.replace_file("config.json", self.contents["config.json"] + b" ")
        self.assert_mismatch("config.json: size")

    def test_missing_file_fails(self):
        (self.snapshot / "tokenizer.json").unlink()
        self.assert_mismatch("tokenizer.json: missing")

    def test_dangling_symlink_fails(self):
        (self.snapshot / "tokenizer.json").unlink()
        os.symlink("../../blobs/absent", self.snapshot / "tokenizer.json")
        self.assert_mismatch("tokenizer.json: dangling symlink")

    def test_unexpected_file_fails(self):
        (self.snapshot / "extra.bin").write_bytes(b"x")
        self.assert_mismatch("extra.bin: file present in snapshot but absent from manifest")

    def test_wrong_schema_version_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["schema_version"] = 2
        self.assert_invalid_manifest("unsupported schema_version 2", manifest)

    def test_duplicate_manifest_keys_are_rejected(self):
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            status = verifier.main(["--manifest", str(manifest_path), "--snapshot", str(self.snapshot)])
        self.assertEqual(status, 2)
        self.assertIn("duplicate JSON keys", stderr.getvalue())

    def test_paths_escaping_snapshot_are_rejected(self):
        for path, fragment in [
            ("../outside.json", "not a normalized relative path"),
            ("/etc/hosts", "is absolute"),
            ("sub\\config.json", "backslash"),
            ("./config.json", "not a normalized relative path"),
            ("", "non-empty string"),
        ]:
            with self.subTest(path=path):
                manifest = copy.deepcopy(self.manifest)
                manifest["files"][0]["path"] = path
                self.assert_invalid_manifest(fragment, manifest)

    def test_symlinked_directory_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "file.bin").write_bytes(b"secret")
        os.symlink(outside, self.snapshot / "linked")
        manifest = copy.deepcopy(self.manifest)
        manifest["files"].append({"path": "linked/file.bin", "size_bytes": 6, "sha256": hashlib.sha256(b"secret").hexdigest()})
        self.assert_mismatch("traverses symlinked directory", manifest)

    def test_revision_directory_mismatch_fails(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["model"]["revision"] = "f" * 40
        self.assert_mismatch("snapshot directory is revision", manifest)

    def test_malformed_headers_fail(self):
        tensors = quantized_tensors()
        valid_header = json.loads(safetensors_bytes(tensors)[8:][: struct.unpack("<Q", safetensors_bytes(tensors)[:8])[0]])
        cases = {
            "truncated length": (b"\x01\x02\x03", "too short for the header length"),
            "oversized length": (safetensors_bytes(tensors, length_override=1 << 40), "exceeds limit"),
            "length past end": (safetensors_bytes(tensors, length_override=10_000), "exceeds remaining file size"),
            "invalid json": (safetensors_bytes(tensors, header_override=b"{not json"), "not valid UTF-8 JSON"),
            "not an object": (safetensors_bytes(tensors, header_override=b"[1, 2]"), "not a JSON object"),
        }
        mutations = {
            "offset past end": (lambda header: header["layer.norm.weight"].update(data_offsets=[44, 1000]), "exceeds data region"),
            "overlap": (lambda header: header["layer.proj.scales"].update(data_offsets=[60, 64]), "overlap"),
            "gap": (lambda header: header.pop("layer.proj.scales"), "gap of"),
            "byte count": (lambda header: header["layer.norm.weight"].update(shape=[5]), "requires"),
            "unknown dtype": (lambda header: header["layer.norm.weight"].update(dtype="Q4"), "unknown dtype"),
            "reversed offsets": (lambda header: header["layer.norm.weight"].update(data_offsets=[8, 0]), "not ordered"),
            "negative shape": (lambda header: header["layer.norm.weight"].update(shape=[-4]), "invalid shape"),
            "metadata type": (lambda header: header.update(__metadata__={"format": 1}), "string-to-string"),
        }
        for name, (mutate, fragment) in mutations.items():
            header = copy.deepcopy(valid_header)
            mutate(header)
            cases[name] = (safetensors_bytes(tensors, header_override=json.dumps(header).encode("utf-8")), fragment)
        for name, (data, fragment) in cases.items():
            with self.subTest(case=name):
                self.replace_file("model.safetensors", data)
                manifest = copy.deepcopy(self.manifest)
                manifest["files"] = [
                    entry if entry["path"] != "model.safetensors" else {"path": entry["path"], "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    for entry in manifest["files"]
                ]
                status, output = self.run_verifier(manifest)
                self.assertEqual(status, 1, output)
                self.assertIn("model.safetensors: malformed safetensors header", output)
                self.assertIn(fragment, output)

    def test_manifest_tensor_mismatch_fails(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["safetensors"][0]["tensors"][3]["shape"] = [8]
        self.assert_mismatch("tensor layer.norm.weight shape is [4]", manifest)

    def test_unlisted_tensor_fails(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["safetensors"][0]["tensors"].pop()
        self.assert_mismatch("present in header but absent from manifest", manifest)

    def test_quantization_shape_mismatch_fails(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["quantization"]["quantized"][0]["logical_shape"] = [2, 128]
        self.assert_mismatch("layer.proj weight is U32[2, 8], expected U32[2, 16]", manifest)

    def test_unclassified_tensor_fails(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["quantization"]["unquantized"] = []
        self.assert_mismatch("layer.norm.weight is neither quantized nor unquantized", manifest)

    def test_special_token_mismatch_fails(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["tokenizer"]["special_tokens"][1]["id"] = 12
        self.assert_mismatch("'<|eot|>' has id 11, manifest expects 12", manifest)

    def test_chat_template_mismatch_fails(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["tokenizer"]["chat_template"]["sha256"] = "0" * 64
        self.assert_mismatch("chat_template sha256", manifest)

    def test_index_mismatch_fails(self):
        index = {"metadata": {"total_size": 1}, "weight_map": {"layer.norm.weight": "model.safetensors"}}
        self.replace_file("model.safetensors.index.json", json.dumps(index).encode("utf-8"), manifest_too=True)
        status, output = self.run_verifier()
        self.assertEqual(status, 1, output)
        self.assertIn("weight_map differs from headers", output)
        self.assertIn("metadata.total_size 1", output)

    def test_verifier_writes_nothing(self):
        before = {path: path.stat().st_mtime_ns for path in self.root.rglob("*")}
        self.run_verifier()
        after = {path: path.stat().st_mtime_ns for path in self.root.rglob("*") if path.name != "manifest.json"}
        self.assertEqual({path: value for path, value in before.items() if path.name != "manifest.json"}, after)

    def test_command_line_reports_nonzero_status(self):
        (self.snapshot / "config.json").write_bytes(b"{}")
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(VERIFIER_PATH), "--manifest", str(manifest_path), "--snapshot", str(self.snapshot)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("mismatch: config.json: size", result.stderr)
        self.assertIn("FAILED", result.stderr)

    def test_missing_snapshot_directory_is_rejected(self):
        status, output = self.run_verifier(snapshot=self.root / "absent")
        self.assertEqual(status, 2)
        self.assertIn("is not a directory", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
