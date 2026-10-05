# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
import argparse
import collections
import hashlib
import json
import os
import re
import stat
import struct
import sys
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
HEADER_BYTES_MAX = 100 * 1024 * 1024
HEADER_LENGTH_BYTES = 8
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IGNORED_TREES = (PurePosixPath(".cache/huggingface"),)
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}
REQUIRED_SECTIONS = {
    "model": dict,
    "loading": dict,
    "inspection": dict,
    "files": list,
    "architecture": dict,
    "safetensors": list,
    "quantization": dict,
    "tokenizer": dict,
}


class ManifestError(Exception):
    pass


class HeaderError(Exception):
    pass


def reject_duplicates(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        counts = collections.Counter(key for key, _ in pairs)
        raise ValueError(f"duplicate JSON keys {sorted(key for key, count in counts.items() if count > 1)}")
    return result


def reject_constant(name):
    raise ValueError(f"non-standard JSON constant {name}")


def parse_json(text):
    return json.loads(text, object_pairs_hook=reject_duplicates, parse_constant=reject_constant)


def load_json_file(path):
    with open(path, encoding="utf-8") as handle:
        return parse_json(handle.read())


def load_snapshot_json(snapshot, relative, errors):
    try:
        return load_json_file(resolve_in_snapshot(snapshot, relative))
    except (OSError, ValueError) as error:
        errors.append(f"{relative}: cannot parse JSON: {error}")
        return None


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def is_shape(value):
    return isinstance(value, list) and all(is_int(dimension) and dimension >= 0 for dimension in value)


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def manifest_path(raw):
    if not isinstance(raw, str) or raw == "":
        raise ManifestError(f"path {raw!r} is not a non-empty string")
    if "\\" in raw or "\x00" in raw:
        raise ManifestError(f"path {raw!r} contains a backslash or NUL byte")
    if raw.startswith("/"):
        raise ManifestError(f"path {raw!r} is absolute")
    parts = raw.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ManifestError(f"path {raw!r} is not a normalized relative path inside the snapshot")
    return PurePosixPath(raw)


def resolve_in_snapshot(snapshot, relative):
    directory = snapshot
    for part in relative.parts[:-1]:
        directory = directory / part
        if directory.is_symlink():
            raise ManifestError(f"path {relative} traverses symlinked directory {part!r}")
    return snapshot.joinpath(*relative.parts)


def validate_schema(manifest):
    if not isinstance(manifest, dict):
        raise ManifestError("manifest root is not a JSON object")
    version = manifest.get("schema_version")
    if not is_int(version) or version != SCHEMA_VERSION:
        raise ManifestError(f"unsupported schema_version {version!r}, expected {SCHEMA_VERSION}")
    for section, kind in REQUIRED_SECTIONS.items():
        if not isinstance(manifest.get(section), kind):
            raise ManifestError(f"section {section!r} is missing or is not a {kind.__name__}")
    revision = manifest["model"].get("revision")
    if not isinstance(revision, str) or not COMMIT_PATTERN.match(revision):
        raise ManifestError(f"model.revision {revision!r} is not a 40-character commit hash")
    if not isinstance(manifest["model"].get("repository"), str):
        raise ManifestError("model.repository is missing")
    seen = set()
    for index, entry in enumerate(manifest["files"]):
        if not isinstance(entry, dict):
            raise ManifestError(f"files[{index}] is not an object")
        path = manifest_path(entry.get("path"))
        if path in seen:
            raise ManifestError(f"files[{index}] duplicates path {path}")
        seen.add(path)
        if not is_int(entry.get("size_bytes")) or entry["size_bytes"] < 0:
            raise ManifestError(f"files[{index}] ({path}) has an invalid size_bytes")
        if not isinstance(entry.get("sha256"), str) or not SHA256_PATTERN.match(entry["sha256"]):
            raise ManifestError(f"files[{index}] ({path}) has an invalid sha256")
    for index, entry in enumerate(manifest["safetensors"]):
        if not isinstance(entry, dict) or not isinstance(entry.get("tensors"), list):
            raise ManifestError(f"safetensors[{index}] is not an object with a tensors list")
        path = manifest_path(entry.get("path"))
        if path not in seen:
            raise ManifestError(f"safetensors[{index}] path {path} is not in the file inventory")


def snapshot_files(snapshot):
    found = set()
    for root, directories, files in os.walk(snapshot, followlinks=False):
        relative_root = PurePosixPath(Path(root).relative_to(snapshot).as_posix())
        directories[:] = [
            name for name in directories if not any((relative_root / name) == tree for tree in IGNORED_TREES)
        ]
        for name in files + [name for name in directories if (Path(root) / name).is_symlink()]:
            found.add(relative_root / name)
    return found


def check_revision(snapshot, manifest, errors):
    name = snapshot.name
    revision = manifest["model"]["revision"]
    if COMMIT_PATTERN.match(name) and name != revision:
        errors.append(f"snapshot directory is revision {name}, manifest pins {revision}")


def check_files(snapshot, manifest, errors):
    expected = {PurePosixPath(entry["path"]): entry for entry in manifest["files"]}
    present = snapshot_files(snapshot)
    for extra in sorted(present - set(expected)):
        errors.append(f"{extra}: file present in snapshot but absent from manifest")
    verified = set()
    for relative, entry in expected.items():
        try:
            target = resolve_in_snapshot(snapshot, relative)
        except ManifestError as error:
            errors.append(str(error))
            continue
        try:
            status = target.stat()
        except FileNotFoundError:
            reason = "dangling symlink" if target.is_symlink() else "missing"
            errors.append(f"{relative}: {reason}")
            continue
        except OSError as error:
            errors.append(f"{relative}: cannot stat: {error}")
            continue
        if not stat.S_ISREG(status.st_mode):
            errors.append(f"{relative}: not a regular file")
            continue
        if status.st_size != entry["size_bytes"]:
            errors.append(f"{relative}: size {status.st_size} bytes, manifest expects {entry['size_bytes']}")
            continue
        try:
            with open(target, "rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
        except OSError as error:
            errors.append(f"{relative}: cannot read: {error}")
            continue
        if digest != entry["sha256"]:
            errors.append(f"{relative}: sha256 {digest}, manifest expects {entry['sha256']}")
            continue
        verified.add(relative)
    return verified


def read_header(path):
    file_bytes = path.stat().st_size
    with open(path, "rb") as handle:
        prefix = handle.read(HEADER_LENGTH_BYTES)
        if len(prefix) != HEADER_LENGTH_BYTES:
            raise HeaderError(f"file has {len(prefix)} bytes, too short for the header length")
        (header_bytes,) = struct.unpack("<Q", prefix)
        if header_bytes > HEADER_BYTES_MAX:
            raise HeaderError(f"header length {header_bytes} exceeds limit {HEADER_BYTES_MAX}")
        if header_bytes > file_bytes - HEADER_LENGTH_BYTES:
            raise HeaderError(f"header length {header_bytes} exceeds remaining file size {file_bytes - HEADER_LENGTH_BYTES}")
        raw = handle.read(header_bytes)
    if len(raw) != header_bytes:
        raise HeaderError(f"header truncated: read {len(raw)} of {header_bytes} bytes")
    try:
        header = parse_json(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise HeaderError(f"header is not valid UTF-8 JSON: {error}") from error
    if not isinstance(header, dict):
        raise HeaderError("header is not a JSON object")
    data_bytes = file_bytes - HEADER_LENGTH_BYTES - header_bytes
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
        raise HeaderError("__metadata__ is not a string-to-string map")
    tensors = {}
    for name, info in header.items():
        tensors[name] = parse_tensor(name, info, data_bytes)
    check_coverage(tensors, data_bytes)
    return {"header_bytes": header_bytes, "data_bytes": data_bytes, "metadata": metadata, "tensors": tensors}


def parse_tensor(name, info, data_bytes):
    if not isinstance(info, dict):
        raise HeaderError(f"tensor {name}: entry is not an object")
    dtype = info.get("dtype")
    if dtype not in DTYPE_BYTES:
        raise HeaderError(f"tensor {name}: unknown dtype {dtype!r}")
    shape = info.get("shape")
    if not is_shape(shape):
        raise HeaderError(f"tensor {name}: invalid shape {shape!r}")
    offsets = info.get("data_offsets")
    if not (isinstance(offsets, list) and len(offsets) == 2 and all(is_int(value) for value in offsets)):
        raise HeaderError(f"tensor {name}: invalid data_offsets {offsets!r}")
    begin, end = offsets
    if begin < 0 or begin > end:
        raise HeaderError(f"tensor {name}: data_offsets {offsets} are not ordered")
    if end > data_bytes:
        raise HeaderError(f"tensor {name}: data_offsets end {end} exceeds data region of {data_bytes} bytes")
    expected = product(shape) * DTYPE_BYTES[dtype]
    if end - begin != expected:
        raise HeaderError(f"tensor {name}: {end - begin} data bytes, {dtype}{shape} requires {expected}")
    return {"dtype": dtype, "shape": shape, "data_offsets": [begin, end], "data_bytes": end - begin}


def check_coverage(tensors, data_bytes):
    position = 0
    for name, info in sorted(tensors.items(), key=lambda item: tuple(item[1]["data_offsets"])):
        begin, end = info["data_offsets"]
        if begin < position:
            raise HeaderError(f"tensor {name}: data_offsets {info['data_offsets']} overlap previous tensor ending at {position}")
        if begin > position:
            raise HeaderError(f"tensor {name}: gap of {begin - position} bytes before data_offsets {info['data_offsets']}")
        position = end
    if position != data_bytes:
        raise HeaderError(f"tensors cover {position} bytes, data region has {data_bytes}")


def check_safetensors(snapshot, manifest, verified, errors):
    headers = {}
    for entry in manifest["safetensors"]:
        relative = PurePosixPath(entry["path"])
        if relative not in verified:
            errors.append(f"{relative}: header not inspected because the file failed inventory checks")
            continue
        try:
            header = read_header(resolve_in_snapshot(snapshot, relative))
        except HeaderError as error:
            errors.append(f"{relative}: malformed safetensors header: {error}")
            continue
        headers[relative] = header
        for key in ("header_bytes", "data_bytes"):
            if entry.get(key) != header[key]:
                errors.append(f"{relative}: {key} {header[key]}, manifest expects {entry.get(key)!r}")
        if "metadata" in entry and entry["metadata"] != header["metadata"]:
            errors.append(f"{relative}: __metadata__ {header['metadata']}, manifest expects {entry['metadata']}")
        compare_tensors(relative, entry["tensors"], header["tensors"], errors)
    return headers


def compare_tensors(relative, listed, actual, errors):
    names = [item.get("name") if isinstance(item, dict) else None for item in listed]
    for duplicate in sorted((name for name, count in collections.Counter(names).items() if count > 1), key=str):
        errors.append(f"{relative}: manifest lists tensor {duplicate!r} more than once")
    for missing in sorted(set(actual) - set(names)):
        errors.append(f"{relative}: tensor {missing} present in header but absent from manifest")
    for item in listed:
        name = item.get("name") if isinstance(item, dict) else None
        if name not in actual:
            errors.append(f"{relative}: tensor {name!r} listed in manifest but absent from header")
            continue
        for key, value in actual[name].items():
            if item.get(key) != value:
                errors.append(f"{relative}: tensor {name} {key} is {value}, manifest expects {item.get(key)!r}")


def check_index(snapshot, manifest, headers, verified, errors):
    index = manifest.get("safetensors_index")
    if index is None:
        return
    relative = manifest_path(index.get("path"))
    if relative not in verified:
        errors.append(f"{relative}: index not inspected because the file failed inventory checks")
        return
    content = load_snapshot_json(snapshot, relative, errors)
    if content is None:
        return
    weight_map = content.get("weight_map") if isinstance(content, dict) else None
    if not isinstance(weight_map, dict):
        errors.append(f"{relative}: weight_map is missing")
        return
    expected = {name: str(path) for path, header in headers.items() for name in header["tensors"]}
    if weight_map != expected:
        missing = sorted(set(expected) - set(weight_map))
        extra = sorted(set(weight_map) - set(expected))
        moved = sorted(name for name in set(expected) & set(weight_map) if expected[name] != weight_map[name])
        errors.append(f"{relative}: weight_map differs from headers (missing {missing[:5]}, extra {extra[:5]}, moved {moved[:5]})")
    total = sum(header["data_bytes"] for header in headers.values())
    declared = (content.get("metadata") or {}).get("total_size")
    if declared != total:
        errors.append(f"{relative}: metadata.total_size {declared!r}, header data regions total {total}")


def check_architecture(snapshot, manifest, verified, errors):
    relative = PurePosixPath("config.json")
    if relative not in verified:
        errors.append("config.json: architecture not compared because the file failed inventory checks")
        return None
    config = load_snapshot_json(snapshot, relative, errors)
    if not isinstance(config, dict):
        return None
    for key, value in manifest["architecture"].items():
        if config.get(key) != value:
            errors.append(f"config.json: {key} is {config.get(key)!r}, manifest architecture records {value!r}")
    return config


def check_quantization(manifest, headers, config, errors):
    quantization = manifest["quantization"]
    bits = quantization.get("bits")
    group_size = quantization.get("group_size")
    if not (is_int(bits) and bits > 0 and 32 % bits == 0 and is_int(group_size) and group_size > 0):
        errors.append(f"quantization: invalid bits {bits!r} or group_size {group_size!r}")
        return
    if config is not None:
        declared = config.get("quantization") or {}
        if declared.get("bits") != bits or declared.get("group_size") != group_size:
            errors.append(f"quantization: config.json declares {declared}, manifest records bits {bits} group_size {group_size}")
    tensors = {name: info for header in headers.values() for name, info in header["tensors"].items()}
    if not tensors:
        return
    claimed = []
    for item in quantization.get("quantized", []):
        claimed.extend(check_quantized(item, tensors, quantization, bits, group_size, errors))
    unquantized = quantization.get("unquantized", [])
    claimed.extend(unquantized)
    for name in unquantized:
        if name not in tensors:
            errors.append(f"quantization: unquantized tensor {name!r} absent from headers")
    for duplicate in sorted((name for name, count in collections.Counter(claimed).items() if count > 1), key=str):
        errors.append(f"quantization: tensor {duplicate!r} is classified more than once")
    for unclassified in sorted(set(tensors) - set(claimed)):
        errors.append(f"quantization: tensor {unclassified} is neither quantized nor unquantized")


def check_quantized(item, tensors, quantization, bits, group_size, errors):
    logical = item.get("logical")
    names = [item.get(role) for role in ("weight", "scales", "biases")]
    if any(name not in tensors for name in names):
        errors.append(f"quantization: {logical!r} references tensors absent from headers: {names}")
        return [name for name in names if name is not None]
    weight, scales, biases = (tensors[name] for name in names)
    logical_shape = item.get("logical_shape")
    if not (is_shape(logical_shape) and len(logical_shape) == 2):
        errors.append(f"quantization: {logical} has invalid logical_shape {logical_shape!r}")
        return names
    rows, columns = logical_shape
    values_per_word = 32 // bits
    if columns % group_size != 0 or columns % values_per_word != 0:
        errors.append(f"quantization: {logical} input dimension {columns} is not divisible by group {group_size}")
        return names
    expectations = [
        (weight, quantization.get("pack_dtype"), [rows, columns // values_per_word], "weight"),
        (scales, quantization.get("scales_dtype"), [rows, columns // group_size], "scales"),
        (biases, quantization.get("biases_dtype"), [rows, columns // group_size], "biases"),
    ]
    for info, dtype, shape, role in expectations:
        if info["dtype"] != dtype or info["shape"] != shape:
            errors.append(f"quantization: {logical} {role} is {info['dtype']}{info['shape']}, expected {dtype}{shape}")
    return names


def check_tokenizer(snapshot, manifest, verified, errors):
    tokenizer = manifest["tokenizer"]
    tokenizer_path = PurePosixPath("tokenizer.json")
    config_path = PurePosixPath("tokenizer_config.json")
    content = load_snapshot_json(snapshot, tokenizer_path, errors) if tokenizer_path in verified else None
    if isinstance(content, dict):
        added = {token.get("content"): token.get("id") for token in content.get("added_tokens", []) if isinstance(token, dict)}
        for token in tokenizer.get("special_tokens", []):
            if added.get(token.get("token")) != token.get("id"):
                errors.append(f"tokenizer.json: {token.get('token')!r} has id {added.get(token.get('token'))!r}, manifest expects {token.get('id')!r}")
    config = load_snapshot_json(snapshot, config_path, errors) if config_path in verified else None
    if isinstance(config, dict):
        template = config.get("chat_template")
        expected = (tokenizer.get("chat_template") or {}).get("sha256")
        actual = hashlib.sha256(template.encode("utf-8")).hexdigest() if isinstance(template, str) else None
        if actual != expected:
            errors.append(f"tokenizer_config.json: chat_template sha256 {actual}, manifest expects {expected!r}")
        for role in ("bos", "eos"):
            expected_token = (tokenizer.get(role) or {}).get("token")
            if config.get(f"{role}_token") != expected_token:
                errors.append(f"tokenizer_config.json: {role}_token {config.get(f'{role}_token')!r}, manifest expects {expected_token!r}")


def verify(manifest, snapshot):
    validate_schema(manifest)
    errors = []
    check_revision(snapshot, manifest, errors)
    verified = check_files(snapshot, manifest, errors)
    headers = check_safetensors(snapshot, manifest, verified, errors)
    check_index(snapshot, manifest, headers, verified, errors)
    config = check_architecture(snapshot, manifest, verified, errors) if manifest["architecture"] else None
    check_quantization(manifest, headers, config, errors)
    check_tokenizer(snapshot, manifest, verified, errors)
    tensor_count = sum(len(header["tensors"]) for header in headers.values())
    return errors, {"files": len(verified), "tensors": tensor_count}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify a pinned model snapshot against its manifest without downloads or writes."
    )
    parser.add_argument("--manifest", required=True, help="manifest.json describing the pinned snapshot")
    parser.add_argument("--snapshot", required=True, help="directory holding the pinned snapshot files")
    arguments = parser.parse_args(argv)
    snapshot = Path(arguments.snapshot)
    if not snapshot.is_dir():
        print(f"error: snapshot {snapshot} is not a directory", file=sys.stderr)
        return 2
    try:
        manifest = load_json_file(arguments.manifest)
        errors, summary = verify(manifest, snapshot)
    except (OSError, ValueError) as error:
        print(f"error: cannot read manifest {arguments.manifest}: {error}", file=sys.stderr)
        return 2
    except ManifestError as error:
        print(f"error: invalid manifest: {error}", file=sys.stderr)
        return 2
    except (AttributeError, KeyError, TypeError) as error:
        print(f"error: invalid manifest structure: {error!r}", file=sys.stderr)
        return 2
    if errors:
        for message in errors:
            print(f"mismatch: {message}", file=sys.stderr)
        print(f"FAILED: {len(errors)} mismatch(es) for {manifest['model']['repository']}@{manifest['model']['revision']}", file=sys.stderr)
        return 1
    print(
        f"OK: {manifest['model']['repository']}@{manifest['model']['revision']}: "
        f"{summary['files']} files hashed, {summary['tensors']} tensors verified"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
