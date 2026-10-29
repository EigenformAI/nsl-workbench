#!/usr/bin/env python3
"""Prepare the pinned materials training snapshot and its initial f16 GGUF.

Run from the source-checkout root after installing the locked Python environment::

    python scripts/prepare_materials_model.py \
        --model-id Qwen/Qwen3-4B-Instruct-2507 \
        --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
        --output-root .model-cache/materials-base --quantization f16

Conversion runs on CPU, directly from the unchanged Hub snapshot. No model is
loaded for training, no optimizer is created, and no Python packages are installed.
Existing initial.gguf or provenance.json files are refused before downloading.
Completed preparation requires both files and a matching GGUF SHA-256 recorded
in provenance.json. Publishing the pair is not atomic across a crash or kill:
a partial pair is explicitly refused on retry and requires a fresh output root
or manual review. Use one bootstrap writer per output root. Downloads resume
through Hub-local metadata when no published output exists.

The converter timeout defaults to 600 seconds; --conversion-timeout-seconds 0
disables that timeout. The GGUF reader timeout remains 120 seconds, and the
toolchain HTTP socket timeout remains 60 seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
LLAMA_TAG = "b8606"
LLAMA_REVISION = "825eb91a66f824242761014c0cd8aea9e9191772"
LLAMA_ARCHIVE_URL = (
    f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{LLAMA_REVISION}"
)
LLAMA_ARCHIVE_SHA256 = (
    "ae0ce319332baca6518cb4b2972d3c1f2b26a8914f42ef0554e4701ed56c55cb"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(root: Path) -> list[dict[str, Any]]:
    """Hash source files, excluding only generated Hub metadata and bytecode."""
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".cache" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise RuntimeError(f"Unexpected source symlink: {relative}")
        if path.is_file():
            files.append(
                {
                    "path": relative.as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return files


def verify_snapshot(snapshot: Path, info: Any) -> list[dict[str, Any]]:
    """Reject incomplete, extra, or locally modified files at a pinned revision."""
    files = file_manifest(snapshot)
    by_name = {entry["path"]: entry for entry in files}
    expected = {entry.rfilename: entry for entry in info.siblings}
    if by_name.keys() != expected.keys():
        raise RuntimeError(
            "Transformers snapshot file set differs from pinned Hub revision"
        )
    for name, remote in expected.items():
        local = by_name[name]
        if local["size_bytes"] != remote.size:
            raise RuntimeError(f"Snapshot size mismatch: {name}")
        if remote.lfs is not None:
            matches = local["sha256"] == remote.lfs.sha256
        else:
            content = (snapshot / name).read_bytes()
            git_digest = hashlib.sha1(
                f"blob {len(content)}\0".encode() + content, usedforsecurity=False
            ).hexdigest()
            matches = git_digest == remote.blob_id
        if not matches:
            raise RuntimeError(f"Snapshot checksum mismatch: {name}")
    return files


def prepare_toolchain(root: Path, *, archive: Path | None = None) -> Path:
    """Fetch a checksum-pinned source archive; never install or build packages."""
    root.mkdir(parents=True, exist_ok=True)
    if archive is None:
        archive = root / f"llama.cpp-{LLAMA_REVISION}.tar.gz"
        if not archive.exists():
            with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
                temporary_archive = Path(handle.name)
                try:
                    with urllib.request.urlopen(
                        LLAMA_ARCHIVE_URL, timeout=60
                    ) as response:
                        shutil.copyfileobj(response, handle)
                except BaseException:
                    temporary_archive.unlink(missing_ok=True)
                    raise
            try:
                if sha256_file(temporary_archive) != LLAMA_ARCHIVE_SHA256:
                    raise RuntimeError("llama.cpp source archive checksum mismatch")
                temporary_archive.replace(archive)
            finally:
                temporary_archive.unlink(missing_ok=True)
    if sha256_file(archive) != LLAMA_ARCHIVE_SHA256:
        raise RuntimeError("llama.cpp source archive checksum mismatch")

    source = root / f"llama.cpp-{LLAMA_REVISION}"
    with tempfile.TemporaryDirectory(prefix=".llama-source-", dir=root) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, "r:gz") as bundle:
            if any(member.issym() or member.islnk() for member in bundle.getmembers()):
                raise RuntimeError("Unexpected link in pinned llama.cpp archive")
            bundle.extractall(staging, filter="data")
        extracted = staging / source.name
        if not (extracted / "convert_hf_to_gguf.py").is_file():
            raise RuntimeError("Pinned llama.cpp archive contains no converter")
        if source.exists():
            if file_manifest(source) != file_manifest(extracted):
                raise RuntimeError(
                    "Existing llama.cpp source differs from pinned archive"
                )
        else:
            extracted.replace(source)
    return source


def _converter_environment() -> dict[str, str]:
    environment = dict(os.environ)
    # Force the converter and reader to use the archive's gguf-py, not an
    # unrelated installed package. The converter receives local files only.
    environment.pop("NO_LOCAL_GGUF", None)
    environment.pop("PYTHONPATH", None)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    return environment


def _conversion_timeout(seconds: int) -> int | None:
    if seconds < 0:
        raise ValueError("conversion_timeout_seconds must be non-negative (0 disables)")
    return None if seconds == 0 else seconds


def inspect_gguf(path: Path, toolchain: Path) -> dict[str, Any]:
    """Parse every tensor with the pinned reader (read-only memory mapping)."""
    code = """
import json, sys
sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader
reader = GGUFReader(sys.argv[2])
def value(key):
    field = reader.get_field(key)
    return field.contents() if field is not None else None
if not reader.tensors:
    raise ValueError('GGUF contains no model tensors')
if value('general.file_type') != 1:
    raise ValueError('GGUF is not an f16 model')
if value('split.count') not in (None, 1):
    raise ValueError('Sharded GGUF cannot be published as one model file')
if not value('general.architecture'):
    raise ValueError('GGUF has no model architecture')
if not value('tokenizer.chat_template'):
    raise ValueError('GGUF has no chat template')
if any(t.n_bytes <= 0 or t.data_offset + t.n_bytes > reader.data.size for t in reader.tensors):
    raise ValueError('GGUF tensor data is missing or truncated')
print(json.dumps({'architecture': value('general.architecture'),
                  'file_type': value('general.file_type'),
                  'tensor_count': len(reader.tensors)}))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, str(toolchain / "gguf-py"), str(path)],
            env=_converter_environment(),
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return json.loads(result.stdout)
    except (subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError(
            "Invalid GGUF: pinned reader could not validate model"
        ) from exc


def convert_to_gguf(
    snapshot: Path,
    destination: Path,
    toolchain: Path,
    *,
    model_name: str = MODEL_ID.split("/")[-1],
    conversion_timeout_seconds: int = 600,
) -> Path:
    """Publish one validated converter output, never an arbitrary first shard."""
    timeout = _conversion_timeout(conversion_timeout_seconds)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".gguf-preparation-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        command = [
            sys.executable,
            str(toolchain / "convert_hf_to_gguf.py"),
            "--outtype",
            "f16",
            "--outfile",
            str(staging / "model.gguf"),
            "--model-name",
            model_name,
            str(snapshot),
        ]
        try:
            subprocess.run(
                command, env=_converter_environment(), check=True, timeout=timeout
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"GGUF converter exited with status {exc.returncode}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"GGUF converter exceeded its {conversion_timeout_seconds}-second timeout"
            ) from exc

        outputs = sorted(staging.rglob("*.gguf"))
        if len(outputs) != 1:
            raise RuntimeError(
                f"Expected exactly one GGUF output; found {len(outputs)}. "
                "Multiple outputs/shards require explicit handling, not first-file selection."
            )
        actual = outputs[0]
        if actual.is_symlink() or not actual.is_file():
            raise RuntimeError("GGUF output is not a regular model file")
        with actual.open("rb") as handle:
            header = handle.read(24)
        if len(header) != 24:
            raise RuntimeError("Invalid or empty GGUF output")
        magic, version, tensors, metadata = struct.unpack("<4sIQQ", header)
        if magic != b"GGUF" or version != 3 or tensors == 0 or metadata == 0:
            raise RuntimeError("Invalid GGUF header or model contains no tensors")
        inspect_gguf(actual, toolchain)
        # A same-filesystem hard link refuses an existing name without copying weights.
        os.link(actual, destination)
    return destination


def model_provenance(snapshot: Path, output: Path, toolchain: Path) -> dict[str, Any]:
    """Return public upstream identities/checksums, without local paths or env."""
    packages = {}
    for name in (
        "huggingface-hub",
        "torch",
        "transformers",
        "numpy",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "tokenizers",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": 1,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "source": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
            "snapshot_directory": "transformers",
            "files": file_manifest(snapshot),
        },
        "toolchain": {
            "source": "https://github.com/ggml-org/llama.cpp",
            "tag": LLAMA_TAG,
            "revision": LLAMA_REVISION,
            "archive_url": LLAMA_ARCHIVE_URL,
            "archive_sha256": LLAMA_ARCHIVE_SHA256,
            "converter": "convert_hf_to_gguf.py",
            "converter_sha256": sha256_file(toolchain / "convert_hf_to_gguf.py"),
            "gguf_python_files": file_manifest(toolchain / "gguf-py"),
            "python_version": sys.version.split()[0],
            "python_packages": packages,
        },
        "conversion": {
            "quantization": "f16",
            "training_performed": False,
            "arguments": [
                "convert_hf_to_gguf.py",
                "--outtype",
                "f16",
                "--outfile",
                "initial.gguf",
                "--model-name",
                MODEL_ID.split("/")[-1],
                "transformers",
            ],
        },
        "output": {
            "path": "initial.gguf",
            "size_bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_prepared_pair(
    prepared: Path, provenance: Path, output: Path, provenance_output: Path
) -> None:
    for path in (prepared, provenance):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    prepared_status = prepared.stat()
    os.link(prepared, output)
    try:
        os.link(provenance, provenance_output)
    except OSError:
        # Roll back only our model inode, not a replacement at the same name.
        try:
            if os.path.samestat(output.lstat(), prepared_status):
                output.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(output.parent)
        raise
    _fsync_directory(output.parent)


def prepare_model(output_root: Path, *, conversion_timeout_seconds: int = 600) -> Path:
    _conversion_timeout(conversion_timeout_seconds)
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "initial.gguf"
    provenance_output = output_root / "provenance.json"
    for path in (output, provenance_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    from huggingface_hub import HfApi, snapshot_download

    # Do not silently operate on a gated, private, or unexpectedly resolved model.
    info = HfApi().model_info(
        MODEL_ID, revision=MODEL_REVISION, token=False, files_metadata=True
    )
    if info.sha != MODEL_REVISION or info.private or info.gated:
        raise RuntimeError("Selected model is not the pinned public, ungated snapshot")
    snapshot = output_root / "transformers"
    print(f"Downloading {MODEL_ID} at {MODEL_REVISION}", flush=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(snapshot),
        token=False,
        max_workers=4,
    )
    # local_dir avoids a second full Hub cache. Hashes before/after conversion
    # attest that the checkpoint and tokenizer were not rewritten by an exporter.
    original_files = verify_snapshot(snapshot, info)
    configuration = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    expected_architecture = configuration.get("model_type")
    if not isinstance(expected_architecture, str) or not expected_architecture:
        raise RuntimeError("Transformers config contains no model architecture")
    toolchain = prepare_toolchain(output_root / "toolchain")
    with tempfile.TemporaryDirectory(
        prefix=".materials-bootstrap-", dir=output_root
    ) as temporary:
        prepared = Path(temporary) / "initial.gguf"
        convert_to_gguf(
            snapshot,
            prepared,
            toolchain,
            conversion_timeout_seconds=conversion_timeout_seconds,
        )
        provenance = model_provenance(snapshot, prepared, toolchain)
        if provenance["model"]["files"] != original_files:
            raise RuntimeError("Transformers snapshot changed during GGUF conversion")
        report = inspect_gguf(prepared, toolchain)
        if report["architecture"] != expected_architecture:
            raise RuntimeError(
                "Prepared GGUF architecture differs from Transformers config"
            )
        provenance["output"]["inspection"] = report
        provenance_path = Path(temporary) / "provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        _publish_prepared_pair(prepared, provenance_path, output, provenance_output)
    print("Prepared initial.gguf and provenance.json", flush=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=[MODEL_ID], default=MODEL_ID)
    parser.add_argument("--revision", choices=[MODEL_REVISION], default=MODEL_REVISION)
    parser.add_argument(
        "--output-root", type=Path, default=Path(".model-cache/materials-base")
    )
    parser.add_argument("--quantization", choices=["f16"], default="f16")
    parser.add_argument(
        "--conversion-timeout-seconds",
        type=int,
        default=600,
        help="Converter timeout in seconds (default: 600; 0 disables)",
    )
    args = parser.parse_args(argv)
    if args.conversion_timeout_seconds < 0:
        parser.error("--conversion-timeout-seconds must be non-negative (0 disables)")
    try:
        prepare_model(
            args.output_root, conversion_timeout_seconds=args.conversion_timeout_seconds
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Model preparation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
