"""Focused bootstrap checks; no model downloads or GPU are needed."""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "prepare_materials_model",
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_materials_model.py",
)
assert SPEC and SPEC.loader
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def gguf_bytes() -> bytes:
    # Structural parsing is exercised separately with the real pinned reader.
    return struct.pack("<4sIQQ", b"GGUF", 3, 1, 1) + b"tensor data"


@pytest.fixture
def conversion(tmp_path, monkeypatch):
    snapshot = tmp_path / "transformers"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type": "qwen3"}')
    (snapshot / "tokenizer_config.json").write_text('{"chat_template": "original"}')
    toolchain = tmp_path / "llama.cpp"
    toolchain.mkdir()
    (toolchain / "convert_hf_to_gguf.py").write_text("# converter")
    calls = []

    def inspect(path, source):
        return {"architecture": "qwen3", "file_type": 1, "tensor_count": 1}

    monkeypatch.setattr(bootstrap, "inspect_gguf", inspect)

    def run(command, **kwargs):
        calls.append(command)
        outfile = Path(command[command.index("--outfile") + 1])
        outfile.write_bytes(gguf_bytes())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    return snapshot, toolchain, tmp_path / "initial.gguf", calls


@pytest.fixture
def preparation(conversion, monkeypatch):
    snapshot, toolchain, _, _ = conversion
    info = SimpleNamespace(sha=bootstrap.MODEL_REVISION, private=False, gated=False)
    hub = SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=lambda *args, **kwargs: info),
        snapshot_download=lambda **kwargs: str(snapshot),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(bootstrap, "prepare_toolchain", lambda root: toolchain)
    monkeypatch.setattr(
        bootstrap, "verify_snapshot", lambda path, info: bootstrap.file_manifest(path)
    )
    return conversion


def test_normalizes_exact_single_output_without_rewriting_snapshot(conversion):
    snapshot, toolchain, destination, calls = conversion
    before = (snapshot / "tokenizer_config.json").read_bytes()
    assert bootstrap.convert_to_gguf(snapshot, destination, toolchain) == destination
    assert destination.read_bytes() == gguf_bytes()
    assert (snapshot / "tokenizer_config.json").read_bytes() == before
    command = calls[0]
    assert command[-1] == str(snapshot)
    assert command[command.index("--outtype") + 1] == "f16"
    assert command[1] == str(toolchain / "convert_hf_to_gguf.py")


@pytest.mark.parametrize("outputs", [[], ["first.gguf", "second.gguf"]])
def test_missing_or_multiple_outputs_fail_without_selecting_first(
    conversion, monkeypatch, outputs
):
    snapshot, toolchain, destination, _ = conversion

    def run(command, **kwargs):
        parent = Path(command[command.index("--outfile") + 1]).parent
        for name in outputs:
            (parent / name).write_bytes(gguf_bytes())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="exactly one"):
        bootstrap.convert_to_gguf(snapshot, destination, toolchain)
    assert not destination.exists()


def test_unexpected_single_filename_is_normalized(conversion, monkeypatch):
    snapshot, toolchain, destination, _ = conversion

    def run(command, **kwargs):
        outfile = Path(command[command.index("--outfile") + 1])
        outfile.with_name("actual.F16.gguf").write_bytes(gguf_bytes())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    assert bootstrap.convert_to_gguf(snapshot, destination, toolchain) == destination
    assert destination.read_bytes() == gguf_bytes()


@pytest.mark.parametrize(
    "payload",
    [b"", b"not a GGUF model", b"GGUF", struct.pack("<4sIQQ", b"GGUF", 3, 0, 1)],
)
def test_empty_fake_or_vocabulary_only_output_rejected(
    conversion, monkeypatch, payload
):
    snapshot, toolchain, destination, _ = conversion

    def run(command, **kwargs):
        Path(command[command.index("--outfile") + 1]).write_bytes(payload)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="GGUF"):
        bootstrap.convert_to_gguf(snapshot, destination, toolchain)
    assert not destination.exists()


def test_pinned_reader_failure_does_not_publish_output(conversion, monkeypatch):
    snapshot, toolchain, destination, _ = conversion

    def inspect(path, source):
        raise RuntimeError("Invalid GGUF tensor data")

    monkeypatch.setattr(bootstrap, "inspect_gguf", inspect)
    with pytest.raises(RuntimeError, match="tensor data"):
        bootstrap.convert_to_gguf(snapshot, destination, toolchain)
    assert not destination.exists()


def test_converter_failure_does_not_publish_partial_file(conversion, monkeypatch):
    snapshot, toolchain, destination, _ = conversion

    def run(command, **kwargs):
        Path(command[command.index("--outfile") + 1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="converter.*2"):
        bootstrap.convert_to_gguf(snapshot, destination, toolchain)
    assert not destination.exists()


def test_existing_destination_is_not_silently_overwritten(conversion):
    snapshot, toolchain, destination, calls = conversion
    destination.write_bytes(b"existing artifact")
    with pytest.raises(FileExistsError):
        bootstrap.convert_to_gguf(snapshot, destination, toolchain)
    assert destination.read_bytes() == b"existing artifact"
    assert not calls


def test_provenance_uses_only_public_sources_and_relative_paths(tmp_path):
    snapshot = tmp_path / "transformers"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    # Hub-local metadata must not become part of public model provenance.
    metadata = snapshot / ".cache" / "huggingface"
    metadata.mkdir(parents=True)
    (metadata / "download.metadata").write_text("private-local-path")
    output = tmp_path / "initial.gguf"
    output.write_bytes(gguf_bytes())
    toolchain = tmp_path / "toolchain"
    toolchain.mkdir()
    (toolchain / "convert_hf_to_gguf.py").write_text("# pinned")
    (toolchain / "gguf-py").mkdir()
    (toolchain / "gguf-py" / "reader.py").write_text("# pinned reader")
    payload = bootstrap.model_provenance(snapshot, output, toolchain)
    text = json.dumps(payload)
    assert str(tmp_path) not in text
    assert "private-local-path" not in text
    assert payload["model"]["id"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert payload["model"]["revision"] == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert payload["output"]["path"] == "initial.gguf"
    assert len(payload["output"]["sha256"]) == 64
    assert (
        payload["toolchain"]["revision"] == "825eb91a66f824242761014c0cd8aea9e9191772"
    )
    assert [f["path"] for f in payload["model"]["files"]] == ["config.json"]


def test_toolchain_checksum_mismatch_fails_before_extraction(tmp_path):
    archive = tmp_path / "llama.cpp.tar.gz"
    archive.write_bytes(b"not the pinned source archive")
    with pytest.raises(RuntimeError, match="checksum"):
        bootstrap.prepare_toolchain(tmp_path, archive=archive)
    assert not (tmp_path / f"llama.cpp-{bootstrap.LLAMA_REVISION}").exists()


def test_snapshot_requires_original_hub_checksums(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"locally changed weights")
    info = SimpleNamespace(
        siblings=[
            SimpleNamespace(
                rfilename=weights.name,
                size=weights.stat().st_size,
                lfs=SimpleNamespace(sha256="0" * 64),
            )
        ]
    )
    with pytest.raises(RuntimeError, match="checksum"):
        bootstrap.verify_snapshot(tmp_path, info)


def test_preparation_publishes_model_and_matching_provenance(preparation):
    snapshot, _, destination, _ = preparation
    assert bootstrap.prepare_model(snapshot.parent) == destination
    assert destination.read_bytes() == gguf_bytes()
    payload = json.loads((snapshot.parent / "provenance.json").read_text())
    assert payload["output"]["sha256"] == bootstrap.sha256_file(destination)
    assert payload["output"]["inspection"]["tensor_count"] == 1


def test_changed_snapshot_cannot_publish_initial_model(preparation, monkeypatch):
    snapshot, _, destination, _ = preparation

    def convert(source, output, toolchain, **kwargs):
        output.write_bytes(gguf_bytes())
        (source / "tokenizer_config.json").write_text('{"chat_template": "changed"}')
        return output

    monkeypatch.setattr(bootstrap, "convert_to_gguf", convert)
    with pytest.raises(RuntimeError, match="snapshot changed"):
        bootstrap.prepare_model(snapshot.parent)
    assert not destination.exists()
    assert not (snapshot.parent / "provenance.json").exists()


def test_mismatched_architecture_cannot_publish_initial_model(preparation, monkeypatch):
    snapshot, _, destination, _ = preparation
    monkeypatch.setattr(
        bootstrap,
        "inspect_gguf",
        lambda path, source: {
            "architecture": "qwen2",
            "file_type": 1,
            "tensor_count": 1,
        },
    )
    with pytest.raises(RuntimeError, match="architecture"):
        bootstrap.prepare_model(snapshot.parent)
    assert not destination.exists()
    assert not (snapshot.parent / "provenance.json").exists()


def test_converter_can_label_another_model_without_candidate_mislabeling(conversion):
    snapshot, toolchain, destination, calls = conversion
    bootstrap.convert_to_gguf(
        snapshot, destination, toolchain, model_name="Custom model"
    )
    command = calls[0]
    assert command[command.index("--model-name") + 1] == "Custom model"


@pytest.mark.parametrize(
    ("arguments", "timeout"),
    [
        ([], 600),
        (["--conversion-timeout-seconds", "13"], 13),
        (["--conversion-timeout-seconds", "0"], 0),
    ],
)
def test_cli_uses_canonical_root_and_forwards_timeout(monkeypatch, arguments, timeout):
    received = []

    def prepare(root, *, conversion_timeout_seconds=600):
        received.append((root, conversion_timeout_seconds))

    monkeypatch.setattr(bootstrap, "prepare_model", prepare)
    assert bootstrap.main(arguments) == 0
    assert received == [(Path(".model-cache/materials-base"), timeout)]


@pytest.mark.parametrize("symlink", [False, True])
def test_stale_provenance_refused_before_download(preparation, monkeypatch, symlink):
    snapshot, _, destination, _ = preparation
    provenance = snapshot.parent / "provenance.json"
    if symlink:
        provenance.symlink_to(snapshot.parent / "missing-provenance")
    else:
        provenance.write_bytes(b"existing provenance")

    def download(**kwargs):
        pytest.fail("Existing provenance must be rejected before download")

    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", download)
    with pytest.raises(FileExistsError, match="provenance.json"):
        bootstrap.prepare_model(snapshot.parent)
    assert not destination.exists()
    if symlink:
        assert provenance.is_symlink()
        assert provenance.readlink() == snapshot.parent / "missing-provenance"
    else:
        assert provenance.read_bytes() == b"existing provenance"


@pytest.mark.parametrize("foreign_replacement", [False, True])
def test_second_publish_failure_rolls_back_only_own_model(
    preparation, monkeypatch, foreign_replacement
):
    snapshot, _, destination, _ = preparation
    provenance = snapshot.parent / "provenance.json"
    real_link = bootstrap.os.link
    real_replace = Path.replace

    def fail_publish():
        if foreign_replacement:
            destination.unlink()
            destination.write_bytes(b"foreign replacement")
        raise OSError("second publication failed")

    def link(source, target, *args, **kwargs):
        if Path(target) == provenance:
            fail_publish()
        return real_link(source, target, *args, **kwargs)

    def replace(source, target):
        if Path(target) == provenance:
            fail_publish()
        return real_replace(source, target)

    monkeypatch.setattr(bootstrap.os, "link", link)
    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError, match="second publication failed"):
        bootstrap.prepare_model(snapshot.parent)
    if foreign_replacement:
        assert destination.read_bytes() == b"foreign replacement"
    else:
        assert not destination.exists()
    assert not provenance.exists()


def test_late_existing_provenance_not_overwritten(preparation, monkeypatch):
    snapshot, _, destination, _ = preparation
    provenance = snapshot.parent / "provenance.json"
    real_link = bootstrap.os.link

    def link(source, target, *args, **kwargs):
        result = real_link(source, target, *args, **kwargs)
        if Path(target) == destination:
            provenance.write_bytes(b"late existing provenance")
        return result

    monkeypatch.setattr(bootstrap.os, "link", link)
    with pytest.raises(FileExistsError):
        bootstrap.prepare_model(snapshot.parent)
    assert provenance.read_bytes() == b"late existing provenance"
    assert not destination.exists()


@pytest.mark.parametrize(("seconds", "expected"), [(13, 13), (0, None)])
def test_converter_timeout_is_configurable(conversion, monkeypatch, seconds, expected):
    snapshot, toolchain, destination, _ = conversion
    original_run = bootstrap.subprocess.run
    received = []

    def run(command, **kwargs):
        received.append(kwargs["timeout"])
        return original_run(command, **kwargs)

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    bootstrap.convert_to_gguf(
        snapshot, destination, toolchain, conversion_timeout_seconds=seconds
    )
    assert received == [expected]


def test_preparation_forwards_conversion_timeout(preparation, monkeypatch):
    snapshot, _, destination, _ = preparation
    received = []
    original_convert = bootstrap.convert_to_gguf

    def convert(source, output, toolchain, **kwargs):
        received.append(kwargs["conversion_timeout_seconds"])
        return original_convert(source, output, toolchain, **kwargs)

    monkeypatch.setattr(bootstrap, "convert_to_gguf", convert)
    assert (
        bootstrap.prepare_model(snapshot.parent, conversion_timeout_seconds=13)
        == destination
    )
    assert received == [13]


def test_negative_cli_timeout_is_usage_error(monkeypatch):
    def prepare(*args, **kwargs):
        pytest.fail("Negative timeout must be rejected before preparing a model")

    monkeypatch.setattr(bootstrap, "prepare_model", prepare)
    with pytest.raises(SystemExit) as error:
        bootstrap.main(["--conversion-timeout-seconds", "-1"])
    assert error.value.code == 2


@pytest.mark.parametrize("entry_point", ["prepare", "convert"])
def test_negative_api_timeout_is_rejected_before_effects(conversion, entry_point):
    snapshot, toolchain, destination, calls = conversion
    with pytest.raises(ValueError, match="non-negative"):
        if entry_point == "prepare":
            bootstrap.prepare_model(snapshot.parent, conversion_timeout_seconds=-1)
        else:
            bootstrap.convert_to_gguf(
                snapshot, destination, toolchain, conversion_timeout_seconds=-1
            )
    assert not destination.exists()
    assert not calls
