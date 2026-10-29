from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import yaml

from tasks.materials_adsorption.compute import (
    MAX_OUTPUT_BYTES,
    MAX_SNIPPET_BYTES,
    build_public_export,
    execute_public_python,
)
from tasks.materials_adsorption.dataset import load_dataset

from tests.materials_adsorption.test_task import _write_synthetic_dataset


def test_public_export_is_energy_invariant_and_exactly_allowlisted(tmp_path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_synthetic_dataset(first_path)
    payload = json.loads(first_path.read_text())
    energies = [record["energy"] for record in payload["records"]]
    for record, energy in zip(payload["records"], reversed(energies), strict=True):
        record["energy"] = energy
    second_path.write_text(json.dumps(payload), encoding="utf-8")
    first = load_dataset(first_path)
    second = load_dataset(second_path)
    evidence = [candidate.public_payload() for candidate in first.universe.candidates[:4]]

    first_dir = tmp_path / "export-a"
    second_dir = tmp_path / "export-b"
    build_public_export(first, evidence, "evidence_public", "Public contract", first_dir)
    build_public_export(second, evidence, "evidence_public", "Public contract", second_dir)

    for name in ("universe.csv", "evidence.csv", "README.md"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
        assert b"energy" not in (first_dir / name).read_bytes().lower()
    assert (first_dir / "universe.csv").read_text().splitlines()[0] == (
        "candidate_alias,coordination,favorability"
    )
    assert (first_dir / "evidence.csv").read_text().splitlines()[0] == (
        "evidence_alias,candidate_alias,system_alias,coordination,favorability"
    )


def test_public_python_enforces_input_output_and_timeout_bounds() -> None:
    container = MagicMock()

    oversized = execute_public_python(container, "x" * (MAX_SNIPPET_BYTES + 1))
    assert not oversized.success
    assert oversized.error == "snippet_too_large"
    container.exec_run.assert_not_called()

    container.exec_run.return_value = SimpleNamespace(
        exit_code=0,
        output=(b"x" * (MAX_OUTPUT_BYTES + 100)),
    )
    truncated = execute_public_python(container, "print('x')")
    assert truncated.success
    assert truncated.output["truncated"] is True
    assert truncated.output["stdout"].endswith("[output truncated]")

    def timeout(*_args, **_kwargs):
        raise TimeoutError("slow")

    timed_out = execute_public_python(container, "while True: pass", executor=timeout)
    assert not timed_out.success
    assert timed_out.error == "execution_timeout"
    assert timed_out.output["timed_out"] is True


def test_compute_sidecar_compose_is_networkless_read_only_and_separate() -> None:
    compose_path = (
        Path(__file__).parents[2]
        / "docker/materials-adsorption-compose/docker-compose.yml"
    )
    service = yaml.safe_load(compose_path.read_text())["services"]["compute-service"]

    assert service["network_mode"] == "none"
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["pids_limit"] == 64
    assert service["tmpfs"] == ["/tmp:rw,size=64m"]
    assert service["volumes"] == [
        {
            "type": "bind",
            "source": "./public-evidence",
            "target": "/data",
            "read_only": True,
        }
    ]
    assert "/work" not in json.dumps(service)
