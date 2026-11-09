from __future__ import annotations

import json
import os
from html.parser import HTMLParser
from pathlib import Path

import pytest

from src.viz import DATA_PLACEHOLDER, assert_safe_output_dir, write_viz_bundle


_TEMPLATE = (
    "<!doctype html><title>t</title>"
    '<script id="viz-data" type="application/json">__VIZ_DATA__</script>'
    "<script>window.after = true;</script>"
)
_HOSTILE = {
    "schema_version": 1,
    "episodes": [
        {
            "record": {
                "hypothesis": "</script><img src=x onerror=alert(1)>",
                "rationale": (
                    "<!-- <script> & </SCRIPT > \u2028 \u2029 \u00e9 __VIZ_DATA__"
                ),
            },
            "score": {"stage2_gain": 0.125, "signal_pass": True},
        }
    ],
}


class _ScriptCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[tuple[str | None, str]] = []
        self._current: str | None = None
        self._inside = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            self._inside = True
            self._current = dict(attrs).get("id")
            self.scripts.append((self._current, ""))

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._inside = False

    def handle_data(self, data: str) -> None:
        if self._inside:
            script_id, text = self.scripts[-1]
            self.scripts[-1] = (script_id, text + data)


def _template(tmp_path: Path, text: str = _TEMPLATE) -> Path:
    path = tmp_path / "template.html"
    path.write_text(text, encoding="utf-8")
    return path


def test_embedded_document_cannot_leave_its_script_and_parses_back_identically(
    tmp_path,
) -> None:
    index = write_viz_bundle(
        _HOSTILE, tmp_path / "out", template_path=_template(tmp_path)
    )
    page = index.read_text(encoding="utf-8")
    parser = _ScriptCollector()
    parser.feed(page)

    assert index == tmp_path / "out" / "index.html"
    embedded = parser.scripts[0][1]
    assert [script_id for script_id, _ in parser.scripts] == ["viz-data", None]
    assert json.loads(embedded) == _HOSTILE
    assert not {"<", ">", "&", "\u2028", "\u2029"} & set(embedded)
    assert "<img" not in page
    assert "\u2028" not in page and "\u2029" not in page
    assert json.loads((tmp_path / "out" / "viz.json").read_text()) == _HOSTILE


def test_viz_json_is_compact_and_key_sorted(tmp_path) -> None:
    write_viz_bundle(
        {"b": [1, 2], "a": {"d": None, "c": "x"}},
        tmp_path / "out",
        template_path=_template(tmp_path),
    )

    assert (tmp_path / "out" / "viz.json").read_text() == (
        '{"a": {"c": "x", "d": null}, "b": [1, 2]}\n'
    )


@pytest.mark.parametrize(
    "template",
    ["<script>no placeholder</script>", f"{DATA_PLACEHOLDER}{DATA_PLACEHOLDER}"],
)
def test_template_needs_exactly_one_placeholder(tmp_path, template: str) -> None:
    with pytest.raises(ValueError, match="exactly once"):
        write_viz_bundle(
            _HOSTILE, tmp_path / "out", template_path=_template(tmp_path, template)
        )

    assert not (tmp_path / "out").exists()


def test_non_finite_numbers_are_refused_before_anything_is_written(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_viz_bundle(
            {"gain": float("nan")},
            tmp_path / "out",
            template_path=_template(tmp_path),
        )

    assert not (tmp_path / "out").exists()


def test_rewriting_a_bundle_replaces_both_files(tmp_path) -> None:
    template = _template(tmp_path)
    write_viz_bundle({"revision": 1}, tmp_path / "out", template_path=template)
    index = write_viz_bundle({"revision": 2}, tmp_path / "out", template_path=template)
    parser = _ScriptCollector()
    parser.feed(index.read_text(encoding="utf-8"))

    assert json.loads(parser.scripts[0][1]) == {"revision": 2}
    assert json.loads((tmp_path / "out" / "viz.json").read_text()) == {"revision": 2}
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "index.html",
        "viz.json",
    ]


@pytest.mark.parametrize(
    "relative",
    [
        "compose",
        "compose/nested/viz",
        "cache/slot_0",
        "linked/viz",
        "a/public-evidence/b",
    ],
)
def test_output_inside_agent_reachable_locations_is_refused(
    tmp_path, relative: str
) -> None:
    compose = tmp_path / "compose"
    cache = tmp_path / "cache"
    compose.mkdir()
    cache.mkdir()
    os.symlink(compose, tmp_path / "linked")

    with pytest.raises(ValueError, match="refusing"):
        assert_safe_output_dir(tmp_path / relative, forbidden_roots=[compose, cache])


@pytest.mark.parametrize("path", ["/data", "/data/viz", "/data/../data/run"])
def test_data_mount_is_always_refused(path: str) -> None:
    with pytest.raises(ValueError, match="refusing"):
        assert_safe_output_dir(path, forbidden_roots=[])


def test_sibling_of_a_forbidden_root_is_allowed(tmp_path) -> None:
    (tmp_path / "compose").mkdir()

    resolved = assert_safe_output_dir(
        tmp_path / "compose-viz" / "out", forbidden_roots=[tmp_path / "compose", ""]
    )

    assert resolved == (tmp_path / "compose-viz" / "out").resolve()


def test_shipped_template_embeds_the_document_in_the_viz_data_script(tmp_path) -> None:
    index = write_viz_bundle(_HOSTILE, tmp_path / "out")
    parser = _ScriptCollector()
    parser.feed(index.read_text(encoding="utf-8"))

    embedded = [text for script_id, text in parser.scripts if script_id == "viz-data"]
    assert len(embedded) == 1
    assert json.loads(embedded[0]) == _HOSTILE
