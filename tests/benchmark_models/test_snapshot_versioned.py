"""Tier-3 snapshot_id upgrade to versioned form.

Pre-fix: ``cli.py`` always called ``build_paths_only_snapshot_id()``.
``build_versioned_snapshot_id`` and ``upgrade_to_versioned`` existed
in ``benchmark_models.snapshot`` but were never called from the
runtime — the documented tier-3 form
``<paths_hash>@<delta_versions_hash>`` was aspirational, not real.

These tests pin the snapshot id math (paths_only + versioned forms
and the ``upgrade_to_versioned`` glue). The actual Delta-version
lookup lives on ``TripleBridge.get_delta_versions`` (proxied through
``EntityService``); its behaviour is covered by
``test_kg_tools_redesign.py::test_entity_service_has_get_delta_versions``
plus the live SLURM smoke runs.

End-to-end CLI integration tested via the runner-side
``SmolagentsModel`` constructor (already covered in
test_tier3_concurrency).
"""

from __future__ import annotations

import re

import pytest

_VERSIONED_RE = re.compile(r"^[0-9a-f]{12}@[0-9a-f]{12}$")


def test_versioned_snapshot_id_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """``build_versioned_snapshot_id`` returns ``<12hex>@<12hex>``."""
    from benchmark_models.snapshot import build_versioned_snapshot_id

    sid = build_versioned_snapshot_id(
        {"kwg": 7, "osmh3kg": 12, "worldkg": 3},
        env={"WORLDKG_NODES_PATH": "/data/worldkg/", "OSMH3KG_NODES_PATH": "/data/osmh3kg/"},
    )
    assert _VERSIONED_RE.match(sid) is not None


def test_upgrade_to_versioned_matches_paths_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``upgrade_to_versioned`` keeps the original paths-hash prefix
    intact — the versions-hash is appended after the ``@`` delimiter."""
    from benchmark_models.snapshot import (
        build_paths_only_snapshot_id,
        upgrade_to_versioned,
    )

    env = {
        "WORLDKG_NODES_PATH": "/data/worldkg/",
        "OSMH3KG_NODES_PATH": "/data/osmh3kg/",
    }
    monkeypatch.setattr("os.environ", env)
    paths_only = build_paths_only_snapshot_id(env)

    class _StubBridge:
        def get_delta_versions(self) -> dict[str, int]:
            return {"kwg": 1}

    versioned = upgrade_to_versioned(paths_only, _StubBridge(), env=env)
    assert versioned.startswith(paths_only + "@")
    assert _VERSIONED_RE.match(versioned) is not None


def test_versioned_snapshot_differs_across_delta_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two versioned snapshots with different Delta version numbers
    produce different hashes — surfaces data drift via the snapshot
    Id alone (resume-warning consumes this)."""
    from benchmark_models.snapshot import build_versioned_snapshot_id

    env = {"WORLDKG_NODES_PATH": "/data/worldkg/"}
    sid_v1 = build_versioned_snapshot_id({"kwg": 1}, env=env)
    sid_v2 = build_versioned_snapshot_id({"kwg": 2}, env=env)
    assert sid_v1 != sid_v2
    # But the paths-hash prefix is identical.
    assert sid_v1.split("@")[0] == sid_v2.split("@")[0]


def test_cli_mlflow_uses_upgraded_snapshot_id() -> None:
    """R4-1: ``cli.py`` must pass the local (already-upgraded)
    ``snapshot_id`` to MLflow, not re-call
    ``build_paths_only_snapshot_id()``. Pre-fix the MLflow run carried
    the paths-only id while every prediction in the JSONL carried the
    versioned id — same run, two different provenance answers.

    Lightweight check: grep the CLI source for any remaining
    ``snapshot_id=build_paths_only_snapshot_id()`` callsite. The
    hoisted-init at the top of ``run()`` is fine; the issue is only
    the duplicate at the log_run call.
    """
    from pathlib import Path

    cli_source = (
        Path(__file__).resolve().parents[2]
        / "benchmark_models" / "cli.py"
    ).read_text()
    # The log_run() invocation must read ``snapshot_id=snapshot_id``
    # (the local variable), not ``snapshot_id=build_paths_only_snapshot_id()``.
    assert "snapshot_id=build_paths_only_snapshot_id()" not in cli_source, (
        "cli.py still recomputes the paths-only snapshot id at the "
        "MLflow log site — should pass the upgraded local "
        "``snapshot_id`` variable instead."
    )
