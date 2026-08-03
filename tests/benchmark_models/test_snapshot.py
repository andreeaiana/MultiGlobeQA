"""Tests for ``benchmark_models.snapshot``.

Covers the snapshot_id construction logic for both forms (paths-only
and paths+versions), determinism, env-var sensitivity, and the
upgrade path from a paths-only id to a versioned one.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from benchmark_models.snapshot import (
    KG_PATH_ENV_VARS,
    SnapshotDriftError,
    build_paths_only_snapshot_id,
    build_versioned_snapshot_id,
    is_versioned,
    upgrade_to_versioned,
)

# ============================================================================
# Helpers
# ============================================================================


def _env(**kwargs: str) -> dict[str, str]:
    """Build a minimal env dict with all KG path vars set."""
    base = {k: f"/path/to/{k.lower()}" for k in KG_PATH_ENV_VARS}
    base.update(kwargs)
    return base


class _FakeBridge:
    def __init__(self, versions: Mapping[str, int]) -> None:
        self._versions = dict(versions)

    def get_delta_versions(self) -> Mapping[str, int]:
        return self._versions


# ============================================================================
# Paths-only form
# ============================================================================


class TestPathsOnlySnapshotId:
    def test_returns_12_hex_chars(self) -> None:
        sid = build_paths_only_snapshot_id(_env())
        assert len(sid) == 12
        assert all(c in "0123456789abcdef" for c in sid)

    def test_deterministic_for_same_env(self) -> None:
        env = _env()
        assert build_paths_only_snapshot_id(env) == build_paths_only_snapshot_id(env)

    def test_changes_when_a_path_changes(self) -> None:
        a = build_paths_only_snapshot_id(_env())
        b = build_paths_only_snapshot_id(
            _env(WORLDKG_TRIPLES_PATH="/different/path")
        )
        assert a != b

    def test_changes_when_a_path_is_unset(self) -> None:
        env_with_all = _env()
        env_missing = dict(env_with_all)
        del env_missing["KWG_TRIPLES_PATH"]
        assert build_paths_only_snapshot_id(env_with_all) != build_paths_only_snapshot_id(
            env_missing
        )

    def test_unrelated_env_vars_do_not_affect_hash(self) -> None:
        env = _env()
        env["UNRELATED_VAR"] = "should not influence the hash"
        env_clean = _env()
        assert build_paths_only_snapshot_id(env) == build_paths_only_snapshot_id(env_clean)

    def test_default_env_uses_os_environ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear all KG vars and set them to known values via monkeypatch.
        for k in KG_PATH_ENV_VARS:
            monkeypatch.setenv(k, f"/test/{k.lower()}")
        sid_default = build_paths_only_snapshot_id()
        sid_explicit = build_paths_only_snapshot_id(
            {k: f"/test/{k.lower()}" for k in KG_PATH_ENV_VARS}
        )
        assert sid_default == sid_explicit

    def test_is_not_versioned(self) -> None:
        sid = build_paths_only_snapshot_id(_env())
        assert "@" not in sid
        assert is_versioned(sid) is False


# ============================================================================
# Paths + versions form
# ============================================================================


class TestVersionedSnapshotId:
    def _versions(self) -> dict[str, int]:
        return {
            "WORLDKG_TRIPLES_PATH": 5,
            "WORLDKG_NODES_PATH": 5,
            "OSMH3KG_NODES_PATH": 12,
            "OSMH3KG_TRIPLES_PATH": 12,
            "KWG_ENTITIES_PATH": 3,
            "KWG_TRIPLES_PATH": 3,
        }

    def test_format_paths_at_versions(self) -> None:
        sid = build_versioned_snapshot_id(self._versions(), env=_env())
        # Format: "<12 hex>@<12 hex>"
        assert "@" in sid
        left, right = sid.split("@")
        assert len(left) == 12
        assert len(right) == 12
        assert all(c in "0123456789abcdef" for c in left + right)

    def test_starts_with_paths_only_hash(self) -> None:
        env = _env()
        paths_id = build_paths_only_snapshot_id(env)
        versioned = build_versioned_snapshot_id(self._versions(), env=env)
        assert versioned.startswith(paths_id + "@")

    def test_deterministic(self) -> None:
        env = _env()
        v = self._versions()
        assert build_versioned_snapshot_id(v, env=env) == build_versioned_snapshot_id(
            v, env=env
        )

    def test_changes_when_a_version_changes(self) -> None:
        env = _env()
        a = build_versioned_snapshot_id(self._versions(), env=env)
        bumped = self._versions()
        bumped["OSMH3KG_NODES_PATH"] = 13
        b = build_versioned_snapshot_id(bumped, env=env)
        assert a != b

    def test_version_dict_order_does_not_matter(self) -> None:
        env = _env()
        v1 = {
            "WORLDKG_TRIPLES_PATH": 1,
            "OSMH3KG_NODES_PATH": 2,
            "KWG_ENTITIES_PATH": 3,
        }
        v2 = {
            "KWG_ENTITIES_PATH": 3,
            "WORLDKG_TRIPLES_PATH": 1,
            "OSMH3KG_NODES_PATH": 2,
        }
        assert build_versioned_snapshot_id(v1, env=env) == build_versioned_snapshot_id(
            v2, env=env
        )

    def test_changes_when_paths_change_too(self) -> None:
        v = self._versions()
        a = build_versioned_snapshot_id(v, env=_env())
        b = build_versioned_snapshot_id(
            v, env=_env(WORLDKG_TRIPLES_PATH="/elsewhere")
        )
        assert a != b
        # Different paths -> the prefix differs.
        assert a.split("@")[0] != b.split("@")[0]
        # Same versions -> the suffix is identical.
        assert a.split("@")[1] == b.split("@")[1]

    def test_is_versioned(self) -> None:
        sid = build_versioned_snapshot_id(self._versions(), env=_env())
        assert is_versioned(sid) is True


# ============================================================================
# Upgrade path
# ============================================================================


class TestUpgradeToVersioned:
    def test_upgrade_from_bridge(self) -> None:
        env = _env()
        bridge = _FakeBridge(
            {
                "WORLDKG_TRIPLES_PATH": 5,
                "OSMH3KG_NODES_PATH": 12,
                "KWG_ENTITIES_PATH": 3,
            }
        )
        paths_id = build_paths_only_snapshot_id(env)
        upgraded = upgrade_to_versioned(paths_id, bridge, env=env)
        assert upgraded.startswith(paths_id + "@")
        assert is_versioned(upgraded)

    def test_upgrade_detects_path_drift_mid_run(self) -> None:
        # Build the paths-only id with one env, then call upgrade with
        # a *different* env. This simulates someone resetting env vars
        # between runner startup and tier-3 setup, which is a serious bug.
        env_before = _env()
        env_after = _env(WORLDKG_TRIPLES_PATH="/changed/under/our/feet")
        paths_id = build_paths_only_snapshot_id(env_before)
        bridge = _FakeBridge({"WORLDKG_TRIPLES_PATH": 5})
        with pytest.raises(SnapshotDriftError) as excinfo:
            upgrade_to_versioned(paths_id, bridge, env=env_after)
        assert "changed mid-run" in str(excinfo.value)
        assert paths_id in str(excinfo.value)
