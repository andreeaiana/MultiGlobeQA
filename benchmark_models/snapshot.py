"""KG snapshot identifier construction.

Every saved :class:`benchmark_models.base.Prediction` carries a
``snapshot_id`` string that lets re-runs and the eval pipeline verify
they're working against the same KG state.

The id has two forms depending on what's available at construction time:

* **Paths-only** (used for tier 1 / tier 2): a 12-character hash of
  the KG path environment variables. Cheap, no Spark required, no
  filesystem I/O. Suitable when the runner doesn't need a Spark
  session for inference.

* **Paths + Delta versions** (used for tier 3): the paths hash, plus
  a 12-character hash of the live Delta table version numbers (one
  per KG table — main + ``subject_index`` + ``name_index`` sidecars
  when present), separated by ``@``. Read via the runner's
  :class:`benchmark_models.kg_tools.entity_service.EntityService`,
  which proxies to
  :class:`benchmark_models.kg_tools.triple_bridge.TripleBridge` over
  the shared Spark session that tier 3 needs anyway.

Both forms are valid ``snapshot_id`` values; the eval pipeline can
recognize either by the presence of ``@``. Re-runs that produce the
same ``snapshot_id`` are guaranteed to have queried the same KG
state (within the precision of the chosen form).

Code provenance (which ``benchmark_models`` / ``benchmark_generation``
commit produced the predictions) is *not* recorded in the snapshot
id — it lives in MLflow run-level params instead, since it's constant
for every prediction in a run and the prediction record should stay
small.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from typing import Protocol

# Environment variables that point at the KG snapshot. Order matters
# for reproducible hashing — never reorder this tuple, only append.
# These names are the same ones question_generator/config.py uses.
KG_PATH_ENV_VARS: tuple[str, ...] = (
    "WORLDKG_TRIPLES_PATH",
    "WORLDKG_NODES_PATH",
    "OSMH3KG_NODES_PATH",
    "OSMH3KG_TRIPLES_PATH",
    "KWG_ENTITIES_PATH",
    "KWG_TRIPLES_PATH",
)

# Hash truncation length. 12 hex chars = 48 bits = ~2.8e14 unique
# values, comfortably collision-free for the handful of distinct
# snapshots we expect to see across the project's lifetime.
_HASH_LEN: int = 12


def _hash(payload: str) -> str:
    """Truncated hex sha256 — see ``_HASH_LEN``."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_HASH_LEN]


def _paths_payload(env: Mapping[str, str] | None = None) -> str:
    """Build the canonical string fed into the paths hash.

    The payload is one ``KEY=value`` line per env var, in fixed
    ``KG_PATH_ENV_VARS`` order, with missing variables represented as
    empty strings (so unset variables are still part of the fingerprint).
    """
    source: Mapping[str, str] = env if env is not None else os.environ
    lines = [f"{k}={source.get(k, '')}" for k in KG_PATH_ENV_VARS]
    return "\n".join(lines)


def build_paths_only_snapshot_id(env: Mapping[str, str] | None = None) -> str:
    """Construct the tier 1 / tier 2 snapshot id (paths-only form).

    Args:
        env: Optional override for the environment mapping. Defaults to
            :data:`os.environ`. Useful in tests for deterministic
            snapshots without monkey-patching the real environment.

    Returns:
        A 12-character hexadecimal hash of the KG path env vars.
    """
    return _hash(_paths_payload(env))


def build_versioned_snapshot_id(
    delta_versions: Mapping[str, int],
    env: Mapping[str, str] | None = None,
) -> str:
    """Construct the tier 3 snapshot id (paths + Delta versions form).

    Args:
        delta_versions: Mapping from KG table identifier (e.g.,
            ``"WORLDKG_TRIPLES_PATH"``) to its current Delta table
            version number, as read by ``EntityService.get_delta_versions``
            (proxying to ``TripleBridge``) over the live Spark
            session. Order is canonicalized (sorted by key) before
            hashing so the result is deterministic.
        env: Optional override for the environment mapping.

    Returns:
        A snapshot id of the form ``"<paths_hash>@<versions_hash>"``,
        with both halves 12 hex chars long.
    """
    paths_hash = build_paths_only_snapshot_id(env)
    canonical = "\n".join(
        f"{k}={delta_versions[k]}" for k in sorted(delta_versions)
    )
    versions_hash = _hash(canonical)
    return f"{paths_hash}@{versions_hash}"


class SupportsDeltaVersions(Protocol):
    """Structural protocol for any object that can report Delta versions.

    :class:`benchmark_models.kg_tools.entity_service.EntityService`
    satisfies this protocol by proxying to
    :class:`benchmark_models.kg_tools.triple_bridge.TripleBridge`.
    The runner uses it to upgrade a paths-only snapshot id to a
    versioned one when a Spark session is available.
    """

    def get_delta_versions(self) -> Mapping[str, int]:
        """Return the current Delta table version for each KG path."""
        ...


def upgrade_to_versioned(
    paths_only_id: str,
    bridge: SupportsDeltaVersions,
    env: Mapping[str, str] | None = None,
) -> str:
    """Upgrade a paths-only snapshot id to the versioned form.

    Used by the runner when starting a tier-3 run: it builds the
    paths-only id at startup (so tier 1/2 records have a valid id
    even if the bridge fails to start), and on successful bridge
    construction it upgrades to the versioned form for tier-3
    predictions.

    Verifies that the current KG path env vars still hash to the
    same paths-only id; raises :class:`SnapshotDriftError` if they've
    changed mid-run (a serious bug — the runner should never observe
    different paths during a single invocation). This catches the
    "someone reset env vars between runner startup and tier-3 setup"
    failure mode.

    Args:
        paths_only_id: The paths-only snapshot id from runner startup,
            typically produced by :func:`build_paths_only_snapshot_id`.
        bridge: Object satisfying :class:`SupportsDeltaVersions`,
            usually the runner's
            :class:`~benchmark_models.kg_tools.entity_service.EntityService`.
        env: Optional override for the environment mapping.

    Returns:
        A versioned snapshot id with the same paths-hash prefix as
        ``paths_only_id``.

    Raises:
        SnapshotDriftError: if the current env's paths hash differs
            from the passed ``paths_only_id``.
    """
    current_paths_id = build_paths_only_snapshot_id(env)
    if current_paths_id != paths_only_id:
        raise SnapshotDriftError(
            f"KG path env vars changed mid-run: snapshot id was "
            f"{paths_only_id!r} at runner startup but is now "
            f"{current_paths_id!r}. The runner must not see different "
            f"KG paths during a single invocation."
        )
    return build_versioned_snapshot_id(bridge.get_delta_versions(), env=env)


class SnapshotDriftError(RuntimeError):
    """Raised when the KG snapshot identity changes mid-run.

    Indicates a process-level bug (env vars mutated after the runner
    started) rather than a data problem. The runner should treat this
    as fatal: continuing would produce predictions whose ``snapshot_id``
    no longer reflects the actual KG state.
    """


_VERSIONED_RE = __import__("re").compile(
    r"^[0-9a-f]{12}@[0-9a-f]{12}$"
)


def is_versioned(snapshot_id: str) -> bool:
    """Return ``True`` if ``snapshot_id`` is the tier-3 (versioned) form.

    Snapshot ids are produced by :func:`build_versioned_snapshot_id`
    which guarantees ``<12-hex>@<12-hex>``; reject strings that
    happen to contain ``@`` but don't match the structure (defence
    against accidental email-shaped or path-shaped values flowing
    through this check).
    """
    return bool(_VERSIONED_RE.match(snapshot_id))
