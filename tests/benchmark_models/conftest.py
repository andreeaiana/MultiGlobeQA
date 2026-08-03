"""Shared fixtures for the benchmark_models test suite.

The bundled production registry only ships TIDs that appear in the
latest benchmark share. Several tests hardcode TID 1.1 / SFID A as a
stable test fixture (the canonical SFID-A distance template).
SFID A is currently absent from the share, so we augment the
production registry with one synthetic entry for TID 1.1.

TIDs ``82.1`` and ``999.999`` are deliberately *not* added — tests
that use those literals exercise the unknown-tid error path and
require the registry lookup to fail.

The augmentation is scoped to the test session and never written to
disk; the bundled registry remains untouched.
"""

from __future__ import annotations

import pytest

from benchmark_models.schemas import (
    TemplateInfo,
    TemplateRegistry,
    load_template_registry,
)

# Minimal set of synthetic TIDs needed by the test suite. Pinned
# canonical names + units mirror what the old (pre-2026-05-05) bundled
# registry exposed for these TIDs, so existing assertions that depend
# on, e.g., ``Numeric (float)``+``km`` for TID 1.1 keep passing.
_SYNTHETIC_TIDS: dict[str, dict[str, str | None]] = {
    "1.1": {
        "sfid": "A",
        "answer_type": "Numeric (float)",
        "answer_unit": "km",
    },
}


def _augment_with_synthetic_tids(reg: TemplateRegistry) -> TemplateRegistry:
    """Return a new registry containing every entry in ``reg`` plus
    the synthetic test TIDs. Synthetic entries do not overwrite an
    existing TID — production data takes precedence."""
    merged: dict[str, TemplateInfo] = dict(reg.templates)
    for tid, fields in _SYNTHETIC_TIDS.items():
        if tid in merged:
            continue
        merged[tid] = TemplateInfo(
            tid=tid,
            sfid=fields["sfid"],  # type: ignore[arg-type]
            answer_type=fields["answer_type"],  # type: ignore[arg-type]
            answer_unit=fields["answer_unit"],
        )
    return TemplateRegistry(version=reg.version, templates=merged)


@pytest.fixture(scope="session")
def registry() -> TemplateRegistry:
    """Bundled template registry plus a few synthetic test TIDs.

    Tests that hardcode TID 1.1 / 82.1 / 999.999 (SFID A — no longer
    in the production share) get registry lookups via this fixture.
    Tests that don't reference those TIDs see the bundled production
    entries unchanged.
    """
    return _augment_with_synthetic_tids(load_template_registry())
