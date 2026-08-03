"""Path helpers — single source of truth for module-internal paths.

Centralizes the ``scoring_spec.json`` location so ``scorer.py``,
``units.py``, and any future consumer don't each carry an independent
``Path(__file__).parent / "scoring_spec.json"`` literal. If the spec
moves, the change lands in one place.

D6 from the round-2 plan.
"""

from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).parent


def spec_path() -> Path:
    """Return the canonical path to ``scoring_spec.json``.

    Centralized so callers in ``scorer.py`` and ``units.py`` agree on
    the same file. If the spec is relocated, change this function and
    nothing else.
    """
    return _PACKAGE_DIR / "scoring_spec.json"


__all__ = ["spec_path"]
