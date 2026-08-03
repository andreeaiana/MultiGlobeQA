"""Default diversity-sampling parameters per SFID family.

Stratified benchmark sampling via `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY hash(..., run_seed))`.
Diversity keys and per-bucket caps are configured per SFID letter and may be
overridden per sub-template in the shape's meta.yaml via:

    sampling:
      diversity_keys: ["a.country_iso3", "a.strata_cell"]
      per_bucket_cap: 5

A deliberate fail-fast on unknown SFID letters forces an explicit decision
when a new family is added.
"""
from __future__ import annotations

DEFAULT_DIVERSITY_KEYS_BY_SFID: dict[str, list[str]] = {
    "A": ["a.country_iso3", "a.strata_cell"],
    "B": ["a.country_iso3", "a.strata_cell"],
    "C": ["a.country_iso3", "a.node_type"],
    "D": ["a.country_iso3", "a.strata_cell"],
    "E": ["a.country_iso3", "a.strata_cell"],
    "F": ["a.country_iso3", "a.strata_cell"],
    "G": ["a.country_iso3", "a.strata_cell"],
    "H": ["a.country_iso3", "a.strata_cell"],
    "I": ["a.country_iso3", "a.node_type"],
    "J": ["a.country_iso3", "a.node_type"],
    "K": ["a.country_iso3", "a.strata_cell"],
    "L": ["a.country_iso3", "a.strata_cell"],
    "M": ["a.country_iso3", "a.strata_cell"],
    "O": ["a.country_iso3", "a.strata_cell"],
    "P": ["a.country_iso3", "a.strata_cell"],
}

DEFAULT_PER_BUCKET_CAP: int = 5


def resolve_diversity_keys(sfid: str, override: list[str] | None = None) -> list[str]:
    if override:
        return list(override)
    if sfid not in DEFAULT_DIVERSITY_KEYS_BY_SFID:
        raise KeyError(
            f"No diversity-keys default for SFID '{sfid}'. Add an entry to "
            "DEFAULT_DIVERSITY_KEYS_BY_SFID or set sampling.diversity_keys per-TID."
        )
    return list(DEFAULT_DIVERSITY_KEYS_BY_SFID[sfid])


def resolve_per_bucket_cap(override: int | None = None) -> int:
    if override is not None:
        return int(override)
    return DEFAULT_PER_BUCKET_CAP


def render_partition_clause(keys: list[str]) -> str:
    """Render a list of diversity keys as a SQL PARTITION BY clause fragment.

    Returns 'PARTITION BY a.country_iso3, a.strata_cell' style string.
    """
    if not keys:
        raise ValueError("diversity keys must be a non-empty list")
    return "PARTITION BY " + ", ".join(keys)
