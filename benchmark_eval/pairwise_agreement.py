"""Pairwise model-agreement helper.

Joins multiple ``score_records.jsonl`` files on ``question_id`` and
emits a cross-model agreement matrix. Each (model_A, model_B) pair
reports:

* ``agreement_rate`` — fraction of questions where ``em_A == em_B``.
* ``both_right_rate`` — both models scored EM=1.
* ``both_wrong_rate`` — both scored EM=0.
* ``a_only_right_rate`` / ``b_only_right_rate`` — disagreement
  breakdown.
* ``cohen_kappa`` — agreement beyond chance. Useful when both models
  are roughly equally accurate but their *failures* are unrelated
  (low kappa → ensemble would help; high kappa → models are
  redundant).

Two model outputs that agree on 70% of questions can mean very
different things depending on the marginals; reporting kappa
disambiguates.

This module is a stand-alone post-eval helper. It does NOT run
during the normal score-records build — model rosters change between
runs, and joining is cheap, so it's better as an ad-hoc tool than as
a runtime aggregation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# Kappa estimates with very small n are unreliable — the
# chance-correction denominator is itself noisy. 30 is a common
# rule-of-thumb threshold for "small sample" warning.
_SMALL_SAMPLE_THRESHOLD = 30


@dataclass(frozen=True)
class PairwiseAgreement:
    """One (model_A, model_B) pair's agreement statistics."""

    model_a: str
    model_b: str
    n_intersection: int
    """Number of ``question_id``s present in both inputs."""
    agreement_rate: float
    """``(both_right + both_wrong) / n_intersection``."""
    both_right_rate: float
    """Fraction where ``em_A == 1 and em_B == 1``."""
    both_wrong_rate: float
    """Fraction where ``em_A == 0 and em_B == 0``."""
    a_only_right_rate: float
    """Fraction where A got it right but B did not."""
    b_only_right_rate: float
    """Fraction where B got it right but A did not."""
    cohen_kappa: float
    """Agreement beyond chance. ``1.0`` = perfect agreement,
    ``0.0`` = chance-level, negative = worse than chance. Useful for
    diagnosing whether two models fail on the same questions
    (``kappa`` near 1) or on disjoint sets (``kappa`` near 0).

    Read alongside :attr:`degenerate_intersection` and
    :attr:`small_sample` — a ``kappa`` value with either flag set is
    not informative."""
    degenerate_intersection: bool = False
    """``True`` when both models scored 100% one-sided on the
    intersection (both always-right or both always-wrong), so the
    chance-correction denominator ``pe`` is 1.0 and ``cohen_kappa``
    is conventionally set to ``1.0`` but is not a meaningful
    "agreement beyond chance" signal."""
    small_sample: bool = False
    """``True`` when ``n_intersection < 30``. The kappa estimate has
    high variance at small n and should be treated as qualitative
    only."""


def load_em_by_question_id(path: Path | str) -> dict[str, int]:
    """Read a ``score_records.jsonl`` and return ``{question_id: em}``.

    Skips malformed lines silently (consistent with the rest of the
    eval pipeline) and ignores rows missing either ``question_id`` or
    ``em``.

    Counts duplicate ``question_id`` rows and emits one warning per
    file when count > 0. Last-write-wins; the warning makes the
    duplication visible so a partial re-run concatenated onto an old
    score file doesn't silently corrupt the agreement matrix.
    """
    import logging

    p = Path(path)
    out: dict[str, int] = {}
    n_duplicates = 0
    with p.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
                qid = row["question_id"]
                em = int(row["em"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if qid in out:
                n_duplicates += 1
            out[qid] = em
    if n_duplicates > 0:
        logging.getLogger(__name__).warning(
            "load_em_by_question_id(%s): %d duplicate question_id "
            "rows found; last value wins. This usually indicates a "
            "partial re-run was concatenated onto an old score file. "
            "Inspect %s if the agreement matrix looks suspect.",
            p, n_duplicates, p,
        )
    return out


_small_sample_warned: set[tuple[str, str]] = set()


def compute_pairwise(
    em_a: dict[str, int],
    em_b: dict[str, int],
    *,
    model_a: str,
    model_b: str,
) -> PairwiseAgreement:
    """Compute pairwise agreement on the ``question_id`` intersection.

    The intersection is critical: comparing models on different question
    sets would produce meaningless rates. We compute only over the
    overlap and report ``n_intersection`` so a downstream reader can
    judge how comparable two models really are.
    """
    import logging

    common = set(em_a.keys()) & set(em_b.keys())
    n = len(common)
    if n == 0:
        return PairwiseAgreement(
            model_a=model_a,
            model_b=model_b,
            n_intersection=0,
            agreement_rate=0.0,
            both_right_rate=0.0,
            both_wrong_rate=0.0,
            a_only_right_rate=0.0,
            b_only_right_rate=0.0,
            cohen_kappa=0.0,
            degenerate_intersection=False,
            small_sample=True,
        )

    both_right = sum(1 for q in common if em_a[q] == 1 and em_b[q] == 1)
    both_wrong = sum(1 for q in common if em_a[q] == 0 and em_b[q] == 0)
    a_only = sum(1 for q in common if em_a[q] == 1 and em_b[q] == 0)
    b_only = sum(1 for q in common if em_a[q] == 0 and em_b[q] == 1)

    po = (both_right + both_wrong) / n

    # Cohen's kappa = (po - pe) / (1 - pe), where pe is the expected
    # agreement assuming the two models pick EM=1/EM=0 independently
    # with their observed marginal rates.
    p_a1 = (both_right + a_only) / n
    p_b1 = (both_right + b_only) / n
    p_a0 = (both_wrong + b_only) / n
    p_b0 = (both_wrong + a_only) / n
    pe = p_a1 * p_b1 + p_a0 * p_b0
    degenerate = pe >= 1.0
    if degenerate:
        # Both models are 100% one-sided (always-right or always-wrong)
        # on the intersection. ``po`` is necessarily 1.0 too, so kappa
        # is conventionally defined as 1.0 — but ``degenerate=True``
        # tells the reader this is not a chance-corrected signal.
        kappa = 1.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    small_sample = n < _SMALL_SAMPLE_THRESHOLD
    if small_sample:
        pair_key = (model_a, model_b)
        if pair_key not in _small_sample_warned:
            _small_sample_warned.add(pair_key)
            logging.getLogger(__name__).warning(
                "compute_pairwise(%s, %s): n_intersection=%d < %d — "
                "kappa estimate has high variance at small n; treat "
                "as qualitative only.",
                model_a, model_b, n, _SMALL_SAMPLE_THRESHOLD,
            )

    return PairwiseAgreement(
        model_a=model_a,
        model_b=model_b,
        n_intersection=n,
        agreement_rate=po,
        both_right_rate=both_right / n,
        both_wrong_rate=both_wrong / n,
        a_only_right_rate=a_only / n,
        b_only_right_rate=b_only / n,
        cohen_kappa=kappa,
        degenerate_intersection=degenerate,
        small_sample=small_sample,
    )


def compute_all_pairs(
    em_by_model: dict[str, dict[str, int]],
) -> list[PairwiseAgreement]:
    """Compute every unique pair from a ``{model_name: em_dict}`` map.

    Returns pairs in lexicographically-sorted model-name order so the
    output is deterministic across runs.
    """
    models = sorted(em_by_model.keys())
    pairs: list[PairwiseAgreement] = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            pairs.append(
                compute_pairwise(
                    em_by_model[a], em_by_model[b],
                    model_a=a, model_b=b,
                )
            )
    return pairs


def write_report(pairs: list[PairwiseAgreement], output_path: Path | str) -> None:
    """Write a list of agreements as JSON for paper-side analysis."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data: list[dict[str, Any]] = [asdict(pair) for pair in pairs]
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def format_table(pairs: list[PairwiseAgreement]) -> str:
    """Render the agreement list as a fixed-width table for the CLI."""
    header = (
        f"{'model_a':<35} {'model_b':<35} {'n':>7} "
        f"{'agree':>7} {'both_R':>7} {'both_W':>7} "
        f"{'A_only':>7} {'B_only':>7} {'kappa':>7}"
    )
    lines = [header, "-" * len(header)]
    for p in pairs:
        lines.append(
            f"{p.model_a:<35} {p.model_b:<35} {p.n_intersection:>7,d} "
            f"{p.agreement_rate:>7.3f} {p.both_right_rate:>7.3f} "
            f"{p.both_wrong_rate:>7.3f} "
            f"{p.a_only_right_rate:>7.3f} {p.b_only_right_rate:>7.3f} "
            f"{p.cohen_kappa:>7.3f}"
        )
    return "\n".join(lines)


__all__ = [
    "PairwiseAgreement",
    "compute_all_pairs",
    "compute_pairwise",
    "format_table",
    "load_em_by_question_id",
    "write_report",
]
