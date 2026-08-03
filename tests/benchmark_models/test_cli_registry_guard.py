"""Tests for ``cli._assert_registry_covers_examples``.

The guard catches a class of bug we hit on the 2026-05-09 renumbered
share: 107 of the benchmark's 180 TIDs weren't in
``template_registry.json`` because the registry hadn't been regenerated
post-renumbering. The runner happily started, the schema-build path
crashed per-row with ``unknown tid 'X.Y'``, and ~50% of every cell's
predictions were silent ``error_class=config`` records — wasted
compute that looked superficially like real refusals.

The guard short-circuits this: at startup, scan the loaded examples'
TIDs against the registry's known set, raise ``ClickException`` with
the regen command before any model call.
"""

from __future__ import annotations

from typing import Any

import click
import pytest

from benchmark_models.cli import _assert_registry_covers_examples


class _StubExample:
    """Minimal Example stand-in: only the ``tid`` attribute the
    guard reads."""

    def __init__(self, tid: str | None) -> None:
        self.tid = tid


class _StubRegistry:
    """Minimal TemplateRegistry stand-in: only the ``templates`` dict
    the guard reads."""

    def __init__(self, tids: list[str]) -> None:
        self.templates = {tid: object() for tid in tids}


class TestRegistryGuardHappyPath:
    def test_no_examples_does_not_raise(self) -> None:
        """Empty list is benign — no TIDs to check, nothing fails."""
        _assert_registry_covers_examples([], _StubRegistry(["1.1", "2.1"]))

    def test_all_tids_present_does_not_raise(self) -> None:
        """Every example's TID is in the registry → guard is a no-op."""
        examples = [
            _StubExample(tid="1.1"),
            _StubExample(tid="2.1"),
            _StubExample(tid="2.1"),  # duplicate is fine
        ]
        registry = _StubRegistry(["1.1", "2.1", "99.9"])
        _assert_registry_covers_examples(examples, registry)

    def test_examples_with_none_tid_skipped(self) -> None:
        """Examples that lack a TID (defensive: shouldn't happen in
        production but the loader is tolerant of malformed rows) are
        ignored rather than treated as 'unknown tid None'."""
        examples = [
            _StubExample(tid="1.1"),
            _StubExample(tid=None),
        ]
        registry = _StubRegistry(["1.1"])
        _assert_registry_covers_examples(examples, registry)


class TestRegistryGuardRaises:
    def test_single_missing_tid_raises_clickexception(self) -> None:
        """One example with an unknown TID raises ClickException —
        not RuntimeError, not KeyError. The CLI exits cleanly."""
        examples = [_StubExample(tid="missing.1")]
        registry = _StubRegistry(["1.1"])
        with pytest.raises(click.ClickException) as exc_info:
            _assert_registry_covers_examples(examples, registry)
        msg = exc_info.value.message
        assert "1 TID(s)" in msg
        assert "missing.1" in msg

    def test_error_message_includes_regen_command(self) -> None:
        """The error message must point operators at the fix path,
        not just complain. The regen one-liner is in every error."""
        examples = [_StubExample(tid="missing.1")]
        registry = _StubRegistry([])
        with pytest.raises(click.ClickException) as exc_info:
            _assert_registry_covers_examples(examples, registry)
        msg = exc_info.value.message
        assert "build_template_registry_from_benchmark" in msg
        assert "template_registry.json" in msg

    def test_multiple_missing_tids_listed_sorted(self) -> None:
        """All unknown TIDs (up to preview_count) appear in the
        message in deterministic sorted order, so two failures from
        the same registry produce identical errors."""
        examples = [
            _StubExample(tid="9.9"),
            _StubExample(tid="3.3"),
            _StubExample(tid="6.6"),
            _StubExample(tid="1.1"),  # this one is in the registry
        ]
        registry = _StubRegistry(["1.1"])
        with pytest.raises(click.ClickException) as exc_info:
            _assert_registry_covers_examples(examples, registry)
        msg = exc_info.value.message
        assert "3 TID(s)" in msg
        # Sorted lexicographically; preview shows all 3 since 3 < 8.
        assert "'3.3'" in msg
        assert "'6.6'" in msg
        assert "'9.9'" in msg
        # Order in the list literal: ['3.3', '6.6', '9.9']
        assert msg.index("'3.3'") < msg.index("'6.6'") < msg.index("'9.9'")

    def test_more_than_preview_count_shows_overflow_suffix(self) -> None:
        """Long lists are truncated to ``preview_count`` with a
        ``+N more`` suffix, so the error stays one screen even if
        every TID is unknown."""
        n_missing = 15
        examples = [_StubExample(tid=f"miss.{i}") for i in range(n_missing)]
        registry = _StubRegistry([])
        with pytest.raises(click.ClickException) as exc_info:
            _assert_registry_covers_examples(
                examples, registry, preview_count=8,
            )
        msg = exc_info.value.message
        assert f"{n_missing} TID(s)" in msg
        # First 8 sorted shown; +7 more in the suffix.
        assert "+7 more" in msg

    def test_error_message_includes_total_benchmark_tid_count(self) -> None:
        """The total benchmark TID count helps the operator validate
        the post-regen registry has the right cardinality."""
        examples = [
            _StubExample(tid="present.1"),
            _StubExample(tid="present.2"),
            _StubExample(tid="missing.3"),
        ]
        registry = _StubRegistry(["present.1", "present.2"])
        with pytest.raises(click.ClickException) as exc_info:
            _assert_registry_covers_examples(examples, registry)
        msg = exc_info.value.message
        assert "all 3 benchmark TIDs" in msg


class TestRegistryGuardCoversCurrentRegistry:
    """Integration: with the *real* template_registry.json on disk,
    a single example with a TID present in the registry must NOT
    trigger the guard.

    The point isn't to test pydantic loading — it's a smoke test that
    confirms the guard's reads (``registry.templates``) match the
    actual TemplateRegistry shape, not just our StubRegistry."""

    def test_real_registry_smoke(self) -> None:
        from benchmark_models.schemas import load_template_registry

        real_registry = load_template_registry()
        # Pick any TID from the real registry as a smoke probe.
        any_tid = next(iter(real_registry.templates.keys()))
        examples = [_StubExample(tid=any_tid)]
        _assert_registry_covers_examples(examples, real_registry)

    def test_real_registry_rejects_garbage_tid(self) -> None:
        from benchmark_models.schemas import load_template_registry

        real_registry = load_template_registry()
        examples = [_StubExample(tid="definitely.not.in.registry")]
        with pytest.raises(click.ClickException) as exc_info:
            _assert_registry_covers_examples(examples, real_registry)
        assert "definitely.not.in.registry" in exc_info.value.message
