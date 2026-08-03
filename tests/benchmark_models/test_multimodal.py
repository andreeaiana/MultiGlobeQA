"""Tests for ``benchmark_models.multimodal``.

Covers the shared helpers used by every backend's image-mode adapter:

* :func:`apply_name_mode` — text-only baseline. Each ``[IMAGE_<slot>]``
  is replaced with the entity's display name (with subcategory /
  category fallbacks for category-only slots).
* :func:`read_image_bytes` and :func:`encode_image_b64` — file I/O on
  the path stored in :class:`ImageContent`.

The per-backend image-block packing is tested separately under
``test_backend_*.py`` (each backend builds a different content shape).
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from benchmark_models.base import EntityRef, Example, ImageContent
from benchmark_models.multimodal import (
    apply_name_mode,
    encode_image_b64,
    read_image_bytes,
)


def _example(
    *,
    entities: dict[str, EntityRef] | None = None,
    images: list[ImageContent] | None = None,
    question: str = "What is in [IMAGE_A]?",
) -> Example:
    """Minimal Example for tests."""
    return Example(
        question_id="q1",
        tid="1.1",
        sfid="A",
        question=question,
        base_template="...",
        kg_source="osmh3kg",
        entities=entities or {},
        images=images or [],
    )


class TestApplyNameMode:
    """Substitutes ``[IMAGE_<slot>]`` with the entity's display name."""

    def test_substitutes_entity_name(self) -> None:
        ex = _example(
            entities={"A": EntityRef(id="ent-1", name="Mt Fuji")},
            question="What is the elevation of [IMAGE_A]?",
        )
        out = apply_name_mode(ex.question, ex)
        assert out == "What is the elevation of Mt Fuji?"

    def test_multiple_slots_each_substituted(self) -> None:
        ex = _example(
            entities={
                "A": EntityRef(id="a", name="Berlin"),
                "B": EntityRef(id="b", name="Paris"),
            },
            question="Distance between [IMAGE_A] and [IMAGE_B]?",
        )
        out = apply_name_mode(ex.question, ex)
        assert out == "Distance between Berlin and Paris?"

    def test_falls_back_to_subcategory(self) -> None:
        # Category-only slot — id and name empty, subcategory is the
        # identifier ("Pharmacy", etc.).
        ex = _example(
            entities={
                "A": EntityRef(id="", name="", subcategory="Pharmacy")
            },
            question="How many [IMAGE_A] are within 5 km?",
        )
        out = apply_name_mode(ex.question, ex)
        assert out == "How many Pharmacy are within 5 km?"

    def test_falls_back_to_category(self) -> None:
        ex = _example(
            entities={
                "A": EntityRef(id="", name="", category="natural features")
            },
            question="Show me [IMAGE_A] near here.",
        )
        out = apply_name_mode(ex.question, ex)
        assert out == "Show me natural features near here."

    def test_missing_slot_leaves_placeholder(self) -> None:
        # No slot A in entities — the placeholder is preserved so the
        # malformed row is visible in error analysis rather than
        # silently lost.
        ex = _example(question="What about [IMAGE_A]?")
        out = apply_name_mode(ex.question, ex)
        assert out == "What about [IMAGE_A]?"

    def test_no_placeholder_pass_through(self) -> None:
        ex = _example(question="What is the distance between A and B?")
        out = apply_name_mode(ex.question, ex)
        assert out == "What is the distance between A and B?"


class TestImageIO:
    """``read_image_bytes`` / ``encode_image_b64`` round-trip."""

    def test_read_returns_raw_bytes(self, tmp_path: Path) -> None:
        path = tmp_path / "img.png"
        payload = b"\x89PNG\r\n\x1a\n-fake-bytes"
        path.write_bytes(payload)
        image = ImageContent(slot="A", path=str(path), mime_type="image/png")
        assert read_image_bytes(image) == payload

    def test_encode_b64_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "img.png"
        payload = b"hello-world-image-bytes"
        path.write_bytes(payload)
        image = ImageContent(slot="A", path=str(path), mime_type="image/png")
        encoded = encode_image_b64(image)
        # Pure ASCII string, decodes back to the original bytes.
        assert isinstance(encoded, str)
        assert base64.b64decode(encoded) == payload

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        image = ImageContent(
            slot="A",
            path=str(tmp_path / "nonexistent.png"),
            mime_type="image/png",
        )
        with pytest.raises(FileNotFoundError):
            read_image_bytes(image)
