"""Per-backend image-mode adapter tests.

Each backend has a small ``_build_message_content`` (OpenAI /
vLLM) or ``_build_contents`` (Google) helper that turns a
prompt + ``Example`` into the provider-specific multimodal request
shape. These tests drive each helper directly — no API calls — and
assert on the output shape.

Three modes per backend:

* ``"off"`` → returns the prompt string unchanged.
* ``"name"`` → substitutes ``[IMAGE_<slot>]`` placeholders with the
  entity's display name; result is still a plain string.
* ``"image"`` → returns the provider's content-block list (or
  google-genai ``Part`` list) with text + one block per image.

Each backend's section is guarded by its own ``importorskip`` so a
machine missing one SDK still runs the rest of the file.
"""

from __future__ import annotations

import base64
import importlib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from benchmark_models.base import EntityRef, Example, ImageContent
from benchmark_models.config import (
    GoogleVariantConfig,
    OpenAIVariantConfig,
)


def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


_PNG_PAYLOAD = b"\x89PNG\r\n\x1a\nfake-but-valid-bytes-for-tests"


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    """A small file standing in for a real image asset."""
    p = tmp_path / "img.png"
    p.write_bytes(_PNG_PAYLOAD)
    return p


def _example_with_image(image_file: Path, slot: str = "A") -> Example:
    """Build an Example with one ``ImageContent`` and a matching entity slot."""
    return Example(
        question_id="q1",
        tid="1.1",
        sfid="A",
        question=f"What is in [IMAGE_{slot}]?",
        base_template="...",
        kg_source="osmh3kg",
        entities={slot: EntityRef(id="ent-1", name="Mt Test")},
        images=[
            ImageContent(
                slot=slot,
                path=str(image_file),
                mime_type="image/png",
            )
        ],
    )


# ----------------------------------------------------------------------------
# OpenAI
# ----------------------------------------------------------------------------


def _openai_model(multimodal_mode: str = "off") -> Any:
    from benchmark_models.backends.openai import OpenAIModel

    config = OpenAIVariantConfig(
        backend="openai",
        model_id="gpt-5.4",
        tier="plain",
        multimodal_mode=multimodal_mode,  # type: ignore[arg-type]
    )
    with patch("benchmark_models.backends.openai.OpenAI"):
        return OpenAIModel("gpt-5.4-test", config, multimodal_mode=multimodal_mode)  # type: ignore[arg-type]


@pytest.mark.skipif(not _has_module("openai"), reason="openai SDK not installed")
class TestOpenAIBuildMessageContent:
    def test_off_returns_prompt_string(self) -> None:
        model = _openai_model("off")
        ex = Example(
            question_id="q",
            tid="1.1",
            sfid="A",
            question="hi",
            base_template="...",
            kg_source="osmh3kg",
        )
        content = model._build_message_content("hello", ex)
        assert content == "hello"

    def test_name_mode_substitutes_entity_name(
        self, image_file: Path
    ) -> None:
        model = _openai_model("name")
        ex = _example_with_image(image_file)
        content = model._build_message_content("Where is [IMAGE_A]?", ex)
        assert content == "Where is Mt Test?"

    def test_image_mode_packs_image_url_block(
        self, image_file: Path
    ) -> None:
        model = _openai_model("image")
        ex = _example_with_image(image_file)
        content = model._build_message_content("hi", ex)
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "hi"}
        assert content[1]["type"] == "image_url"
        url = content[1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        # The base64 payload decodes back to the original bytes.
        b64 = url.split(",", 1)[1]
        assert base64.b64decode(b64) == _PNG_PAYLOAD


# ----------------------------------------------------------------------------
# Google
# ----------------------------------------------------------------------------


def _google_model(multimodal_mode: str = "off") -> Any:
    from benchmark_models.backends.google import GoogleModel

    config = GoogleVariantConfig(
        backend="google",
        model_id="gemini-3-flash-preview",
        tier="plain",
        multimodal_mode=multimodal_mode,  # type: ignore[arg-type]
    )
    with patch("benchmark_models.backends.google.genai.Client"):
        return GoogleModel(
            "gemini-3-flash-test",
            config,
            multimodal_mode=multimodal_mode,  # type: ignore[arg-type]
        )


@pytest.mark.skipif(
    not _has_module("google.genai"), reason="google.genai SDK not installed"
)
class TestGoogleBuildContents:
    def test_off_returns_prompt_string(self) -> None:
        model = _google_model("off")
        ex = Example(
            question_id="q",
            tid="1.1",
            sfid="A",
            question="hi",
            base_template="...",
            kg_source="osmh3kg",
        )
        out = model._build_contents("hello", ex)
        assert out == "hello"

    def test_name_mode_substitutes_entity_name(
        self, image_file: Path
    ) -> None:
        model = _google_model("name")
        ex = _example_with_image(image_file)
        out = model._build_contents("Where is [IMAGE_A]?", ex)
        assert out == "Where is Mt Test?"

    def test_image_mode_returns_part_list_with_inline_data(
        self, image_file: Path
    ) -> None:
        model = _google_model("image")
        ex = _example_with_image(image_file)
        out = model._build_contents("hi", ex)
        # Two parts: one text, one inline_data.
        assert isinstance(out, list)
        assert len(out) == 2
        # Part objects have a non-None text attribute on the first
        # entry, and a non-None inline_data attribute on the second.
        # We don't look at the SDK private types — just the surface
        # we care about.
        assert getattr(out[0], "text", None) == "hi"
        blob = getattr(out[1], "inline_data", None)
        assert blob is not None
        assert blob.mime_type == "image/png"
        assert blob.data == _PNG_PAYLOAD


# ----------------------------------------------------------------------------
# vLLM
# ----------------------------------------------------------------------------


# vLLM is heavy — only run these tests if it imports.
vllm_available = True
# In-process vLLM multimodal routing tests were removed alongside the
# backends/vllm.py drop. Multimodal Qwen / Gemma variants are not in
# models.yaml any more (vllm-serve OpenAI-compat vision support is a
# separate piece of work). Re-add a multimodal-vllm_http test class
# here if/when those variants come back.
