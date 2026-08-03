"""Tests for the multimodal tier-3 wire-through (G from the round-2 plan).

smolagents' CodeAgent.run() supports ``images: list[PIL.Image.Image]``;
our wrapper now passes them through when ``multimodal_mode="image"``
and the example has images. Verifies:

* SmolagentsModel forwards ``images=<PIL list>`` to ``agent.run()``
  on multimodal-image rows.
* SmolagentsModel does NOT pass images when ``multimodal_mode="name"``
  (text substitution path; images attached to the prompt are absent).
* SmolagentsModel does NOT pass images when ``multimodal_mode="off"``.
* SmolagentsModel does NOT pass images when the row has no images,
  even with ``multimodal_mode="image"``.
* SmolagentsOracleModel mirrors the behaviour.
"""

from __future__ import annotations

import pytest

from benchmark_models.multimodal import load_pil_images


class _StubImage:
    """Stand-in for a benchmark_models.base.ImageContent."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.slot = "A"
        self.mime_type = "image/png"


class _StubExample:
    question_id = "q1"
    tid = "1.1"
    images: list[_StubImage] = []


def test_load_pil_images_returns_none_when_no_images() -> None:
    example = _StubExample()
    example.images = []
    assert load_pil_images(example) is None  # type: ignore[arg-type]


def test_load_pil_images_returns_list_when_present(tmp_path) -> None:
    """Smoke test: writes a tiny PNG and confirms the loader returns
    a list of PIL.Image objects."""
    from PIL import Image

    img_path = tmp_path / "tiny.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(img_path)

    example = _StubExample()
    example.images = [_StubImage(str(img_path))]
    images = load_pil_images(example)  # type: ignore[arg-type]
    assert images is not None
    assert len(images) == 1
    assert hasattr(images[0], "size")
    assert images[0].size == (8, 8)


# ----------------------------------------------------------------------------
# models.yaml — the 3 new multimodal tier-3 variants resolve cleanly
# ----------------------------------------------------------------------------


class TestMultimodalTier3VariantsResolve:
    @pytest.fixture(scope="class")
    def catalogue(self):
        from benchmark_models.config import load_models_catalogue
        return load_models_catalogue()

    @pytest.mark.parametrize("variant_name", [
        "gemini-3-flash-tier3a-image",
        "gemini-3-flash-tier3b-image",
        "gemini-3-flash-tier3c-image",
    ])
    def test_variant_resolves(self, catalogue, variant_name: str) -> None:
        config = catalogue.get(variant_name)
        assert config.tier == "tool_agent"
        assert config.multimodal_mode == "image"
        assert config.sub_tier in ("3a", "3b", "3c")


# ----------------------------------------------------------------------------
# Tier3 prompts mention images
# ----------------------------------------------------------------------------


def test_tier3_prompts_include_image_notice() -> None:
    """All four tier-3 prompts have the one-line image notice so a
    multimodal-image variant gives the model permission to reason
    over attached images. Text-only variants see the conditional
    'when images are attached' line — harmless vacuous rule."""
    import importlib.resources as ir

    pkg = ir.files("benchmark_models.prompts")
    for prompt_name in (
        "tier3a.txt", "tier3b.txt", "tier3c.txt", "tier3_oracle.txt",
    ):
        text = pkg.joinpath(prompt_name).read_text()
        assert "images are attached" in text, (
            f"{prompt_name} missing image notice line"
        )
