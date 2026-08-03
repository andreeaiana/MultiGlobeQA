"""Shared multimodal helpers used by every backend.

The backends differ in how they pack image data into a request — Google
takes ``inline_data`` parts, OpenAI takes ``input_image`` blocks with a
data URI, and vLLM serves the OpenAI-compat surface — but the
*upstream* logic is the same: load the file, base64-encode, decide
what to put where based on
``multimodal_mode``. This module owns that shared logic.

Three operations live here:

* :func:`apply_name_mode` — text-only baseline. Substitutes the
  ``[IMAGE_<slot>]`` placeholders in the question with the
  corresponding entity's display name (and subcategory / category
  context when present), so the model gets the same problem with
  no image bytes.
* :func:`read_image_bytes` — read the file at ``ImageContent.path``.
* :func:`encode_image_b64` — convenience: read + base64-encode in one
  step (the shape every provider's content block needs).

The image-mode dispatch (which content block to build) lives per
backend because the SDKs disagree on shape. ``apply_name_mode`` is
shared because it's plain text manipulation.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

from benchmark_models.base import Example, ImageContent

# ``[IMAGE_A]``, ``[IMAGE_B]``, ... — one slot letter per match. Same
# shape the cleanup script writes; both ends use the regex so the round
# trip is exact.
_IMAGE_PLACEHOLDER_RE = re.compile(r"\[IMAGE_([A-Z])\]")


def _entity_display(example: Example, slot: str) -> str:
    """Return the substitution string for ``[IMAGE_<slot>]`` in name mode.

    Resolution order: the entity's ``name``, then a ``subcategory``
    (e.g., ``"Pharmacy"`` for category-only slots that carry a sub-
    category but no name), then ``category`` (covers the rare slots
    that have a top-level / parameters category but no subcategory),
    then the placeholder text itself when nothing is available. The
    fallback path keeps a malformed row from silently losing the
    placeholder — the model sees a literal ``[IMAGE_A]`` and the
    failure is visible in error analysis rather than buried as a
    confusing prompt.
    """
    entity = example.entities.get(slot)
    if entity is not None:
        if entity.name:
            return entity.name
        if entity.subcategory:
            return entity.subcategory
        if entity.category:
            return entity.category
    return f"[IMAGE_{slot}]"


def apply_name_mode(prompt: str, example: Example) -> str:
    """Replace ``[IMAGE_<slot>]`` placeholders with the entity name.

    Used only when ``multimodal_mode == "name"``. The returned string
    has the same shape as the original question but with the entity
    text substituted in — so a question like
    ``"What is the distance between [IMAGE_A] and Ōtani (...)?"``
    becomes ``"What is the distance between Mt Fuji and Ōtani (...)?"``
    and can be sent to a text-only backend unchanged.

    Robust to the unusual case of a placeholder that has no matching
    entity slot: the placeholder text is left in place and the call
    completes without raising. That keeps the failure mode obvious
    rather than crashing the run.
    """

    def _sub(match: re.Match[str]) -> str:
        slot = match.group(1)
        return _entity_display(example, slot)

    return _IMAGE_PLACEHOLDER_RE.sub(_sub, prompt)


def read_image_bytes(image: ImageContent) -> bytes:
    """Read the file pointed to by an :class:`ImageContent`.

    The cleanup script writes paths relative to the project root; the
    caller is expected to have ``cwd`` at the project root (the runner
    and CLI do). When the path is absolute, this works unchanged.
    """
    return Path(image.path).read_bytes()


def encode_image_b64(image: ImageContent) -> str:
    """Read an :class:`ImageContent` and return its base64 string.

    Convenience wrapper used by every backend's image-block packer.
    Returns an ASCII string (no padding / wrapping); callers add
    provider-specific framing (data URI, source dict, ...).
    """
    return base64.b64encode(read_image_bytes(image)).decode("ascii")


def load_pil_images(example: Example) -> list["PIL.Image.Image"] | None:  # type: ignore[name-defined]  # noqa: F821
    """Load the example's images as a list of ``PIL.Image.Image``.

    Used by the tier-3 wrapper (``SmolagentsModel`` /
    ``SmolagentsOracleModel``) — smolagents' ``CodeAgent.run()``
    accepts an ``images: list[PIL.Image.Image] | None`` parameter,
    and the underlying LiteLLM bridge serialises the bytes to each
    provider's content-block shape.

    Returns ``None`` when the example has no images (the agent's
    ``run()`` accepts ``None``). Returns the loaded list otherwise,
    in slot order (``"A"``, ``"B"``, ...). PIL is imported lazily
    so the dependency is only required when this function is used.
    """
    if not example.images:
        return None
    from PIL import Image  # type: ignore[import-not-found]
    import io as _io

    out = []
    for img in example.images:
        # Open inside a context manager and copy so the returned image
        # is fully decoded and independent of the BytesIO buffer; PIL's
        # default lazy-load would keep an active reference to the
        # temporary file pointer.
        with Image.open(_io.BytesIO(read_image_bytes(img))) as im:
            im.load()
            out.append(im.copy())
    return out


__all__ = [
    "apply_name_mode",
    "encode_image_b64",
    "load_pil_images",
    "read_image_bytes",
]
