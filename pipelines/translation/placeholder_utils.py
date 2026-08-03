"""Utilities for protecting and restoring {{placeholders}} during translation."""

import re

PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}")


def protect_placeholders(text: str) -> str:
    """Wrap {{placeholders}} in <span class="notranslate"> for Google Translate."""
    return PLACEHOLDER_RE.sub(
        lambda m: f'<span class="notranslate">{m.group()}</span>',
        text,
    )


def restore_placeholders(text: str) -> str:
    """Strip the notranslate span tags added by protect_placeholders."""
    text = re.sub(r'<span class="notranslate">', "", text)
    text = re.sub(r"</span>", "", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Fix whitespace artifacts around placeholders introduced by translation."""
    # Fix stray brace before placeholder: "{ {{x}}" or "{ {{x}}" → "{{x}}"
    text = re.sub(r"\{\s+(\{\{[^}]+\}\})", r"\1", text)
    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)
    # Remove space before punctuation: . , ? ! ; : )
    text = re.sub(r"\s+([.,?!;:)])", r"\1", text)
    # Remove space after opening paren/bracket
    text = re.sub(r"([(])\s+", r"\1", text)
    return text.strip()


def validate_placeholders(source: str, translated: str) -> list[str]:
    """Check that all source placeholders appear unchanged in the translation."""
    src_ph = set(PLACEHOLDER_RE.findall(source))
    tgt_ph = set(PLACEHOLDER_RE.findall(translated))
    errors = []
    missing = src_ph - tgt_ph
    extra = tgt_ph - src_ph
    if missing:
        errors.append(f"Missing placeholders: {missing}")
    if extra:
        errors.append(f"Unexpected placeholders: {extra}")
    return errors
