"""Look up the post-edited per-language template for a given vid.

Reads ``data/templates/templates_<lang>.tsv``, one file per language, each
keyed by ``vid``. The post-edit pipeline asks the LLM to compare TEMPLATE
against QUESTION; for a non-English language the informative TEMPLATE is that
language's own, so the LLM sees both fields in one language. The English
``base_template`` is the fallback when a vid is missing.
"""

from __future__ import annotations

import csv
import re
import threading
from pathlib import Path

from loguru import logger


_LANG_COLS = (
    "eng", "bul", "deu", "ell", "spa", "fra", "ita", "jpn", "kat",
    "por_PT", "ron", "rus", "sqi", "tur", "urd", "vie", "zho_CN",
)


# Templates registry stores placeholders as ``{entity_A}`` (single
# brace) but the rest of the pipeline uses ``{{entity_A}}`` (double).
# Convert on load so prompts stay internally consistent.
_SINGLE_BRACE_RE = re.compile(r"(?<!\{)\{(\w+)\}(?!\})")


def _normalize_braces(text: str) -> str:
    return _SINGLE_BRACE_RE.sub(r"{{\1}}", text)


class TemplateRegistry:
    """In-memory map from ``(vid, lang)`` to translated template text.

    Lookups fall back to the English template if the requested
    ``(vid, lang)`` is missing, and emit a one-time warning per missing
    vid so the pipeline log surfaces gaps without spamming.
    """

    def __init__(self, by_vid_lang: dict[tuple[str, str], str]):
        self._by_vid_lang = by_vid_lang
        # The two "already-warned" sets are touched concurrently from
        # ``--question-concurrency`` worker threads. Lock writes so two
        # threads don't both add+log for the same vid/lang.
        self._warned_missing_vids: set[str] = set()
        self._warned_missing_langs: set[tuple[str, str]] = set()
        self._warn_lock = threading.Lock()

    @classmethod
    def load(cls, templates_dir: Path) -> "TemplateRegistry":
        """Load every ``templates_<lang>.tsv`` under ``templates_dir``."""
        by_vid_lang: dict[tuple[str, str], str] = {}
        missing = []
        for lang in _LANG_COLS:
            path = Path(templates_dir) / f"templates_{lang}.tsv"
            if not path.exists():
                missing.append(lang)
                continue
            with path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    txt = (row.get("variant") or "").strip()
                    if txt:
                        by_vid_lang[(row["vid"], lang)] = _normalize_braces(txt)
        if missing:
            raise ValueError(
                f"{templates_dir}: no template table for {missing}"
            )
        logger.info(
            "Loaded {} (vid, lang) template entries from {}",
            len(by_vid_lang), templates_dir,
        )
        return cls(by_vid_lang)

    def lookup(self, vid: str, lang: str, *, eng_fallback: str | None = None) -> str:
        """Return the template for ``(vid, lang)``.

        Falls back to the English template (registry's ``eng`` column),
        then to the row's ``base_template`` if the registry has nothing
        for this vid. Logs a warning if both the vid and the
        ``eng_fallback`` are absent — prompt would otherwise be empty.
        """
        if not vid:
            if not eng_fallback:
                logger.warning(
                    "lookup called with empty vid AND empty eng_fallback "
                    "— prompt's TEMPLATE field will be empty"
                )
            return eng_fallback or ""

        text = self._by_vid_lang.get((vid, lang))
        if text:
            return text

        # Lang-specific column missing for this vid — fall back via eng.
        with self._warn_lock:
            if (vid, lang) not in self._warned_missing_langs:
                self._warned_missing_langs.add((vid, lang))
                if vid not in self._warned_missing_vids:
                    eng_text = self._by_vid_lang.get((vid, "eng"))
                    if eng_text:
                        logger.debug(
                            "vid={} has no '{}' template, falling back to 'eng'",
                            vid, lang,
                        )
                    else:
                        self._warned_missing_vids.add(vid)
                        logger.warning(
                            "vid={} not in templates registry — using "
                            "row's base_template", vid,
                        )

        eng_text = self._by_vid_lang.get((vid, "eng"))
        if eng_text:
            return eng_text
        return eng_fallback or ""
