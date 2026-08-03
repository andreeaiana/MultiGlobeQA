"""Look up the human-edited per-language template for a given vid.

The renumbered templates registry at
``benchmark/raw_eng_renumbered/templates.csv`` carries one column per
target language (``eng``, ``bul``, ``deu``, …, ``zho_CN``). The
post-edit pipeline asks the LLM to compare TEMPLATE vs QUESTION; for
non-English languages the most informative TEMPLATE is the
language-specific one (so the LLM sees both fields in the same
language), with the English ``base_template`` as a fallback when a
row's vid happens to miss the registry.
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
    def load(cls, csv_path: Path) -> "TemplateRegistry":
        by_vid_lang: dict[tuple[str, str], str] = {}
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            missing_cols = [c for c in _LANG_COLS if c not in fieldnames]
            if missing_cols:
                raise ValueError(
                    f"templates.csv missing language columns: {missing_cols}"
                )
            for row in reader:
                vid = row["ftidv"]
                for lang in _LANG_COLS:
                    txt = (row.get(lang) or "").strip()
                    if txt:
                        by_vid_lang[(vid, lang)] = _normalize_braces(txt)
        logger.info(
            "Loaded {} (vid, lang) template entries from {}",
            len(by_vid_lang), csv_path,
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
