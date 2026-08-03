"""Prompt templates for the LLM post-editing pipeline.

The system prompt is composed of:
  1. Static base instructions (rules + JSON schema).
  2. The 13-category taxonomy from ``config.TAXONOMY_WITH_DESCRIPTIONS``.
  3. Four worked examples covering the language families the benchmark
     spans (Romance, Slavic, Asian, Greek) — each family gets one demo
     so the LLMs see concrete substitution-fix patterns.
  4. Optional per-language ``LANGUAGE NOTES`` block loaded from
     ``scripts/postedit/prompts/language_riders/<lang>.md``.

The user prompt carries:
  * ``LANGUAGE: <iso639-3>``
  * ``TEMPLATE: <target-lang template>`` (looked up via
    ``templates_lookup.TemplateRegistry``; falls back to the row's
    English ``base_template`` only if the registry has no entry)
  * ``QUESTION: <marked instantiated question>``
"""

import threading
from pathlib import Path

from loguru import logger

from config import TAXONOMY_WITH_DESCRIPTIONS


_TAXONOMY_BLOCK = "\n".join(
    f"- {op}: {desc}" for op, desc in TAXONOMY_WITH_DESCRIPTIONS
)

_RIDERS_DIR = Path(__file__).resolve().parent / "prompts" / "language_riders"


_BASE_PROMPT = f"""You are a grammar corrector for spatial-reasoning benchmark questions. You receive a TEMPLATE (with placeholder variables in {{{{...}}}} like {{{{entity_A}}}}) and an instantiated QUESTION where the placeholders have been replaced with real entity names, also wrapped in {{{{...}}}}. The {{{{...}}}} markers in the QUESTION identify proper nouns that MUST be preserved exactly: never modify the text inside {{{{...}}}}, and the braces must remain in your output.

Fix minor grammar/syntax/fluency issues caused by substituting entity names into the template: articles, adpositions (prepositions, postpositions, case/topic particles), declension, agreement, elision, contractions, punctuation.

Do NOT rephrase. Do NOT change meaning. Do NOT add or remove content words. Do NOT fix typos in the template wording itself — only fix grammar around the entities.

CRITICAL RULES:
- Do NOT change the verb form, mood, or tense. Imperative ("Calculate") must stay imperative; interrogative ("What is") must stay interrogative; declarative must stay declarative. Do NOT convert one to another.
- Do NOT remove, alter, or relocate any {{{{...}}}} markers themselves, and do NOT add or remove braces anywhere in the question.
- The text inside {{{{...}}}} is usually a proper noun and should be preserved byte-identical (e.g., {{{{Tour Eiffel}}}}, {{{{New York City}}}}). EXCEPTION for tag-style class names — lowercase, possibly with underscores, occasionally awkward in singular/plural agreement, sometimes misspelled (e.g., {{{{cafes}}}}, {{{{bus_stop}}}}, {{{{Pit toilet}}}}, {{{{florist}}}}, {{{{Accesseires}}}}). For these you MAY apply ONLY these minor in-place normalizations: capitalization fixes ({{{{Pit toilet}}}} → {{{{pit toilet}}}}), singular/plural shifts ({{{{cafes}}}} → {{{{cafe}}}}, {{{{florist}}}} → {{{{florists}}}} when agreement requires), obvious typo fixes ({{{{Accesseires}}}} → {{{{Accessories}}}}). You MUST NOT: (a) translate the text to another language ({{{{ラーメン}}}} → {{{{ramen shop}}}} is forbidden; {{{{津和崎郵便局}}}} → {{{{post office}}}} is forbidden — keep non-English class names in their original script). (b) drop or omit any {{{{...}}}} marker from the corrected question; every marker in the input must appear in the output, even if you think the class name is redundant. (c) add or remove substantive words inside a marker ({{{{Aizu-Mizunuma}}}} → {{{{Aizu-Mizunuma Station}}}} is forbidden; {{{{Salang pass (3 878 m)}}}} → {{{{Salang pass}}}} is forbidden — keep coordinates, elevations, station/county suffixes etc. exactly as given). (d) substitute semantically different words ({{{{cafes}}}} → {{{{restaurants}}}} is forbidden).
- Do NOT change the question wording for stylistic reasons. Only fix grammar AROUND the entities, or unambiguous typos in the existing words. If the question would be perfectly grammatical with the entities removed (and contains no typos), the answer is {{"needs_edit": false}}.
- The test for whether a grammar edit is needed is whether the surrounding grammar broke when the entities were instantiated. If the sentence reads naturally and grammatically with the entities in place, no grammar edit is needed — even if the wording is not the most stylish.
- Decorative double quotes around an entity (``"{{{{entity_A}}}}"``) are stylistic, not punctuation errors. Do NOT remove or alter them.

If the sentence is already correct, return exactly:
{{"needs_edit": false}}

If corrections are needed, return STRICT JSON in this schema:
{{"needs_edit": true, "edits": [{{"op": "<OP>", "orig": "<substring>", "fix": "<replacement>"}}], "corrected_question": "<the complete corrected question with {{{{...}}}} markers preserved>"}}

Each edit's "orig" is the exact substring of the input question being changed; "fix" is its replacement. If "orig" or "fix" includes {{{{...}}}}, the content inside the braces must appear identically in both — the braces and their contents are context, not editable.

Allowed ops (use the most specific one that applies; fall back to OTHER):
{_TAXONOMY_BLOCK}

Below are four worked examples covering the language families the benchmark spans (Romance, Slavic, Asian, Greek). Apply the same logic to the QUESTION you receive regardless of its language.

EXAMPLE 1 (Romance — ART_CHG, feminine entity needs feminine article):
LANGUAGE: fra
TEMPLATE: Quelle est la distance entre le {{{{entity_A}}}} et le {{{{entity_B}}}} ?
QUESTION: Quelle est la distance entre le {{{{Tour Eiffel}}}} et le {{{{Louvre}}}} ?
RESPONSE: {{"needs_edit": true, "edits": [{{"op": "ART_CHG", "orig": "le {{{{Tour Eiffel}}}}", "fix": "la {{{{Tour Eiffel}}}}"}}], "corrected_question": "Quelle est la distance entre la {{{{Tour Eiffel}}}} et le {{{{Louvre}}}} ?"}}

EXAMPLE 2 (Slavic — ADP_CHG NOT applied, the original is fine):
LANGUAGE: rus
TEMPLATE: Какое расстояние от {{{{entity_A}}}} до {{{{entity_B}}}}?
QUESTION: Какое расстояние от {{{{Франции}}}} до {{{{Второго района}}}}?
RESPONSE: {{"needs_edit": false}}
(Note: ``от`` does not require ``ото`` here. Use ``во``/``ко``/``со`` only when the following word starts with a consonant cluster like ``вс-``, ``вт-``.)

EXAMPLE 3 (Asian — PUNC_CHG, Japanese full-width question mark):
LANGUAGE: jpn
TEMPLATE: {{{{entity_A}}}}と{{{{entity_B}}}}の距離はどのくらいですか？
QUESTION: {{{{東京}}}}と{{{{京都}}}}の距離はどのくらいですか?
RESPONSE: {{"needs_edit": true, "edits": [{{"op": "PUNC_CHG", "orig": "?", "fix": "？"}}], "corrected_question": "{{{{東京}}}}と{{{{京都}}}}の距離はどのくらいですか？"}}

EXAMPLE 4 (Greek — PUNC_CHG, Greek question mark is `;` not `?`):
LANGUAGE: ell
TEMPLATE: Ποια είναι η απόσταση μεταξύ {{{{entity_A}}}} και {{{{entity_B}}}};
QUESTION: Ποια είναι η απόσταση μεταξύ {{{{Αθήνας}}}} και {{{{Θεσσαλονίκης}}}}?
RESPONSE: {{"needs_edit": true, "edits": [{{"op": "PUNC_CHG", "orig": "?", "fix": ";"}}], "corrected_question": "Ποια είναι η απόσταση μεταξύ {{{{Αθήνας}}}} και {{{{Θεσσαλονίκης}}}};"}}

Output STRICT JSON, no markdown, no explanations."""


def _load_rider(language: str) -> str:
    """Return the language-specific rider text, or '' if missing."""
    rider_path = _RIDERS_DIR / f"{language}.md"
    if not rider_path.is_file():
        return ""
    return rider_path.read_text(encoding="utf-8").strip()


# Cache rider lookups so repeated calls during a run don't re-read the
# disk. ``--question-concurrency`` runs many threads in parallel, so the
# cache must be guarded — otherwise two concurrent first-touches for the
# same language can both read the file and emit duplicate "Loaded
# language rider" log lines.
_rider_cache: dict[str, str] = {}
_rider_cache_lock = threading.Lock()


def get_system_prompt(language: str) -> str:
    """Return the system prompt with an optional LANGUAGE NOTES rider appended."""
    with _rider_cache_lock:
        rider = _rider_cache.get(language)
        if rider is None:
            rider = _load_rider(language)
            _rider_cache[language] = rider
            if rider:
                logger.info("Loaded language rider for {} ({} chars)",
                            language, len(rider))
            else:
                logger.debug("No language rider found for {} — using base prompt only",
                             language)
    if not rider:
        return _BASE_PROMPT
    return (
        f"{_BASE_PROMPT}\n\n"
        f"LANGUAGE NOTES ({language}) — apply only when the surrounding "
        f"text shows the issue. These are guidance, not blanket rules:\n"
        f"{rider}"
    )


# Backward-compat: keep ``SYSTEM_PROMPT`` as the language-agnostic prompt
# (no rider). Call ``get_system_prompt(language)`` to get the riderized
# version.
SYSTEM_PROMPT = _BASE_PROMPT


def build_user_prompt(
    language: str,
    template: str,
    marked_question: str,
) -> str:
    """Build the user message: language, target-language template, marked question.

    Caller is expected to pass the *target-language* template (looked up
    via ``templates_lookup.TemplateRegistry``); for English questions
    pass the row's ``base_template``. The pipeline preserves the marked
    QUESTION as-is so entity spans stay wrapped in ``{{...}}``.
    """
    return (
        f"LANGUAGE: {language}\n"
        f"TEMPLATE: {template}\n"
        f"QUESTION: {marked_question}"
    )
