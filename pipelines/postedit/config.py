"""Shared configuration for the LLM post-editing pipeline."""

import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "benchmark" / "sampled_multilingual"
DEFAULT_TEMPLATES_CSV = PROJECT_ROOT / "benchmark" / "raw_eng_renumbered" / "templates.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "postedit"
# The post-edited benchmark the pipeline writes. Has the same
# layout as DEFAULT_BENCHMARK_ROOT (<kg>/sfid_*/<lang>/benchmark.jsonl/part-*,
# plus multimodal/<inner_kg>/<lang>/...), so cell_walker.iter_cells walks it
# unchanged. Stats join against it to recover the true per-language row
# universe (every source row lands here, including ones never post-edited).
DEFAULT_FINAL_ROOT = PROJECT_ROOT / "benchmark" / "final"

# --- Taxonomy (13 categories) ---
# Each entry: (acronym, short description shown to the LLM and in the README).
# The acronym is what the LLM emits; the description tells it what the
# acronym means so it can assign categories consistently.
TAXONOMY_WITH_DESCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("NO_CHANGE", "the question is already correct, no edit needed"),
    ("ART_ADD", "add an article (e.g., 'the', 'a', 'le', 'la', 'der')"),
    ("ART_DEL", "delete an article that is now redundant"),
    ("ART_CHG", "change one article to another (e.g., 'le' → 'la' for gender)"),
    ("ADP_ADD", "add an adposition — preposition, postposition, or case/topic particle (e.g., 'of', 'à', Hindi 'में', Japanese 'を', Turkish '-de')"),
    ("ADP_DEL", "delete an adposition (preposition, postposition, or particle) that is now redundant"),
    ("ADP_CHG", "change one adposition to another, including case/topic particles (e.g., 'in' → 'on'; Japanese 'は' → 'が')"),
    ("DEC_CHG", "change the declension/case ending of a word (e.g., genitive, dative, accusative)"),
    ("AGR_CHG", "change agreement of a word (gender, number, or person)"),
    ("CONTR_CHG", "add, remove, or change a contraction (e.g., 'de le' → 'du', 'a el' → 'al')"),
    ("ELISION", "elide a vowel before another vowel (e.g., 'le ami' → \"l'ami\", 'di Italia' → \"d'Italia\")"),
    ("PUNC_CHG", "fix punctuation around the entity, including script-specific marks (e.g., add/remove a comma, fix spacing, Greek '?' → ';', full-width '？' '、', Arabic '،')"),
    ("OTHER", "any other minor fix that doesn't fit the categories above"),
)

TAXONOMY = tuple(op for op, _ in TAXONOMY_WITH_DESCRIPTIONS)

# --- Model identifiers ---
MODEL_NAMES = ("gemini", "qwen", "deepseek")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-27B-FP8")

# --- Endpoints ---
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")

# --- API generation settings ---
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 2048
REQUEST_TIMEOUT = float(os.environ.get("POSTEDIT_TIMEOUT", "60"))
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.5  # exponential base, seconds

# --- Pipeline ---
DEFAULT_BATCH_SIZE = 50
DEFAULT_LANGUAGE = "eng"
DEFAULT_QUESTION_CONCURRENCY = 16
# Worst-case per-LLM-call budget is REQUEST_TIMEOUT * MAX_RETRIES ≈ 180 s
# (3 retries × 60 s each); 3 calls run concurrently. Set the per-question
# wall to comfortably exceed that so the timeout doesn't fire while
# retries are still in flight (which would leak the worker thread).
DEFAULT_PER_QUESTION_TIMEOUT_S = 210

# --- Resume modes ---
RESUME_OFF = "off"
RESUME_SKIP_DONE = "skip-done"
RESUME_RETRY_FAILED = "retry-failed"
RESUME_MODES = (RESUME_OFF, RESUME_SKIP_DONE, RESUME_RETRY_FAILED)

# --- Languages with riders shipped (used for tests + SLURM generator) ---
TARGET_LANGUAGES = (
    "bul", "deu", "ell", "spa", "fra", "ita", "jpn", "kat",
    "por_PT", "ron", "rus", "sqi", "tur", "urd", "vie", "zho_CN",
)

# --- KG buckets in the benchmark layout ---
KG_BUCKETS = ("kwg", "osmh3kg", "worldkg", "multimodal")
