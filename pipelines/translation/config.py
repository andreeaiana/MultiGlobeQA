"""Shared configuration for the translation pipeline."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "templates"
VARIANTS_FILE = DATA_DIR / "benchmark_template_variants.tsv"
TRANSLATIONS_DIR = DATA_DIR / "translations"

# --- Target languages (code → display name) ---
TARGET_LANGUAGES = {
    "de": "German",
    "it": "Italian",
    "fr": "French",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "pt-PT": "Portuguese (Portugal)",
    "es": "Spanish (Spain)",
    "el": "Greek",
    "ka": "Georgian",
    "sq": "Albanian",
    "tr": "Turkish",
    "zh-CN": "Chinese",
    "vi": "Vietnamese",
    "ur": "Urdu",
    "ru": "Russian",
    "ja": "Japanese",
}
