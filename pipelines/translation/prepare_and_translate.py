"""Task 3: Translate template variants using Google Cloud Translation LLM.

Usage:
    uv run python scripts/translation/prepare_and_translate.py --dry-run
    uv run python scripts/translation/prepare_and_translate.py --languages all
    uv run python scripts/translation/prepare_and_translate.py --languages de,fr,it

Environment variables:
    GOOGLE_CLOUD_PROJECT  - GCP project ID (required)
    GOOGLE_CLOUD_LOCATION - API location (default: us-central1)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from google.cloud import translate_v3

from config import VARIANTS_FILE, TRANSLATIONS_DIR, TARGET_LANGUAGES
from placeholder_utils import (
    normalize_whitespace,
    protect_placeholders,
    restore_placeholders,
    validate_placeholders,
)


def load_data(input_path: Path = VARIANTS_FILE) -> pd.DataFrame:
    """Load and clean the variants TSV."""
    df = pd.read_csv(input_path, sep="\t", dtype=str)
    # The first column may be named " " or "FTID" depending on the export;
    # normalise to "FTID".
    df.columns = [c.strip() if c.strip() else "FTID" for c in df.columns]
    if df.columns[0] != "FTID":
        df = df.rename(columns={df.columns[0]: "FTID"})
    df["FTID"] = df["FTID"].str.strip()
    # Keep only rows with non-empty Template variant
    df = df[df["Template variant"].notna() & (df["Template variant"].str.strip() != "")]
    df["Template variant"] = df["Template variant"].str.strip()
    return df.reset_index(drop=True)



def translate_batch(
    client: translate_v3.TranslationServiceClient,
    parent: str,
    texts: list[str],
    target_language: str,
) -> list[str]:
    """Translate a batch of texts using the Translation LLM model."""
    # Protect placeholders with HTML notranslate spans
    protected = [protect_placeholders(t) for t in texts]

    response = client.translate_text(
        parent=parent,
        contents=protected,
        target_language_code=target_language,
        source_language_code="en",
        mime_type="text/html",
        model=f"{parent}/models/general/translation-llm",
    )

    results = []
    for i, translation in enumerate(response.translations):
        restored = restore_placeholders(translation.translated_text)
        restored = normalize_whitespace(restored)
        errors = validate_placeholders(texts[i], restored)
        if errors:
            print(f"  WARNING placeholder issue: {'; '.join(errors)}")
            print(f"    Source:     {texts[i]}")
            print(f"    Translated: {restored}")
        results.append(restored)

    return results


def translate_language(
    client: translate_v3.TranslationServiceClient,
    parent: str,
    df: pd.DataFrame,
    lang_code: str,
    lang_name: str,
) -> pd.DataFrame:
    """Translate all template variants for one language."""
    texts = df["Template variant"].tolist()
    batch_size = 100
    all_translations = []

    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"  Translating {len(texts)} texts in {total_batches} batches...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"    Batch {batch_num}/{total_batches} ({len(batch)} texts)...", end=" ")

        translated = translate_batch(client, parent, batch, lang_code)
        all_translations.extend(translated)
        print("done")

        # Small delay between batches to respect quotas
        if i + batch_size < len(texts):
            time.sleep(0.1)

    result = df[["FTID", "TID", "STID"]].copy()
    result["Template variant (en)"] = df["Template variant"]
    result[f"Template variant ({lang_name})"] = all_translations
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Translate benchmark template variants")
    parser.add_argument(
        "--languages",
        type=str,
        default="all",
        help='Comma-separated language codes (e.g., "de,fr,it") or "all"',
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(VARIANTS_FILE),
        help="Path to input TSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(TRANSLATIONS_DIR),
        help="Directory for output TSV files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be translated without calling the API",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save a TSV with original and HTML-protected texts for inspection, then exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine target languages
    if args.languages == "all":
        languages = TARGET_LANGUAGES
    else:
        codes = [c.strip() for c in args.languages.split(",")]
        languages = {}
        for code in codes:
            if code not in TARGET_LANGUAGES:
                print(f"Error: unknown language code '{code}'")
                print(f"Available: {', '.join(TARGET_LANGUAGES.keys())}")
                sys.exit(1)
            languages[code] = TARGET_LANGUAGES[code]

    # Load data
    input_path = Path(args.input)
    df = load_data(input_path)
    print(f"Loaded {len(df)} template variants from {input_path}")
    print(f"Target languages: {', '.join(f'{n} ({c})' for c, n in languages.items())}")

    if args.preview:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        preview = df[["FTID", "TID", "STID"]].copy()
        preview["Template variant (en)"] = df["Template variant"]
        preview["Protected HTML"] = df["Template variant"].apply(protect_placeholders)
        preview_file = output_dir / "preview_protected.tsv"
        preview.to_csv(preview_file, sep="\t", index=False)
        print(f"Saved preview with {len(preview)} rows to: {preview_file}")
        print("Inspect the 'Protected HTML' column to verify placeholder wrapping.")
        return

    if args.dry_run:
        total_chars = df["Template variant"].str.len().sum()
        protected_chars = df["Template variant"].apply(protect_placeholders).str.len().sum()
        print(f"\n--- Dry Run ---")
        print(f"Texts to translate:    {len(df)}")
        print(f"Raw characters:        {total_chars:,}")
        print(f"Characters with markup: {protected_chars:,}")
        print(f"Languages:             {len(languages)}")
        print(f"Total API calls:       ~{len(languages) * ((len(df) + 99) // 100)}")
        print(f"\nNo API calls made. Remove --dry-run to translate.")
        return

    # Set up API client
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("Error: GOOGLE_CLOUD_PROJECT environment variable is required")
        sys.exit(1)

    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    parent = f"projects/{project_id}/locations/{location}"

    client = translate_v3.TranslationServiceClient()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Translate each language
    for lang_code, lang_name in languages.items():
        print(f"\n{'='*60}")
        print(f"Translating to {lang_name} ({lang_code})...")

        result = translate_language(client, parent, df, lang_code, lang_name)

        output_file = output_dir / f"benchmark_template_variants_{lang_code}.tsv"
        result.to_csv(output_file, sep="\t", index=False)
        print(f"  Saved: {output_file}")

    print(f"\n{'='*60}")
    print(f"Translation complete. {len(languages)} languages processed.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
