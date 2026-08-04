# pipelines/postedit

Instantiating a translated template with concrete entities (`{{entity_A}}` ->
`"Eiffel Tower"`) can break the surrounding grammar: wrong articles, missing
prepositions, declension mismatches. This pipeline repairs those artefacts with
an ensemble of three LLMs that vote on the corrected question, and records a
per-row `decision` label carrying the voting confidence.

## Ensemble

| Model | Provider |
|---|---|
| Gemini 3 Flash | Google, `google-genai` SDK |
| DeepSeek v4-flash | DeepSeek API via the OpenAI SDK |
| Qwen 3.5 27B (FP8) | self-hosted vLLM, OpenAI-compatible HTTP |

Voting (`ensemble.py`) compares the NFC-normalised text of the models that succeeded:

| `decision` | Condition |
|---|---|
| `unanimous` | at least two models succeeded and all of them agree |
| `majority` | more than half of the successful models agree |
| `fallback_gemini` | no majority, or fewer than two models succeeded; the text falls back to Gemini, else the first successful model in `gemini, qwen, deepseek` order, else the original question |

## Edit taxonomy

| Acronym | Description |
|---|---|
| `NO_CHANGE` | the question is already correct, no edit needed |
| `ART_ADD` | add an article (English "the", French "la", German "der") |
| `ART_DEL` | delete an article that is now redundant |
| `ART_CHG` | change one article to another (French "le" -> "la" for gender) |
| `ADP_ADD` | add an adposition: preposition, postposition, or case/topic particle |
| `ADP_DEL` | delete an adposition that is now redundant |
| `ADP_CHG` | change one adposition to another, including case/topic particles ("in" -> "on") |
| `DEC_CHG` | change the declension/case ending of a word (genitive, dative, accusative) |
| `AGR_CHG` | change agreement of a word (gender, number, or person) |
| `CONTR_CHG` | add, remove, or change a contraction (French "de le" -> "du", Spanish "a el" -> "al") |
| `ELISION` | elide a vowel before another vowel ("le ami" -> "l'ami") |
| `PUNC_CHG` | fix punctuation around the entity, including script-specific marks (Greek question mark, full-width CJK punctuation, Arabic comma) |
| `OTHER` | any other minor fix that does not fit the categories above |

`config.TAXONOMY_WITH_DESCRIPTIONS` holds these descriptions in full, with their
non-Latin examples, and is injected into the system prompt. An op outside the
taxonomy is coerced to `OTHER` rather than rejected.

## Prerequisites

`GEMINI_API_KEY` and `DEEPSEEK_API_KEY` in the environment, plus a vLLM server
hosting the Qwen model. vLLM is an optional extra, not installed by `uv sync`:

```bash
uv sync --extra vllm
uv run vllm serve Qwen/Qwen3.5-27B-FP8 --host 0.0.0.0 --port 8000
```

Optional environment variables: `GEMINI_MODEL`, `DEEPSEEK_MODEL`,
`DEEPSEEK_BASE_URL`, `VLLM_MODEL`, `VLLM_BASE_URL`, `VLLM_API_KEY`,
`POSTEDIT_TIMEOUT` (seconds per LLM call, default 60).

## Usage

The pipeline walks
`benchmark/sampled_multilingual/<kg>/<sfid>/<lang>/benchmark.jsonl/` and writes
two outputs per (language, kg, sfid) cell:

* `output/postedit/<lang>/<kg>/<sfid>/results.jsonl` -- the ensemble audit
  trail, one nested record per post-edited question.
* `benchmark/final/large/<kg>/<sfid>/<lang>/benchmark.jsonl/part-00000.json` --
  the post-edited benchmark, mirroring the input layout. Every source row lands
  here: `question` is replaced by `final_question`, or kept as the original for
  rows the pipeline did not touch (rows `normalize.py` could not mark
  unambiguously). `--no-final` skips this output.

```bash
# Count cells and rows without calling any API
uv run python pipelines/postedit/postedit_pipeline.py --language deu --kg kwg --dry-run

# One language, one KG bucket, then a single cell within it
uv run python pipelines/postedit/postedit_pipeline.py --language deu --kg kwg
uv run python pipelines/postedit/postedit_pipeline.py --language deu --kg kwg --sfid sfid_a

# One model only, no vLLM
uv run python pipelines/postedit/postedit_pipeline.py \
    --language deu --kg multimodal \
    --models gemini --limit 10 --skip-vllm-health-check
```

`--models` takes a comma-separated subset of `gemini`, `qwen`, `deepseek`.
`--input` / `--output` switch to single-file mode and bypass the cell walk.
`--help` lists the remaining flags.

### Resume

| `--resume` | Behaviour |
|---|---|
| `off` (default) | truncate the cell's output file and start fresh |
| `skip-done` | append only the qids missing from the existing output |
| `retry-failed` | rewrite the output, dropping and re-running rows where any model returned `ok=False` or `selected_model` is null |

### Prompt inputs

The system prompt gains a `LANGUAGE NOTES` block from
`pipelines/postedit/prompts/language_riders/<lang>.md` when a rider exists for
`--language`. The prompt's `TEMPLATE` field is the target-language template,
read from `data/templates/templates_<lang>.tsv` by `vid`, so the LLM sees
TEMPLATE and QUESTION in one language; a missing `(vid, lang)` pair falls back
to the English template, then to the row's `base_template`.

## Output format

One JSON object per line in `results.jsonl`:

| Field | Contents |
|---|---|
| `question_id` | canonical key for joining back to the input |
| `language` | ISO 639-3 code |
| `original_question`, `marked_question`, `base_template`, `instantiations` | the input row |
| `source_metadata` | `vid`, `tid`, `sfid`, `kg_source` |
| `model_outputs` | per model: `raw`, `ops`, `final_text`, `prompt_tokens`, `completion_tokens`, `ok`, `error`, and `entity_edits` (any non-exact polish applied inside `{{...}}` markers) |
| `decision` | `unanimous`, `majority` or `fallback_gemini` |
| `selected_model` | the model whose text became `final_question`, null if every model failed |
| `final_question` | the chosen corrected text |

## Export and statistics

Both scripts discover every per-cell `results.jsonl` under `output/postedit/`
and write to `output/postedit/exports/`.

```bash
# Per-language JSONL/CSV/XLSX plus a combined all-languages workbook
uv run python pipelines/postedit/export_postedit.py --languages deu,fra --format csv

# Per-language statistics_<lang>.json plus statistics_aggregate.json
uv run python pipelines/postedit/postedit_statistics.py
```

The statistics cover the decision distribution, edit and fallback rates, edit
distances, counts per taxonomy op, per-model behaviour and pairwise agreement,
templates with the highest edit rate, op co-occurrence, disagreement patterns
and a cross-language op heatmap. They are computed against the released
benchmark under `--final-root` (default `benchmark/final`), which supplies the
true row universe; `--no-final` reports over the post-edited rows only.

## MLflow and failures

Runs are tracked in MLflow by default, backed by `sqlite:///logs/mlflow.db` and
overridable with `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME`; `--no-mlflow`
disables it. LLM calls are retried three times with exponential backoff, a
permanently failing model is dropped from the vote, and a question exceeding
`--per-question-timeout-s` is left unwritten so that `--resume retry-failed`
picks it up.
