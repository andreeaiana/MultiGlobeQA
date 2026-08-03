# scripts/postedit


After templates are translated and post-edited by humans, they are instantiated with concrete entities (e.g., `{{entity_A}}` → `"Eiffel Tower"`). The instantiation can introduce minor grammar issues — wrong articles, missing prepositions, declension mismatches, etc. — that vary by language. The LLM ensemble post-editing pipeline automates these fixes; the per-row `decision` label (`unanimous` / `majority` / `fallback_gemini`) records voting confidence so downstream workflows can sample low-confidence cases if needed.

## Ensemble

Three LLMs are queried in parallel for each question:

| Model | Provider | Role |
|---|---|---|
| Gemini 3 Flash | Google `google-genai` SDK | Tiebreaker / fallback |
| DeepSeek v4-flash | DeepSeek API via OpenAI SDK (`base_url=https://api.deepseek.com`) | Voter |
| Qwen 3.5 27B (FP8) | self-hosted via vLLM (HTTP, OpenAI-compatible) | Voter |

**Voting:**
- Unanimous → apply directly
- 2/3 majority → apply majority correction
- No clear majority → fall back to Gemini's answer; row tagged `decision: fallback_gemini` for downstream sampling

## Edit Taxonomy (13 categories)

The taxonomy is the single source of truth in [`scripts/postedit/config.py`](scripts/postedit/config.py) (the `TAXONOMY_WITH_DESCRIPTIONS` constant). The same descriptions are injected into the LLM system prompt at runtime, so the models see both the acronym and its meaning when assigning categories.

| Acronym | Description |
|---|---|
| `NO_CHANGE` | The question is already correct, no edit needed |
| `ART_ADD` | Add an article (e.g., `the`, `a`, `le`, `la`, `der`) |
| `ART_DEL` | Delete an article that is now redundant |
| `ART_CHG` | Change one article to another (e.g., `le` → `la` for gender) |
| `ADP_ADD` | Add an adposition — preposition, postposition, or case/topic particle (e.g., `of`, `à`, Hindi `में`, Japanese `を`, Turkish `-de`) |
| `ADP_DEL` | Delete an adposition (preposition, postposition, or particle) that is now redundant |
| `ADP_CHG` | Change one adposition to another, including case/topic particles (e.g., `in` → `on`; Japanese `は` → `が`) |
| `DEC_CHG` | Change the declension/case ending of a word (e.g., genitive, dative, accusative) |
| `AGR_CHG` | Change agreement of a word (gender, number, or person) |
| `CONTR_CHG` | Add, remove, or change a contraction (e.g., `de le` → `du`, `a el` → `al`) |
| `ELISION` | Elide a vowel before another vowel (e.g., `le ami` → `l'ami`, `di Italia` → `d'Italia`) |
| `PUNC_CHG` | Fix punctuation around the entity, including script-specific marks (e.g., add/remove a comma, fix spacing, Greek `?` → `;`, full-width `？` `、`, Arabic `،`) |
| `OTHER` | Any other minor fix that doesn't fit the categories above |

The LLM returns the complete corrected question plus a per-edit `op`/`orig`/`fix` audit trail. Token-efficient: `NO_CHANGE` responses are ~12 tokens; full edits are ~80 tokens (measured on the GPT-5.4 reference run; DeepSeek v4-flash produces similar token budgets).

## Prerequisites

1. **Gemini API key**: `export GEMINI_API_KEY=...`
2. **DeepSeek API key**: `export DEEPSEEK_API_KEY=...`
3. **vLLM running with the Qwen model** (downloads from HuggingFace).

   vLLM is an **optional** dependency (heavy CUDA-only install ~2-5 GB; not pulled in by `uv sync` to keep the default dev environment lightweight). On the GPU server that will host Qwen, install it with:
   ```bash
   uv sync --extra vllm
   ```
   Qwen3.5 uses gated delta-net linear attention, which vLLM JIT-compiles at startup and therefore requires `nvcc` on `PATH`. On HPC systems this typically means loading the CUDA toolkit module first (the version should match `torch.version.cuda`):
   ```bash
   module load devel/cuda/12.4
   ```
   Then start the server:
   ```bash
   uv run vllm serve Qwen/Qwen3.5-27B-FP8 --host 0.0.0.0 --port 8000
   ```
   Qwen3.5-27B-FP8 is a dense 28B-parameter model in FP8 precision. FP8 execution requires a Hopper-class GPU, so it fits on a single H100 80GB (Ampere cards such as A100 cannot run the FP8 kernels). When the post-edit pipeline runs on a different node from vLLM, point it at the GPU node's resolvable hostname — not `localhost` — e.g. `export VLLM_BASE_URL=http://uc3n082:8000`.

Optional environment variables: `GEMINI_MODEL`, `DEEPSEEK_MODEL`, `DEEPSEEK_BASE_URL`, `VLLM_MODEL`, `VLLM_API_KEY`, `POSTEDIT_TIMEOUT` (default 60s).

> **Note:** if you installed vLLM into a separate venv (e.g. to work around a Python version mismatch with this project), `deactivate` it before running the pipeline. Otherwise `uv run` will warn `VIRTUAL_ENV=... does not match the project environment path .venv and will be ignored` and use the project's `.venv` anyway.

## Usage

The pipeline reads from the sampled multilingual benchmark
(`benchmark/sampled_multilingual/<kg>/<sfid>/<lang>/benchmark.jsonl/`)
and writes two outputs per (lang, kg, sfid) cell:

* `output/postedit/<lang>/<kg>/<sfid>/results.jsonl` — full nested
  ensemble records: per-model raw responses, edit ops, decision label,
  selected model, prompt + completion tokens. This is the audit trail.
* `benchmark/final/large/<kg>/<sfid>/<lang>/benchmark.jsonl/part-00000.json`
  — the post-edited benchmark questions, mirroring the
  `sampled_multilingual/` layout. Each row is the original
  benchmark row with `question` replaced by the ensemble's
  `final_question` (or the original if the postedit pipeline didn't
  touch the row, e.g. multimodal cells normalize.py skips). Every
  question lands here — none get dropped. Filter `--no-final` to skip.

SLURM granularity is **per (language, kg)**: 16 langs × 4 kgs = up to
64 jobs.

```bash
# Install dependencies (postedit deps land in the dev group)
uv sync

# Dry run — counts cells + rows without making API calls
uv run python scripts/postedit/postedit_pipeline.py --language deu --kg kwg --dry-run

# Process every cell of one language (typical SLURM-job granularity is per (lang, kg))
uv run python scripts/postedit/postedit_pipeline.py --language deu --kg kwg

# Process one cell only (for debugging)
uv run python scripts/postedit/postedit_pipeline.py --language deu --kg kwg --sfid sfid_a

# Resume after a kill: skip qids that already wrote a row
uv run python scripts/postedit/postedit_pipeline.py --language deu --kg kwg --resume skip-done

# Re-run only rows where any model previously failed
uv run python scripts/postedit/postedit_pipeline.py --language deu --kg kwg --resume retry-failed

# Run with only one model (e.g., for testing without vLLM)
uv run python scripts/postedit/postedit_pipeline.py \
    --language deu --kg multimodal \
    --models gemini --limit 10 --skip-vllm-health-check
```

### Resume modes

| Mode | Behavior |
|---|---|
| `off` (default) | Truncate the output file if it exists; start fresh. |
| `skip-done` | Read existing output, append-only on missing qids. |
| `retry-failed` | Drop rows where any model returned `ok=False` *or* `selected_model is None`; re-run only those qids. Output is rewritten atomically (`.tmp` + `os.replace`). |

### Concurrency knobs

* `--question-concurrency N` (default 16) — number of questions
  processed in parallel. Each question still spawns 3 internal LLM
  threads (one per model), so the total in-flight LLM-call ceiling is
  3·N. Tuning advice: 16 saturates Gemini Flash (~30 RPS) without
  hitting rate limits; 8 is safer if your API tier is lower.
* `--per-question-timeout-s` (default 210s) — hard wall per question.
  Tuned to comfortably exceed the worst-case retry budget
  (`REQUEST_TIMEOUT × MAX_RETRIES ≈ 180s` per LLM call); shortening
  this risks leaking worker threads when a slow API is mid-retry.
  Stuck questions get marked retryable (re-run them with
  `--resume retry-failed`).

### Per-language prompt riders

Each of the 16 target languages ships a markdown rider at
`scripts/postedit/prompts/language_riders/<lang>.md` covering the
language-specific gotchas the LLM should watch for (case agreement,
particles, script-specific punctuation, etc.). Each rider is
intentionally short (3–4 bullets, ~700–870 chars) so it doesn't bloat
the cached input prefix. The rider is auto-loaded by `--language` and
appended to `SYSTEM_PROMPT` as a `LANGUAGE NOTES` block. To revise:
edit the markdown file — no code change needed.

### Target-language template lookup

The `TEMPLATE` field of the user prompt is the **target-language**
template from the per-language column of
`benchmark/raw_eng_renumbered/templates.csv` (so the LLM sees both
TEMPLATE and QUESTION in the same language). The lookup falls back to
the row's English `base_template` if a `(vid, lang)` pair is missing
from the registry.

## MLflow tracking

Every run is tracked in MLflow by default — one parent run per SLURM
job, one nested child run per cell. Backend defaults to the project's
shared SQLite store at `sqlite:///logs/mlflow.db`; override with
`MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME`. Disable with
`--no-mlflow` (e.g. for smoke tests).

Per-cell metrics logged:

* `rows_total`, `rows_written`, `rows_errored`,
  `rows_fallback_gemini`, `rows_edit_made`, `edit_rate`,
  `fallback_rate`
* `decision_<unanimous|majority|fallback_gemini>` counts
* `op_<NO_CHANGE|ART_CHG|ADP_ADD|…>` per-category edit counts
* `total_prompt_tokens`, `total_completion_tokens`

Per-cell artifacts: the cell's `results.jsonl` is logged whole if it's
under 100 MB (the typical case for one (lang, kg, sfid)).

## Failure handling

* **Per-LLM-call retry** — 3 attempts with exponential backoff in
  `BaseLLMClient.generate` (`llm_clients.py`). Transient HTTP/JSON
  errors are recovered.
* **Partial ensemble** — if 1 of 3 models fails permanently, the
  ensemble votes among the survivors (see `ensemble.py`). The row is
  still written; if the survivors disagree the row gets
  `decision: fallback_gemini` so downstream consumers can sample it.
* **Per-question hard timeout** — `--per-question-timeout-s` (default
  210s) bails on a stuck question; the row is *not* written so it
  gets re-run on the next `--resume retry-failed` pass.
* **Pre-flight key check** — pipeline refuses to start if
  `GEMINI_API_KEY` / `DEEPSEEK_API_KEY` are missing for the requested
  models, and probes the vLLM `/health` endpoint unless
  `--skip-vllm-health-check` is set.

## Local smoke test (no SLURM, no vLLM)

For a quick correctness check on a laptop / login node — Gemini API
only, ~50 questions per language, ~2 min wall-clock:

### 1. Generate the curated sample

```bash
uv run python scripts/postedit/select_smoke_sample.py
```

Writes:
* ``output/postedit/smoke/eng.jsonl`` and ``ron.jsonl`` — 50 rows each
  (20 NO-EDIT-likely, 20 SINGLE-EDIT-likely, 10 HARD/multi-entity).
* ``output/postedit/smoke/<lang>_manifest.csv`` — bucket label +
  heuristic reason per qid, so you can review the picks before / after
  the run.

The selector is deterministic (fixed seed 42); re-running picks the
same 100 rows. Add ``--languages bul,deu,...`` for other languages or
``--per-bucket 30,30,15`` to pick more.

### 2. Run on English (control)

```bash
export GEMINI_API_KEY=...
uv run python scripts/postedit/postedit_pipeline.py \
    --input output/postedit/smoke/eng.jsonl \
    --output output/postedit/smoke/eng_results.jsonl \
    --language eng \
    --templates-csv benchmark/raw_eng_renumbered/templates.csv \
    --models gemini \
    --skip-vllm-health-check \
    --no-mlflow
```

### 3. Run on Romanian

```bash
uv run python scripts/postedit/postedit_pipeline.py \
    --input output/postedit/smoke/ron.jsonl \
    --output output/postedit/smoke/ron_results.jsonl \
    --language ron \
    --templates-csv benchmark/raw_eng_renumbered/templates.csv \
    --models gemini \
    --skip-vllm-health-check \
    --no-mlflow
```

### 4. What to look for

For each output JSONL:

```bash
# Decision distribution
jq -r '.decision' output/postedit/smoke/eng_results.jsonl | sort | uniq -c

# How many rows had any edit applied
jq -r 'select(.original_question != .final_question) | .question_id' \
    output/postedit/smoke/eng_results.jsonl | wc -l

# Which categories the model proposed
jq -r '.model_outputs.gemini.ops[]?' \
    output/postedit/smoke/eng_results.jsonl | sort | uniq -c
```

## Output Format

JSONL, one record per question, including:
- `question_id` (canonical key for joining back to the input)
- `original_question`, `marked_question`, `instantiations`
- `source_metadata` (`vid`, `tid`, `sfid`, `kg_source`)
- `model_outputs` — per-model `raw`, `ops`, `final_text`, token usage
- `decision` — `unanimous` | `majority` | `fallback_gemini`
- `selected_model` — name of the model whose output became `final_question`
- `final_question` — the chosen corrected text

## Export and Statistics

After running the pipeline (which writes one JSONL per language at `output/postedit/results_<lang>.jsonl`), use these two scripts to flatten the results and compute statistics:

```bash
# Export per-language JSONL/CSV/XLSX files plus a combined all-languages workbook
uv run python scripts/postedit/export_postedit.py

# Filter to specific languages or formats
uv run python scripts/postedit/export_postedit.py --languages en,fr,de --format csv

# Compute per-language and aggregate statistics
uv run python scripts/postedit/postedit_statistics.py
```

Both scripts auto-discover every `results_*.jsonl` file in `output/postedit/` by default, but you can also pass `--input` to point at a single file.

**Export output structure:**

```
output/postedit/exports/
├── en/
│   ├── postedit_en.jsonl       # full nested records
│   ├── postedit_en.csv         # flattened, one row per question
│   ├── postedit_en.xlsx        # 2 sheets: Questions, Disagreements
│   └── statistics_en.json
├── fr/...
├── all_languages.jsonl
├── all_languages.csv
├── all_languages.xlsx          # 2 sheets: All Questions, Per Language Summary
└── statistics_aggregate.json
```

**Statistics computed (per language and aggregated):**

- Decision distribution (`unanimous` / `majority` / `fallback_gemini`)
- Edit rate, fallback rate (`decision == 'fallback_gemini'`), edit-distance histogram and average
- Counts per edit category (the 13 taxonomy ops)
- Per-model behavior: proposed edit rate, failure rate, ops distribution, total tokens
- Pairwise model agreement (gemini vs deepseek, gemini vs qwen, deepseek vs qwen)
- Top 10 templates (`tid`) with the highest edit rate
- Op co-occurrence (which categories appear together in edited questions)
- Disagreement patterns (which model is the outlier when models disagree)
- Entity-position analysis (does the edit fall near the first or second entity)
- Op transitions (different models proposing different ops for the same question)
- Cross-language edit-type heatmap (13 ops × N languages)

## Out of scope (deferred)

* **`gold_triples.{jsonl,parquet}` copy + parquet regen** into
  `benchmark/final/large/`. After the postedit run completes, copy each
  cell's gold triples across (one level above the per-language
  `benchmark.jsonl/`) and regenerate parquet:
  ```bash
  for kg in kwg osmh3kg worldkg multimodal; do
      for sfid_dir in benchmark/sampled_multilingual/$kg/*/; do
          cp -a "${sfid_dir}gold_triples."{jsonl,parquet} \
              "benchmark/final/large/$kg/$(basename $sfid_dir)/"
      done
  done
  uv run python -m scripts.regenerate_benchmark_parquet --benchmark-root benchmark/final/large
  ```

## Tests

```bash
uv run pytest tests/postedit/ -v
```

