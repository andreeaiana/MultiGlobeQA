# benchmark_models

Evaluates LLMs on the generated benchmark. Models are organized into **capability tiers**: tier 1 (plain parametric knowledge), tier 2 (reasoning-enhanced with CoT or native thinking), and tier 3 (tool-augmented via Smolagents CodeAgent, with KG / web / KG+web sub-tiers). Produces structured JSONL prediction records that the eval pipeline (`benchmark_eval/`) scores.

## Installation

```bash
# Install the model section's extra dependencies (vLLM, provider SDKs, etc.)
uv sync --extra benchmark_models
```

## API Key Configuration

API keys are needed for hosted model backends. Set the environment variable, or
pass the key per run:

```bash
export GOOGLE_API_KEY=...
python -m benchmark_models smoke -v gemini-3-flash-plain --api-key YOUR_KEY --n 5
```

The following environment variables are recognized:

| Variable | Backend |
|---|---|
| `GOOGLE_API_KEY` | Google (Gemini 3 Flash) |

## Quickstart — Smoke Test

```bash
# DummyModel (no API key needed — validates the pipeline end-to-end)
python -m benchmark_models smoke -v dummy --n 5

# Gemini tier 1 (plain, minimal thinking)
python -m benchmark_models smoke -v gemini-3-flash-plain --n 10

# Gemini tier 2 (reasoning, high thinking)
python -m benchmark_models smoke -v gemini-3-flash-thinking-tier2 --n 5

# With concurrent batching (5 parallel API calls per batch)
python -m benchmark_models smoke -v gemini-3-flash-plain --n 20 --batch-size 5

# Tier 3b — web search agent (Smolagents CodeAgent)
python -m benchmark_models smoke -v gemini-3-flash-tier3b --n 3

# Tier 3a — KG agent (requires KG snapshot access)
python -m benchmark_models smoke -v gemini-3-flash-tier3a --n 3
```

Predictions are written to `results/smoke/<variant>/predictions.jsonl`. A `run.log` file is created alongside for persistent logging. Resume is automatic — re-running the same command skips already-completed questions.

Tier 3 predictions include a full execution trace in the `tool_calls` field — each step records the Python code the model wrote, the execution output, and any errors. This trace is used by the eval pipeline for tool-use analysis.

## Running at scale

The share is chunked by KG bucket, spatial-function family (SFID) and language:
`<size>/<kg>/<sfid>/<lang>/benchmark.jsonl/`. Distribute work along those axes, one
job per cell, rather than by line count. The runner's walker concatenates whatever
sub-tree you point it at, so a job can target anything from a single (kg, sfid, lang)
cell up to the whole share.

Granularity that works in practice:

| tier | granularity | why |
|---|---|---|
| 1, 2 | `(lang, kg)` | single-turn; a cell finishes inside a normal job wall |
| 3 | `(lang, kg, sfid)` | agent loops run 3-10 s per question on web search, ~30 s on oracle modes and up to ~250 s on KG retrieval, so a whole KG bucket overruns |

Per-KG SFID coverage is 13 + 13 + 9 = 35 standard cells plus 2 multimodal sub-cells,
so one tier-3 variant across all languages is 17 x 37 = 629 jobs.

The multimodal bucket fans out into one sub-job per underlying KG so a tier-3a / 3c
job can pass a real `--kg-source`: the KG bridge has no `multimodal` schema and would
raise `ValueError: Unknown KG: 'multimodal'`.

Runs resume by default. Point a rerun at the same output path and the runner skips
questions that already have predictions; `--no-resume` forces a clean run.

For vLLM variants, start one `vllm serve` per job and point the harness at it with
`VLLM_BASE_URL`. `--enable-prefix-caching` lets vLLM reuse KV cache across the
questions in a cell.

### Tier-3 tuning knobs

Each knob has a default → per-variant override in `models.yaml` → CLI override (highest priority). Tier-3 only; passing them on tier-1 / tier-2 raises a CLI error.

| Knob | CLI flag | Default | Purpose |
|---|---|---|---|
| Max agent steps | `--max-steps N` | 10 (headline-run cap) | Per-question step cap. Raised from Smolagents' upstream 6 after observing 80 % cap_hit on qwen35-27b-tier3b at 6 — the open-weights 27B class needs the extra reasoning budget to be apples-to-apples with the closed-source tier-3b column. Substituted into the prompt's `$max_tool_calls` slot so prompt and agent loop never disagree. |
| Per-step `max_tokens` | `--max-tokens N` | 4096 (3a/3b/3c) | Per-LLM-call output cap. With WKT simplification active on both the oracle and `EntityService.get_entity` paths, 4096 covers shapely / final_answer outputs. On vLLM variants that serve at 32 K context this leaves ~28 K of input headroom per call; on the YaRN-extended `qwen35-27b-tier3a` / `qwen35-27b-tier3c` variants (65 K, see "vLLM context windows" below) it leaves ~60 K. |
| Concurrency | `--max-concurrent N` | 4 | `predict_batch` worker count. Multiplied by `max_steps` gives provider RPM pressure. |
| Web-search timeout | `--web-search-timeout S` | 15.0 | Per-request DDG-lite timeout. Tier-3b / 3c only (3a has no web tool). |

## CLI Subcommands

```bash
# Smoke test a single variant (see above)
python -m benchmark_models smoke -v <variant> [options]

# List all variants from models.yaml
python -m benchmark_models list-variants
```

## Model Variant Catalogue

All model variants are defined in [`benchmark_models/models.yaml`](benchmark_models/models.yaml). Each entry specifies a backend, tier, model identifier, and runtime knobs. The variants in scope for the current iteration:

| Backend | Tier 1 (plain) | Tier 2 (reasoning) | Tier 3a (KG+Py) | Tier 3b (Web+Py) | Tier 3c (All) |
|---|---|---|---|---|---|
| **Google** | gemini-3-flash-plain | gemini-3-flash-thinking-tier2 | gemini-3-flash-tier3a | gemini-3-flash-tier3b | gemini-3-flash-tier3c |
| **vLLM** (Qwen 35B) | qwen35-a3b-plain | qwen35-a3b-thinking-tier2 (SC@3) | qwen35-a3b-tier3a | qwen35-a3b-tier3b | qwen35-a3b-tier3c |
| **vLLM** (Qwen 27B) | qwen35-27b-plain | qwen35-27b-tier2 | qwen35-27b-tier3a | qwen35-27b-tier3b | qwen35-27b-tier3c |
| **vLLM** (Gemma 3) | gemma-3-27b-it-plain | gemma-3-27b-it-tier2 | gemma-3-27b-it-tier3a | gemma-3-27b-it-tier3b | gemma-3-27b-it-tier3c |

Multimodal `-image` variants exist for Google (`gemini-3-flash-tier3{a,b,c}-image`) and the three vLLM models (`{qwen35-27b,qwen35-a3b,gemma-3-27b-it}-{tier1,tier2,tier3a,tier3b,tier3c}-image`). Text-only `-name` ablation variants mirror them.

### Multimodal slice

The `multimodal` bucket is a *slice* of `osmh3kg` and `worldkg` — the same questions but with an image attached to one entity slot — not its own KG. On-disk layout under `benchmark/final/<size>/` (`<size>` ∈ {`large`, `small`}):

```
benchmark/final/<size>/multimodal/
  osmh3kg/
    images/<qid>_<slot>.jpg          # shared across all languages
    <lang>/benchmark.jsonl/part-*.json
    gold_triples.jsonl/part-*.json   # shared across all languages
  worldkg/
    images/<qid>_<slot>.jpg
    <lang>/benchmark.jsonl/part-*.json
    gold_triples.jsonl/part-*.json
```

**Image path convention**: each JSONL row stores `images[*].path` as a short relative segment (`images/<file>`) interpreted by the loader as relative to `<dataset_root>/multimodal/<kg_source>/`, where `<dataset_root>` is the ancestor of the shard containing the `multimodal/` directory. `Example.from_benchmark_row` resolves this to an absolute path at load time (`benchmark_models/image_paths.py`), so downstream code sees absolute filesystem paths. A legacy-format shim accepts the old generator stub `benchmark/eng/multimodal/<kg>/images/<file>` for trees that haven't been migrated.

**Maintenance**: if the question generator emits the legacy stored-path stub on new benchmark builds, run the one-shot rewriter to migrate the paths to the published convention. The script is atomic per-shard and idempotent:

```bash
uv run python -m scripts.normalize_multimodal_image_paths --root benchmark/final/large/ [--dry-run]
```

**Per-KG dispatch**: fan the `multimodal` bucket out into one job per underlying KG so tier-3a/3c jobs can pass a real `--kg-source` (the bridge's `schema_for(...)` doesn't know a `multimodal` schema and would raise `ValueError: Unknown KG: 'multimodal'`). Generated layout:

```
jobs/benchmark_models/<size>/<variant>/seed42/
    eng_multimodal_osmh3kg_usekg.sh   (867 osmh3kg multimodal questions)
    eng_multimodal_worldkg_usekg.sh   ( 80 worldkg multimodal questions)
```

with output mirroring the data tree:

```
results/runs/<size>/<variant>/usekg/seed42/eng/multimodal/
    osmh3kg/predictions.jsonl
    worldkg/predictions.jsonl
```

The split is uniform across tiers — tier-1/2/3b multimodal scripts also emit per-KG sub-cells, even though only tier-3a/3c functionally needs the dispatch. Keeps the output layout consistent and predictions cleanly separated.

**Smolagents-specific note**: tier-3 multimodal runs require a step callback to prevent the task image from being replayed on every `ActionStep`. Without it, vLLM's `--limit-mm-per-prompt '{"image":1}'` rejects Step ≥2 with `At most 1 image(s) may be provided in one prompt`. The callback is registered automatically in `SmolagentsModel._build_code_agent` / `SmolagentsOracleModel._build_code_agent` — see `_clear_replayed_step_image` in `benchmark_models/tier3.py` for the mechanism (mirrors the upstream `vision_web_browser.save_screenshot` pattern).

**Implementation status**: Google + vLLM-served backends are implemented for tier 1 + tier 2. Tier 3 is implemented via Smolagents CodeAgent with three sub-tiers (3a: KG + Python, 3b: Web + Python, 3c: all tools). API backends use concurrent batching via ThreadPoolExecutor; the vLLM-served backend (`vllm_http`) talks to a per-cell `vllm serve` over the OpenAI Chat-Completions API and inherits the same client-side batching. MLflow experiment tracking is integrated.

**Tier 3 sub-tiers**:
- **Tier 3a** (KG + Python): model has KG access tools (`find_entity`, `get_entity`, `find_entities_by_type`, `list_categories`) + Python sandbox (shapely, geopy, pyproj, h3). Must write spatial computations from raw KG data.
- **Tier 3b** (Web + Python): model has web search + page fetch + Python sandbox. No KG access. Tests whether web info can substitute for a structured KG.
- **Tier 3c** (KG + Web + Python): all tools combined. Most realistic scenario.

**Backend notes**:
- **Google (Gemini)**: structured output via `response_json_schema`. Thinking via `thinking_config` with `thinking_level` parameter. Full reasoning trace exposed in response.

### vLLM context windows

All open-weights vLLM variants serve at their native context windows except for tier-3a / tier-3c on qwen35-27b, where the agent loop's per-step input growth (~5-7 K tokens of cumulative tool outputs replayed each turn) exhausts the native 32 K window within 4–5 steps. Those two variants extend to **65 K** via YaRN rope scaling (factor=2.0, `original_max_position_embeddings=32768`) — the standard inference-time technique documented on the [Qwen3.5 model card](https://huggingface.co/Qwen/Qwen3.5-27B-Instruct). Pass this to vLLM as `--hf-overrides '{"rope_scaling": {...}}'`, which overrides the HuggingFace config.json at engine init. Perplexity degradation at factor=2.0 is well under 1 % in the 28 K–60 K regime where these loops actually consume the budget. Tier-1 / tier-2 / tier-3b qwen variants stay at native 32 K because they don't need the extended window. Configuration lives on the `_qwen35_27b_yarn_64k` YAML anchor in `models.yaml`.

## Prompt Caching

Closed-source backends report cache hits via different fields. ``Prediction.usage.cached_tokens`` is populated end-to-end so the eval pipeline can quote real (post-cache) input cost on every ``AggregateReport``:

| Backend | Source field | Notes |
|---|---|---|
| Google | `response.usage_metadata.cached_content_token_count` | Explicit context caching via `client.caches.create()`. The backend splits the rendered prompt at `Question:\n`, gates on a 4096-token minimum via `count_tokens`, creates one cache per `sha256(prefix + model_id)` with 1-hour TTL, and references `cached_content=<name>` on subsequent calls. Cached input bills at ~25% of the normal rate. Cache-expired errors are detected, the registry is invalidated, and the call is retried without the cache reference. Disable per-variant via `enable_context_cache: false` on `GoogleVariantConfig`. |
| vLLM | n/a | Internal KV cache; no API-level cache concept. `cached_tokens` stays at 0. |

`ScoreRecord.cached_tokens` (denormalised from `Prediction.usage.cached_tokens`) rolls up into `AggregateReport.total_cached_tokens`.

## KG Layout (tier-3a / 3c)

Production knowledge graphs live at `benchmark/kgs/<kg_name>/` as Delta tables of `(subject, predicate, object)` triples (osmh3kg additionally has a `date` column). Two optional Z-ordered sidecar tables sit alongside the main table to keep query latency bounded — `subject_index/` for fast subject-keyed reads, `name_index/` for fuzzy `find_entity(name)` substring search. The bridge auto-detects both sidecars and falls back to the slower main-table scan when they are absent.

```
benchmark/kgs/
├── kwg/
│   ├── _delta_log/
│   ├── part-*.snappy.parquet
│   ├── subject_index/         # Z-ordered on subject
│   │   ├── _delta_log/
│   │   └── part-*.snappy.parquet
│   └── name_index/            # label-only, Z-ordered on name_lower
│       ├── _delta_log/
│       └── part-*.snappy.parquet
├── osmh3kg/   (same layout)
└── worldkg/   (same layout)
```

Invoke one runner per `(lang, kg)` cell and pass `--kg-source <kg>`; the runner resolves `kg_path = kg_root / kg_source` (default root `benchmark/kgs/`, overridable via `--kg-root` or the `BENCHMARK_KGS_ROOT` env var) and passes it to `EntityService(TripleBridge(...))`. The bridge expects a Delta-format directory (parquet shards + `_delta_log/`).
- **vLLM-served (open weights)**: start a per-cell `vllm serve`; the harness talks to it over the OpenAI Chat-Completions API. Structured output via `response_format={"type": "json_schema", ...}` (vLLM serve supports the same envelope as OpenAI for guided decoding under the hood). `--enable-prefix-caching` lets vLLM reuse KV cache across questions in the same cell. Thinking-toggle handling (Qwen 3+ / Gemma 4) is driven via the per-variant `thinking` field and serialised through the chat-completions request — no in-process tokenizer-template gymnastics needed.

## Output Format

Each prediction is one JSON line in the predictions JSONL file:

```json
{
  "question_id": "1285b5ef-...",
  "tid": "1.1",
  "sfid": "A",
  "answer_type": "Numeric (float)",
  "model_id": "gemini-3-flash-plain",
  "tier": "tier1",
  "language": "en",
  "answer": 12.5,
  "unit": "km",
  "refusal": false,
  "usage": {"input_tokens": 150, "output_tokens": 20, ...},
  "latency_ms": 1234.5,
  ...
}
```

Each run pins the JSON response schema from the per-row `ground_truth_type`, so models are told the *shape* of the expected answer (number, boolean, coordinate pair, etc.) before reasoning. This is a deliberate separation of format compliance from spatial-reasoning capability — the free-form ablation on workhorse models measures the gap.

## Template Registry

The template registry maps each template ID to its answer type, spatial function, and conventional unit. It is derived from the benchmark JSONL itself (not from any external sheet) and bundled at `benchmark_models/schemas/template_registry.json`:

```bash
# Regenerate from the finalized benchmark
uv run python -m scripts.build_template_registry_from_benchmark \
    --benchmark benchmark/final/large/ \
    --output benchmark_models/schemas/template_registry.json
```

Runtime dispatch uses `resolve_canonical_answer_type(ground_truth_type, fallback, tid=...)`, which consults `TID_TO_CANONICAL_OVERRIDE` first (handles benchmark mislabels and overloaded vocab tags like `categorical` / `grid_cell_id`), then the row's `ground_truth_type`, then the registry's `TemplateInfo.answer_type` as a final fallback. The override map is the single source of truth: the build script and the runtime resolver both import the same constant from `benchmark_models/schemas/type_registry.py`. Eval-side `loaders.py` reads the registry's already-overridden `answer_type` for known TIDs, and consults the override directly for new TIDs the registry hasn't been regenerated for.

## MLflow Experiment Tracking

Runs are logged to MLflow by default. Params and metrics cover variant config, token / latency / tool-call counters, and prediction-shape statistics; artifacts include `predictions.jsonl`, `run.log`, and the prompt templates used.

```bash
# View runs
mlflow ui --backend-store-uri sqlite:///logs/mlflow.db

# Disable for a run
python -m benchmark_models smoke -v gemini-3-flash-plain --n 5 --no-mlflow
```

The experiment name defaults to `Benchmarked_Models`; override via `MLFLOW_EXPERIMENT_NAME`.

## Oracle Baselines

The oracle baselines inject each question's gold triples into the prompt. Four modes:

| Mode | Tier | Triples are formatted as |
|---|---|---|
| `--oracle-mode structured` | 1 | JSON array, short predicate names (`rdf:type`, `geo:asWKT`), prefixed by a PREFIXES header |
| `--oracle-mode raw` | 1 | N-Triples-like text, full URIs, one `<s> <p> <o> .` per line |
| `--oracle-mode verbalized` | 1 | Natural-language prose, one sentence per entity (entity-aggregated) |
| `--oracle-mode t3` | 3 | Structured JSON injected into a tier-3 CodeAgent with no KG/web tools, Python sandbox only |

Example invocations:

```bash
# Tier-1 oracle — structured JSON injection
python -m benchmark_models smoke -v gemini-3-flash-plain --oracle-mode structured --n 5

# Tier-1 oracle — raw N-Triples injection
python -m benchmark_models smoke -v gemini-3-flash-plain --oracle-mode raw --n 5

# Tier-1 oracle — verbalized prose injection
python -m benchmark_models smoke -v gemini-3-flash-plain --oracle-mode verbalized --n 5

# Tier-3 oracle — gold triples + Python sandbox, no KG/web tools
python -m benchmark_models smoke -v gemini-3-flash-tier3a --oracle-mode t3 --n 5

# Custom gold-triples path (default: benchmark/final/large/, walked by discover_gold_shards)
python -m benchmark_models smoke -v gemini-3-flash-plain --oracle-mode structured \
    --gold-triples path/to/triples_share/ --n 5
```

**Gold triples file format:** JSONL with one `{question_id, gold_triples}` row per question; `gold_triples` is a JSON-encoded list of `{subject, predicate, object}` triples. See `output/sample_gold_triples.jsonl` for an example. Coverage of every benchmark `question_id` is required — runs abort if a question is missing gold.

## Recovering from Transient Errors

Backends retry transient SDK failures (rate limits, 5xx, timeouts) with exponential backoff. Non-retryable errors produce a `final_state="error"` prediction. To re-run only errored questions from a prior batch:

```bash
python -m benchmark_models smoke -v gemini-3-flash-plain --n 100 --retry-errors
```

The current `predictions.jsonl` is copied to a timestamped `.bak`, errored rows are dropped, and the normal resume logic re-runs their question_ids. Genuine abstentions and cap-hits are preserved.

## Testing

```bash
uv run pytest tests/benchmark_models/ -v
```
