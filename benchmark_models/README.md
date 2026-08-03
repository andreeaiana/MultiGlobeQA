# benchmark_models

Evaluates LLMs on the generated benchmark. Models are organized into **capability tiers**: tier 1 (plain parametric knowledge), tier 2 (reasoning-enhanced with CoT or native thinking), and tier 3 (tool-augmented via Smolagents CodeAgent, with KG / web / KG+web sub-tiers). Produces structured JSONL prediction records that the eval pipeline (`benchmark_eval/`) scores.

## Installation

```bash
# Install the model section's extra dependencies (vLLM, provider SDKs, etc.)
uv sync --extra benchmark_models
```

## API Key Configuration

API keys are needed for closed-source model backends. Two options:

```bash
# Option 1: copy the template and fill in your keys (gitignored)
cp scripts/set_api_keys.sh.example scripts/set_api_keys.sh
# ... edit scripts/set_api_keys.sh with your real keys ...
source scripts/set_api_keys.sh

# Option 2: pass directly via CLI flag
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

## Running at Scale — Per-Cell Parallel Runs

The post-edited share `benchmark/final/<size>/<kg>/<sfid>/<lang>/benchmark.jsonl/` (`<size>` ∈ {`large`, `small`}, default `large`) is already chunked by KG bucket, structural function (SFID), and language. Headline runs distribute work along those axes — one SLURM job per cell — rather than artificial line-count chunks. The runner's share-root walker concatenates within whatever sub-tree you point it at, so a job can target any granularity from a single (kg, sfid, lang) cell up to the whole share.

The recommended invocation is the SLURM generator in `scripts/benchmark_models/generate_slurm.py`. It emits one script per cell with the right defaults per variant (GPU partition for vLLM, CPU partition for API; granularity = `cell` for tier 1/2, `sfid` for tier 3 since per-row agent loops would blow past any single-job time budget).

```bash
# Emit per-cell jobs for one closed-source tier-1 variant (4 KG buckets × 17 langs = 68 scripts).
uv run python scripts/benchmark_models/generate_slurm.py \
    --variant gemini-3-flash-plain

# Tier-3 variant: auto-splits to per-(lang, kg, sfid). Per-KG SFID
# coverage on disk is 13 + 13 + 9 = 35 standard cells (kwg has no D,
# osmh3kg has no I, worldkg populates only 9), plus 2 multimodal
# sub-cells (the multimodal bucket fans out into one per-underlying-KG
# sub-script — osmh3kg + worldkg — so tier-3a/3c jobs can pass a real
# --kg-source value): 17 × (35 + 2) = 629 scripts.
uv run python scripts/benchmark_models/generate_slurm.py \
    --variant qwen35-a3b-tier3a

# Single-language smoke; pick the kgs you want.
uv run python scripts/benchmark_models/generate_slurm.py \
    --variant gemma-3-27b-it-plain --languages eng --kgs kwg,worldkg

# Run on the small benchmark instead (--size small; default large).
uv run python scripts/benchmark_models/generate_slurm.py \
    --variant qwen35-a3b-tier3b --size small

# Submit whichever subset you want.
sbatch jobs/benchmark_models/large/gemini-3-flash-plain/eng_kwg.sh
```

Each script ends up calling the smoke command against the right sub-tree:

```bash
uv run python -m benchmark_models smoke \
    --variant <variant> \
    --benchmark benchmark/final/<size>/<kg>/[<sfid>/] \
    --language <lang> \
    --output results/runs/<size>/<variant>/<lang>/<kg>/[<sfid>/]predictions.jsonl \
    --n 0 \
    --batch-size <tier-dependent>
```

`--n 0` is the sentinel for "process every row in the loaded shard" — the smoke command's default 10 is for ad-hoc development only.

**Per-tier guidance:**

- **Tier 1 / 2 (API backends)**: granularity `cell` is enough. One (lang, kg) cell is ~hundreds of questions; the ThreadPoolExecutor + provider rate limit is the bottleneck.
- **Tier 3 (any backend)**: granularity `sfid`. Per-question agent loops run 3–10 s/q on tier-3b, ~30 s/q on oracle modes, and ~250 s/q on tier-3a/3c with `--use-kg` (each step issues a Spark query). The generator's `_compute_time_budget` scales the SLURM `--time` cap by `--n`, concurrency, and a cluster-specific cold-start (Helix vLLM ≈ 15 min, bwUC3 ≈ 5 min, API ≈ 30 s), then rounds up to the next 30 min and clips at the 24 h partition wall (resume picks up after a wall-clip).
- **vLLM (HTTP-served)**: one GPU per job (the script header allocates `--gres=gpu:1` on bwUniCluster or `--gres=gpu:H200:1` on Helix). Each SLURM cell starts its own `vllm serve` in the background, polls `/health`, runs the smoke against `localhost:<port>/v1`, then traps to kill serve on exit. Cold-start is ~3-5 min on first run (HF weight download + JIT) and ~30-60s on subsequent jobs (warm Torchinductor + vLLM cache).

**`--cluster bwuc3 | helix`** picks the partition + GPU type + workspace path bundle for the target cluster. Default `bwuc3` keeps existing behaviour. `helix` swaps `gpu_h100` → `gpu-single`, `gpu:1` → `gpu:H200:1`, `single` → `cpu-single`, and wraps `vllm serve` in a Singularity container (Helix host glibc is too old for the in-tree vLLM wheel). The Helix path requires `$SBR_WS` and `$SBR_VLLM_SIF` env vars set in `~/.bashrc` — see `scripts/postedit/README.md` for the one-time setup.

**Cost / time pre-flight:**

```bash
# Order-of-magnitude check before submitting. See benchmark_models/cost_estimator.py.
uv run python -m benchmark_models cost-estimate --variant all --n 65000
```

**Caveats:**

- All cells must use the **same scoring spec** (`benchmark_eval/scoring_spec.json`) and the **same prompt templates** for the predictions to be comparable. Don't mix runs with manual prompt edits in flight.
- MLflow logging is per-process; one MLflow run per cell is the default. Pass `--no-mlflow` to the script template (via `generate_slurm.py --no-mlflow`) if the cluster runner shouldn't write to MLflow.
- Score-aggregation joins on `question_id` and is order-independent — concatenating per-cell `predictions.jsonl` files for the eval pipeline is safe at any cell order.

### Headline-run script generation — variant × flag tables

The tables below enumerate the in-scope variants per tier × regime and the `generate_slurm.py` flags needed to emit the SLURM scripts for that regime. Each row corresponds to one invocation per variant in the row.

**Flag legend**

| Flag | Purpose |
|---|---|
| `--variant <name>` | Which variant from `models.yaml` |
| `--seeds 42` | Pin seed; emits `seed42/` under `jobs/` + `results/` |
| `--languages eng` | Restrict to one language (omit → all 17) |
| `--kgs <list>` | KG bucket(s); comma-separated |
| `--use-kg` | Tier-3a/3c only — talk to the live KG bridge at runtime |
| `--oracle-mode <m>` | Inject gold triples; `structured`/`raw`/`verbalized` (tier 1 only), `t3` (tier 3). Rejected on tier-2 by the CLI. |
| `--cluster <c>` | `bwuc3` (default) or `helix` |

Multimodal variants (`-image`, `-name`) carry their `multimodal_mode` in YAML — no extra flag needed. `--use-kg` and `--oracle-mode` are mutually exclusive (the runner short-circuits use-kg when an oracle mode is set).

#### Text-only runs (all 3 KGs)

**Tier 1 — plain (4 variants)**

| Regime | Variants | Flags after `--variant <V> --seeds 42` |
|---|---|---|
| Plain (no oracle) | `gemini-3-flash-plain`, `qwen35-27b-plain`, `qwen35-a3b-plain`, `gemma-3-27b-it-plain` | `--kgs kwg,osmh3kg,worldkg` |
| + Oracle structured | same 4 | `--kgs kwg,osmh3kg,worldkg --oracle-mode structured` |
| + Oracle raw | same 4 | `--kgs kwg,osmh3kg,worldkg --oracle-mode raw` |
| + Oracle verbalized | same 4 | `--kgs kwg,osmh3kg,worldkg --oracle-mode verbalized` |

**Tier 2 — reasoning (4 variants)**

| Regime | Variants | Flags after `--variant <V> --seeds 42` |
|---|---|---|
| Reasoning (no oracle) | `gemini-3-flash-thinking-tier2`, `qwen35-27b-tier2`, `qwen35-a3b-thinking-tier2`, `gemma-3-27b-it-tier2` | `--kgs kwg,osmh3kg,worldkg` |

Tier-2 + oracle is rejected by the CLI (`cli.py:515-519`). Run oracle ablations at tier-1 only.

**Tier 3a — KG + Python (4 variants)**

| Regime | Variants | Flags after `--variant <V> --seeds 42` |
|---|---|---|
| With KG bridge | `gemini-3-flash-tier3a`, `qwen35-27b-tier3a`, `qwen35-a3b-tier3a`, `gemma-3-27b-it-tier3a` | `--kgs kwg,osmh3kg,worldkg --use-kg` |
| Oracle-t3 (gold triples + Python, no KG) | same 4 | `--kgs kwg,osmh3kg,worldkg --oracle-mode t3` |

**Tier 3b — Web + Python (4 variants)**

| Regime | Variants | Flags after `--variant <V> --seeds 42` |
|---|---|---|
| Web tools (no KG) | `gemini-3-flash-tier3b`, `qwen35-27b-tier3b`, `qwen35-a3b-tier3b`, `gemma-3-27b-it-tier3b` | `--kgs kwg,osmh3kg,worldkg` |

Tier-3b + oracle-t3 is code-identical to tier-3a + oracle-t3 (both disable tools and inject gold triples through the same `SmolagentsOracleModel`). Run it under tier-3a only.

**Tier 3c — KG + Web + Python (4 variants)**

| Regime | Variants | Flags after `--variant <V> --seeds 42` |
|---|---|---|
| With KG bridge + Web | `gemini-3-flash-tier3c`, `qwen35-27b-tier3c`, `qwen35-a3b-tier3c`, `gemma-3-27b-it-tier3c` | `--kgs kwg,osmh3kg,worldkg --use-kg` |

Tier-3c + oracle-t3 is also code-identical to tier-3a + oracle-t3. The oracle path uses `SmolagentsOracleModel` with `tools=[]` and the `tier3_oracle` prompt template regardless of the variant's `sub_tier`, so all three tier-3 sub-tiers collapse to the same agent setup under `--oracle-mode t3`. Run it under tier-3a only.

#### Multimodal runs (`osmh3kg` + `worldkg` slice only)

The `multimodal` bucket fans into 2 sub-cells per language (`osmh3kg` and `worldkg`) — one SLURM script per underlying KG. `kwg` has no multimodal questions.

**Tier 1 — image / name (8 variants)**

| Regime | Variants | Flags after `--variant <V> --seeds 42` |
|---|---|---|
| Image (no oracle) | `gemini-3-flash-image`, `qwen35-27b-image`, `qwen35-a3b-image`, `gemma-3-27b-it-image` | `--kgs multimodal` |
| Name ablation (no oracle) | `gemini-3-flash-name`, `qwen35-27b-name`, `qwen35-a3b-name`, `gemma-3-27b-it-name` | `--kgs multimodal` |
| Image + oracle structured | image variants | `--kgs multimodal --oracle-mode structured` |
| Image + oracle raw | image variants | `--kgs multimodal --oracle-mode raw` |
| Image + oracle verbalized | image variants | `--kgs multimodal --oracle-mode verbalized` |
| Name + oracle structured/raw/verbalized | name variants | `--kgs multimodal --oracle-mode <mode>` |

**Tier 2 — image / name (8 variants)**

| Regime | Variants | Flags after `--variant <V> --seeds 42` |
|---|---|---|
| Image (no oracle) | `gemini-3-flash-thinking-tier2-image`, `qwen35-27b-tier2-image`, `qwen35-a3b-thinking-tier2-image`, `gemma-3-27b-it-tier2-image` | `--kgs multimodal` |
| Name ablation | corresponding `*-name` variants | `--kgs multimodal` |

Same restriction as text-only tier-2: oracle modes are rejected on tier-2 image and name variants by the CLI.

**Tier 3a — image / name (8 variants)**

| Regime | Variants | Flags after `--variant <V> --seeds 42` |
|---|---|---|
| Image + KG bridge | `gemini-3-flash-tier3a-image`, `qwen35-27b-tier3a-image`, `qwen35-a3b-tier3a-image`, `gemma-3-27b-it-tier3a-image` | `--kgs multimodal --use-kg` |
| Name + KG bridge | `*-tier3a-name` (same 4 prefixes) | `--kgs multimodal --use-kg` |
| Image + oracle-t3 | image variants | `--kgs multimodal --oracle-mode t3` |
| Name + oracle-t3 | name variants | `--kgs multimodal --oracle-mode t3` |

**Tier 3b — image / name (8 variants)**

| Regime | Variants | Flags after `--variant <V> --seeds 42` |
|---|---|---|
| Image + Web (no KG) | `gemini-3-flash-tier3b-image`, `qwen35-27b-tier3b-image`, `qwen35-a3b-tier3b-image`, `gemma-3-27b-it-tier3b-image` | `--kgs multimodal` |
| Name + Web | corresponding `*-name` | `--kgs multimodal` |

**Tier 3c — image / name (8 variants)**

| Regime | Variants | Flags after `--variant <V> --seeds 42` |
|---|---|---|
| Image + KG + Web | `gemini-3-flash-tier3c-image`, `qwen35-27b-tier3c-image`, `qwen35-a3b-tier3c-image`, `gemma-3-27b-it-tier3c-image` | `--kgs multimodal --use-kg` |
| Name + KG + Web | corresponding `*-name` | `--kgs multimodal --use-kg` |

Same collapse as the text-only case: tier-3c + oracle-t3 (image or name) is code-identical to tier-3a + oracle-t3 (image or name) — same `SmolagentsOracleModel`, same `tier3_oracle` prompt, same empty tool list. Run it under tier-3a only.

#### One-liner shape

```bash
uv run python scripts/benchmark_models/generate_slurm.py \
    --variant <variant-from-table> \
    --seeds 42 \
    --kgs <kg-list-from-table> \
    [--use-kg]              \
    [--oracle-mode <mode>]  \
    [--languages eng]       # add to restrict to English; omit for all 17
```

#### Copy-paste examples

```bash
# tier 1 plain — all 4 models, all 3 KGs, all 17 langs, no oracle
for V in gemini-3-flash-plain qwen35-27b-plain qwen35-a3b-plain gemma-3-27b-it-plain; do
    uv run python scripts/benchmark_models/generate_slurm.py \
        --variant $V --seeds 42 --kgs kwg,osmh3kg,worldkg
done

# tier 3a image + KG bridge (multimodal)
for V in gemini-3-flash-tier3a-image qwen35-27b-tier3a-image qwen35-a3b-tier3a-image gemma-3-27b-it-tier3a-image; do
    uv run python scripts/benchmark_models/generate_slurm.py \
        --variant $V --seeds 42 --kgs multimodal --use-kg
done

# tier 1 oracle-raw on multimodal image
for V in gemini-3-flash-image qwen35-27b-image qwen35-a3b-image gemma-3-27b-it-image; do
    uv run python scripts/benchmark_models/generate_slurm.py \
        --variant $V --seeds 42 --kgs multimodal --oracle-mode raw
done

# oracle-t3 multimodal at tier 3a (replaces tier-3b/3c oracle-t3 — code-identical)
for V in gemini-3-flash-tier3a-image qwen35-27b-tier3a-image qwen35-a3b-tier3a-image gemma-3-27b-it-tier3a-image; do
    uv run python scripts/benchmark_models/generate_slurm.py \
        --variant $V --seeds 42 --kgs multimodal --oracle-mode t3
done
```

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

**Per-KG SLURM dispatch**: the SLURM generator fans the `multimodal` bucket out into one sub-script per underlying KG so tier-3a/3c jobs can pass a real `--kg-source` (the bridge's `schema_for(...)` doesn't know a `multimodal` schema and would raise `ValueError: Unknown KG: 'multimodal'`). Generated layout:

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

All open-weights vLLM variants serve at their native context windows except for tier-3a / tier-3c on qwen35-27b, where the agent loop's per-step input growth (~5-7 K tokens of cumulative tool outputs replayed each turn) exhausts the native 32 K window within 4–5 steps. Those two variants extend to **65 K** via YaRN rope scaling (factor=2.0, `original_max_position_embeddings=32768`) — the standard inference-time technique documented on the [Qwen3.5 model card](https://huggingface.co/Qwen/Qwen3.5-27B-Instruct). The SLURM template passes this to vLLM as `--hf-overrides '{"rope_scaling": {...}}'`, which overrides the HuggingFace config.json at engine init. Perplexity degradation at factor=2.0 is well under 1 % in the 28 K–60 K regime where these loops actually consume the budget. Tier-1 / tier-2 / tier-3b qwen variants stay at native 32 K because they don't need the extended window. Configuration lives on the `_qwen35_27b_yarn_64k` YAML anchor in `models.yaml`.

## Prompt Caching

Closed-source backends report cache hits via different fields. ``Prediction.usage.cached_tokens`` is populated end-to-end so the eval pipeline can quote real (post-cache) input cost on every ``AggregateReport``:

| Backend | Source field | Notes |
|---|---|---|
| Google | `response.usage_metadata.cached_content_token_count` | Explicit context caching via `client.caches.create()`. The backend splits the rendered prompt at `Question:\n`, gates on a 4096-token minimum via `count_tokens`, creates one cache per `sha256(prefix + model_id)` with 1-hour TTL, and references `cached_content=<name>` on subsequent calls. Cached input bills at ~25% of the normal rate. Cache-expired errors are detected, the registry is invalidated, and the call is retried without the cache reference. Disable per-variant via `enable_context_cache: false` on `GoogleVariantConfig`. |
| vLLM | n/a | Internal KV cache; no API-level cache concept. `cached_tokens` stays at 0. |

`ScoreRecord.cached_tokens` (denormalised from `Prediction.usage.cached_tokens`) rolls up into `AggregateReport.total_cached_tokens`.

## KG Layout (tier-3a / 3c)

Production knowledge graphs live at `benchmark/kgs/<kg_name>/` as Delta tables of `(subject, predicate, object)` triples (osmh3kg additionally has a `date` column). Two optional Z-ordered sidecar tables sit alongside the main table to keep query latency bounded — `subject_index/` for fast subject-keyed reads, `name_index/` for fuzzy `find_entity(name)` substring search. Both sidecars are built once via `scripts/build_kg_indexes.py`; the bridge auto-detects them and falls back to the slower main-table scan if missing.

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

The SLURM generator invokes one runner per `(lang, kg)` cell and passes `--kg-source <kg>`; the runner resolves `kg_path = kg_root / kg_source` (default root `benchmark/kgs/`, overridable via `--kg-root` or the `BENCHMARK_KGS_ROOT` env var) and passes it to `EntityService(TripleBridge(...))`. The bridge expects a Delta-format directory (parquet shards + `_delta_log/`).
- **vLLM-served (open weights)**: the SLURM template starts a per-cell `vllm serve` and the harness talks to it over the OpenAI Chat-Completions API. Structured output via `response_format={"type": "json_schema", ...}` (vLLM serve supports the same envelope as OpenAI for guided decoding under the hood). The SLURM template's `--enable-prefix-caching` lets vLLM reuse KV cache across questions in the same cell. Thinking-toggle handling (Qwen 3+ / Gemma 4) is driven via the per-variant `thinking` field and serialised through the chat-completions request — no in-process tokenizer-template gymnastics needed.

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
