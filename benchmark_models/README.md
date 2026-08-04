# benchmark_models

Runs models against the benchmark and writes prediction records for `benchmark_eval` to
score. Covers tier-1 and tier-2 single-call inference, the tier-3 code agent (KG
retrieval, web search, Python sandbox) and the oracle conditions that replace retrieval
with the question's gold triples. The root README covers the tier ladder and the
metrics; this file documents the harness interface.

## Install

The `models` extra brings in the backends and the tier-3 agent; add `kg` for tier-3a /
3c knowledge-graph retrieval and `vllm` to serve open weights locally. Real KG
retrieval also needs the graph snapshots from Zenodo
(<https://doi.org/PLACEHOLDER>), pointed at by `BENCHMARK_KGS_ROOT`.

```bash
uv sync --extra models --extra kg
```

## Environment

| Variable | Used by |
|---|---|
| `GOOGLE_API_KEY` | the `google` backend |
| `OPENAI_API_KEY` | the OpenAI-compatible backend; a placeholder suffices for a local `vllm serve` |
| `VLLM_BASE_URL` | base URL of the `vllm serve` process (default `http://localhost:8000/v1`) |
| `BENCHMARK_KGS_ROOT` | root holding the per-KG Delta tables (default `benchmark/kgs/`) |

`--api-key` overrides the key for a single run.

## Quickstart

```bash
python -m benchmark_models smoke -v dummy --n 5                     # canned, no API key
python -m benchmark_models smoke -v gemini-3-flash-plain --n 10     # tier 1
python -m benchmark_models smoke -v gemini-3-flash-tier3b --n 3     # tier 3b, web agent
python -m benchmark_models smoke -v gemini-3-flash-tier3a --use-kg --kg-source osmh3kg --n 3
```

Predictions are written to `results/smoke/<variant>/predictions.jsonl`, with a `run.log`
alongside. Without `--use-kg`, tier 3a/3c run against mock KG data.

## Commands

| Command | Purpose |
|---|---|
| `smoke` | run one variant over a benchmark path |
| `list-variants` | print every variant in `models.yaml`, grouped by backend and tier |
| `cost-estimate` | approximate API cost for a variant and row count before launching |

`smoke` options:

| Flag | Effect |
|---|---|
| `-b, --benchmark` | a JSONL file, a leaf `benchmark.jsonl/` shard, or a share root that is walked (default `benchmark/final/large/`) |
| `-l, --language` | ISO 639-3 code; restricts the walk and stamps the language on every row |
| `--n` | number of examples; `0` runs the whole loaded shard |
| `--no-resume` | start fresh instead of skipping questions that already have predictions |
| `--retry-errors` | re-run only rows with `final_state="error"`; answered, abstained and cap-hit rows are kept |
| `--use-kg`, `--kg-root`, `--kg-source` | real KG retrieval for tier 3a/3c; needs Java and `JAVA_HOME` |
| `--multimodal-mode` | `off`, `image` or `name`, overriding the variant's setting |
| `--max-tokens` | output budget: the whole response on tier 1/2, each agent step on tier 3 |

`--no-resume` and `--retry-errors` are mutually exclusive. `--max-steps` (agent step
cap, also substituted into the prompt), `--max-concurrent` (`predict_batch` workers) and
`--web-search-timeout` (3b/3c only) are rejected outside tier 3. Sampling overrides
(`--temperature`, `--top-p`, `--seed`, `--self-consistency-n`) and every tier-3 knob
resolve as default -> `models.yaml` -> CLI flag.

Every run is logged to MLflow unless `--no-mlflow` is passed; `MLFLOW_TRACKING_URI` and
`MLFLOW_EXPERIMENT_NAME` set the store and experiment name.

## Variants

`benchmark_models/models.yaml` defines every variant: backend, tier, model identifier
and runtime settings.

| Backend | Tier 1 | Tier 2 | Tier 3a | Tier 3b | Tier 3c |
|---|---|---|---|---|---|
| Google | gemini-3-flash-plain | gemini-3-flash-thinking-tier2 | gemini-3-flash-tier3a | gemini-3-flash-tier3b | gemini-3-flash-tier3c |
| vLLM, Qwen3.5-35B-A3B | qwen35-a3b-plain | qwen35-a3b-thinking-tier2 | qwen35-a3b-tier3a | qwen35-a3b-tier3b | qwen35-a3b-tier3c |
| vLLM, Qwen3.5-27B | qwen35-27b-plain | qwen35-27b-tier2 | qwen35-27b-tier3a | qwen35-27b-tier3b | qwen35-27b-tier3c |
| vLLM, Gemma 3 27B | gemma-3-27b-it-plain | gemma-3-27b-it-tier2 | gemma-3-27b-it-tier3a | gemma-3-27b-it-tier3b | gemma-3-27b-it-tier3c |

Multimodal variants add an `-image` or `-name` suffix. Tier-1 names drop the `-plain`
segment (`gemini-3-flash-image`, `qwen35-27b-name`); tier-2 and tier-3 names keep the
base (`qwen35-27b-tier2-image`, `gemini-3-flash-tier3a-name`).

Tier-3 tool sets: 3a has the KG tools (`find_entity`, `get_entity`,
`find_entities_by_type`, `list_categories`) plus a Python sandbox; 3b has web search,
page fetch and the sandbox; 3c has all of them. Tier-3 predictions carry the full agent
trace in `tool_calls` -- the code written at each step, the execution output and any
error.

## Oracle modes

`--oracle-mode` injects the question's gold triples in place of retrieval.

| Mode | Tier | Triples appear as |
|---|---|---|
| `structured` | 1 | JSON array with short predicate names, behind a PREFIXES header |
| `raw` | 1 | N-Triples-like lines with full URIs |
| `verbalized` | 1 | prose, one sentence per entity |
| `t3` | 3 | structured JSON in a code agent with the Python sandbox but no KG or web tools |

```bash
python -m benchmark_models smoke -v gemini-3-flash-plain --oracle-mode structured \
    --gold-triples <data>/large/ --n 5
```

Output defaults to `results/smoke/<variant>/oracle_<mode>/predictions.jsonl`.

Gold shards hold one row per triple -- `{question_id, entity_role, subject, predicate,
object}` -- discovered under the `<kg>/<sfid>/gold_triples.jsonl/` layout. Every question
in the run must have gold; the run aborts otherwise. `--wkt-simplify-tolerance` sets the
Douglas-Peucker tolerance applied to gold geometries in lat/lon degrees (default 0.001;
`0` keeps them verbatim).

## Multimodal slice

The `multimodal` bucket is a slice of `osmh3kg` and `worldkg` with an image attached to
one entity slot, not a knowledge graph of its own. Rows store `images[*].path` as a short
relative segment, resolved at load time against `<dataset_root>/multimodal/<kg_source>/`
so downstream code sees absolute paths. Tier-3a and 3c need a real `--kg-source`, so the
bucket is run one job per underlying KG. `-image` and `-name` are the ablation pair:
`image` packs the image bytes into a provider content block, `name` substitutes the
entity's display name into the `[IMAGE_<slot>]` placeholder for a text-only baseline.

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

Runs resume by default. Point a rerun at the same output path and the runner skips
questions that already have predictions; `--no-resume` forces a clean run.

For vLLM variants, start one `vllm serve` per job and point the harness at it with
`VLLM_BASE_URL`. `--enable-prefix-caching` lets vLLM reuse KV cache across the
questions in a cell.

## Output format

Each prediction is one JSON line:

```json
{"question_id": "1285b5ef-...", "tid": "1.1", "sfid": "A",
 "answer_type": "Numeric (float)", "model_id": "gemini-3-flash-plain",
 "tier": "tier1", "language": "eng", "answer": 12.5, "unit": "km",
 "refusal": false, "latency_ms": 1234.5}
```

Provenance (`question_id`, `tid`, `sfid`, `answer_type`, `model_id`, `tier`, `language`,
`seed`, `snapshot_id`, `prompt_template_id`), the response (`answer`, `unit`, `refusal`),
`usage` and `latency_ms` are always present; `answer` is `null` for an abstention and
`refusal` mirrors that. Tier-2 rows may add `reasoning` and `sc_samples`; tier-3 rows add
`tool_calls`, `cap_hit`, `final_state` and `error_class`; oracle rows add
`triple_context`. Each run pins the JSON response schema from the row's
`ground_truth_type`, so the model is told the shape of the expected answer before it
reasons.
