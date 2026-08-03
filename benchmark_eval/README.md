# benchmark_eval


Primary-metric scoring for predictions produced by the model section. Reads a `predictions.jsonl` and a benchmark JSONL, writes `score_records.jsonl` (one per prediction) and `aggregate_report.json` (rolled-up numbers) to an output directory. Logs the same metrics to MLflow by default.

## Quickstart

```bash
python -m benchmark_eval score \
    --predictions results/smoke/<variant>/predictions.jsonl \
    --benchmark benchmark/final/large/ \
    --aliases output/aliases/eng.json \
    --output results/smoke/<variant>/scores/
```

`--benchmark` accepts either a directory walked recursively for Spark `part-*.json` shards (layout `benchmark/final/<size>/<kg>/<sfid>/<lang>/benchmark.jsonl/`, `<size>` ∈ {`large`, `small`}) or a single JSONL file — point it at one size, not the bare `benchmark/final/`. `--aliases` is required (no default — pass the per-language alias store for multilingual runs).

Output files:

- `score_records.jsonl` — one `ScoreRecord` per prediction, with `em`, `ne`, `covered`, `raw_distance`, `raw_tolerance`, per-type drill-down fields, and all identity fields denormalized for analysis (`question_id`, `tid`, `sfid`, `answer_type`, `model_id`, `tier`, `language`, `strata_cell`).
- `aggregate_report.json` — headline numbers and per-group breakdowns.

## Primary Metrics

Two headline metrics applied uniformly across every answer type; the per-type rules vary but the interface doesn't.

**Exact Match (EM)** — binary, higher is better.

```
EM = 1 [ d(â, a*) < τ(a*) ]
```

For discrete types (Boolean / Entity / Cardinal direction / Predefined value / Entity hex string / Set enumeration) the distance `d` is 0 or 1 and the tolerance `τ` is 1, so the rule collapses to strict equality (up to alias resolution for entity-like types). For continuous types (Numeric / Coordinates / Geometry lines) the rule is a hit-at-tolerance: the prediction counts as correct if the distance to gold is below the type-appropriate `τ`. Polygon geometry uses `EM = 1[IoU ≥ polygon_iou_threshold]` (default 0.5). Dates use components-match (gold-specified components must exactly match).

**Normalized Error (NE)** — real-valued in `[0, 1]`, lower is better.

```
NE = min(1, d(â, a*) / τ(a*))
```

Set enumeration uses `NE = 1 − Jaccard(Â, A*)` directly. Polygon geometry uses `NE = 1 − IoU`. Dates use `NE = min(1, days_between / τ_days)` with `τ_days` picked from the coarser of prediction and gold precision (1 / 15 / 365 for day / month / year).

## Per-Type Rules

| Answer type | Distance `d` | Tolerance `τ` | Drill-down fields |
|---|---|---|---|
| Boolean | 0 if equal else 1 | 1 | — |
| Entity name (single) | 0 if `resolve(â) == gold_id` else 1 | 1 | `resolve_failure` |
| Entity hex string (single) | 0 if case-insensitive exact match else 1 | 1 | — |
| Cardinal direction | 0 if `resolve(â) == gold_id` else 1 | 1 | `resolve_failure` |
| Predefined value (left/right, upstream/downstream) | 0 if `resolve(â) == gold_id` else 1 | 1 | `resolve_failure` |
| Set enumeration (entities) | `1 − Jaccard` (members resolved via alias) | 1 | `set_precision`, `set_recall`, `set_f1` |
| Set enumeration (dates) | `1 − Jaccard` (members canonicalized) | 1 | `set_precision`, `set_recall`, `set_f1` |
| Numeric | `|ŷ − y*|` (strict unit match required for EM=1) | `max(δ, α·|y*|)` | `unit_match`, `numeric_failure_category` |
| Coordinates (single point) | geodesic distance (km) | `max(δ_coord, α·d_scale)` | — |
| Geometry: lines | Hausdorff distance (km) | `line_tolerance_km` | — |
| Geometry: polygons | `1 − IoU` | 1 (EM uses `iou_threshold` directly) | — |
| Date | components-match for EM; day-distance for NE | 1 / 15 / 365 days (precision-adaptive) | — |

Default tolerance hyperparameters (configurable via CLI flags):

| Parameter | Default | Flag |
|---|---|---|
| `α` (relative tolerance) | 0.05 | `--alpha` |
| `δ` distance (km) | 5.0 | `--delta-distance` |
| `δ` area (km²) | 25.0 | `--delta-area` |
| `δ` count | 1.0 | (no flag) |
| Polygon IoU threshold | 0.5 | `--polygon-iou-threshold` |
| Scale-from-magnitude | **on** | `--no-scale-from-magnitude` to disable |

By default the `δ` for distance / area is picked from the gold's magnitude: urban (`|y*| < 50` → `δ=0.5`), regional (`50 ≤ |y*| < 500` → `δ=5`), national (`|y*| ≥ 500` → `δ=10`). Pass `--no-scale-from-magnitude` to revert to a single flat global `δ` (set via `--delta-distance` / `--delta-area`).

These defaults live in `benchmark_eval/scoring_spec.json` — the canonical scoring rules. CLI flag values that differ from the spec are recorded on every run as `scoring_spec_deviations` in the aggregate report and MLflow, so readers can tell whether numbers were produced under spec defaults or a custom config.

## Refusals and Coverage

Predictions with `answer == None` (or `refusal=True`) are treated as refusals: `em=0`, `ne=1`, `covered=False`. Predictions that fail to parse into the expected Python type (e.g. string where a number was expected) are treated as format errors: `em=0`, `ne=1`, `covered=False`, `format_error=True`.

**False-premise rows are a deliberate exception**: when `is_true_premise=False` and the model refuses (`answer=None`), the refusal is the *correct* answer to an unanswerable question. The row is scored `em=1`, `covered=True`, `refusal=True`. So `coverage_rate` counts:

- Real-premise rows where the model produced a parseable, non-refusal answer.
- False-premise rows where the model correctly refused.

It does NOT count:

- Real-premise rows where the model refused (these are scored `em=0`, `covered=False`).
- Format errors on any row.
- Gold-missing rows.

The aggregate report includes:

- `coverage_rate` — fraction of rows that produced a scoreable response per the rules above. Includes correct false-premise refusals. **Do not read this as "fraction of questions the model attempted an answer for"** — read `refusal_rate` (true-premise refusals only) and `negative_rejection_rate` (false-premise correct-refusal rate) together to get that picture.
- `em_conditional` — mean EM conditional on `covered=True`. Reports the model's accuracy on the questions it actually attempted, plus its correct-refusal rate on false-premise rows.
- `format_error_rate` — fraction of predictions that failed to parse.
- `refusal_rate` — fraction of true-premise rows where the model refused (i.e. erroneously declined).
- `negative_rejection_rate` — fraction of false-premise rows where the model correctly refused.

## Micro vs Macro

Both are reported:

- `em_micro` / `ne_micro` — simple mean across all predictions. Reflects the benchmark's actual distribution of answer types.
- `em_macro` / `ne_macro` — mean across per-answer-type means. Each answer type contributes equally.

## Per-Group Breakdowns

The aggregate report carries `dict[str, GroupStats]` rollups keyed on:

- `per_answer_type` — by canonical answer-type name.
- `per_sfid` — by Spatial Function ID family.
- `per_language` — by ISO language code.
- `per_strata_cell` — by full `strata_cell` tag (e.g. `HIC_high`, `LIC_low`). Surfaces the income-tier × density-bucket cross product the benchmark stratifies on.
- `per_income_tier` — coarser companion: the prefix before `_` in `strata_cell` (`HIC` / `UMC` / `LMC` / `LIC`). For paper tables that don't need the full 12-cell matrix.

Both strata rollups are empty dicts when no record carries a `strata_cell`. Each `GroupStats` carries `n`, `n_covered`, `em`, `ne`, `em_conditional`.

## Diagnostic Metrics

Populated when the relevant sub-population is non-empty:

- `resolve_failure_rate` — alias lookup returned `None` for an entity / cardinal / predefined prediction.
- `unit_match_rate` — predicted unit matched gold after canonicalization (numeric only).
- `numeric_failure_breakdown` — counts of `em=0` numeric failures, split by whether unit and/or value were wrong.

## Unit-Normalized Companions

Strict `em_micro` requires the model's declared unit to match gold (so `10000 m` vs gold `10 km` is EM=0). To separate *spatial-reasoning capability* from *unit-instruction compliance*, the aggregate report also reports `numeric_em_unit_normalized` / `numeric_ne_unit_normalized`, computed after converting prediction units into the gold unit. Conversion is restricted to within a dimensional group (distance / area); cross-group or unknown-unit cases fall back to the strict values.

## Alias Store

Entity / cardinal / predefined resolution goes through `output/entity_aliases.json`, a flat lookup keyed by canonical id:

```json
{
  "osm:2476384356": {
    "canonical": "Cafe Kanel",
    "aliases": ["Cafe Kanel", "Café Kanel"]
  },
  "cardinal:N": {
    "canonical": "N",
    "aliases": ["N", "North", "Nord", "Norte", "北"]
  },
  "predef:left": {
    "canonical": "left",
    "aliases": ["left", "gauche", "izquierda", "links"]
  }
}
```

`output/entity_aliases.json` is gitignored (generated, not tracked). Build it once after pulling benchmark data:

```bash
python scripts/build_entity_aliases.py
```

The script populates the file from OSM entities in the benchmark, plus multilingual aliases for cardinal directions and predefined values.

## MLflow Tracking

Each `score` invocation logs one MLflow run by default (`--no-mlflow` to skip). Pass `--parent-run-id <id>` to nest the eval run under a `benchmark_models` run in the UI. Artifacts: `score_records.jsonl`, `aggregate_report.json`.

## Testing

```bash
uv run pytest tests/benchmark_eval/ -v
```

Test coverage (~210 tests): canonicalization, alias store, per-type distance and tolerance, per-type scorer end-to-end, aggregation (micro / macro / diagnostics), MLflow logging (mocked), CLI end-to-end.
