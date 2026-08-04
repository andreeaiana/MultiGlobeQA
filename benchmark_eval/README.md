# benchmark_eval

Scoring for predictions produced by `benchmark_models`. Reads a `predictions.jsonl` and
the benchmark gold, joins them on `question_id`, and writes `score_records.jsonl` (one
row per prediction) and `aggregate_report.json` to an output directory.

## score

```bash
python -m benchmark_eval score \
    --predictions results/<variant>/predictions.jsonl \
    --benchmark   <data>/large/ \
    --aliases     <data>/aliases/eng.json \
    --output      results/<variant>/scores/
```

`--benchmark` takes a single JSONL file or a directory walked for
`<kg>/<sfid>/<lang>/benchmark.jsonl/part-*.json` shards; point it at one size, not at a
root holding both. `--aliases` is required and takes one alias file, normally the store
for the language being scored. `--language <iso639-3>` restricts the walk to that
language's shards; omitting it loads all 17 and collapses them last-wins.
`--include-invalid` keeps rows the generator flagged `is_valid=False`, dropped by
default. The remaining flags set tolerances, listed below.

`score_records.jsonl` carries `em`, `ne`, `covered`, the per-type drill-down fields
below, and the identity fields (`question_id`, `tid`, `sfid`, `answer_type`, `model_id`,
`tier`, `language`, `kg_source`, `strata_cell`) denormalised so the file stands alone.
`raw_distance` and `raw_tolerance` preserve `d` and `tau`, so both metrics can be
recomputed under other tolerances without re-running a model.

## pairwise-agreement

Joins two or more score-record files on `question_id` and reports, per pair, the
agreement rate, the both-right / both-wrong / one-only rates, and Cohen's kappa.

```bash
python -m benchmark_eval pairwise-agreement \
    --records gemini=results/gemini/score_records.jsonl \
    --records qwen=results/qwen/score_records.jsonl \
    --output results/agreement.json
```

## Primary metrics

```
EM = 1[ d(pred, gold) < tau(gold) ]        binary, higher is better
NE = min(1, d(pred, gold) / tau(gold))     in [0, 1], lower is better
```

For discrete types `d` is 0 or 1 and `tau` is 1, so EM is equality, after alias
resolution where the type is alias-backed. For continuous types EM is a hit at
tolerance. Set enumeration takes `NE = 1 - Jaccard` and polygons `NE = 1 - IoU`.

## Per-type rules

| Answer type | Distance `d` | Tolerance `tau` | Drill-down fields |
|---|---|---|---|
| Boolean (yes/no, true/false) | 0 if equal else 1 | 1 | - |
| Ternary (true/false/plausibly) | 0 if equal after case folding else 1 | 1 | - |
| Entity name (single) | 0 if `resolve(pred)` is the gold id else 1 | 1 | `resolve_failure` |
| Entity hex string (single), H3 cell index, S2 cell index | equality after hex / decimal / signed-int64 normalisation | 1 | `em_spatial`, `ne_spatial` (H3 only) |
| Geohash string | case-insensitive equality; non-base-32 predictions are format errors | 1 | - |
| Composite string (H3+S2) | equality of both pipe-separated halves | 1 | `em_spatial`, `ne_spatial` (H3 half) |
| Cardinal direction (predefined) | 0 if `resolve(pred)` is the gold id else 1 | 1 | `resolve_failure` |
| Predefined value (left/right, upstream/downstream) | 0 if `resolve(pred)` is the gold id else 1 | 1 | `resolve_failure` |
| Set enumeration (list of entity names) | `1 - Jaccard` over alias-resolved members; EM needs set equality | 1 | `set_precision`, `set_recall`, `set_f1` |
| Set enumeration (list of dates) | `1 - Jaccard` over canonicalised dates; an unparseable element is a format error | 1 | `set_precision`, `set_recall`, `set_f1` |
| Numeric (integer, float) | `abs(pred - gold)`; EM also needs the declared unit to match | `max(delta, alpha*abs(gold))` | `unit_match`, `numeric_failure_category` |
| Coordinates (lat/long pair) | geodesic distance (km) | `max(delta_coord, alpha*d_scale)` | - |
| Polar coordinates (angle, distance) | max of the normalised bearing and radial errors | EM needs both within tolerance | - |
| Bounding box (4-float) | `1 - IoU` | EM is `1[IoU >= threshold]` | - |
| Geometry (list of lat/long pairs), lines | Hausdorff distance (km) | `line_tolerance_km` | - |
| Geometry (list of lat/long pairs), polygons | `1 - IoU`, computed in EPSG:6933 | EM is `1[IoU >= threshold]` | - |
| Date | components match for EM, day distance for NE | 1 / 15 / 365 days, from the coarser of prediction and gold precision | - |

Geometry routes to the polygon rule when the gold is a closed ring enclosing a
non-trivial area, and to the line rule otherwise.

| Parameter | Default | Flag |
|---|---|---|
| `alpha`, relative tolerance | 0.05 | `--alpha` |
| `delta` distance (km) | 5.0 | `--delta-distance` |
| `delta` area (km2) | 25.0 | `--delta-area` |
| IoU threshold, polygons and boxes | 0.5 | `--polygon-iou-threshold` |
| Polar bearing tolerance (degrees) | 5.0 | `--polar-angle-tol` |
| Scale-from-magnitude | on | `--no-scale-from-magnitude` disables |

Counts use `delta = 1.0` and dimensionless ratios `delta = 0.0`, making the ratio
tolerance purely relative. Coordinates use `delta_coord = 5 km` and `d_scale = 200 km`,
so the default tolerance is 10 km. None of those four has a flag.

With scale-from-magnitude on, distance and area answers take `delta` from the gold's
magnitude instead of the flat default: distances under 50 km get 0.5, under 500 km get
5, otherwise 10; areas under 2500 km2 get 0.25, under 250000 km2 get 25, otherwise 100.
Counts and ratios never scale.

The defaults live in `benchmark_eval/scoring_spec.json`, the canonical rule set. Flag
values that differ from it are recorded on each run as `scoring_spec_deviations` in the
aggregate report.

## Refusals and coverage

| Prediction | Gold | Scored as |
|---|---|---|
| refusal (`answer` is null, or `refusal`) | true premise | `em=0`, `ne=1`, `covered=False`, `refusal=True` |
| refusal | false premise | `em=1`, `ne=0`, `covered=True`, `refusal=True` |
| refusal | no `ground_truth`, no false-premise marker | `em=0`, `covered=False`, `gold_missing=True` |
| typed answer | false premise | `em=0`, `covered=True` |
| unparseable as the declared type | any | `em=0`, `ne=1`, `covered=False`, `format_error=True` |
| `final_state == "error"` | any | `em=0`, `ne=1`, `covered=False`, not a refusal |

`coverage_rate`, also reported as `answer_rate`, is the share of rows with
`covered=True`: parseable non-refusal answers plus correct refusals on false-premise
rows. Alongside it:

- `em_conditional` - mean EM over covered rows.
- `refusal_rate` - share of all rows that are uncredited refusals; gold-missing rows and
  correct false-premise refusals are excluded from the numerator.
- `false_refusal_rate` - share of true-premise rows the model refused.
- `negative_rejection_rate` - mean EM over false-premise rows, the correct-refusal rate.
- `em_adjusted` / `ne_adjusted` - the headline means with harness failures (`error_class`
  in `infra`, `context_overflow`, `code_gen`) out of the denominator.
- `missing_prediction_rate` - gold question ids with no prediction, over the gold size.
  Those rows are absent from every other rate rather than scored as EM=0.

## Aggregate report

`em_micro` / `ne_micro` are means over all predictions; `em_macro` / `ne_macro` are
means over the per-answer-type means, so each type carries equal weight.
`em_true_premise` / `ne_true_premise` restrict to real-premise rows, and
`majority_baseline_em` is the EM of always emitting each TID's modal gold label.

Breakdowns are `dict[str, GroupStats]` under `per_answer_type`, `per_sfid`,
`per_language`, `per_kg`, `per_strata_cell` (the full tag, e.g. `HIC_high`) and
`per_income_tier` (the prefix before `_`). Each `GroupStats` carries `n`, `n_covered`,
`em`, `ne`, `em_conditional`, `ne_conditional`.

Diagnostics, populated when the relevant subset is non-empty:

- `resolve_failure_rate` - alias lookup returned nothing for an entity, cardinal or
  predefined prediction.
- `unit_match_rate` - predicted unit matched gold after canonicalisation (numeric only).
- `numeric_failure_breakdown` - numeric `em=0` rows counted by
  `number_wrong_unit_correct`, `number_correct_unit_wrong`, `both_wrong`.
- `numeric_em_unit_normalized` / `numeric_ne_unit_normalized` - numeric scores after
  converting the prediction into the gold unit. Conversion applies only within a
  dimensional group (distance, area); other rows keep their strict values.
- `grid_em_spatial` / `grid_ne_spatial` - over rows whose H3 cell decodes, scored by
  hierarchy containment rather than exact id, so a right place at a different resolution
  counts. `grid_em_strict` / `grid_ne_strict` are the exact-id numbers on that subset.

## Alias store

Entity, cardinal and predefined resolution goes through a per-language alias store: a
flat JSON object keyed by canonical id.

```json
{
  "osm:2476384356": {"canonical": "Cafe Kanel", "aliases": ["Cafe Kanel"]},
  "cardinal:N": {"canonical": "N", "aliases": ["N", "North", "Nord", "Norte"]},
  "predef:left": {"canonical": "left", "aliases": ["left", "gauche", "links"]}
}
```

Lookup canonicalises both sides (NFC, case folding, punctuation stripping). Where
several ids claim one alias, resolution prefers the gold id if it is among them and
otherwise takes the first in sorted order. A prediction that resolves to nothing scores
`em=0` with `resolve_failure=True`. Entity hex strings bypass the store.

The stores ship with the benchmark data, one file per language. `AliasStore.load` reads
one file; `AliasStore.for_language` reads `<aliases_dir>/<lang>.json` and falls back to
its `fallback` language when that file is absent.

## MLflow

Each `score` run logs one MLflow run with the report metrics and both output files as
artifacts. `--no-mlflow` skips it; `--parent-run-id` nests it under an existing run.
