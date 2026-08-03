# question_generator

Builds the benchmark from knowledge-graph data: ingestion, entity sampling,
template instantiation, answer computation, validation and export.

Every answer is produced by spatial SQL over the graph, so it is verifiable and
reproducible from a seed.

## Pipeline

```
raw KG dumps
   → nt_parser_rs (Rust)            N-Triples → RDF triples
   → data_extraction.nt_to_delta    triples → entity Delta tables
   → PageRank                       entity prominence
   → shape_executor                 sampling + spatial SQL per sub-template
   → validator                      answer sanity checks
   → exporter                       JSONL / Parquet
```

## Knowledge graphs

| KG | source | coverage | geometries |
|---|---|---|---|
| WorldKG | OpenStreetMap | global | points |
| OSMH3KG | OpenStreetMap + H3 grid | configurable regions | points, lines, polygons |
| KnowWhereGraph | USGS, EPA, Census, USDA and others | US-focused | points, polygons, topology |

Coverage varies: not every spatial family can be asked of every graph.

## Spatial-function families

Fourteen families, defined in
`benchmark_models/schemas/spatial_functions.py`:

| | | | |
|---|---|---|---|
| A | Metric distance & proximity | H | Grid & spatial indexing |
| B | Metric containment | I | Spatio-temporal events |
| C | Topological relationships | J | Directional & orientation |
| D | Network & path-based | K | Shape & geometry properties |
| E | Containment & administrative hierarchy | L | Uncertainty & fuzzy reasoning |
| F | Spatial comparison & ordering | M | Transformation & reference change |
| G | Centrality & medial geometry | N | Spatial reference & geometry |

## SQL shapes

Each of the 129 sub-templates is a triplet under `sql_shapes/<sfid>/`:

| file | contents |
|---|---|
| `<tid>.meta.yaml` | applicable graphs, entity pools, sampling strategy, parameters, which predicates to collect as gold triples |
| `<tid>.sql` | the question-generation query |
| `<tid>.gold.sql` | the gold-triple extraction query |

A shape may declare several **variants** — different entity pairings over the same
SQL. Reference one as `<tid>:<variant>`; a bare `<tid>` resolves when the shape
declares only one.

## CLI

```
multiglobeqa extract   --kg {worldkg,osmh3kg,kwg,all}    fetch raw KG data
multiglobeqa convert   --kg {worldkg,osmh3kg,kwg,all}    triples → Delta tables
multiglobeqa pagerank  --kg {worldkg,osmh3kg,kwg,all}    entity prominence
multiglobeqa enrich    --kg {worldkg,osmh3kg,kwg,all}    cell coverage, population, strata
multiglobeqa generate  --run <spec>.yaml                 instantiate questions
multiglobeqa validate  --input benchmark.jsonl           check an existing benchmark
multiglobeqa run       --kg all --run <spec>.yaml        extract → convert → pagerank → generate
```

### Run spec

`generate` is driven by a YAML spec, validated by `RunSpec` in `shape_loader.py`:

```yaml
run_name: example
tids:                      # <tid> or <tid>:<variant>
  - 1.1:poi
kgs:      [kwg, osmh3kg, worldkg]
strata:   [HIC_high, LIC_high]     # <income>_<density>, or omit for all
run_seed: 12345
total_sample: 2000
output_dir: data/benchmark_output/example
countries_per_cell: 2      # optional, default 2
overrides: {}              # optional per-tid sampling overrides
```

Strata cells are `<income>_<density>`, income in {HIC, UMC, LMC, LIC} and density in
{high, medium, low}.

## Data preparation

| KG | steps |
|---|---|
| WorldKG | download the NT dumps, then `cd nt_parser_rs && cargo run --release`, then `multiglobeqa convert --kg worldkg` |
| OSMH3KG | `bash data_extraction/download_osm.sh`, then `multiglobeqa extract --kg osmh3kg`, then `convert` |
| KnowWhereGraph | set the SPARQL endpoint, then `multiglobeqa extract --kg kwg` |

Population rasters come from `data_extraction/download_population.sh`; the strata
assignment in `enrich` needs them.

## Configuration

| variable | default | purpose |
|---|---|---|
| `MULTIGLOBEQA_DATA_DIR` | `./data` | root data directory |
| `MULTIGLOBEQA_INCLUDED_TIDS` | all | comma-separated allowlist, e.g. `1,13,64` |
| `MULTIGLOBEQA_FAIL_ON_ERROR` | `false` | abort instead of skipping a failing template |
| `MULTIGLOBEQA_MAX_WKT_BYTES` | — | drop geometries above this size |
| `MULTIGLOBEQA_MAX_POOL_SIZE` | — | cap entities per pool |
| `MULTIGLOBEQA_GOLD_TRIPLES_ERROR_THRESHOLD` | — | tolerated share of gold-triple failures |
| `ENTITIES_PER_TEMPLATE` | 30 | entities sampled per template |
| `WORLDKG_NODES_PATH`, `OSMH3KG_NODES_PATH`, `KWG_ENTITIES_PATH` | under `data/` | Delta table locations |
| `SPARK_MASTER`, `SPARK_DRIVER_MEMORY`, `SPARK_EXECUTOR_MEMORY`, `SPARK_TEMP_DIR` | `local[*]`, `4g` | Spark |
| `PAGERANK_DAMPING`, `PAGERANK_MAX_ITER` | 0.85, — | PageRank |
| `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME` | local | run tracking |

## Output schema

One JSON object per question:

| column | description |
|---|---|
| `question_id` | UUID; the join key for predictions and gold triples |
| `tid` / `vid` | sub-template id (`13.1`) and paraphrase-variant id (`13.1.2`) |
| `variant_stid` | variant index within the sub-template |
| `sfid` | spatial-function family, A–N |
| `question` | the rendered question |
| `base_template` | the template with `{{entity_A}}`-style placeholders, English |
| `ground_truth` / `ground_truth_type` | the computed answer and its type |
| `entity_a_id` / `entity_a_name` | first operand; `entity_b_*` and `entity_c_*` when the template is binary or ternary |
| `parameters` | JSON of the sampling parameters the shape used |
| `kg_source` | originating graph |
| `strata_cell` | `<income>_<density>` |
| `is_true_premise` | false-premise questions have no valid answer |
| `is_valid` | passed answer validation |
| `language` | ISO 639-3 code |

Gold triples are written alongside, keyed by `question_id`: the KG triples the answer
was computed from.
