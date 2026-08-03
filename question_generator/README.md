# question_generator

Builds the spatial QA benchmark from raw KG data: ingestion, entity sampling, template instantiation, validation, and export.

## Architecture Overview

```
Raw Data (KWG / OSM / WorldKG)
    │
    ▼
NT Parser (Rust) → RDF Triples
    │
    ▼
nt_to_delta.transformer → Entity Delta Tables
    │
    ▼
PageRank Computation → Ranked Entities
    │
    ▼
Sampler (Pragmatic or Hierarchical) → Sampled Entities
    │
    ▼
TemplateManager → SQL/UDF Execution via SedonaExecutor or StandardSparkExecutor
    │
    ▼
BenchmarkValidator → Validated Answers
    │
    ▼
BenchmarkExporter → JSONL / Parquet Output
```

### Knowledge Graphs

| KG                 | Source                                 | Coverage             | Entity Types                                  |
| ------------------ | -------------------------------------- | -------------------- | --------------------------------------------- |
| **WorldKG**        | OpenStreetMap via geographic triples   | Global               | POIs, landmarks, natural features             |
| **OSMH3KG**        | OpenStreetMap + H3 DGG                 | Configurable regions | OSM features with H3 spatial indexing         |
| **KnowWhereGraph** | Multi-source (USGS, EPA, Census, etc.) | US-focused           | Observations, administrative regions, hazards |

### Template System

19 benchmark templates across 10 Spatial Function ID (SFID) categories:

| SFID | Category           | Example Question                         |
| ---- | ------------------ | ---------------------------------------- |
| A    | Metric Distance    | "What is the distance between X and Y?"  |
| B    | Metric Containment | "How many X are within 1km of Y?"        |
| C    | Topological        | "Does X intersect Y?"                    |
| D    | Network/Path       | "What is the shortest path from X to Y?" |
| E    | Administrative     | "Does boundary of X contain Y?"          |
| F    | Ordering           | "What is the largest X in Y?"            |
| H    | Hierarchy          | "What region contains X?"                |
| K    | Directional        | "Is X north of Y?"                       |
| L    | Shape/Geometry     | "What is the fractal dimension of X?"    |
| P    | Coordinates        | "What are the coordinates of X?"         |

## Quick Start

```bash
# Generate benchmark with dummy data (no real KG data needed)
uv run spatialbenchrag generate --dry-run --samples 100 --seed 42 --output-format jsonl,parquet

# Full pipeline with real data
uv run spatialbenchrag run --kg all --samples 3000 --seed 42

# Validate an existing benchmark
uv run spatialbenchrag validate --input data/benchmark_output/benchmark.jsonl
```

### CLI Subcommands

```bash
  # 1. Extract raw data
  uv run spatialbenchrag extract --kg all

  # 2. Convert triples → entity table
  uv run spatialbenchrag convert --kg all

  # 3. PageRank (needs triples, writes separate sibling table)
  uv run spatialbenchrag pagerank --kg all

  # 4. Enrich (needs entity table + PageRank already joined)
  uv run spatialbenchrag enrich --kg all

  # 5. Generate benchmark
  uv run spatialbenchrag generate --pipeline-mode compute-first --seed 42
```

```
spatialbenchrag extract   --kg {worldkg,osmh3kg,kwg,all}     # Extract raw data
spatialbenchrag convert   --kg {worldkg,osmh3kg,kwg,all}     # Convert to Delta tables
spatialbenchrag pagerank  --kg {worldkg,osmh3kg,kwg,all}     # Compute PageRank
spatialbenchrag generate  --samples N --seed S               # Generate benchmark
spatialbenchrag validate  --input benchmark.jsonl             # Validate answers
spatialbenchrag run       --kg all --samples 3000 --seed 42  # Full pipeline
```

## Data Preparation

### WorldKG

1. Download WorldKG NT files to `data/worldkg/`
2. Parse with Rust parser: `cd nt_parser_rs && cargo run --release`
3. Convert: `uv run spatialbenchrag convert --kg worldkg`

### OSMH3KG

1. Download OSM extracts: `bash data_extraction/download_osm.sh`
2. Generate H3 grid: `uv run spatialbenchrag extract --kg osmh3kg`
3. Convert: `uv run spatialbenchrag convert --kg osmh3kg`

### KnowWhereGraph

1. Configure SPARQL endpoints in environment
2. Crawl: `uv run spatialbenchrag extract --kg kwg`

## Configuration

All configuration can be overridden via environment variables:

| Variable                     | Default                            | Description                 |
| ---------------------------- | ---------------------------------- | --------------------------- |
| `SPATIALBENCH_DATA_DIR`      | `./data`                           | Root data directory         |
| `WORLDKG_NODES_PATH`         | `data/worldkg/worldkg_nodes`       | WorldKG Delta table         |
| `OSMH3KG_NODES_PATH`         | `data/osmh3kg/osmh3kg_nodes`       | OSMH3KG Delta table         |
| `KWG_ENTITIES_PATH`          | `data/kwg/knowwheregraph_entities` | KWG Delta table             |
| `SPARK_MASTER`               | `local[*]`                         | Spark master URL            |
| `SPARK_DRIVER_MEMORY`        | `4g`                               | Spark driver memory         |
| `SPATIALBENCH_FAIL_ON_ERROR` | `false`                            | Fail on template errors     |
| `DEFAULT_SAMPLE_SIZE`        | `3000`                             | Default entity sample count |
| `MLFLOW_TRACKING_URI`        | `file://./logs/mlruns`             | MLflow tracking URI         |
| `OUTPUT_BENCHMARK_DIR`       | `data/benchmark_output`            | Benchmark output directory  |

## Output Format

The benchmark is exported in JSONL and/or Parquet format with the following schema:

| Column              | Type    | Description                                         |
| ------------------- | ------- | --------------------------------------------------- |
| `question_id`       | string  | Unique identifier (UUID)                            |
| `tid`               | string  | Template ID                                         |
| `sfid`              | string  | Spatial Function ID                                 |
| `question`          | string  | Interpolated question text                          |
| `base_template`     | string  | Original template with placeholders                 |
| `ground_truth`      | string  | Computed answer                                     |
| `ground_truth_type` | string  | Answer type (distance_meters, count, boolean, etc.) |
| `entity_a_id`       | string  | Entity A identifier                                 |
| `entity_a_name`     | string  | Entity A human-readable name                        |
| `entity_b_id`       | string  | Entity B identifier (if binary)                     |
| `entity_b_name`     | string  | Entity B human-readable name (if binary)            |
| `parameters`        | string  | JSON parameters used                                |
| `kg_source`         | string  | Source knowledge graph                              |
| `is_valid`          | boolean | Whether ground truth passes validation              |
| `validation_errors` | string  | Validation error messages (if any)                  |

