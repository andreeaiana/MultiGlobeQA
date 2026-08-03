"""CLI entry point for SpatialBenchRAG pipeline orchestration.

Provides subcommands for each stage of the benchmark generation pipeline:
    spatialbenchrag extract   --kg {worldkg,osmh3kg,kwg,all}
    spatialbenchrag convert   --kg {worldkg,osmh3kg,kwg,all}
    spatialbenchrag pagerank  --kg {worldkg,osmh3kg,kwg,all}
    spatialbenchrag generate  --run runs/<spec>.yaml
    spatialbenchrag validate  --input benchmark.jsonl
    spatialbenchrag run       --kg all --run runs/<spec>.yaml
"""

import argparse
import sys

from loguru import logger


def _add_kg_argument(parser: argparse.ArgumentParser) -> None:
    """Add the common --kg argument to a subparser."""
    parser.add_argument(
        "--kg",
        choices=["worldkg", "osmh3kg", "kwg", "all"],
        default="all",
        help="Knowledge graph source to process",
    )


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract raw data from knowledge graph sources."""
    kg = args.kg
    logger.info("Extracting data for KG: {}", kg)

    if kg in ("worldkg", "all"):
        try:
            from data_extraction.worldkg_construction import construct_worldkg

            construct_worldkg()
        except ImportError:
            logger.warning("WorldKG extraction module not available.")
        except Exception as e:
            logger.error("WorldKG extraction failed: {}", e)

    if kg in ("kwg", "all"):
        try:
            from data_extraction.kwg_crawler import crawl_kwg

            crawl_kwg()
        except ImportError:
            logger.warning("KWG crawler module not available.")
        except Exception as e:
            logger.error("KWG extraction failed: {}", e)

    if kg in ("osmh3kg", "all"):
        try:
            from data_extraction.osmh3kg.data_preparation.h3_generation import create_h3_grid

            create_h3_grid()
        except ImportError:
            logger.warning("OSMH3KG H3 generation module not available.")
        except Exception as e:
            logger.error("OSMH3KG extraction failed: {}", e)

    logger.info("Extraction complete.")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert raw data to Delta tables."""
    kg = args.kg
    logger.info("Converting data to Delta for KG: {}", kg)

    try:
        from data_extraction.nt_to_delta import NtToDeltaConfig, TriplesTransformer
        from question_generator.config import (
            KWG_ENTITIES_PATH,
            KWG_TRIPLES_PATH,
            OSMH3KG_NODES_PATH,
            OSMH3KG_TRIPLES_PATH,
            WORLDKG_NODES_PATH,
            WORLDKG_TRIPLES_PATH,
        )
        from question_generator.main import init_spark

        kg_configs = {
            "worldkg": {
                "triples_path": WORLDKG_TRIPLES_PATH,
                "entities_path": WORLDKG_NODES_PATH,
                "kg_source": "worldkg",
                "name_en_predicate": "http://www.worldkg.org/schema/nameEn",
                "geometry_link_predicates": [
                    "http://www.worldkg.org/schema/spatialObject",
                    "http://www.opengis.net/ont/geosparql#hasGeometry",
                ],
                "geometry_predicates": [
                    "http://www.opengis.net/ont/geosparql#asWKT",
                ],
                "temporal_mode": None,
            },
            "osmh3kg": {
                "triples_path": OSMH3KG_TRIPLES_PATH,
                "entities_path": OSMH3KG_NODES_PATH,
                "kg_source": "osmh3kg",
                "name_en_predicate": "name",
                "geometry_link_predicates": [
                    "http://www.opengis.net/ont/geosparql#hasGeometry",
                ],
                "geometry_predicates": [
                    "http://www.opengis.net/ont/geosparql#asWKT",
                ],
                "temporal_mode": "column",
                "address_street_predicate": "addr:street",
                "address_housenumber_predicate": "addr:housenumber",
                "address_suburb_predicate": "addr:suburb",
            },
            "kwg": {
                "triples_path": KWG_TRIPLES_PATH,
                "entities_path": KWG_ENTITIES_PATH,
                "kg_source": "kwg",
                "name_en_predicate": "http://www.w3.org/2000/01/rdf-schema#label",
                "geometry_link_predicates": [
                    "http://www.opengis.net/ont/geosparql#hasGeometry",
                ],
                "geometry_predicates": [
                    "http://www.opengis.net/ont/geosparql#asWKT",
                ],
                "temporal_mode": "predicate",
            },
        }

        targets = [kg] if kg != "all" else list(kg_configs.keys())

        spark = None
        try:
            spark = init_spark()
            for kg_name in targets:
                if kg_name not in kg_configs:
                    logger.warning("Convert not supported for KG: {}", kg_name)
                    continue
                cfg = kg_configs[kg_name]
                logger.info("Converting {} triples to entities...", kg_name)
                config = NtToDeltaConfig(
                    input_path="/dev/null",
                    triples_delta_path=cfg["triples_path"],
                    entities_delta_path=cfg["entities_path"],
                    kg_source=cfg["kg_source"],
                    name_en_predicate=cfg["name_en_predicate"],
                    geometry_link_predicates=cfg["geometry_link_predicates"],
                    geometry_predicates=cfg["geometry_predicates"],
                    temporal_mode=cfg["temporal_mode"],
                    address_street_predicate=cfg.get("address_street_predicate"),
                    address_housenumber_predicate=cfg.get("address_housenumber_predicate"),
                    address_suburb_predicate=cfg.get("address_suburb_predicate"),
                    overwrite=True,
                )
                transformer = TriplesTransformer(spark, config)
                transformer.transform()
                logger.info("Conversion complete for {}", kg_name)
        finally:
            if spark is not None:
                spark.stop()

    except ImportError:
        logger.warning("nt_to_delta module not available. Run: uv sync")
    except Exception as e:
        logger.error("Conversion failed: {}", e)
        return 1

    logger.info("Conversion complete.")
    return 0


def cmd_enrich(args: argparse.Namespace) -> int:
    """Run enrichment pipeline: cell coverage, population grid, strata assignment."""
    from question_generator.config import REFERENCE_DATA_PATH, cells_path, strata_path
    from question_generator.main import init_spark

    kg = args.kg
    logger.info("Running enrichment pipeline for KG: {}", kg)

    spark = init_spark()

    try:
        from sedona.spark import SedonaContext

        spark = SedonaContext.create(spark)
    except Exception as e:
        logger.warning("Sedona context creation had issues: {}", e)

    # Determine entity path
    if kg == "worldkg":
        from question_generator.config import WORLDKG_NODES_PATH
        entity_paths = [("worldkg", WORLDKG_NODES_PATH)]
    elif kg == "osmh3kg":
        from question_generator.config import OSMH3KG_NODES_PATH
        entity_paths = [("osmh3kg", OSMH3KG_NODES_PATH)]
    elif kg == "kwg":
        from question_generator.config import KWG_ENTITIES_PATH
        entity_paths = [("kwg", KWG_ENTITIES_PATH)]
    else:
        from question_generator.config import (
            KWG_ENTITIES_PATH,
            OSMH3KG_NODES_PATH,
            WORLDKG_NODES_PATH,
        )
        entity_paths = [
            ("worldkg", WORLDKG_NODES_PATH),
            ("osmh3kg", OSMH3KG_NODES_PATH),
            ("kwg", KWG_ENTITIES_PATH),
        ]

    # Step 1: Cell coverage (non-POINT geometry explosion)
    if not args.skip_cells:
        from data_extraction.cell_coverage import cell_coverage_calculation

        for kg_label, entity_path in entity_paths:
            output = cells_path(kg_label)
            logger.info("Computing cell coverage for {} -> {}", kg_label, output)
            try:
                cell_coverage_calculation(entity_path, output, spark)
            except Exception as e:
                logger.warning("Cell coverage for {} failed: {}", kg_label, e)

    spark.stop()

    # Step 2: Population grid (one-time, global)
    if not args.skip_population:
        from data_extraction.population_aggregation import (
            aggregate_population,
            init_spark_session as init_pop_spark,
        )

        pop_output = str(REFERENCE_DATA_PATH / "population_h3_res3")
        raster_path = args.raster_path or "data/population/global_pop_2025.tif"
        logger.info("Aggregating population raster -> {}", pop_output)
        pop_spark = None
        try:
            pop_spark = init_pop_spark()
            from sedona.spark import SedonaContext as _SC
            pop_spark = _SC.create(pop_spark)
            aggregate_population(raster_path, pop_output, pop_spark)
        except Exception as e:
            logger.error("Population aggregation failed: {}", e)
            logger.error("Run separately with: uv run python -m data_extraction.population_aggregation")
        finally:
            if pop_spark is not None:
                try:
                    pop_spark.stop()
                except Exception:
                    pass

    # Step 3: Country + income + strata assignment
    if not args.skip_strata:
        from data_extraction.country_enrichment import enrich_with_strata

        strata_spark = None
        try:
            strata_spark = init_spark()
            from sedona.spark import SedonaContext as _SC2
            strata_spark = _SC2.create(strata_spark)

            pop_grid_path = str(REFERENCE_DATA_PATH / "population_h3_res3")
            for kg_label, entity_path in entity_paths:
                output = strata_path(kg_label)
                logger.info("Computing strata assignment for {} -> {}", kg_label, output)
                try:
                    enrich_with_strata(entity_path, pop_grid_path, output, strata_spark)
                except Exception as e:
                    logger.warning("Strata assignment for {} failed: {}", kg_label, e)
        except Exception as e:
            logger.error("Could not start Spark for strata assignment: {}", e)
        finally:
            if strata_spark is not None:
                try:
                    strata_spark.stop()
                except Exception:
                    pass

    logger.info("Enrichment complete.")
    return 0


def cmd_pagerank(args: argparse.Namespace) -> int:
    """Compute PageRank for entity importance scoring."""
    kg = args.kg
    logger.info("Computing PageRank for KG: {}", kg)

    try:
        from data_extraction.pagerank_computation import init_spark, run_pagerank_for_kg

        spark = init_spark()
        if not run_pagerank_for_kg(spark, kg):
            logger.warning("PageRank returned no results for {}", kg)
        spark.stop()
    except ImportError:
        logger.warning("PageRank computation module not available.")
    except Exception as e:
        logger.error("PageRank computation failed: {}", e)
        return 1

    logger.info("PageRank computation complete.")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate benchmark questions from a run spec.

    The run spec (``runs/*.yaml``) names the templates, KGs, strata, seed and
    sample size. Everything else — SQL, pools, sampling strategy, gold-triple
    predicates — comes from the matching shape under ``sql_shapes/``.
    """
    from question_generator.main import generate_from_run_spec

    try:
        generate_from_run_spec(args.run)
        return 0
    except Exception as e:
        logger.error("Benchmark generation failed: {}", e)
        import traceback

        traceback.print_exc()
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate an existing benchmark file."""
    input_path = args.input
    logger.info("Validating benchmark at: {}", input_path)

    try:
        from pyspark.sql import SparkSession

        from question_generator.validator import BenchmarkValidator

        spark = (
            SparkSession.builder.appName("SpatialBenchRAG-Validate")
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

        if input_path.endswith(".jsonl") or input_path.endswith(".json"):
            df = spark.read.json(input_path)
        elif input_path.endswith(".parquet"):
            df = spark.read.parquet(input_path)
        else:
            df = spark.read.format("delta").load(input_path)

        validator = BenchmarkValidator()
        validated = validator.validate(df)

        total = validated.count()
        valid = validated.filter(validated.is_valid == True).count()  # noqa: E712
        logger.info("Results: {}/{} valid ({:.1f}%)", valid, total, valid / total * 100 if total else 0)

        invalid = validated.filter(validated.is_valid == False)  # noqa: E712
        if invalid.count() > 0:
            logger.warning("Invalid rows:")
            invalid.select("question_id", "tid", "sfid", "validation_errors").show(
                20, truncate=False
            )

        spark.stop()
        return 0

    except Exception as e:
        logger.error("Validation failed: {}", e)
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full pipeline: extract, convert, pagerank, generate."""
    logger.info("Running full pipeline for KG: {}", args.kg)

    extract_args = argparse.Namespace(kg=args.kg)
    ret = cmd_extract(extract_args)
    if ret != 0:
        logger.warning("Extraction had issues (exit code {}), continuing...", ret)

    convert_args = argparse.Namespace(kg=args.kg)
    ret = cmd_convert(convert_args)
    if ret != 0:
        logger.warning("Conversion had issues (exit code {}), continuing...", ret)

    pr_args = argparse.Namespace(kg=args.kg)
    ret = cmd_pagerank(pr_args)
    if ret != 0:
        logger.warning("PageRank had issues (exit code {}), continuing...", ret)

    return cmd_generate(args)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="spatialbenchrag",
        description="SpatialBenchRAG — Spatial Q&A Benchmark Generation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract raw data from KG sources")
    _add_kg_argument(p_extract)

    # convert
    p_convert = subparsers.add_parser("convert", help="Convert raw data to Delta tables")
    _add_kg_argument(p_convert)

    # pagerank
    p_pagerank = subparsers.add_parser("pagerank", help="Compute PageRank scores")
    _add_kg_argument(p_pagerank)

    # enrich
    p_enrich = subparsers.add_parser("enrich", help="Run enrichment: cell coverage, population grid, strata assignment")
    _add_kg_argument(p_enrich)
    p_enrich.add_argument("--skip-cells", action="store_true", help="Skip cell coverage computation")
    p_enrich.add_argument("--skip-population", action="store_true", help="Skip population grid aggregation")
    p_enrich.add_argument("--skip-strata", action="store_true", help="Skip strata assignment")
    p_enrich.add_argument("--raster-path", type=str, default=None, help="Path to population raster GeoTIFF")

    # generate
    p_generate = subparsers.add_parser("generate", help="Generate benchmark questions")
    p_generate.add_argument(
        "--run", type=str, required=True,
        help="Path to a runs/*.yaml run spec",
    )

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate an existing benchmark file")
    p_validate.add_argument("--input", required=True, help="Path to benchmark file (JSONL/Parquet/Delta)")

    # run (full pipeline)
    p_run = subparsers.add_parser("run", help="Run the full pipeline end-to-end")
    _add_kg_argument(p_run)
    p_run.add_argument(
        "--run", type=str, required=True,
        help="Path to a runs/*.yaml run spec",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "extract": cmd_extract,
        "convert": cmd_convert,
        "pagerank": cmd_pagerank,
        "enrich": cmd_enrich,
        "generate": cmd_generate,
        "validate": cmd_validate,
        "run": cmd_run,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
