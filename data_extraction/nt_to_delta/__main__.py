"""CLI for N-Triples to Delta conversion pipeline.

This module orchestrates the two-stage conversion:
1. Rust parser: Fast N-Triples parsing to triples Delta table
2. PySpark transformer: Pivot triples to entity format with H3 indexing
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from pyspark.sql import SparkSession

from data_extraction.nt_to_delta.config import NtToDeltaConfig
from data_extraction.nt_to_delta.transformer import TriplesTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_rust_binary() -> Path:
    """Find the nt_parser Rust binary.

    Searches in common locations:
    1. System PATH
    2. Local Cargo target directory
    3. Project nt_parser_rs/target/release
    """
    # Check PATH first
    binary = shutil.which("nt_parser")
    if binary:
        return Path(binary)

    # Check local Cargo target
    project_root = Path(__file__).parent.parent.parent
    release_path = project_root / "nt_parser_rs" / "target" / "release" / "nt_parser"
    if release_path.exists():
        return release_path

    debug_path = project_root / "nt_parser_rs" / "target" / "debug" / "nt_parser"
    if debug_path.exists():
        return debug_path

    raise FileNotFoundError(
        "Could not find nt_parser binary. "
        "Please build it with: cd nt_parser_rs && cargo build --release"
    )


def run_rust_parser(config: NtToDeltaConfig) -> None:
    """Run the Rust N-Triples parser (Stage 1).

    Args:
        config: Pipeline configuration.

    Raises:
        subprocess.CalledProcessError: If the parser fails.
    """
    binary_path = config.rust_binary_path
    if not binary_path.exists():
        binary_path = find_rust_binary()

    cmd = [
        str(binary_path),
        "--input", str(config.input_path),
        "--output", str(config.triples_delta_path),
        "--batch-size", str(config.batch_size),
    ]

    if config.overwrite:
        cmd.append("--overwrite")

    logger.info(f"Running Rust parser: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Rust parser failed:\n{result.stderr}")
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    logger.info(f"Rust parser output:\n{result.stdout}")


def run_spark_transformer(config: NtToDeltaConfig) -> None:
    """Run the PySpark transformer (Stage 2).

    Builds a SparkSession with Delta Lake + Sedona, honoring env vars
    SPARK_MASTER / SPARK_DRIVER_MEMORY / SPARK_EXECUTOR_MEMORY / SPARK_TEMP_DIR
    and SEDONA_PACKAGES from question_generator.config.

    Args:
        config: Pipeline configuration.
    """
    from sedona.spark import SedonaContext

    from question_generator.config import (
        SEDONA_PACKAGES,
        SPARK_DRIVER_MEMORY,
        SPARK_EXECUTOR_MEMORY,
        SPARK_MASTER,
        SPARK_TEMP_DIR,
    )

    logger.info(
        "Initializing Spark session (master=%s, driver=%s, executor=%s)",
        SPARK_MASTER, SPARK_DRIVER_MEMORY, SPARK_EXECUTOR_MEMORY,
    )

    builder = (
        SparkSession.builder
        .appName("NtToDeltaTransformer")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sedona.global.index", "true")
        .config("spark.sedona.global.indextype", "rtree")
        .config("spark.sedona.join.autoBroadcastJoinThreshold", "10485760")
        .config("spark.sql.shuffle.partitions", "4000")
        .config("spark.default.parallelism", "4000")
        # zstd shuffle compression — ~2x denser than lz4 default on WKT text
        # data. The planet OSM shuffle is the gating disk constraint here
        # (~315 GB lz4 vs ~158 GB zstd-6), and 40 cores have plenty of CPU
        # headroom while the join waits on shuffle I/O.
        .config("spark.io.compression.codec", "zstd")
        .config("spark.io.compression.zstd.level", "6")
        .config("spark.driver.host", "localhost")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config(
            "spark.driver.extraJavaOptions",
            "-Dsun.java2d.d3d=false -Djava.awt.headless=true -Djts.overlay=ng",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "-Dsun.java2d.d3d=false -Djava.awt.headless=true -Djts.overlay=ng",
        )
        .config("spark.jars.packages", SEDONA_PACKAGES)
    )

    if SPARK_TEMP_DIR:
        builder = builder.config("spark.local.dir", SPARK_TEMP_DIR)

    spark = builder.getOrCreate()
    spark = SedonaContext.create(spark)

    try:
        transformer = TriplesTransformer(spark, config)
        transformer.transform()
    finally:
        spark.stop()


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Convert N-Triples files to Delta tables for benchmark generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline (Rust + Spark)
    python -m data_extraction.nt_to_delta --input data.nt --output /tmp/entities

    # Only Stage 2 (if triples already exist)
    python -m data_extraction.nt_to_delta --triples /tmp/triples --output /tmp/entities --skip-rust

    # Custom H3 resolution
    python -m data_extraction.nt_to_delta --input data.nt --output /tmp/entities --h3-resolution 6
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input N-Triples file path",
    )
    parser.add_argument(
        "--triples", "-t",
        type=Path,
        help="Intermediate triples Delta table path (auto-generated if not provided)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output entities Delta table path",
    )
    parser.add_argument(
        "--h3-resolution",
        type=int,
        default=7,
        help="H3 resolution for spatial indexing (0-15, default: 7)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Batch size for Rust parser (default: 100000)",
    )
    parser.add_argument(
        "--rust-binary",
        type=Path,
        help="Path to nt_parser Rust binary (auto-detected if not provided)",
    )
    parser.add_argument(
        "--skip-rust",
        action="store_true",
        help="Skip Stage 1 (Rust parsing), use existing triples table",
    )
    parser.add_argument(
        "--skip-spark",
        action="store_true",
        help="Skip Stage 2 (Spark transformation), only run Rust parser",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Delta tables",
    )
    parser.add_argument(
        "--class-lookup",
        type=Path,
        help="Path to class_lookup.csv for category resolution (auto-detected if not provided)",
    )
    parser.add_argument(
        "--kg-source",
        choices=["worldkg", "osmh3kg", "kwg"],
        default="worldkg",
        help="KG source; selects class_lookup filter (default: worldkg)",
    )
    parser.add_argument(
        "--temporal-mode",
        choices=["column", "predicate", "none"],
        default="none",
        help="Temporal extraction mode: 'column' (osmh3kg 4th date col), 'predicate' (kwg date triples), 'none' (worldkg)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.skip_rust and not args.input:
        parser.error("--input is required unless --skip-rust is specified")

    if args.skip_rust and not args.triples:
        parser.error("--triples is required when --skip-rust is specified")

    # Set up paths
    if args.triples:
        triples_path = args.triples
    else:
        triples_path = args.output.parent / f"{args.output.name}_triples"

    # Create config
    config_kwargs = dict(
        input_path=args.input or Path("/dev/null"),  # Placeholder if skipping Rust
        triples_delta_path=triples_path,
        entities_delta_path=args.output,
        h3_resolution=args.h3_resolution,
        batch_size=args.batch_size,
        rust_binary_path=args.rust_binary or Path("nt_parser"),
        overwrite=args.overwrite,
        kg_source=args.kg_source,
        temporal_mode=None if args.temporal_mode == "none" else args.temporal_mode,
    )
    if args.class_lookup:
        config_kwargs["class_lookup_path"] = args.class_lookup
    config = NtToDeltaConfig(**config_kwargs)

    # Run pipeline
    try:
        if not args.skip_rust:
            logger.info("=== Stage 1: Rust N-Triples Parser ===")
            run_rust_parser(config)

        if not args.skip_spark:
            logger.info("=== Stage 2: PySpark Transformer ===")
            run_spark_transformer(config)

        logger.info("Pipeline complete!")

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Rust parser failed with exit code {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
