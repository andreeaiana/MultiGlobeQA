"""
Question Generator Module for Spatial KG RAG Benchmark.

This module provides functionality for generating benchmark questions
from spatial knowledge graphs using PySpark and Apache Sedona.
"""

from question_generator.config import (
    TEMPLATES_TSV,
    EXCLUDED_TIDS,
    FAIL_ON_TEMPLATE_ERROR,
    KWG_ENTITIES_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    OSMH3KG_NODES_PATH,
    OUTPUT_BENCHMARK_DELTA_PATH,
    SEDONA_PACKAGES,
    SEDONA_SFIDS,
    SPARK_DRIVER_MEMORY,
    SPARK_EXECUTOR_MEMORY,
    SPARK_MASTER,
    STANDARD_SPARK_SFIDS,
    WORLDKG_NODES_PATH,
    WORLDKG_TRIPLES_PATH,
)
from question_generator.executors import (
    BaseExecutor,
    RegistryExecutor,
    SedonaExecutor,
    StandardSparkExecutor,
)
from question_generator.exporter import BenchmarkExporter
from question_generator.tracker import (
    log_data_source_info,
    log_execution_metrics,
    setup_mlflow,
)
from question_generator.validator import BenchmarkValidator

__all__ = [
    # Config - Paths
    "WORLDKG_TRIPLES_PATH",
    "WORLDKG_NODES_PATH",
    "OSMH3KG_NODES_PATH",
    "KWG_ENTITIES_PATH",
    "TEMPLATES_TSV",
    "OUTPUT_BENCHMARK_DELTA_PATH",
    # Config - MLflow
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    # Config - Spark
    "SPARK_MASTER",
    "SPARK_DRIVER_MEMORY",
    "SPARK_EXECUTOR_MEMORY",
    "SEDONA_PACKAGES",
    # Config - SFIDs
    "SEDONA_SFIDS",
    "STANDARD_SPARK_SFIDS",
    # Config - TID filtering
    "EXCLUDED_TIDS",
    # Config - Error handling
    "FAIL_ON_TEMPLATE_ERROR",
    # Pipeline
    # Executors
    "BaseExecutor",
    "SedonaExecutor",
    "StandardSparkExecutor",
    "RegistryExecutor",
    # Exporter
    "BenchmarkExporter",
    # Validator
    "BenchmarkValidator",
    # Tracker
    "setup_mlflow",
    "log_execution_metrics",
    "log_data_source_info",
]
