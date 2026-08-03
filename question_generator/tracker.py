"""
MLflow Tracking Integration.

Provides utilities for experiment tracking.
"""

from typing import Any

import mlflow

from question_generator.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)


def setup_mlflow() -> str:
    """
    Initialize MLflow tracking URI and experiment.

    Returns:
        The experiment ID for the current experiment.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return experiment.experiment_id


def log_sampling_params(
    total_samples: int,
    pagerank_ratio: float,
    h3_resolution: int,
) -> None:
    """
    Log sampling parameters to MLflow.

    Args:
        total_samples: Total number of samples requested.
        pagerank_ratio: Ratio of PageRank-based sampling.
        h3_resolution: H3 resolution used for density calculation.
    """
    mlflow.log_params({
        "total_samples_requested": total_samples,
        "pagerank_ratio": pagerank_ratio,
        "h3_resolution": h3_resolution,
    })


def log_execution_metrics(
    actual_samples: int,
    templates_processed: int,
    benchmark_rows: int,
) -> None:
    """
    Log execution metrics to MLflow.

    Args:
        actual_samples: Actual number of samples generated.
        templates_processed: Number of templates successfully processed.
        benchmark_rows: Total rows in final benchmark output.
    """
    mlflow.log_metrics({
        "actual_samples_generated": actual_samples,
        "templates_processed": templates_processed,
        "benchmark_rows_written": benchmark_rows,
    })


def log_data_source_info(
    sources: dict[str, Any],
) -> None:
    """
    Log data source information to MLflow.

    Args:
        sources: Dictionary mapping source names to their metadata.
    """
    for name, info in sources.items():
        mlflow.log_param(f"data_source_{name}", str(info))
