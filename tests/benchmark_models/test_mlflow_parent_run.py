"""P1-7: ``log_run`` reopens ``parent_run_id`` before nesting.

Pre-fix: ``mlflow.start_run(nested=parent_run_id is not None)`` only
nests under an *already active* parent run. Calling ``log_run`` with
a ``parent_run_id`` standalone (outside an active run context) didn't
attach to that parent at all — the param was silently ignored.

The eval side has done this correctly since Round 1 (see
``benchmark_eval/mlflow_logging.py:127-132``). This commit brings
benchmark_models into parity.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


def test_log_run_reopens_parent_run_id() -> None:
    """When ``parent_run_id`` is set, ``log_run`` first calls
    ``mlflow.start_run(run_id=parent_run_id)`` and THEN
    ``mlflow.start_run(run_name=variant_name, nested=True)``."""
    from benchmark_models.config import OpenAIVariantConfig
    from benchmark_models.mlflow_logging import log_run

    config = OpenAIVariantConfig(
        backend="openai", model_id="gpt-5.4", tier="plain"
    )

    # Stub one prediction so log_run does its full body.
    prediction = MagicMock(
        question_id="q1",
        latency_ms=100.0,
        final_state="answered",
        tool_calls=None,
        usage=MagicMock(input_tokens=10, output_tokens=5, cached_tokens=0),
        cap_hit=False,
    )

    with patch("mlflow.start_run") as start_run, patch(
        "mlflow.log_param"
    ), patch("mlflow.log_metric"), patch("mlflow.log_artifact"):
        # Each start_run call returns a context manager whose __enter__
        # yields a run-like object.
        mock_run = MagicMock()
        mock_run.info.run_id = "child-id"
        mock_run.info.run_name = "test-variant"
        start_run.return_value.__enter__.return_value = mock_run

        log_run(
            variant_name="test-variant",
            config=config,
            predictions=[prediction],
            output_path=Path("/tmp/fake.jsonl"),
            snapshot_id="snap-xyz",
            parent_run_id="parent-id-123",
        )

    # Assert: start_run was called twice — once with run_id=parent,
    # once with run_name+nested=True. The order matters: parent first.
    assert start_run.call_count == 2
    first_call_kwargs = start_run.call_args_list[0].kwargs
    second_call_kwargs = start_run.call_args_list[1].kwargs

    assert first_call_kwargs.get("run_id") == "parent-id-123"
    assert second_call_kwargs.get("run_name") == "test-variant"
    assert second_call_kwargs.get("nested") is True


def test_log_run_skips_parent_reopen_when_parent_run_id_is_none() -> None:
    """Without ``parent_run_id``, ``start_run`` is called exactly once
    with ``nested=False`` (top-level run)."""
    from benchmark_models.config import OpenAIVariantConfig
    from benchmark_models.mlflow_logging import log_run

    config = OpenAIVariantConfig(
        backend="openai", model_id="gpt-5.4", tier="plain"
    )
    prediction = MagicMock(
        question_id="q1",
        latency_ms=100.0,
        final_state="answered",
        tool_calls=None,
        usage=MagicMock(input_tokens=10, output_tokens=5, cached_tokens=0),
        cap_hit=False,
    )

    with patch("mlflow.start_run") as start_run, patch(
        "mlflow.log_param"
    ), patch("mlflow.log_metric"), patch("mlflow.log_artifact"):
        mock_run = MagicMock()
        mock_run.info.run_id = "child-id"
        mock_run.info.run_name = "test-variant"
        start_run.return_value.__enter__.return_value = mock_run

        log_run(
            variant_name="test-variant",
            config=config,
            predictions=[prediction],
            output_path=Path("/tmp/fake.jsonl"),
            snapshot_id="snap-xyz",
            parent_run_id=None,
        )

    assert start_run.call_count == 1
    assert start_run.call_args.kwargs.get("nested") is False
