"""Tests for :class:`SmolagentsOracleModel` — the T3-oracle variant.

Covers construction (with ``tools=[]``), the strict-error fallback
when a question_id is not in the map, and correct prompt/template_id
tagging in the happy path. All smolagents imports are mocked so the
tests run without the real agent framework installed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from benchmark_models.base import Example
from benchmark_models.oracle import Triple


@pytest.fixture
def example() -> Example:
    return Example.from_benchmark_row(
        {
            "question_id": "qid-t3o",
            "tid": "1.1",
            "sfid": "A",
            "question": "Distance X to Y?",
            "base_template": "Distance {{entity_A}} to {{entity_B}}?",
            "kg_source": "osmh3kg",
            "entity_a_id": "1",
            "entity_a_name": "X",
            "entity_b_id": "2",
            "entity_b_name": "Y",
        }
    )


@pytest.fixture
def gold_triples_map(example: Example) -> dict[str, list[Triple]]:
    return {
        example.question_id: [
            {
                "subject": "1",
                "predicate": "http://www.opengis.net/ont/geosparql#hasGeometry",
                "object": "geo1",
            },
            {
                "subject": "geo1",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": "POINT (1 2)",
            },
        ]
    }


def _mock_smolagents_module(
    *,
    final_answer_called: bool = True,
    n_steps: int = 1,
    return_value: Any = None,
) -> MagicMock:
    """Construct a mock smolagents module with enough surface to satisfy
    :class:`SmolagentsOracleModel`.

    Args:
        final_answer_called: when True (default), at least one
            mocked action step has ``is_final_answer=True``. When
            False, no step has it set — exercises the
            ``no_final_answer`` detection path.
        n_steps: how many action steps the mock memory exposes.
        return_value: what ``agent.run`` returns (defaults to a
            valid answer dict for the happy path).
    """
    mock_smol = MagicMock()
    agent = MagicMock()

    steps = []
    for i in range(n_steps):
        step = MagicMock()
        step.code_action = "final_answer({'answer': 1.5, 'unit': 'km'})"
        # Only the last step is the "final answer" step (when applicable).
        step.is_final_answer = (
            final_answer_called and i == n_steps - 1
        )
        step.observations = "ok"
        step.error = None
        step.token_usage = MagicMock(input_tokens=10, output_tokens=5)
        steps.append(step)
    agent.memory.steps = steps

    agent.run.return_value = (
        return_value
        if return_value is not None
        else {"answer": 1.5, "unit": "km"}
    )
    mock_smol.CodeAgent.return_value = agent
    return mock_smol


class TestSmolagentsOracleModelConstruction:
    def test_constructs_with_no_tools(
        self,
        registry: Any,
        example: Example,
        gold_triples_map: dict[str, list[Triple]],
    ) -> None:
        """The oracle model must construct the CodeAgent with an empty
        tool list. Construction happens *inside* ``predict()`` (thread
        safety — see ``SmolagentsModel.__init__``), so we have to
        trigger a predict call to observe the constructor args.
        """
        mock_smol = _mock_smolagents_module()
        with patch.dict("sys.modules", {"smolagents": mock_smol}):
            from benchmark_models.tier3 import SmolagentsOracleModel

            model = SmolagentsOracleModel(
                "test-t3o",
                provider="google",
                provider_model_id="gemini-3-flash-preview",
                gold_triples_map=gold_triples_map,
                api_key="fake",
                registry=registry,
            )
            model.predict(example)

            # Inspect the CodeAgent constructor call.
            call = mock_smol.CodeAgent.call_args
            assert call.kwargs["tools"] == []


class TestSmolagentsOracleModelPredict:
    def test_happy_path_uses_oracle_template(
        self,
        registry: Any,
        example: Example,
        gold_triples_map: dict[str, list[Triple]],
    ) -> None:
        """Predict emits ``prompt_template_id='tier3.oracle'`` and the
        prompt passed to ``agent.run`` contains the formatted triples.
        Also asserts ``final_state='answered'`` — the success tail now
        sets this explicitly (previously left as ``None``)."""
        mock_smol = _mock_smolagents_module()
        with patch.dict("sys.modules", {"smolagents": mock_smol}):
            from benchmark_models.tier3 import SmolagentsOracleModel

            model = SmolagentsOracleModel(
                "test-t3o",
                provider="google",
                provider_model_id="gemini-3-flash-preview",
                gold_triples_map=gold_triples_map,
                api_key="fake",
                registry=registry,
            )
            pred = model.predict(example)

            # Prompt sent to the agent contains the structured triples.
            # ``mock_smol.CodeAgent.return_value`` is the mock agent
            # instance returned by each (per-call) CodeAgent(...) call;
            # ``.run.call_args`` captures the most recent invocation.
            sent_prompt: str = mock_smol.CodeAgent.return_value.run.call_args.args[0]
            assert "TRIPLES:" in sent_prompt
            assert "POINT (1 2)" in sent_prompt
            # T3-oracle-specific framing appears.
            assert "NOT have access" in sent_prompt
            # Prediction records which template was used.
            assert pred.prompt_template_id == "tier3.oracle"
            # Every tier-3 prediction must declare a terminal state
            # so the eval pipeline can group on it without rederiving
            # from refusal / cap_hit.
            assert pred.final_state == "answered"
            assert pred.refusal is False

    def test_abstention_emits_abstained_state(
        self,
        registry: Any,
        example: Example,
        gold_triples_map: dict[str, list[Triple]],
    ) -> None:
        """Explicit ``final_answer({"answer": None})`` → ``abstained``
        (distinct from ``no_final_answer``, which is the malformed /
        unparseable case)."""
        mock_smol = _mock_smolagents_module(
            return_value={"answer": None, "unit": None}
        )
        with patch.dict("sys.modules", {"smolagents": mock_smol}):
            from benchmark_models.tier3 import SmolagentsOracleModel

            model = SmolagentsOracleModel(
                "test-t3o",
                provider="google",
                provider_model_id="gemini-3-flash-preview",
                gold_triples_map=gold_triples_map,
                api_key="fake",
                registry=registry,
            )
            pred = model.predict(example)
            assert pred.final_state == "abstained"
            assert pred.refusal is True
            assert pred.answer is None

    def test_missing_qid_emits_error_prediction(
        self,
        registry: Any,
        example: Example,
    ) -> None:
        """A question_id absent from the map produces a strict-error
        Prediction (final_state='error'), not a silent fallback."""
        mock_smol = _mock_smolagents_module()
        with patch.dict("sys.modules", {"smolagents": mock_smol}):
            from benchmark_models.tier3 import SmolagentsOracleModel

            model = SmolagentsOracleModel(
                "test-t3o",
                provider="google",
                provider_model_id="gemini-3-flash-preview",
                gold_triples_map={},  # deliberately empty
                api_key="fake",
                registry=registry,
            )
            pred = model.predict(example)
            assert pred.final_state == "error"
            assert pred.refusal is True
            # agent.run must NOT have been called — we errored before
            # invocation. With per-call agent construction, we also
            # check that CodeAgent was never constructed: the missing
            # qid check fires BEFORE _build_agent.
            mock_smol.CodeAgent.assert_not_called()

    def test_no_final_answer_emits_refusal(
        self,
        registry: Any,
        example: Example,
        gold_triples_map: dict[str, list[Triple]],
    ) -> None:
        """Agent runs to completion under the cap but never registers
        a final_answer call → ``final_state='no_final_answer'``,
        ``refusal=True``, ``answer=None``.

        Most common cause: model wraps the call inside a function
        definition. The detection looks for any action_step with
        ``is_final_answer=True``; when none is present, this branch
        fires.
        """
        mock_smol = _mock_smolagents_module(
            final_answer_called=False,
            n_steps=3,
            # Smolagents would return e.g. the raw code text in this
            # situation; we don't care what it returns because the
            # predict() should ignore it.
            return_value="from geopy import ...",
        )
        with patch.dict("sys.modules", {"smolagents": mock_smol}):
            from benchmark_models.tier3 import SmolagentsOracleModel

            model = SmolagentsOracleModel(
                "test-t3o",
                provider="google",
                provider_model_id="gemini-3-flash-preview",
                gold_triples_map=gold_triples_map,
                api_key="fake",
                registry=registry,
            )
            pred = model.predict(example)
            assert pred.final_state == "no_final_answer"
            assert pred.refusal is True
            assert pred.answer is None
            assert pred.cap_hit is False  # not the cap, the detection
