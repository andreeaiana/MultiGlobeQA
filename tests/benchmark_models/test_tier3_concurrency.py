"""Thread-safety regression tests for :class:`SmolagentsModel` /
:class:`SmolagentsOracleModel`.

Earlier versions of the tier-3 code kept a single ``self.agent`` on
the model instance and called ``self.agent.run()`` from every
:meth:`predict_batch` worker. Smolagents' ``run()`` mutates
``agent.memory`` / ``agent.state`` / ``agent.python_executor`` on
every invocation, so concurrent calls raced:

* ``_extract_trace`` / ``_extract_usage`` read shared ``memory.steps``
  *after* ``run()`` returned, but by then another thread could have
  reset memory for its own run — traces ended up empty or mixed
  between questions.
* ``cap_hit`` / ``no_final_answer`` detection read the same shared
  memory and could mis-fire.
* The returned ``answer`` itself was per-thread (returned up the
  stack synchronously), so headline EM usually looked fine — but
  every diagnostic field that feeds the paper's tool-use-analysis
  table was unreliable.

The 2026-04-23 refactor moved ``CodeAgent`` construction from
``__init__`` into a per-call ``_build_agent()`` helper. These tests
pin that invariant: under concurrent ``predict_batch``, every
question's trace is isolated, no qid's trace contains another qid's
steps.

All smolagents imports are mocked so the tests run without the real
framework installed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from benchmark_models.base import Example


def _make_example(qid: str, tid: str = "1.1") -> Example:
    return Example.from_benchmark_row(
        {
            "question_id": qid,
            "tid": tid,
            "sfid": "A",
            "question": f"Q {qid}?",
            "base_template": "Distance {{entity_A}} to {{entity_B}}?",
            "kg_source": "osmh3kg",
            "entity_a_id": "1",
            "entity_a_name": "X",
            "entity_b_id": "2",
            "entity_b_name": "Y",
        }
    )


def _mock_smol_with_per_call_agents() -> Any:
    """Mock smolagents where every ``CodeAgent(...)`` call returns a
    **fresh** MagicMock agent.

    This is the critical invariant: a new ``CodeAgent`` instance per
    ``predict()`` call gives each call its own isolated memory. If the
    source code reverts to a shared ``self.agent``, this mock would
    still work (it returns new MagicMocks either way) — so the
    isolation assertion in the tests below looks at which **mock
    instance** each agent.run call went to, confirming per-call
    allocation.
    """
    mock_smol = MagicMock()

    created_agents: list[Any] = []

    def make_agent(**_kwargs: Any) -> Any:
        agent = MagicMock()
        # Each agent mock gets a unique marker so tests can assert
        # traces came from the right one.
        agent._marker = len(created_agents)
        # Mock memory.steps: one fake action_step with a code_action
        # that embeds the agent's marker — lets tests cross-check
        # that _extract_trace reads from the correct agent.
        step = MagicMock()
        step.code_action = f"final_answer({{'answer': {agent._marker}}})"
        step.is_final_answer = True
        step.observations = f"marker_{agent._marker}"
        step.error = None
        step.token_usage = MagicMock(
            input_tokens=10 + agent._marker, output_tokens=5
        )
        agent.memory.steps = [step]
        agent.tools = {}
        agent.run.return_value = {
            "answer": float(agent._marker),
            "unit": "km",
        }
        created_agents.append(agent)
        return agent

    mock_smol.CodeAgent.side_effect = make_agent
    mock_smol._created_agents = created_agents
    return mock_smol


class TestPerCallAgentConstruction:
    """The refactor's central invariant: ``predict()`` constructs a
    fresh ``CodeAgent`` every time, not once at ``__init__``."""

    def test_smolagents_model_init_does_not_create_agent(
        self,
        registry: Any,
    ) -> None:
        """Constructing the model must NOT build a CodeAgent. The
        agent is built lazily inside ``predict()`` so concurrent
        workers each get their own instance."""
        mock_smol = _mock_smol_with_per_call_agents()
        with patch.dict("sys.modules", {"smolagents": mock_smol}):
            from benchmark_models.tier3 import SmolagentsModel

            SmolagentsModel(
                "test-t3",
                provider="google",
                provider_model_id="fake",
                sub_tier="3a",
                api_key="fake",
                registry=registry,
                kg_backend=None,
            )
            mock_smol.CodeAgent.assert_not_called()

    def test_each_predict_call_constructs_a_new_agent(
        self,
        registry: Any,
    ) -> None:
        """N predict() calls → N CodeAgent constructions, each
        receiving its own prompt and producing its own trace."""
        mock_smol = _mock_smol_with_per_call_agents()
        with patch.dict("sys.modules", {"smolagents": mock_smol}):
            from benchmark_models.tier3 import SmolagentsModel

            model = SmolagentsModel(
                "test-t3",
                provider="google",
                provider_model_id="fake",
                sub_tier="3a",
                api_key="fake",
                registry=registry,
                kg_backend=None,
            )
            examples = [_make_example(f"q{i}") for i in range(5)]
            predictions = [model.predict(ex) for ex in examples]

            # One CodeAgent constructed per predict() call.
            assert mock_smol.CodeAgent.call_count == 5
            # Each agent was called once (its own run).
            for agent in mock_smol._created_agents:
                assert agent.run.call_count == 1
            # Answers preserved per call (no cross-talk in the return
            # path): each prediction has a unique answer matching
            # that call's mock agent marker.
            answers = sorted(p.answer for p in predictions)
            assert answers == [0.0, 1.0, 2.0, 3.0, 4.0]


class TestConcurrentPredictBatch:
    """Stress test: ``predict_batch`` with 8 workers on 20 examples.

    With the per-call construction, every worker has its own agent.
    Traces, usages, and answers must be isolated per question. The
    test runs quickly (all mocked — no real agent loop) but exercises
    the exact code path the production runner uses under concurrency.
    """

    def test_predict_batch_isolates_traces(
        self,
        registry: Any,
    ) -> None:
        mock_smol = _mock_smol_with_per_call_agents()
        with patch.dict("sys.modules", {"smolagents": mock_smol}):
            from benchmark_models.tier3 import SmolagentsModel

            model = SmolagentsModel(
                "test-t3",
                provider="google",
                provider_model_id="fake",
                sub_tier="3a",
                api_key="fake",
                registry=registry,
                kg_backend=None,
                max_concurrent=8,
            )
            examples = [_make_example(f"q{i}") for i in range(20)]
            predictions = model.predict_batch(examples)

            # One prediction per input, in input order.
            assert len(predictions) == 20
            assert [p.question_id for p in predictions] == [
                f"q{i}" for i in range(20)
            ]

            # One agent per predict call — full isolation. If the old
            # shared-agent code path regressed, CodeAgent would be
            # called once at __init__ and never again.
            assert mock_smol.CodeAgent.call_count == 20

            # Every prediction's trace must be a single-step trace
            # coming from that prediction's own agent. The mock makes
            # each agent's step's observations carry a unique marker;
            # the extracted trace's result field should match.
            for i, pred in enumerate(predictions):
                assert pred.tool_calls is not None
                assert len(pred.tool_calls) == 1
                # The trace points back to a specific marker; that
                # marker must NOT appear in another prediction.
                observed_marker = pred.tool_calls[0].result
                other_markers = [
                    other.tool_calls[0].result
                    for j, other in enumerate(predictions)
                    if j != i and other.tool_calls
                ]
                assert observed_marker not in other_markers, (
                    f"qid {pred.question_id} shares trace marker "
                    f"{observed_marker!r} with another prediction — "
                    f"concurrency isolation broken."
                )

    def test_predict_batch_usage_is_per_question(
        self,
        registry: Any,
    ) -> None:
        """Per-call usage numbers must differ — the mock assigns
        ``input_tokens = 10 + marker``, so every prediction should
        have a distinct input_tokens value."""
        mock_smol = _mock_smol_with_per_call_agents()
        with patch.dict("sys.modules", {"smolagents": mock_smol}):
            from benchmark_models.tier3 import SmolagentsModel

            model = SmolagentsModel(
                "test-t3",
                provider="google",
                provider_model_id="fake",
                sub_tier="3a",
                api_key="fake",
                registry=registry,
                kg_backend=None,
                max_concurrent=8,
            )
            examples = [_make_example(f"q{i}") for i in range(10)]
            predictions = model.predict_batch(examples)

            input_tokens = [p.usage.input_tokens for p in predictions]
            # Every value should be unique (different markers → different
            # input_tokens). If shared state leaked, some would collide.
            assert len(set(input_tokens)) == len(input_tokens)

    def test_predict_batch_preserves_input_order(
        self,
        registry: Any,
    ) -> None:
        """ThreadPoolExecutor.map preserves order. Even under
        concurrent execution the output list matches the input list
        positionally."""
        mock_smol = _mock_smol_with_per_call_agents()
        with patch.dict("sys.modules", {"smolagents": mock_smol}):
            from benchmark_models.tier3 import SmolagentsModel

            model = SmolagentsModel(
                "test-t3",
                provider="google",
                provider_model_id="fake",
                sub_tier="3a",
                api_key="fake",
                registry=registry,
                kg_backend=None,
                max_concurrent=8,
            )
            examples = [_make_example(f"q{i}") for i in range(30)]
            predictions = model.predict_batch(examples)

            assert [p.question_id for p in predictions] == [
                ex.question_id for ex in examples
            ]
