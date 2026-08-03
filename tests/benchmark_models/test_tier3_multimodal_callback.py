"""Tests for :func:`benchmark_models.tier3._clear_replayed_step_image`.

The callback prevents smolagents from replaying the static task image
on every ActionStep — see the helper's docstring for the full
mechanism. These tests cover the four behavioural cases (ActionStep
with images, ActionStep without images, non-ActionStep type, multi-step
sequence) plus an integration-style end-to-end check via
``write_memory_to_messages`` to confirm the LLM input only carries the
TaskStep image regardless of how many steps have run.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from benchmark_models.tier3 import _clear_replayed_step_image


def _make_pil_image():
    """Build a tiny PIL image — keeps the dependency optional in CI
    by importing PIL only when an image fixture is needed."""
    from PIL import Image

    return Image.new("RGB", (4, 4), color="red")


class TestClearReplayedStepImage:
    """Unit-test the callback in isolation against synthetic memory steps."""

    def test_clears_action_step_observations_images(self) -> None:
        from smolagents.memory import ActionStep, Timing

        step = ActionStep(
            step_number=1,
            timing=Timing(start_time=0.0, end_time=1.0),
            observations_images=[_make_pil_image()],
        )
        assert step.observations_images is not None

        _clear_replayed_step_image(step, agent=MagicMock())

        assert step.observations_images is None

    def test_noop_when_observations_images_is_none(self) -> None:
        from smolagents.memory import ActionStep, Timing

        step = ActionStep(
            step_number=1,
            timing=Timing(start_time=0.0, end_time=1.0),
            observations_images=None,
        )
        _clear_replayed_step_image(step, agent=MagicMock())
        assert step.observations_images is None

    def test_noop_when_observations_images_is_empty(self) -> None:
        from smolagents.memory import ActionStep, Timing

        step = ActionStep(
            step_number=1,
            timing=Timing(start_time=0.0, end_time=1.0),
            observations_images=[],
        )
        _clear_replayed_step_image(step, agent=MagicMock())
        # Empty list is falsy → guard short-circuits, untouched.
        assert step.observations_images == []

    def test_noop_on_task_step(self) -> None:
        """The callback must never null TaskStep.task_images — that's the
        one place we WANT the image to live."""
        from smolagents.memory import TaskStep

        img = _make_pil_image()
        step = TaskStep(task="What is X?", task_images=[img])

        _clear_replayed_step_image(step, agent=MagicMock())

        assert step.task_images is not None
        assert step.task_images[0] is img

    def test_noop_on_planning_step(self) -> None:
        from smolagents.memory import PlanningStep, Timing
        from smolagents.models import ChatMessage, MessageRole

        step = PlanningStep(
            model_input_messages=[
                ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "x"}])
            ],
            model_output_message=ChatMessage(
                role=MessageRole.ASSISTANT, content=[{"type": "text", "text": "y"}]
            ),
            plan="plan text",
            timing=Timing(start_time=0.0, end_time=1.0),
        )
        # Should not raise / not touch anything.
        _clear_replayed_step_image(step, agent=MagicMock())


class TestCallbackPreventsImageReplayEndToEnd:
    """Integration: walk a multi-step memory through smolagents'
    ``write_memory_to_messages`` machinery and confirm the LLM input
    only carries one image, regardless of how many ActionSteps ran."""

    def _count_image_blocks(self, messages: list[Any]) -> int:
        """Count message-content blocks of type ``image`` or
        ``image_url`` across a message list — covers both smolagents'
        internal representation and the OpenAI-compat shape."""
        n = 0
        for msg in messages:
            content = msg.content if hasattr(msg, "content") else msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") in {"image", "image_url"}:
                    n += 1
        return n

    def test_messages_carry_one_image_after_three_steps(self) -> None:
        """Simulate what happens during agent.run() — TaskStep with an
        image, three ActionSteps each auto-populated with the same
        ``observations_images``, callback fires after each step. After
        the third step the model input messages should still only carry
        the single TaskStep image."""
        from smolagents.memory import ActionStep, AgentMemory, TaskStep, Timing

        memory = AgentMemory(system_prompt="you are an agent")
        task_image = _make_pil_image()
        action_image = _make_pil_image()

        # Step 1
        memory.steps.append(TaskStep(task="What is X?", task_images=[task_image]))
        step1 = ActionStep(
            step_number=1,
            timing=Timing(start_time=0.0, end_time=1.0),
            observations_images=[action_image],
            model_output="some code",
            observations="some output",
        )
        _clear_replayed_step_image(step1, agent=MagicMock())
        memory.steps.append(step1)

        # Step 2
        step2 = ActionStep(
            step_number=2,
            timing=Timing(start_time=1.0, end_time=2.0),
            observations_images=[action_image],
            model_output="more code",
            observations="more output",
        )
        _clear_replayed_step_image(step2, agent=MagicMock())
        memory.steps.append(step2)

        # Step 3
        step3 = ActionStep(
            step_number=3,
            timing=Timing(start_time=2.0, end_time=3.0),
            observations_images=[action_image],
            model_output="final code",
            observations="final output",
        )
        _clear_replayed_step_image(step3, agent=MagicMock())
        memory.steps.append(step3)

        # Compose what write_memory_to_messages would emit.
        messages = memory.system_prompt.to_messages()
        for step in memory.steps:
            messages.extend(step.to_messages())

        # Only the TaskStep image should remain.
        n_images = self._count_image_blocks(messages)
        assert n_images == 1, (
            f"Expected 1 image (TaskStep only) but got {n_images}. "
            "If the count grew with steps, the callback isn't clearing "
            "observations_images correctly — this is the same bug that "
            "tripped vLLM's --limit-mm-per-prompt = 1 at Step 2."
        )

    def test_without_callback_image_count_grows_with_steps(self) -> None:
        """Regression baseline: if we DON'T fire the callback, the image
        count grows linearly with the number of steps. This proves the
        callback is doing real work (and documents what the bug looked
        like before the fix)."""
        from smolagents.memory import ActionStep, AgentMemory, TaskStep, Timing

        memory = AgentMemory(system_prompt="you are an agent")
        task_image = _make_pil_image()
        action_image = _make_pil_image()

        memory.steps.append(TaskStep(task="What is X?", task_images=[task_image]))
        for n in range(1, 4):
            memory.steps.append(
                ActionStep(
                    step_number=n,
                    timing=Timing(start_time=float(n - 1), end_time=float(n)),
                    observations_images=[action_image],
                    model_output="code",
                    observations="out",
                )
            )

        messages = memory.system_prompt.to_messages()
        for step in memory.steps:
            messages.extend(step.to_messages())

        # 1 TaskStep image + 3 ActionStep images = 4. This is what
        # vLLM rejected at Step 2 (where the count was 2).
        n_images = self._count_image_blocks(messages)
        assert n_images == 4
