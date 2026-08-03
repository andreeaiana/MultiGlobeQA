"""Tests for KG path resolution (N from the round-2 plan).

The CLI resolves ``kg_path = kg_root / kg_source`` and passes it to
``EntityService(TripleBridge(...))``. The :func:`resolve_kg_path`
helper centralizes the precedence:

  --kg-root flag > BENCHMARK_KGS_ROOT env > default 'benchmark/kgs/'

These tests pin the path resolution without booting a real Spark
session.
"""

from __future__ import annotations

import pytest

from benchmark_models.kg_tools.executor_bridge import (
    BENCHMARK_KGS_ROOT_ENV,
    DEFAULT_KGS_ROOT,
    resolve_kg_path,
)


class TestResolveKgPath:
    def test_returns_none_when_no_kg_source(self) -> None:
        assert resolve_kg_path(kg_source=None) is None

    def test_uses_default_root_when_kg_root_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(BENCHMARK_KGS_ROOT_ENV, raising=False)
        path = resolve_kg_path(kg_source="kwg")
        assert path == f"{DEFAULT_KGS_ROOT}/kwg"

    def test_kg_root_arg_takes_precedence_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(BENCHMARK_KGS_ROOT_ENV, "/env/kgs")
        path = resolve_kg_path(kg_source="kwg", kg_root="/cli/kgs")
        assert path == "/cli/kgs/kwg"

    def test_env_var_used_when_arg_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(BENCHMARK_KGS_ROOT_ENV, "/env/kgs")
        path = resolve_kg_path(kg_source="osmh3kg")
        assert path == "/env/kgs/osmh3kg"

    def test_handles_all_three_production_kgs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The three production KGs (kwg/osmh3kg/worldkg) all resolve
        cleanly under the default root."""
        monkeypatch.delenv(BENCHMARK_KGS_ROOT_ENV, raising=False)
        assert resolve_kg_path(kg_source="kwg") == f"{DEFAULT_KGS_ROOT}/kwg"
        assert (
            resolve_kg_path(kg_source="osmh3kg") == f"{DEFAULT_KGS_ROOT}/osmh3kg"
        )
        assert (
            resolve_kg_path(kg_source="worldkg") == f"{DEFAULT_KGS_ROOT}/worldkg"
        )


class TestSlurmKgSourceFlag:
    """N3: the SLURM generator passes --kg-source per (lang, kg) cell
    so the bridge loads the right Delta table."""

    def test_slurm_emits_kg_source_for_tier3a_use_kg(self) -> None:
        from scripts.benchmark_models.generate_slurm import (
            _Cell,
            _extra_flags,
        )
        from benchmark_models.config import OpenAIVariantConfig

        config = OpenAIVariantConfig(
            backend="openai",
            model_id="gpt-5.4",
            tier="tool_agent",
            sub_tier="3a",
        )
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        flags = _extra_flags(
            config,
            cell=cell,
            oracle_mode=None,
            use_kg=True,
            no_mlflow=False,
        )
        assert "--kg-source kwg" in flags
        assert "--use-kg" in flags

    def test_slurm_omits_kg_source_when_use_kg_false(self) -> None:
        from scripts.benchmark_models.generate_slurm import (
            _Cell,
            _extra_flags,
        )
        from benchmark_models.config import OpenAIVariantConfig

        config = OpenAIVariantConfig(
            backend="openai",
            model_id="gpt-5.4",
            tier="tool_agent",
            sub_tier="3a",
        )
        cell = _Cell(lang="eng", kg="osmh3kg", sfid="")
        flags = _extra_flags(
            config,
            cell=cell,
            oracle_mode=None,
            use_kg=False,
            no_mlflow=False,
        )
        assert "--kg-source" not in flags

    def test_slurm_omits_kg_source_for_tier3b(self) -> None:
        """Tier 3b has no KG access (web+python only)."""
        from scripts.benchmark_models.generate_slurm import (
            _Cell,
            _extra_flags,
        )
        from benchmark_models.config import OpenAIVariantConfig

        config = OpenAIVariantConfig(
            backend="openai",
            model_id="gpt-5.4",
            tier="tool_agent",
            sub_tier="3b",
        )
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        flags = _extra_flags(
            config,
            cell=cell,
            oracle_mode=None,
            use_kg=True,  # ignored on 3b
            no_mlflow=False,
        )
        assert "--kg-source" not in flags
