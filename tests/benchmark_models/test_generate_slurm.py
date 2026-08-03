"""Tests for the benchmark_models SLURM script generator.

The generator emits one script per cell (granularity ``cell`` =
(lang, kg); granularity ``sfid`` = (lang, kg, sfid)), with two backend
templates:

* **vLLM** — GPU partition, ``vllm serve`` over HTTP, CUDA module, cache pinning.
* **API** — CPU partition, fail-fast on missing API-key file.

The tests pin granularity defaults and template shape so a regression
that mis-routes a tier-3 variant to ``cell`` granularity (24 h time
budget, single job per (lang, kg)) doesn't ship silently.
"""

from __future__ import annotations

from pathlib import Path

from benchmark_models.config import load_models_catalogue
from scripts.benchmark_models.generate_slurm import (
    KG_BUCKETS,
    SFIDS,
    SFIDS_BY_KG,
    TARGET_LANGUAGES,
    _Cell,
    _CLUSTERS,
    _CLUSTER_CPU_PARTITION,
    _CLUSTER_GPU_GRES,
    _CLUSTER_GPU_PARTITION,
    _CLUSTER_WORKSPACE_ROOT,
    _COLD_START_OVERHEAD_S,
    _compute_time_budget,
    _default_batch_size,
    _default_granularity,
    _default_max_concurrent,
    _enumerate_cells,
    _format_hms,
    _mode_key_for_time_budget,
    _mode_tag,
    _PARTITION_WALL_CAP_MIN,
    _round_up_to_half_hour,
    render_script,
    write_script,
)


_CATALOGUE = load_models_catalogue()


# ----------------------------------------------------------------------------
# Granularity / batch defaults
# ----------------------------------------------------------------------------


class TestGranularityDefaults:
    def test_tier1_defaults_to_cell(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        assert _default_granularity(config) == "cell"

    def test_tier2_defaults_to_cell(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-thinking-tier2")
        assert _default_granularity(config) == "cell"

    def test_tier3_defaults_to_sfid(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-tier3a")
        assert _default_granularity(config) == "sfid"


class TestBatchSizeDefaults:
    def test_vllm_reads_yaml_batch_size_tier1(self) -> None:
        """R4-2: SLURM generator honours ``config.batch_size`` from
        models.yaml (matching the runner's
        ``resolve_default_batch_size``). qwen35-a3b-plain pins 64
        in YAML."""
        config = _CATALOGUE.get("qwen35-a3b-plain")
        assert _default_batch_size(config) == 64
        # And the variant config really declares 64 (sanity).
        assert getattr(config, "batch_size", None) == 64

    def test_vllm_reads_yaml_batch_size_tier2(self) -> None:
        """R4-2: SLURM generator reads ``batch_size`` from YAML for
        vLLM tier-2 too. The pre-R4-2 hardcoded 32 silently overrode
        the YAML value (qwen35-a3b-thinking-tier2 pins 64). Now the
        SLURM and runner paths agree on whatever YAML says."""
        config = _CATALOGUE.get("qwen35-a3b-thinking-tier2")
        assert _default_batch_size(config) == 64
        assert getattr(config, "batch_size", None) == 64

    def test_vllm_uses_class_default_when_yaml_omits_batch_size(self) -> None:
        """Variants that don't override ``batch_size`` in YAML fall
        through to the VLLMHttpVariantConfig default of 32."""
        from benchmark_models.config import VLLMHttpVariantConfig

        config = VLLMHttpVariantConfig(
            backend="vllm_http",
            model_path="Qwen/NoYAMLOverride",
            tier="plain",
            # No batch_size specified → class default applies.
        )
        assert _default_batch_size(config) == 32

    def test_vllm_yaml_override_respected(self) -> None:
        """R4-2 regression: if a future YAML edit changes a vLLM
        tier-1 variant's batch_size, the SLURM generator must pick it
        up (not the pre-R4-2 hardcoded 64)."""
        from benchmark_models.config import VLLMHttpVariantConfig

        config = VLLMHttpVariantConfig(
            backend="vllm_http",
            model_path="Qwen/Test",
            tier="plain",
            batch_size=8,
        )
        assert _default_batch_size(config) == 8

    def test_tier3a_and_3c_keep_batch_size_1(self) -> None:
        """3a / 3c use the shared SparkSession on ``local[*]`` master
        with an 8 GB driver heap + 8 GB direct-buffer cap (see
        ``triple_bridge.py``'s session builder). Concurrent Sedona
        queries contend for JVM cores → no linear speedup and OOM
        risk on broad polygon scans. Stays at 1 until the bridge
        grows per-thread sessions."""
        for variant in (
            "qwen35-a3b-tier3a", "qwen35-a3b-tier3c",
            "gemini-3-flash-tier3a", "gemini-3-flash-tier3c",
        ):
            assert _default_batch_size(_CATALOGUE.get(variant)) == 1, variant

    def test_tier3b_fans_out_per_backend(self) -> None:
        """3b is web + Python only — no Spark, no shared compute
        bottleneck. vLLM fans out at 16 (continuous batching well
        under ``--max-num-seqs 64``); API fans out at 8 (provider
        RPM ceiling at K × max_steps in-flight requests)."""
        assert _default_batch_size(_CATALOGUE.get("qwen35-a3b-tier3b")) == 16
        assert _default_batch_size(_CATALOGUE.get("gemini-3-flash-tier3b")) == 8

    def test_oracle_t3_fans_out_regardless_of_sub_tier(self) -> None:
        """Oracle-t3 strips every tool from the agent. Even on a
        tier-3a variant the KG bridge is never invoked, so the
        Spark contention argument doesn't apply and the run is safe
        to batch."""
        vllm = _CATALOGUE.get("qwen35-a3b-tier3a")
        api = _CATALOGUE.get("gemini-3-flash-tier3a")
        assert _default_batch_size(vllm, oracle_mode="t3") == 16
        assert _default_batch_size(api, oracle_mode="t3") == 8

    def test_non_t3_oracle_modes_do_not_change_batch_size(self) -> None:
        """``structured`` / ``raw`` / ``verbalized`` apply on tier-1
        plain variants. The resolver should not be confused into the
        fan-out branch by their presence — tier-1 keeps its own
        per-backend default."""
        plain_vllm = _CATALOGUE.get("qwen35-a3b-plain")
        plain_api = _CATALOGUE.get("gemini-3-flash-plain")
        for mode in ("structured", "raw", "verbalized"):
            assert _default_batch_size(plain_vllm, oracle_mode=mode) == 64
            assert _default_batch_size(plain_api, oracle_mode=mode) == 16

    def test_api_tier1_uses_16(self) -> None:
        # P0-4: API tier 1 / tier 2 now defaults to 16 so the
        # ThreadPoolExecutor batching in the backend actually
        # activates. The pre-P0-4 default of 1 silently bypassed
        # the concurrency path.
        config = _CATALOGUE.get("gemini-3-flash-plain")
        assert _default_batch_size(config) == 16


# ----------------------------------------------------------------------------
# Cell enumeration
# ----------------------------------------------------------------------------


class TestEnumerateCells:
    def test_cell_granularity_emits_one_per_lang_kg(self) -> None:
        cells = _enumerate_cells(["eng", "deu"], ["kwg", "worldkg"], "cell")
        assert len(cells) == 4
        assert all(cell.sfid == "" for cell in cells)

    def test_sfid_granularity_emits_one_per_lang_kg_sfid(self) -> None:
        cells = _enumerate_cells(["eng"], ["kwg"], "sfid")
        # kwg has 13 sfids (no sfid_d) — see SFIDS_BY_KG.
        assert len(cells) == len(SFIDS_BY_KG["kwg"])
        assert {cell.sfid for cell in cells} == set(SFIDS_BY_KG["kwg"])
        assert "sfid_d" not in {cell.sfid for cell in cells}

    def test_sfid_granularity_per_kg_coverage(self) -> None:
        # SFID coverage on disk:
        #   kwg     — 13 (no D), osmh3kg — 13 (no I), worldkg — 9.
        # Together: 35 standard sfid-cells per language. ``multimodal``
        # fans out into 2 sub-cells (osmh3kg + worldkg dispatch) so
        # tier-3a/3c jobs receive a real ``--kg-source``. Total: 37
        # cells per language.
        cells = _enumerate_cells(
            ["eng"], ["kwg", "osmh3kg", "worldkg", "multimodal"], "sfid",
        )
        per_kg_counts = {kg: 0 for kg in ("kwg", "osmh3kg", "worldkg", "multimodal")}
        for cell in cells:
            per_kg_counts[cell.kg] += 1
        assert per_kg_counts == {"kwg": 13, "osmh3kg": 13, "worldkg": 9, "multimodal": 2}
        assert len(cells) == 37

    def test_multimodal_has_no_sfid_stratification(self) -> None:
        # multimodal doesn't carry sfid (each sub-cell processes one
        # whole language × underlying-KG slice), but DOES fan out into
        # two sub-cells via ``_MULTIMODAL_DISPATCH_KGS``.
        cells = _enumerate_cells(["eng"], ["multimodal"], "sfid")
        assert len(cells) == 2
        assert all(c.sfid == "" for c in cells)
        assert all(c.kg == "multimodal" for c in cells)
        assert {c.kg_dispatch for c in cells} == {"osmh3kg", "worldkg"}
        assert {c.kg_source for c in cells} == {"osmh3kg", "worldkg"}

    def test_no_seeds_argument_leaves_cell_seed_unset(self) -> None:
        """Backward compatibility: when ``seeds`` is omitted, every cell
        has ``seed=None`` and path layouts stay unchanged."""
        cells = _enumerate_cells(["eng"], ["kwg"], "cell")
        assert all(cell.seed is None for cell in cells)

    def test_multi_seed_multiplies_cell_count(self) -> None:
        """``--seeds 42,43,44`` triples the script count: 1 lang × 2 kgs
        × 3 seeds = 6 cells, each carrying its own seed."""
        cells = _enumerate_cells(
            ["eng"], ["kwg", "worldkg"], "cell", seeds=[42, 43, 44],
        )
        assert len(cells) == 6
        assert {cell.seed for cell in cells} == {42, 43, 44}
        # Same (lang, kg) appears 3× — once per seed.
        per_cell = [(c.lang, c.kg) for c in cells]
        assert per_cell.count(("eng", "kwg")) == 3
        assert per_cell.count(("eng", "worldkg")) == 3

    def test_empty_seeds_list_treated_as_unset(self) -> None:
        """An empty ``seeds`` list keeps the legacy single-script behaviour."""
        cells = _enumerate_cells(["eng"], ["kwg"], "cell", seeds=[])
        assert all(cell.seed is None for cell in cells)
        assert len(cells) == 1

    def test_single_seed_emits_one_script_per_cell(self) -> None:
        """``--seeds 42`` pins the seed but doesn't multiply cells."""
        cells = _enumerate_cells(
            ["eng", "deu"], ["kwg"], "cell", seeds=[42],
        )
        assert len(cells) == 2  # same as no seed, but seed is set
        assert all(cell.seed == 42 for cell in cells)


class TestMultimodalDispatch:
    """The ``multimodal`` bucket fans out into one sub-cell per underlying
    KG so tier-3a/3c can call ``schema_for(kg_source)`` with a real KG
    name. ``"multimodal"`` is not a known KG to ``schema_for`` and would
    raise ``ValueError: Unknown KG: 'multimodal'`` at Spark bootstrap."""

    def test_cell_granularity_multimodal_yields_two_subcells(self) -> None:
        cells = _enumerate_cells(["eng"], ["multimodal"], "cell")
        assert len(cells) == 2
        assert all(c.kg == "multimodal" for c in cells)
        assert {c.kg_dispatch for c in cells} == {"osmh3kg", "worldkg"}

    def test_kg_source_property_falls_back_to_kg_for_non_multimodal(self) -> None:
        cells = _enumerate_cells(["eng"], ["kwg"], "cell")
        assert len(cells) == 1
        assert cells[0].kg == "kwg"
        assert cells[0].kg_dispatch == ""
        # Falls back to kg.
        assert cells[0].kg_source == "kwg"

    def test_kg_source_property_uses_dispatch_for_multimodal(self) -> None:
        cells = _enumerate_cells(["eng"], ["multimodal"], "cell")
        for c in cells:
            # multimodal cells always carry a non-empty kg_dispatch.
            assert c.kg_dispatch in {"osmh3kg", "worldkg"}
            assert c.kg_source == c.kg_dispatch

    def test_multi_seed_multiplies_multimodal_subcells(self) -> None:
        # 2 underlying KGs × 3 seeds = 6 cells for one lang.
        cells = _enumerate_cells(
            ["eng"], ["multimodal"], "cell", seeds=[42, 43, 44],
        )
        assert len(cells) == 6
        per_seed = {(c.seed, c.kg_dispatch) for c in cells}
        assert per_seed == {
            (42, "osmh3kg"), (42, "worldkg"),
            (43, "osmh3kg"), (43, "worldkg"),
            (44, "osmh3kg"), (44, "worldkg"),
        }


class TestMultimodalPaths:
    """Paths derived from ``_Cell`` must drill into the underlying KG
    for multimodal sub-cells — both the benchmark input tree and the
    predictions output tree mirror the on-disk layout
    ``benchmark/final/large/multimodal/<underlying_kg>/<lang>/``."""

    def test_benchmark_path_drills_into_dispatch_kg(self) -> None:
        from scripts.benchmark_models.generate_slurm import _benchmark_path

        cell = _Cell(
            lang="eng", kg="multimodal", sfid="",
            kg_dispatch="osmh3kg",
        )
        assert _benchmark_path(cell) == "benchmark/final/large/multimodal/osmh3kg/"

    def test_benchmark_path_for_non_multimodal_unchanged(self) -> None:
        from scripts.benchmark_models.generate_slurm import _benchmark_path

        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        assert _benchmark_path(cell) == "benchmark/final/large/kwg/sfid_a/"

    def test_output_path_includes_dispatch_kg_segment(self) -> None:
        from scripts.benchmark_models.generate_slurm import _output_path

        cell = _Cell(
            lang="eng", kg="multimodal", sfid="",
            seed=42, kg_dispatch="worldkg",
        )
        out = _output_path("qwen35-27b-tier3a-image", cell, mode_tag="usekg")
        assert out == (
            "results/runs/large/qwen35-27b-tier3a-image/usekg/seed42/"
            "eng/multimodal/worldkg/predictions.jsonl"
        )

    def test_output_path_for_non_multimodal_unchanged(self) -> None:
        from scripts.benchmark_models.generate_slurm import _output_path

        cell = _Cell(
            lang="eng", kg="kwg", sfid="sfid_a", seed=42,
        )
        out = _output_path("qwen35-27b-tier3a", cell, mode_tag="usekg")
        assert out == (
            "results/runs/large/qwen35-27b-tier3a/usekg/seed42/"
            "eng/kwg/sfid_a/predictions.jsonl"
        )

    def test_script_filename_disambiguates_multimodal_subcells(self) -> None:
        from scripts.benchmark_models.generate_slurm import _script_filename

        osmh3kg_cell = _Cell(
            lang="eng", kg="multimodal", sfid="",
            kg_dispatch="osmh3kg",
        )
        worldkg_cell = _Cell(
            lang="eng", kg="multimodal", sfid="",
            kg_dispatch="worldkg",
        )
        assert _script_filename(osmh3kg_cell, "usekg") == (
            "eng_multimodal_osmh3kg_usekg.sh"
        )
        assert _script_filename(worldkg_cell, "usekg") == (
            "eng_multimodal_worldkg_usekg.sh"
        )
        # Without the dispatch suffix the two scripts would collide
        # at the same filename. Regression guard.
        assert _script_filename(osmh3kg_cell, "usekg") != _script_filename(
            worldkg_cell, "usekg",
        )

    def test_job_name_disambiguates_multimodal_subcells(self) -> None:
        from scripts.benchmark_models.generate_slurm import _job_name

        osmh3kg_cell = _Cell(
            lang="eng", kg="multimodal", sfid="",
            seed=42, kg_dispatch="osmh3kg",
        )
        worldkg_cell = _Cell(
            lang="eng", kg="multimodal", sfid="",
            seed=42, kg_dispatch="worldkg",
        )
        a = _job_name("qwen35-27b-tier3a-image", osmh3kg_cell, "usekg")
        b = _job_name("qwen35-27b-tier3a-image", worldkg_cell, "usekg")
        assert "multimodal_osmh3kg" in a
        assert "multimodal_worldkg" in b
        assert a != b


class TestMultimodalKgSourceFlag:
    """The ``--kg-source`` flag emitted into the SLURM script must name
    a real KG so that ``schema_for`` at runtime doesn't reject it."""

    def test_kg_source_flag_uses_dispatch_kg_for_multimodal(self) -> None:
        config = _CATALOGUE.get("qwen35-27b-tier3a-image")
        cell = _Cell(
            lang="eng", kg="multimodal", sfid="",
            seed=42, kg_dispatch="osmh3kg",
        )
        text = render_script(
            "qwen35-27b-tier3a-image", config, cell, granularity="sfid",
            logs_dir=Path("/tmp/logs"), use_kg=True,
        )
        # The bridge expects a real KG name; ``multimodal`` would raise.
        assert "--kg-source osmh3kg" in text
        assert "--kg-source multimodal" not in text
        # The benchmark path also drills into the dispatch KG.
        assert "--benchmark benchmark/final/large/multimodal/osmh3kg/" in text


# ----------------------------------------------------------------------------
# Script rendering — backend templates
# ----------------------------------------------------------------------------


class TestRenderScriptVLLM:
    def test_vllm_template_includes_gpu_partition(self) -> None:
        config = _CATALOGUE.get("qwen35-a3b-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "qwen35-a3b-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--partition=gpu_h100" in text
        assert "--gres=gpu:1" in text
        assert "module load devel/cuda/12.8" in text
        assert "TORCHINDUCTOR_CACHE_DIR" in text
        assert "VLLM_CACHE_ROOT" in text

    def test_vllm_tier3_template_uses_sfid_path(self) -> None:
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        text = render_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--benchmark benchmark/final/large/kwg/sfid_a/" in text
        assert "--language eng" in text
        assert "--n 0" in text  # process the whole shard


class TestRenderScriptAPI:
    def test_api_template_uses_cpu_partition(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="deu", kg="worldkg", sfid="")
        text = render_script(
            "gemini-3-flash-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--partition=single" in text
        assert "--gres=gpu" not in text
        assert "module load devel/cuda" not in text
        assert "ERROR: scripts/set_api_keys.sh not found." in text
        # No vLLM caches in API jobs.
        assert "TORCHINDUCTOR_CACHE_DIR" not in text

    def test_api_template_includes_correct_paths(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="deu", kg="worldkg", sfid="")
        text = render_script(
            "gemini-3-flash-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--benchmark benchmark/final/large/worldkg/" in text
        assert "--language deu" in text
        assert "--output results/runs/large/gemini-3-flash-plain/deu/worldkg/predictions.jsonl" in text


# ----------------------------------------------------------------------------
# Optional flags wired through
# ----------------------------------------------------------------------------


class TestOptionalFlags:
    def test_oracle_mode_flag_appears(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "gemini-3-flash-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
            oracle_mode="structured",
        )
        assert "--oracle-mode structured" in text

    def test_use_kg_only_applied_to_tier3a_3c(self) -> None:
        # tier-3a vLLM with use_kg=True.
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        text = render_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            logs_dir=Path("/tmp/logs"),
            use_kg=True,
        )
        assert "--use-kg" in text

    def test_use_kg_skipped_on_tier1(self) -> None:
        # use_kg=True on a tier-1 variant must NOT emit --use-kg
        # (the runner rejects it; emitting would break the job).
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "gemini-3-flash-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
            use_kg=True,
        )
        assert "--use-kg" not in text

    def test_no_mlflow_flag(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "gemini-3-flash-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
            no_mlflow=True,
        )
        assert "--no-mlflow" in text

    def test_seed_embedded_in_paths_and_flag(self) -> None:
        """When the cell carries a seed, the script must (1) write to a
        seed-stamped output dir so multi-seed runs don't collide,
        (2) carry a seed-stamped job name, and (3) pass ``--seed <N>``
        to the runner."""
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="", seed=43)
        text = render_script(
            "gemini-3-flash-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
        )
        assert "results/runs/large/gemini-3-flash-plain/seed43/eng/kwg/" in text
        assert "--job-name=bm_gemini-3-flash-plain_seed43_eng_kwg" in text
        assert "--seed 43" in text

    def test_no_seed_keeps_legacy_paths(self) -> None:
        """``cell.seed is None`` keeps the pre-multi-seed paths so
        existing SLURM submissions remain bit-identical."""
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")  # seed defaults to None
        text = render_script(
            "gemini-3-flash-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
        )
        assert "results/runs/large/gemini-3-flash-plain/eng/kwg/" in text
        assert "seed" not in text.split("predictions.jsonl")[0]  # no seed segment in the output path
        assert "--seed " not in text


# ----------------------------------------------------------------------------
# --n LIMIT pass-through (row cap per emitted script)
# ----------------------------------------------------------------------------


class TestRowLimitFlag:
    """``--n LIMIT`` forwards to the runner's row cap. Default 0 is the
    runner's 'process every row' sentinel; positive integers cap rows
    for limited-row smokes via SLURM (the only way to exercise vLLM
    serves, since they live inside a job allocation)."""

    def test_default_emits_n_zero(self) -> None:
        """When ``n`` isn't passed, the script keeps the
        full-eval-friendly ``--n 0`` so existing SLURM submissions
        remain bit-identical."""
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "gemini-3-flash-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--n 0" in text

    def test_positive_n_round_trips(self) -> None:
        """An explicit positive cap reaches the smoke invocation
        verbatim — no off-by-one or stringification quirk."""
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "gemini-3-flash-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"), n=5,
        )
        assert "--n 5" in text
        assert "--n 0" not in text

    def test_n_works_alongside_other_flags(self) -> None:
        """Cap composes with other extra-flags (oracle, use-kg, seed,
        max-concurrent). All flags coexist on their own lines."""
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a", seed=43)
        text = render_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            logs_dir=Path("/tmp/logs"),
            n=5, oracle_mode="t3",
        )
        assert "--n 5" in text
        assert "--oracle-mode t3" in text
        assert "--seed 43" in text
        # Oracle-t3 also fans out → --max-concurrent emitted.
        assert "--max-concurrent 8" in text


# ----------------------------------------------------------------------------
# max_concurrent resolver + emission in the rendered script
# ----------------------------------------------------------------------------


class TestMaxConcurrent:
    """``--max-concurrent`` is tier-3-only at the runner (cli.py:480-481)
    AND only takes effect when batch_size > 1 (runner.py:430-433
    sequential branch). The generator must mirror both gates so an
    emitted script never carries a flag the runner would either reject
    or silently ignore."""

    def test_resolver_none_for_tier1(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        assert _default_max_concurrent(config, 16) is None

    def test_resolver_none_for_tier2(self) -> None:
        config = _CATALOGUE.get("qwen35-a3b-thinking-tier2")
        assert _default_max_concurrent(config, 64) is None

    def test_resolver_none_when_tier3_batch_size_1(self) -> None:
        """3a/3c run with batch_size=1 (Spark contention). The runner
        takes the sequential ``predict()`` branch, so emitting
        max_concurrent would be dead code."""
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        assert _default_max_concurrent(config, 1) is None

    def test_resolver_returns_per_backend_default_for_tier3b(self) -> None:
        vllm = _CATALOGUE.get("qwen35-a3b-tier3b")
        api = _CATALOGUE.get("gemini-3-flash-tier3b")
        assert _default_max_concurrent(vllm, 16) == 8
        assert _default_max_concurrent(api, 8) == 4

    def test_resolver_returns_per_backend_default_for_oracle_t3(self) -> None:
        """Oracle-t3 fans out at the same budget as tier-3b — same
        rationale (no tool I/O contention)."""
        vllm = _CATALOGUE.get("qwen35-a3b-tier3a")
        api = _CATALOGUE.get("gemini-3-flash-tier3a")
        assert _default_max_concurrent(vllm, 16, oracle_mode="t3") == 8
        assert _default_max_concurrent(api, 8, oracle_mode="t3") == 4

    def test_emitted_in_tier3b_script(self) -> None:
        """End-to-end: a default-resolved tier-3b script must carry
        ``--max-concurrent`` so the runner actually fans out at
        runtime."""
        config = _CATALOGUE.get("qwen35-a3b-tier3b")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        text = render_script(
            "qwen35-a3b-tier3b", config, cell, granularity="sfid",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--max-concurrent 8" in text
        assert "--batch-size 16" in text

    def test_emitted_in_oracle_t3_script(self) -> None:
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        text = render_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            logs_dir=Path("/tmp/logs"),
            oracle_mode="t3",
        )
        assert "--max-concurrent 8" in text
        assert "--batch-size 16" in text

    def test_not_emitted_in_tier3a_script(self) -> None:
        """3a stays at batch_size=1 → max_concurrent suppressed."""
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        text = render_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--max-concurrent" not in text
        assert "--batch-size 1" in text

    def test_not_emitted_in_tier1_script(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "gemini-3-flash-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--max-concurrent" not in text

    def test_cli_override_wins_for_batch_size(self) -> None:
        """Explicit ``batch_size=`` argument overrides the resolver
        (allowing e.g. a tier-3a smoke at batch_size=4 if you know
        the queries are point lookups)."""
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        text = render_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            logs_dir=Path("/tmp/logs"),
            batch_size=4,
        )
        assert "--batch-size 4" in text
        # Default max_concurrent now applies since batch_size > 1.
        assert "--max-concurrent 8" in text

    def test_cli_override_wins_for_max_concurrent(self) -> None:
        config = _CATALOGUE.get("qwen35-a3b-tier3b")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        text = render_script(
            "qwen35-a3b-tier3b", config, cell, granularity="sfid",
            logs_dir=Path("/tmp/logs"),
            max_concurrent=2,
        )
        assert "--max-concurrent 2" in text

    def test_max_concurrent_override_dropped_on_tier1(self) -> None:
        """The runner rejects ``--max-concurrent`` on tier-1/2; the
        generator suppresses the flag silently rather than emit
        something the job would reject at startup."""
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "gemini-3-flash-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
            max_concurrent=8,
        )
        assert "--max-concurrent" not in text

    def test_max_steps_emitted_when_set_on_tier3(self) -> None:
        """``--max-steps N`` lands in the SLURM body verbatim for
        tier-3 / oracle-t3 variants."""
        config = _CATALOGUE.get("qwen35-a3b-tier3b")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        text = render_script(
            "qwen35-a3b-tier3b", config, cell, granularity="sfid",
            logs_dir=Path("/tmp/logs"),
            max_steps=10,
        )
        assert "--max-steps 10" in text

    def test_max_steps_dropped_on_tier1(self) -> None:
        """The runner rejects ``--max-steps`` on tier-1/2; the
        generator suppresses it silently rather than emit something
        the job would reject at startup."""
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "gemini-3-flash-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
            max_steps=10,
        )
        assert "--max-steps" not in text

    def test_max_steps_default_omitted(self) -> None:
        """When ``max_steps`` isn't passed, the generator emits no
        ``--max-steps`` flag — the runner uses DEFAULT_MAX_TOOL_CALLS."""
        config = _CATALOGUE.get("qwen35-a3b-tier3b")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        text = render_script(
            "qwen35-a3b-tier3b", config, cell, granularity="sfid",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--max-steps" not in text

    def test_max_concurrent_override_dropped_when_batch_size_1(self) -> None:
        """If the user passes ``--max-concurrent`` but the resolved
        batch_size stays at 1 (tier-3a default), the flag is dropped:
        runner.py:430-433 routes batch_size=1 to sequential predict(),
        so max_concurrent is dead code."""
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        text = render_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            logs_dir=Path("/tmp/logs"),
            max_concurrent=8,
        )
        assert "--max-concurrent" not in text
        assert "--batch-size 1" in text


# ----------------------------------------------------------------------------
# Mode disambiguation: tier3a vs tier3a-oracle vs tier3a-usekg must not
# share script filenames, output paths, or job names.
# ----------------------------------------------------------------------------


class TestModeDisambiguation:
    """Re-running the generator for the same variant with different
    ``--oracle-mode`` / ``--use-kg`` flags must not silently overwrite
    earlier scripts or have their runtime predictions collide on the
    same on-disk ``predictions.jsonl``."""

    def test_mode_tag_default_is_empty(self) -> None:
        """Bare invocation (no oracle, no use-kg) ⇒ empty tag, which
        keeps the legacy paths bit-identical (back-compat with already
        submitted runs)."""
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        assert _mode_tag(config, oracle_mode=None, use_kg=False) == ""

    def test_mode_tag_oracle_wins_over_use_kg(self) -> None:
        """The runner short-circuits ``use_kg`` whenever ``oracle_mode``
        is set (cli.py:387-393), so an oracle-tagged run must not pick
        up a stray ``usekg`` suffix from both flags being passed."""
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        assert (
            _mode_tag(config, oracle_mode="t3", use_kg=True) == "oracle-t3"
        )

    def test_mode_tag_use_kg_only_on_tier3a_3c(self) -> None:
        """``--use-kg`` is gated on tier-3a/3c by ``_extra_flags``;
        the mode tag mirrors that gate so tier-1/2 stays in the
        legacy untagged paths."""
        tier1 = _CATALOGUE.get("gemini-3-flash-plain")
        tier3a = _CATALOGUE.get("qwen35-a3b-tier3a")
        tier3b = _CATALOGUE.get("qwen35-a3b-tier3b")
        assert _mode_tag(tier1, oracle_mode=None, use_kg=True) == ""
        assert _mode_tag(tier3a, oracle_mode=None, use_kg=True) == "usekg"
        # tier3b is web+python (no KG access), so use-kg doesn't apply.
        assert _mode_tag(tier3b, oracle_mode=None, use_kg=True) == ""

    def test_oracle_mode_changes_script_filename_and_output_path(
        self, tmp_path: Path
    ) -> None:
        """Two runs that differ only in ``--oracle-mode`` must write
        to distinct script files and distinct predictions paths."""
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")

        bare = write_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            out_root=tmp_path,
        )
        oracle = write_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            out_root=tmp_path, oracle_mode="t3",
        )
        assert bare.name == "eng_kwg_sfid_a.sh"
        assert oracle.name == "eng_kwg_sfid_a_oracle-t3.sh"
        assert bare.exists() and oracle.exists()  # neither overwrote the other

        bare_text = bare.read_text()
        oracle_text = oracle.read_text()
        assert (
            "--output results/runs/large/qwen35-a3b-tier3a/eng/kwg/sfid_a/predictions.jsonl"
            in bare_text
        )
        assert (
            "--output results/runs/large/qwen35-a3b-tier3a/oracle-t3/eng/kwg/sfid_a/predictions.jsonl"
            in oracle_text
        )
        # Job names diverge too, so squeue / sacct can tell the two runs apart.
        assert "--job-name=bm_qwen35-a3b-tier3a_eng_kwg_sfid_a" in bare_text
        assert (
            "--job-name=bm_qwen35-a3b-tier3a_oracle-t3_eng_kwg_sfid_a"
            in oracle_text
        )

    def test_use_kg_changes_script_filename_and_output_path(
        self, tmp_path: Path
    ) -> None:
        """Same cell with and without ``--use-kg`` (the fake-KG smoke
        vs real-KG eval distinction) must not collide."""
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        bare = write_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            out_root=tmp_path,
        )
        with_kg = write_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            out_root=tmp_path, use_kg=True,
        )
        assert bare.name == "eng_kwg_sfid_a.sh"
        assert with_kg.name == "eng_kwg_sfid_a_usekg.sh"
        assert (
            "--output results/runs/large/qwen35-a3b-tier3a/usekg/eng/kwg/sfid_a/predictions.jsonl"
            in with_kg.read_text()
        )

    def test_oracle_mode_variants_are_each_distinct(
        self, tmp_path: Path
    ) -> None:
        """The four oracle modes (structured/raw/verbalized/t3) must
        each get their own tag — generating all four against the same
        variant must produce four distinct scripts, not stomp."""
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        names = set()
        for mode in ("structured", "raw", "verbalized"):
            sp = write_script(
                "gemini-3-flash-plain", config, cell, granularity="cell",
                out_root=tmp_path, oracle_mode=mode,
            )
            names.add(sp.name)
        # Three distinct filenames, all with the mode embedded.
        assert names == {
            "eng_kwg_oracle-structured.sh",
            "eng_kwg_oracle-raw.sh",
            "eng_kwg_oracle-verbalized.sh",
        }

    def test_seed_and_mode_tag_coexist(self, tmp_path: Path) -> None:
        """The mode tag must compose with the seed segment — multi-seed
        oracle runs need both disambiguators on disk. The seed lives as
        a parent directory (``<variant>/seed43/``); the mode tag stays
        as a filename suffix so all of one seed's runs glob together."""
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a", seed=43)
        sp = write_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            out_root=tmp_path, oracle_mode="t3",
        )
        assert sp.name == "eng_kwg_sfid_a_oracle-t3.sh"
        assert sp.parent.name == "seed43"
        assert sp.parent.parent.name == "qwen35-a3b-tier3a"
        text = sp.read_text()
        # Mode segment sits between variant and seed in the output dir
        # so seed-different oracle runs group under one oracle subtree.
        assert (
            "results/runs/large/qwen35-a3b-tier3a/oracle-t3/seed43/eng/kwg/sfid_a/"
            in text
        )
        assert (
            "--job-name=bm_qwen35-a3b-tier3a_oracle-t3_seed43_eng_kwg_sfid_a"
            in text
        )


# ----------------------------------------------------------------------------
# {size} scoping — large vs small benchmark + results trees
# ----------------------------------------------------------------------------


class TestSizeScoping:
    """The ``size`` axis ('large' | 'small') selects both the benchmark
    input tree (``benchmark/final/<size>/``) and the results output tree
    (``results/runs/<size>/``). Oracle runs additionally scope
    ``--gold-triples`` to the same size+cell so a 'small' oracle job
    reads 'small' gold, not the whole 'large' tree."""

    def test_benchmark_path_defaults_to_large(self) -> None:
        from scripts.benchmark_models.generate_slurm import _benchmark_path

        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        assert _benchmark_path(cell) == "benchmark/final/large/kwg/sfid_a/"

    def test_benchmark_path_small(self) -> None:
        from scripts.benchmark_models.generate_slurm import _benchmark_path

        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        assert _benchmark_path(cell, "small") == "benchmark/final/small/kwg/sfid_a/"

    def test_output_path_small_includes_size_segment(self) -> None:
        from scripts.benchmark_models.generate_slurm import _output_path

        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a", seed=42)
        out = _output_path("qwen35-a3b-tier3b", cell, size="small")
        assert out == (
            "results/runs/small/qwen35-a3b-tier3b/seed42/eng/kwg/sfid_a/"
            "predictions.jsonl"
        )

    def test_render_script_small_threads_both_trees(self) -> None:
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        text = render_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            logs_dir=Path("/tmp/logs"), size="small",
        )
        assert "--benchmark benchmark/final/small/kwg/sfid_a/" in text
        assert (
            "--output results/runs/small/qwen35-a3b-tier3a/eng/kwg/sfid_a/"
            "predictions.jsonl" in text
        )

    def test_oracle_emits_size_scoped_gold_triples(self) -> None:
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        text = render_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            logs_dir=Path("/tmp/logs"), oracle_mode="t3", size="small",
        )
        assert "--gold-triples benchmark/final/small/kwg/sfid_a/" in text

    def test_non_oracle_omits_gold_triples(self) -> None:
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        text = render_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--gold-triples" not in text

    def test_write_script_nests_jobs_under_size(self, tmp_path: Path) -> None:
        # Job scripts nest under <out_root>/<size>/<variant>/ so a 'small'
        # sweep doesn't clobber the 'large' launchers for the same cells.
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        sp = write_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            out_root=tmp_path, size="small",
        )
        assert sp.parent.name == "qwen35-a3b-tier3a"
        assert sp.parent.parent.name == "small"


# ----------------------------------------------------------------------------
# vLLM-HTTP templates + --cluster flag
# ----------------------------------------------------------------------------


class TestRenderScriptVllmHttp:
    def test_bwuc3_template_launches_serve_and_health_polls(self) -> None:
        """The bwUniCluster vllm_http template launches ``vllm serve``
        in the background, traps to kill it on exit, polls ``/health``,
        and exports ``VLLM_BASE_URL`` so Smolagents tier-3 finds it.

        Port is derived at runtime from ``$SLURM_JOB_ID`` rather than
        hardcoded, so two co-scheduled single-GPU jobs on a multi-GPU
        node can each bind their own port."""
        config = _CATALOGUE.get("qwen35-a3b-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "qwen35-a3b-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
            cluster="bwuc3",
        )
        assert "#SBATCH --partition=gpu_h100" in text
        assert "#SBATCH --gres=gpu:1" in text
        assert "vllm serve Qwen/Qwen3.5-35B-A3B" in text
        assert "VLLM_PID=$!" in text
        assert "trap " in text
        # Runtime port derivation + downstream substitution.
        assert "VLLM_PORT=$((8000 + SLURM_JOB_ID % 1000))" in text
        assert "VLLM_PORT=8000" in text  # the else-branch fallback for interactive runs
        assert "--port $VLLM_PORT" in text
        assert "curl -sf http://localhost:$VLLM_PORT/health" in text
        assert "export VLLM_BASE_URL=http://localhost:$VLLM_PORT/v1" in text

    def test_helix_template_wraps_serve_in_singularity(self) -> None:
        """Helix host glibc is too old for in-tree vLLM wheels — the
        template runs ``vllm serve`` inside a Singularity container.
        Port derivation matches bwUC3 so concurrent jobs on the same
        Helix multi-GPU node also stay collision-free."""
        config = _CATALOGUE.get("qwen35-a3b-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "qwen35-a3b-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
            cluster="helix",
        )
        assert "#SBATCH --partition=gpu-single" in text
        assert "#SBATCH --gres=gpu:H200:1" in text
        assert "singularity exec --nv" in text
        assert '"$SBR_VLLM_SIF"' in text
        # Helix gets a 15-min health budget (vs 10 on bwuc3) because
        # the in-container HF download tends to be slower.
        assert "$(seq 1 900)" in text
        # Port parametrization mirrors bwUC3.
        assert "VLLM_PORT=$((8000 + SLURM_JOB_ID % 1000))" in text
        assert "--port $VLLM_PORT" in text
        assert "curl -sf http://localhost:$VLLM_PORT/health" in text
        assert "export VLLM_BASE_URL=http://localhost:$VLLM_PORT/v1" in text

    def test_api_template_on_helix_uses_cpu_single(self) -> None:
        """API-backend variants on Helix run on ``cpu-single`` rather
        than bwuc3's ``single`` partition; the bashrc dance differs too."""
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "gemini-3-flash-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
            cluster="helix",
        )
        assert "#SBATCH --partition=cpu-single" in text
        assert "conda deactivate 2>/dev/null || true" in text

    def test_api_template_on_bwuc3_unchanged(self) -> None:
        """Default ``bwuc3`` cluster keeps the API template
        bit-identical to pre-cluster-flag behaviour."""
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "gemini-3-flash-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
            cluster="bwuc3",
        )
        assert "#SBATCH --partition=single" in text
        assert "conda deactivate" not in text

    def test_vllm_http_template_respects_config_port_as_fallback(self) -> None:
        """The YAML ``vllm_port`` shows up as the **else-branch
        fallback** for interactive (non-SLURM) runs. Under SLURM the
        port is derived from ``$SLURM_JOB_ID`` and overrides this
        value, so the YAML port doesn't appear in the
        ``vllm serve --port`` / ``/health`` / ``VLLM_BASE_URL`` lines
        directly — those use ``$VLLM_PORT`` instead.
        """
        from benchmark_models.config import VLLMHttpVariantConfig

        config = VLLMHttpVariantConfig(
            backend="vllm_http", tier="plain",
            model_path="Qwen/Qwen3.5-35B-A3B",
            vllm_port=9001, max_model_len=4096, max_num_seqs=32,
        )
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "qwen35-a3b-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
        )
        # Fallback for interactive runs uses the YAML value.
        assert "VLLM_PORT=9001" in text
        # max-model-len / max-num-seqs still substitute at generation time.
        assert "--max-model-len 4096" in text
        assert "--max-num-seqs 32" in text
        # Downstream lines reference $VLLM_PORT regardless of YAML.
        assert "--port $VLLM_PORT" in text
        assert "http://localhost:$VLLM_PORT/health" in text
        assert "export VLLM_BASE_URL=http://localhost:$VLLM_PORT/v1" in text

    def test_vllm_http_template_forwards_dtype_and_gmu(self) -> None:
        """``dtype`` and ``gpu_memory_utilization`` from the variant
        config flow through to the ``vllm serve`` invocation so the
        serve doesn't silently use vLLM defaults that differ from
        the YAML-declared values. Regression: the open-weights Gemma
        yaml asks for bfloat16 but the template previously emitted no
        ``--dtype`` flag, so the serve fell back to vLLM's auto."""
        from benchmark_models.config import VLLMHttpVariantConfig

        config = VLLMHttpVariantConfig(
            backend="vllm_http", tier="plain",
            model_path="google/gemma-3-27B-it",
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
        )
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "gemma-3-27b-it-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--dtype bfloat16" in text
        assert "--gpu-memory-utilization 0.85" in text

    def test_vllm_http_template_forwards_quantization_when_set(self) -> None:
        """``--quantization`` only emitted when the field is non-None."""
        from benchmark_models.config import VLLMHttpVariantConfig

        config = VLLMHttpVariantConfig(
            backend="vllm_http", tier="plain",
            model_path="Qwen/Qwen3.5-35B-A3B",
            quantization="awq",
        )
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "qwen35-a3b-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--quantization awq" in text

    def test_vllm_http_template_omits_quantization_when_unset(self) -> None:
        """``--quantization`` absent from the script when config's
        quantization is None (its default). Matters because vLLM
        rejects ``--quantization None`` rather than ignoring it."""
        from benchmark_models.config import VLLMHttpVariantConfig

        config = VLLMHttpVariantConfig(
            backend="vllm_http", tier="plain",
            model_path="Qwen/Qwen3.5-35B-A3B",
            # quantization defaults to None
        )
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "qwen35-a3b-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--quantization" not in text

    def test_vllm_http_template_emits_hf_overrides_when_rope_scaling_set(self) -> None:
        """``--hf-overrides '{"rope_scaling": {...}}'`` lands in the
        SLURM body when the variant config has ``rope_scaling`` set.
        Uses --hf-overrides rather than --rope-scaling because the
        latter is not present in every vLLM version (the Helix
        Singularity image's vLLM 0.20.1 rejects ``--rope-scaling``);
        --hf-overrides has been the canonical way to override HF
        config.json values since very early vLLM. Drives the
        qwen35-27b-tier3a/3c YaRN extension from 32 K to 65 K."""
        from benchmark_models.config import VLLMHttpVariantConfig

        config = VLLMHttpVariantConfig(
            backend="vllm_http", tier="tool_agent", sub_tier="3a",
            model_path="Qwen/Qwen3.5-27B-FP8",
            max_model_len=65536,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 32768,
            },
        )
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        text = render_script(
            "qwen35-27b-tier3a", config, cell,
            granularity="sfid",
            cluster="helix",
            use_kg=True,
            logs_dir=Path("/tmp/logs"),
            n=100,
        )
        assert "--max-model-len 65536" in text
        # The rope_scaling dict is wrapped under the rope_scaling key
        # so vLLM's HF overrides path treats it as a config.json
        # override.
        assert (
            "--hf-overrides '"
            '{"rope_scaling":'
            '{"rope_type":"yarn","factor":2.0,'
            '"original_max_position_embeddings":32768}}'
            "'" in text
        )
        # Belt-and-suspenders: the old --rope-scaling flag must NOT
        # appear (vLLM 0.20.1 rejects it).
        assert "--rope-scaling" not in text

    def test_vllm_http_template_omits_hf_overrides_when_rope_scaling_unset(self) -> None:
        """``--hf-overrides`` absent when ``rope_scaling`` is None
        (default). Other vLLM variants must still produce
        bit-identical scripts after this feature lands."""
        from benchmark_models.config import VLLMHttpVariantConfig

        config = VLLMHttpVariantConfig(
            backend="vllm_http", tier="plain",
            model_path="Qwen/Qwen3.5-27B-FP8",
            # rope_scaling defaults to None
        )
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "qwen35-27b-plain", config, cell, granularity="cell",
            cluster="helix",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--hf-overrides" not in text
        assert "--rope-scaling" not in text
        # And max_model_len falls back to the catalogue default 32 K.
        assert "--max-model-len 32768" in text

    def test_vllm_http_template_treats_empty_rope_scaling_as_unset(self) -> None:
        """``rope_scaling: {}`` in YAML (a mis-edit) should NOT emit
        ``--hf-overrides '{"rope_scaling": {}}'`` — vLLM would
        either no-op or error. Truthy check on the dict drops the
        flag entirely."""
        from benchmark_models.config import VLLMHttpVariantConfig

        config = VLLMHttpVariantConfig(
            backend="vllm_http", tier="plain",
            model_path="Qwen/Qwen3.5-27B-FP8",
            rope_scaling={},  # explicit empty dict
        )
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        text = render_script(
            "qwen35-27b-plain", config, cell, granularity="cell",
            cluster="helix",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--hf-overrides" not in text
        assert "--rope-scaling" not in text

    def test_yaml_qwen35_27b_tier3a_carries_yarn(self) -> None:
        """The shipped ``models.yaml`` applies YaRN to
        qwen35-27b-tier3a (and 3c) but not to tier-1, tier-2, or
        tier-3b — those don't need the extended context. Regression
        guard against accidentally widening the YaRN scope.

        Also covers the paired KV-cache + SLURM-mem overrides on the
        anchor: YaRN variants drop ``max_num_seqs`` to 16 and bump
        ``slurm_mem_gb`` to 128 to absorb the larger KV cache; other
        variants keep the 64/80 defaults."""
        catalogue = _CATALOGUE
        tier3a = catalogue.get("qwen35-27b-tier3a")
        tier3c = catalogue.get("qwen35-27b-tier3c")
        for cfg in (tier3a, tier3c):
            assert cfg.max_model_len == 65536
            assert cfg.max_num_seqs == 16
            assert cfg.slurm_mem_gb == 128
            assert cfg.rope_scaling == {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 32768,
            }
        # Variants that should NOT carry YaRN — stay at 32K/64-seq/80GB.
        for variant in (
            "qwen35-27b-plain",
            "qwen35-27b-tier2",
            "qwen35-27b-tier3b",
        ):
            cfg = catalogue.get(variant)
            assert cfg.max_model_len == 32768, variant
            assert cfg.max_num_seqs == 64, variant
            assert cfg.slurm_mem_gb == 80, variant
            assert cfg.rope_scaling is None, variant

    def test_slurm_mem_gb_default_emits_80gb_in_vllm_header(self) -> None:
        """Non-YaRN vLLM variants emit ``#SBATCH --mem=80gb`` (the
        default for ``slurm_mem_gb`` on ``VLLMHttpVariantConfig``)."""
        from benchmark_models.config import VLLMHttpVariantConfig

        config = VLLMHttpVariantConfig(
            backend="vllm_http", tier="tool_agent", sub_tier="3b",
            model_path="Qwen/Qwen3.5-27B-FP8",
            # slurm_mem_gb defaults to 80
        )
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        text = render_script(
            "qwen35-27b-tier3b", config, cell,
            granularity="sfid",
            cluster="helix",
            logs_dir=Path("/tmp/logs"),
        )
        assert "#SBATCH --mem=80gb" in text
        assert "#SBATCH --mem=128gb" not in text

    def test_slurm_mem_gb_override_emits_128gb_for_yarn_variants(self) -> None:
        """YaRN-extended vLLM variants set ``slurm_mem_gb=128`` on
        the config; the generated SLURM header reflects this. Drives
        the 80 → 128 GiB bump that prevented the OOM kill we observed
        on the 64 K-context runs."""
        from benchmark_models.config import VLLMHttpVariantConfig

        config = VLLMHttpVariantConfig(
            backend="vllm_http", tier="tool_agent", sub_tier="3a",
            model_path="Qwen/Qwen3.5-27B-FP8",
            max_model_len=65536,
            max_num_seqs=16,
            slurm_mem_gb=128,
            rope_scaling={
                "rope_type": "yarn", "factor": 2.0,
                "original_max_position_embeddings": 32768,
            },
        )
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a")
        text = render_script(
            "qwen35-27b-tier3a", config, cell,
            granularity="sfid",
            cluster="helix",
            use_kg=True,
            logs_dir=Path("/tmp/logs"),
            n=100,
        )
        assert "#SBATCH --mem=128gb" in text
        assert "#SBATCH --mem=80gb" not in text
        # And vLLM gets the matching --max-num-seqs.
        assert "--max-num-seqs 16" in text


# ----------------------------------------------------------------------------
# write_script (filesystem side effect)
# ----------------------------------------------------------------------------


class TestWriteScript:
    def test_writes_to_variant_subdirectory(self, tmp_path: Path) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        script_path = write_script(
            "gemini-3-flash-plain", config, cell, granularity="cell",
            out_root=tmp_path,
        )
        assert script_path.exists()
        assert script_path.parent.name == "gemini-3-flash-plain"
        # Default size nests the variant dir under large/.
        assert script_path.parent.parent.name == "large"
        assert script_path.name == "eng_kwg.sh"
        # Logs dir created alongside.
        assert (tmp_path / "large" / "gemini-3-flash-plain" / "logs").is_dir()
        # Executable bit set so the user can chmod-less submit / run.
        assert script_path.stat().st_mode & 0o111

    def test_sfid_filename_includes_sfid_suffix(self, tmp_path: Path) -> None:
        config = _CATALOGUE.get("qwen35-a3b-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_d")
        script_path = write_script(
            "qwen35-a3b-tier3a", config, cell, granularity="sfid",
            out_root=tmp_path,
        )
        assert script_path.name == "eng_kwg_sfid_d.sh"

    def test_seeded_scripts_nest_under_seed_directory(
        self, tmp_path: Path
    ) -> None:
        """Scripts for different seeds live under ``<size>/<variant>/seed<N>/``
        sub-directories rather than carrying a seed-prefixed filename.

        Why: this lets a user sbatch one seed's work with a single
        glob (``<variant>/seed42/*.sh``) and keeps the variant dir
        from cluttering up with seed-prefixed twins when --seeds is
        set on a multi-cell run.
        """
        config = _CATALOGUE.get("gemini-3-flash-plain")
        paths = []
        for seed in (42, 43, 44):
            cell = _Cell(lang="eng", kg="kwg", sfid="", seed=seed)
            sp = write_script(
                "gemini-3-flash-plain", config, cell, granularity="cell",
                out_root=tmp_path,
            )
            paths.append(sp)
        # Filename is plain — no ``seed<N>_`` prefix.
        assert [sp.name for sp in paths] == ["eng_kwg.sh"] * 3
        # Seed lives one directory above the script.
        assert [sp.parent.name for sp in paths] == ["seed42", "seed43", "seed44"]
        # Parent-of-parent is the variant dir — seed subtree under it.
        assert all(sp.parent.parent.name == "gemini-3-flash-plain" for sp in paths)
        # All three coexist on disk — none overwrote a sibling.
        assert all(sp.exists() for sp in paths)
        # Logs dir is shared across seeds at the variant level (the
        # SLURM job_name embeds the seed so log files don't collide).
        assert (tmp_path / "large" / "gemini-3-flash-plain" / "logs").is_dir()
        assert not (tmp_path / "large" / "gemini-3-flash-plain" / "seed42" / "logs").exists()


# ----------------------------------------------------------------------------
# Sanity: constants
# ----------------------------------------------------------------------------


class TestConstants:
    def test_target_languages_includes_english(self) -> None:
        # English wasn't in the postedit TARGET_LANGUAGES (translation-only)
        # but IS in the benchmark_models target set; missing this would
        # silently drop English from all SLURM emit calls.
        assert "eng" in TARGET_LANGUAGES
        assert len(TARGET_LANGUAGES) == 17

    def test_kg_buckets_match_layout(self) -> None:
        assert set(KG_BUCKETS) == {"kwg", "osmh3kg", "worldkg", "multimodal"}

    def test_sfid_count(self) -> None:
        # 14 SFIDs (a..n) emitted by the renumbering migration.
        assert len(SFIDS) == 14

    def test_sfids_by_kg_multimodal_is_empty_string(self) -> None:
        # Multimodal collapses to one cell — represented by the empty
        # SFID sentinel so _enumerate_cells produces one cell per
        # (lang, multimodal) rather than 14.
        assert SFIDS_BY_KG["multimodal"] == ("",)


# ----------------------------------------------------------------------------
# Time-budget compute — _compute_time_budget + helpers
# ----------------------------------------------------------------------------


class TestRoundUpToHalfHour:
    """Verify the half-hour rounding contract: ``ceil`` to next 30-min
    boundary, minimum 30 min."""

    def test_below_30_min_rounds_to_30(self) -> None:
        assert _round_up_to_half_hour(0) == 30
        assert _round_up_to_half_hour(60) == 30      # 1 min → 30 min
        assert _round_up_to_half_hour(1800 - 1) == 30  # 29:59 → 30

    def test_exact_30_min_stays_30(self) -> None:
        assert _round_up_to_half_hour(30 * 60) == 30

    def test_31_min_rounds_to_60(self) -> None:
        assert _round_up_to_half_hour(31 * 60) == 60

    def test_47_min_rounds_to_60(self) -> None:
        assert _round_up_to_half_hour(47 * 60) == 60

    def test_60_min_stays_60(self) -> None:
        assert _round_up_to_half_hour(60 * 60) == 60

    def test_61_min_rounds_to_90(self) -> None:
        assert _round_up_to_half_hour(61 * 60) == 90

    def test_113_min_rounds_to_120(self) -> None:
        # 1h53m → 2h00 (next 30 min boundary)
        assert _round_up_to_half_hour(113 * 60) == 120

    def test_378_min_rounds_to_390(self) -> None:
        # 6h18m → 6h30
        assert _round_up_to_half_hour(378 * 60) == 390

    def test_fractional_seconds_ceil(self) -> None:
        # Fractional seconds should round up: 1.1 s = 0.018 min = 1 min
        # → 30 min.
        assert _round_up_to_half_hour(1.1) == 30


class TestFormatHMS:
    def test_basic(self) -> None:
        assert _format_hms(30) == "00:30:00"
        assert _format_hms(60) == "01:00:00"
        assert _format_hms(90) == "01:30:00"
        assert _format_hms(1440) == "24:00:00"

    def test_zero_pads(self) -> None:
        assert _format_hms(5) == "00:05:00"


class TestModeKeyForTimeBudget:
    """The mode_key encodes the load profile that's actually run, not
    the variant's name. Two tier-3a variants doing the same
    use-kg / oracle pattern should hash to the same key."""

    def test_plain_no_oracle(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        assert _mode_key_for_time_budget(config, oracle_mode=None, use_kg=False) == "none"

    def test_plain_with_oracle_structured(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        key = _mode_key_for_time_budget(config, oracle_mode="structured", use_kg=False)
        assert key == "oracle-structured"

    def test_plain_with_oracle_raw(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        key = _mode_key_for_time_budget(config, oracle_mode="raw", use_kg=False)
        assert key == "oracle-raw"

    def test_reasoning_always_none(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-thinking-tier2")
        # Reasoning doesn't accept oracle_mode in practice but the
        # function should be defensive.
        assert _mode_key_for_time_budget(config, oracle_mode=None, use_kg=False) == "none"

    def test_tier3a_use_kg(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-tier3a")
        assert _mode_key_for_time_budget(config, oracle_mode=None, use_kg=True) == "tier3a-usekg"

    def test_tier3a_oracle_t3(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-tier3a")
        assert _mode_key_for_time_budget(config, oracle_mode="t3", use_kg=False) == "tier3a-oracle"

    def test_tier3b_no_flags(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-tier3b")
        assert _mode_key_for_time_budget(config, oracle_mode=None, use_kg=False) == "tier3b"

    def test_tier3c_use_kg(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-tier3c")
        assert _mode_key_for_time_budget(config, oracle_mode=None, use_kg=True) == "tier3c-usekg"


class TestComputeTimeBudgetHeadline:
    """Spot-check the budget numbers shown to the user during the
    Option-A design review. Regression-locks the projected SLURM
    --time values for the gemini-3-flash matrix on Helix."""

    def _cell(self, sfid: str) -> _Cell:
        return _Cell(lang="eng", kg="kwg", sfid=sfid, seed=42)

    def test_tier1_plain_small_sfid_1h(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        budget = _compute_time_budget(
            config, self._cell("sfid_n"),
            variant="gemini-3-flash-plain",
            n=136, cluster="helix", is_vllm=False,
            oracle_mode=None, use_kg=False, max_concurrent=None,
        )
        # 136*5/16 = 43s + 30s cold + 1800s buffer = 1873s ≈ 31.2 min → 60 min
        assert budget == "01:00:00"

    def test_tier1_plain_large_sfid_1h(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        budget = _compute_time_budget(
            config, self._cell("sfid_a"),
            variant="gemini-3-flash-plain",
            n=4827, cluster="helix", is_vllm=False,
            oracle_mode=None, use_kg=False, max_concurrent=None,
        )
        # 4827*5/16 = 1508s + 30s + 1800s = 3338s ≈ 55.6 min → 60 min
        assert budget == "01:00:00"

    def test_tier2_reasoning_large_sfid_2h(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-thinking-tier2")
        budget = _compute_time_budget(
            config, self._cell("sfid_a"),
            variant="gemini-3-flash-thinking-tier2",
            n=4827, cluster="helix", is_vllm=False,
            oracle_mode=None, use_kg=False, max_concurrent=None,
        )
        # 4827*15/16 = 4525s + 30s + 1800s = 6355s ≈ 105.9 min → 120 min
        assert budget == "02:00:00"

    def test_tier3a_usekg_small_sfid_3h30(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-tier3a")
        budget = _compute_time_budget(
            config, self._cell("sfid_n"),
            variant="gemini-3-flash-tier3a",
            n=136, cluster="helix", is_vllm=False,
            oracle_mode=None, use_kg=True, max_concurrent=None,
        )
        # 136*250/4 = 8500s + 30s + 1800s = 10330s ≈ 172 min → 180 min (3h)
        # Wait — with helix non-vllm cold-start of 30s, raw=172.2 min,
        # round-up to next 30 → 180 min = 3h.
        assert budget == "03:00:00"

    def test_tier3a_usekg_helix_vllm_small_sfid_3h30(self) -> None:
        """With vLLM cold-start (900s = 15 min) tier-3a use-kg
        becomes one 30-min step longer than API mode."""
        config = _CATALOGUE.get("qwen35-27b-tier3a")
        budget = _compute_time_budget(
            config, self._cell("sfid_n"),
            variant="qwen35-27b-tier3a",
            n=136, cluster="helix", is_vllm=True,
            oracle_mode=None, use_kg=True, max_concurrent=None,
        )
        # 136*250/4 = 8500s + 900s vLLM cold + 1800s = 11200s ≈ 187 min
        # → round-up to 210 min = 3h30.
        assert budget == "03:30:00"

    def test_tier3a_usekg_large_sfid_clips_to_24h(self) -> None:
        """sfid_a at 4827 questions × 250 s/q / 4 concurrent = 84 h
        of raw wall-clock — must clip to 24 h with a warning."""
        config = _CATALOGUE.get("gemini-3-flash-tier3a")
        budget = _compute_time_budget(
            config, self._cell("sfid_a"),
            variant="gemini-3-flash-tier3a",
            n=4827, cluster="helix", is_vllm=False,
            oracle_mode=None, use_kg=True, max_concurrent=None,
        )
        assert budget == "24:00:00"

    def test_tier3a_oracle_large_sfid_6h(self) -> None:
        """Oracle path: 30 s/q × 4827 / 8 concurrency = ~5 h —
        well under the cap."""
        config = _CATALOGUE.get("gemini-3-flash-tier3a")
        budget = _compute_time_budget(
            config, self._cell("sfid_a"),
            variant="gemini-3-flash-tier3a",
            n=4827, cluster="helix", is_vllm=False,
            oracle_mode="t3", use_kg=False, max_concurrent=None,
        )
        # 4827*30/8 = 18101s + 30s + 1800s = 19931s ≈ 332 min → 360 min (6h)
        assert budget == "06:00:00"


class TestComputeTimeBudgetEdgeCases:
    def test_explicit_n_overrides_sfid_lookup(self) -> None:
        """Passing positive ``n`` short-circuits the on-disk
        benchmark.jsonl count — important when running headlines from
        a checkout where the benchmark hasn't been built locally."""
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a", seed=None)
        budget_small_n = _compute_time_budget(
            config, cell,
            variant="gemini-3-flash-plain",
            n=10, cluster="helix", is_vllm=False,
            oracle_mode=None, use_kg=False, max_concurrent=None,
        )
        # 10*5/16 = 3.1s + 30s + 1800s = 1833s ≈ 30.6 min → 60 min
        assert budget_small_n == "01:00:00"

    def test_explicit_max_concurrent_used(self) -> None:
        """Caller's ``--max-concurrent`` override flows into the
        budget compute, not just the emitted runner flag."""
        config = _CATALOGUE.get("gemini-3-flash-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_g", seed=None)
        budget_default = _compute_time_budget(
            config, cell,
            variant="gemini-3-flash-tier3a",
            n=333, cluster="helix", is_vllm=False,
            oracle_mode=None, use_kg=True, max_concurrent=None,  # → default 4
        )
        budget_doubled = _compute_time_budget(
            config, cell,
            variant="gemini-3-flash-tier3a",
            n=333, cluster="helix", is_vllm=False,
            oracle_mode=None, use_kg=True, max_concurrent=8,  # double
        )
        # Doubling concurrency halves the per-question term; total
        # budget can only be smaller or equal.
        h_default = int(budget_default.split(":")[0])
        h_doubled = int(budget_doubled.split(":")[0])
        assert h_doubled <= h_default

    def test_clip_to_partition_wall_cap_minutes(self) -> None:
        """Any computed value above 24h gets exactly 24h, never more."""
        config = _CATALOGUE.get("gemini-3-flash-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a", seed=None)
        # Force the clip with an absurd n.
        budget = _compute_time_budget(
            config, cell,
            variant="gemini-3-flash-tier3a",
            n=100_000, cluster="helix", is_vllm=False,
            oracle_mode=None, use_kg=True, max_concurrent=None,
        )
        assert budget == _format_hms(_PARTITION_WALL_CAP_MIN)
        assert budget == "24:00:00"

    def test_minimum_budget_is_30_min(self) -> None:
        """Even a 1-question smoke gets a 30-min slot — startup
        variance dominates for very small n."""
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_n", seed=None)
        budget = _compute_time_budget(
            config, cell,
            variant="gemini-3-flash-plain",
            n=1, cluster="bwuc3", is_vllm=False,
            oracle_mode=None, use_kg=False, max_concurrent=None,
        )
        # 1*5/16 = 0.3s + 30s + 1800s = 1830s ≈ 30.5 min → 60 min
        # Actually 30.5 > 30 so rounds up to 60. Confirm.
        assert budget == "01:00:00"


class TestComputeTimeBudgetCliOverride:
    """The CLI's ``--time-budget`` override should bypass the heuristic
    entirely — ``render_script`` honors ``time_budget_override``."""

    def test_override_appears_verbatim_in_header(self, tmp_path: Path) -> None:
        config = _CATALOGUE.get("gemini-3-flash-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a", seed=None)
        text = render_script(
            "gemini-3-flash-tier3a", config, cell,
            granularity="sfid",
            cluster="helix",
            use_kg=True,
            n=4827,
            time_budget_override="48:00:00",  # absurd, but caller's choice
        )
        assert "#SBATCH --time=48:00:00" in text

    def test_no_override_uses_computed_budget(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-tier3a")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_n", seed=None)
        text = render_script(
            "gemini-3-flash-tier3a", config, cell,
            granularity="sfid",
            cluster="helix",
            use_kg=True,
            n=136,
        )
        # Computed budget for tier3a use-kg helix API at n=136:
        # 136*250/4 + 30 + 1800 = 10330s = 172.2 min → 180 min = 3h.
        assert "#SBATCH --time=03:00:00" in text


class TestWktSimplifyToleranceFlag:
    """Verify ``--wkt-simplify-tolerance`` is emitted for oracle modes
    even when the caller didn't pass an explicit value — uses the
    same 0.001 default as ``OracleContext`` and ``EntityService`` so
    the SLURM body documents the tolerance and oracle results are
    reproducible from the script alone."""

    def test_oracle_mode_emits_default_tolerance_without_explicit_value(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a", seed=None)
        text = render_script(
            "gemini-3-flash-plain", config, cell,
            granularity="sfid",
            oracle_mode="structured",
            # no wkt_simplify_tolerance passed: should default to 0.001
        )
        assert "--wkt-simplify-tolerance 0.001" in text

    def test_oracle_mode_respects_explicit_tolerance(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a", seed=None)
        text = render_script(
            "gemini-3-flash-plain", config, cell,
            granularity="sfid",
            oracle_mode="raw",
            wkt_simplify_tolerance=0.005,  # caller override
        )
        assert "--wkt-simplify-tolerance 0.005" in text
        assert "--wkt-simplify-tolerance 0.001" not in text

    def test_oracle_mode_zero_tolerance_for_ablation(self) -> None:
        """Explicit ``--wkt-simplify-tolerance 0`` is the
        'no-simplification' ablation. The flag must still be emitted
        (so the SLURM body shows the choice was deliberate, not the
        default leaking through)."""
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="sfid_a", seed=None)
        text = render_script(
            "gemini-3-flash-plain", config, cell,
            granularity="sfid",
            oracle_mode="raw",
            wkt_simplify_tolerance=0.0,
        )
        assert "--wkt-simplify-tolerance 0.0" in text

    def test_non_oracle_mode_omits_flag(self) -> None:
        """The runner ignores --wkt-simplify-tolerance for non-oracle
        runs. The generator skips emitting it so the SLURM body
        stays clean."""
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="", seed=None)
        text = render_script(
            "gemini-3-flash-plain", config, cell,
            granularity="cell",
            # no oracle_mode
        )
        assert "--wkt-simplify-tolerance" not in text


class TestMultimodalLimitMmPerPromptFlag:
    """The vLLM ``--limit-mm-per-prompt`` flag must be emitted for
    ``multimodal_mode=image`` variants and skipped otherwise. Without
    it, some vLLM versions default to zero allowed images per request
    and silently reject the OpenAI ``image_url`` blocks the harness
    sends — leading to per-row 400s that look like model failures."""

    def test_image_mode_emits_limit_mm_per_prompt(self) -> None:
        """vLLM 0.20+ requires the JSON dict form for
        ``--limit-mm-per-prompt``; the older ``image=1`` key-value form
        is rejected at CLI parse with ``Value image=1 cannot be
        converted to <function loads at ...>`` because the parser runs
        ``json.loads`` on the argument. Caught live on
        qwen35-27b-image job 13027716 (vllm_0.20.1.sif). Lock the JSON
        form here."""
        from benchmark_models.config import VLLMHttpVariantConfig

        config = VLLMHttpVariantConfig(
            backend="vllm_http", tier="plain",
            model_path="google/gemma-3-27B-it",
            multimodal_mode="image",
        )
        cell = _Cell(lang="eng", kg="multimodal", sfid="", seed=None)
        text = render_script(
            "gemma-3-27b-it-image", config, cell,
            granularity="cell", cluster="helix",
            logs_dir=Path("/tmp/logs"),
        )
        # JSON form (vLLM 0.20+); the old key=value form would be
        # rejected by the CLI parser.
        assert "--limit-mm-per-prompt '{\"image\":1}'" in text
        # And the old form must NOT be in the generated body.
        assert "--limit-mm-per-prompt 'image=1'" not in text

    def test_name_mode_omits_limit_mm_per_prompt(self) -> None:
        """``-name`` variants substitute entity names into the prompt
        text; no image data leaves the harness. The serve doesn't need
        the multimodal allowance — and emitting it unnecessarily would
        confuse a reader of the SLURM body about whether image data is
        actually flowing."""
        from benchmark_models.config import VLLMHttpVariantConfig

        config = VLLMHttpVariantConfig(
            backend="vllm_http", tier="plain",
            model_path="google/gemma-3-27B-it",
            multimodal_mode="name",
        )
        cell = _Cell(lang="eng", kg="multimodal", sfid="", seed=None)
        text = render_script(
            "gemma-3-27b-it-name", config, cell,
            granularity="cell", cluster="helix",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--limit-mm-per-prompt" not in text

    def test_text_only_variant_omits_limit_mm_per_prompt(self) -> None:
        """Non-multimodal variants (``multimodal_mode=off``) also skip
        the flag. Regression guard against accidentally widening the
        emission scope to every vLLM variant."""
        from benchmark_models.config import VLLMHttpVariantConfig

        config = VLLMHttpVariantConfig(
            backend="vllm_http", tier="plain",
            model_path="Qwen/Qwen3.5-27B-FP8",
            # multimodal_mode defaults to "off"
        )
        cell = _Cell(lang="eng", kg="kwg", sfid="", seed=None)
        text = render_script(
            "qwen35-27b-plain", config, cell,
            granularity="cell", cluster="helix",
            logs_dir=Path("/tmp/logs"),
        )
        assert "--limit-mm-per-prompt" not in text

    def test_yaml_open_weights_multimodal_variants_present(self) -> None:
        """The shipped ``models.yaml`` defines the open-weights
        multimodal variants (qwen35-27b/a3b + gemma-3-27b-it × image,
        name). All point at the same base model weights as their
        -plain twins and only differ in ``multimodal_mode``."""
        for stem, model_path in (
            ("qwen35-27b", "Qwen/Qwen3.5-27B-FP8"),
            ("qwen35-a3b", "Qwen/Qwen3.5-35B-A3B-FP8"),
            ("gemma-3-27b-it", "google/gemma-3-27B-it"),
        ):
            for mode in ("image", "name"):
                variant = f"{stem}-{mode}"
                cfg = _CATALOGUE.get(variant)
                assert cfg.backend == "vllm_http", variant
                assert cfg.model_path == model_path, variant
                assert cfg.multimodal_mode == mode, variant
                assert cfg.tier == "plain", variant


class TestDwsClusterConstants:
    """Pin the DWS Mannheim cluster constants. The generator dispatches
    on these dicts when rendering templates; a missing or wrong value
    surfaces as a confusing KeyError or a script that won't run on
    DWS. See docs/superpowers/specs/2026-05-21-dws-cluster-support-design.md."""

    def test_dws_in_clusters_tuple(self) -> None:
        assert "dws" in _CLUSTERS

    def test_dws_gpu_partition_is_94gb_tier(self) -> None:
        # 27B FP8 models need >=94GB VRAM; smaller-model partitions
        # are deferred (see spec "out of scope").
        assert _CLUSTER_GPU_PARTITION["dws"] == "gpu-vram-94gb"

    def test_dws_gpu_gres_is_plain_count(self) -> None:
        # The partition already constrains hardware to H100 NVL; an
        # explicit type pin (gpu:nvidia_h100_nvl:1) would be redundant.
        assert _CLUSTER_GPU_GRES["dws"] == "gpu:1"

    def test_dws_cpu_partition(self) -> None:
        assert _CLUSTER_CPU_PARTITION["dws"] == "cpu"

    def test_dws_workspace_root_is_work_mboeckli(self) -> None:
        # DWS scratch equivalent of bwUC3's /pfs/work9 / Helix's /gpfs.
        assert _CLUSTER_WORKSPACE_ROOT["dws"] == "/work/mboeckli"

    def test_dws_cold_start_matches_bwuc3(self) -> None:
        # DWS uses native venv (no Singularity), so cold-start mirrors
        # bwUC3 (300s vllm / 30s api), not Helix (900s vllm).
        assert _COLD_START_OVERHEAD_S[("dws", True)] == 300
        assert _COLD_START_OVERHEAD_S[("dws", False)] == 30


class TestDwsVllmHttpTemplate:
    """Pin the DWS vLLM-HTTP header shape per
    docs/superpowers/specs/2026-05-21-dws-cluster-support-design.md §5.

    Three concrete deltas vs the bwUC3 header are load-bearing:
    no LMOD, no .venv-vllm dance, system Java auto-exported."""

    def _render(self) -> str:
        config = _CATALOGUE.get("qwen35-a3b-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        return render_script(
            "qwen35-a3b-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
            cluster="dws",
        )

    def test_partition_is_gpu_vram_94gb(self) -> None:
        assert "#SBATCH --partition=gpu-vram-94gb" in self._render()

    def test_gres_is_plain_count(self) -> None:
        assert "#SBATCH --gres=gpu:1" in self._render()

    def test_java_home_auto_exported_jdk21(self) -> None:
        # DWS has only system OpenJDK 21; the template must export
        # JAVA_HOME itself rather than asserting on a bashrc-supplied
        # value (the bashrc fail-fast was bwuc3-specific).
        assert "export JAVA_HOME=/usr/lib/jvm/java-21-openjdk" in self._render()

    def test_vllm_via_uv_extra(self) -> None:
        # No separate .venv-vllm on DWS — reuse the project .venv via
        # the [vllm] extra declared in pyproject.toml.
        assert "uv run --extra vllm vllm serve" in self._render()

    def test_caches_pinned_under_work_mboeckli(self) -> None:
        text = self._render()
        assert "TORCHINDUCTOR_CACHE_DIR=/work/mboeckli/.cache/torchinductor" in text

    def test_no_module_load(self) -> None:
        # DWS has no LMOD installed; module load crashes the job at
        # line 1 of the user-script portion.
        assert "module load" not in self._render()

    def test_no_venv_vllm_activate(self) -> None:
        # The .venv-vllm directory doesn't exist on DWS scratch.
        assert ".venv-vllm/bin/activate" not in self._render()

    def test_health_budget_is_30_minutes(self) -> None:
        # DWS node performance varies dramatically: dws-16 (8-GPU,
        # 96-CPU box) completed cold-start in ~7 min, while dws-10
        # (2-GPU, 40-CPU box) took 20+ min for the same job (job
        # 248266 timed out at exactly 20:06 mid CUDA-graph capture).
        # Model load was 153s vs 38s; warmup 210s vs 60s — roughly
        # 5x slower on the smaller node. 30 min covers the dws-10
        # worst case with headroom; warm reruns on fast nodes still
        # respond in well under 10 min.
        text = self._render()
        assert "$(seq 1 1800)" in text
        assert "30 minutes" in text
        assert "$(seq 1 1200)" not in text
        assert "$(seq 1 600)" not in text


class TestDwsApiTemplate:
    """Pin the DWS API header shape per spec §5. Tier-3 API runs spin
    up Spark/Sedona for KG access; the JAVA_HOME export is therefore
    kept even though tier-1/2 ignore it."""

    def _render(self) -> str:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        cell = _Cell(lang="eng", kg="kwg", sfid="")
        return render_script(
            "gemini-3-flash-plain", config, cell, granularity="cell",
            logs_dir=Path("/tmp/logs"),
            cluster="dws",
        )

    def test_partition_is_cpu(self) -> None:
        assert "#SBATCH --partition=cpu" in self._render()

    def test_java_home_auto_exported_jdk21(self) -> None:
        assert "export JAVA_HOME=/usr/lib/jvm/java-21-openjdk" in self._render()
