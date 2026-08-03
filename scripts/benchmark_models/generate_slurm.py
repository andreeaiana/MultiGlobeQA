"""Emit per-cell SLURM scripts for benchmark_models runs.

Granularity
-----------

* ``--granularity cell`` — one job per ``(language, kg)``. The runner
  consumes ``benchmark/final/<size>/<kg>/`` with ``--language <lang>``; the
  walker concatenates every SFID. Right for tier 1 / tier 2 where
  per-batch throughput dominates and SFID overhead is negligible.
* ``--granularity sfid`` — one job per ``(language, kg, sfid)``. Right
  for tier 3, where the agent loop is per-row 30–60s wall-clock and
  a single ``(lang, kg)`` would blow past any reasonable SLURM time
  budget.

Granularity defaults are picked from the variant's tier:

* ``plain`` / ``reasoning`` → ``cell``
* ``tool_agent`` → ``sfid``

Backend-specific templates
--------------------------

* **vLLM backends** (``qwen35-a3b-*``, ``qwen35-27b-*``,
  ``gemma-3-27b-it-*``): GPU partition (``gpu_h100`` on bwUC3,
  ``gpu-single`` on Helix), one H100/H200, 80 GB host RAM. The
  template launches ``vllm serve`` in the background, polls
  ``/health``, exports ``VLLM_BASE_URL=http://localhost:<port>/v1``,
  then runs the benchmark client against it. The OpenAI-compatible
  HTTP path is used by every vLLM variant — there is no in-process
  ``LLM(...)`` path anywhere in ``benchmark_models``.
* **API backends** (``gpt-5.4-*``, ``gemini-3-flash-*``,
  ``claude-sonnet-4.6-*``): CPU partition (``single``), no GPU,
  32 GB host RAM. Sources ``scripts/set_api_keys.sh`` so the per-row
  ThreadPoolExecutor has the right credentials.

Output layout
-------------

Scripts: ``jobs/benchmark_models/<size>/<variant>/<lang>_<kg>[_<sfid>].sh``
Logs:    ``jobs/benchmark_models/<size>/<variant>/logs/<...>_%j.{out,err}``
Predictions (written by the runner inside the job):
         ``results/runs/<variant>/<lang>/<kg>[/<sfid>]/predictions.jsonl``

Submission
----------

This generator emits scripts only; it does **not** auto-submit. The
caller is expected to ``sbatch`` whichever subset they want — staying
under cluster + upstream-API rate limits is a human decision.

Examples
--------

::

    # All scripts for one tier-1 closed-source variant, defaulting to
    # per-(lang, kg) granularity, all 17 languages + 4 KGs.
    uv run python scripts/benchmark_models/generate_slurm.py \\
        --variant gemini-3-flash-plain

    # Tier-3 vLLM run, automatically per-(lang, kg, sfid) split.
    uv run python scripts/benchmark_models/generate_slurm.py \\
        --variant qwen35-a3b-tier3a

    # English smoke for one variant; only kwg.
    uv run python scripts/benchmark_models/generate_slurm.py \\
        --variant gemma-3-27b-it-plain --languages eng --kgs kwg
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from benchmark_models.config import VariantConfig, load_models_catalogue


# Project conventions: 17 target languages (English + the 16 multilingual
# translations) and 4 KG buckets. Listed inline rather than imported from
# pipelines/postedit/config so the generator can be exercised without
# pulling in the postedit dependency chain.
TARGET_LANGUAGES: tuple[str, ...] = (
    "eng",
    "bul", "deu", "ell", "spa", "fra", "ita", "jpn", "kat",
    "por_PT", "ron", "rus", "sqi", "tur", "urd", "vie", "zho_CN",
)

KG_BUCKETS: tuple[str, ...] = ("kwg", "osmh3kg", "worldkg", "multimodal")

# 14 SFIDs (a..n). This is the full alphabet; per-KG coverage in
# ``SFIDS_BY_KG`` is a subset.
SFIDS: tuple[str, ...] = tuple(f"sfid_{c}" for c in "abcdefghijklmn")

# Per-KG SFID coverage, mirroring what's actually populated under
# ``benchmark/sampled_multilingual/`` (and therefore ``benchmark/final/large/``
# after the postedit pass). Generating jobs for a (kg, sfid) cell that
# doesn't exist on disk burns a SLURM allocation on a no-op walker.
#
# Coverage gaps:
#   * kwg     — 13 SFIDs (no ``sfid_d``).
#   * osmh3kg — 13 SFIDs (no ``sfid_i``).
#   * worldkg —  9 SFIDs ({a, b, c, f, h, j, l, m, n}); the missing
#                families either don't fit the WorldKG schema or weren't
#                seeded by the question generator.
#   * multimodal — slice of osmh3kg + worldkg with images attached
#                  (kwg has no multimodal questions). No SFID
#                  stratification (empty sentinel), but ``_enumerate_cells``
#                  fans out into one sub-cell per underlying KG via
#                  ``_MULTIMODAL_DISPATCH_KGS`` so tier-3a/3c jobs can
#                  pass a real ``--kg-source`` to the bridge.
SFIDS_BY_KG: dict[str, tuple[str, ...]] = {
    "kwg": tuple(f"sfid_{c}" for c in "abcefghijklmn"),
    "osmh3kg": tuple(f"sfid_{c}" for c in "abcdefghjklmn"),
    "worldkg": tuple(f"sfid_{c}" for c in "abcfhjlmn"),
    "multimodal": ("",),
}

# Underlying KGs that the multimodal bucket fans out to. Multimodal
# questions inherit their geometry / entity grounding from one of these
# two KGs (kwg has no multimodal slice). Mapped at enumeration time so
# every ``kg=="multimodal"`` cell carries a ``kg_dispatch`` field
# naming a real KG — required by tier-3a/3c which calls
# ``schema_for(kg_source)`` at Spark bootstrap time; ``"multimodal"``
# is not a known KG name there.
_MULTIMODAL_DISPATCH_KGS: tuple[str, ...] = ("osmh3kg", "worldkg")


REPO_ROOT = Path(__file__).resolve().parents[2]
JOBS_ROOT = REPO_ROOT / "jobs" / "benchmark_models"


# SLURM time-budget compute (see ``_compute_time_budget``):
#
#   wall_s = (n_questions / effective_concurrency) * per_question_cost
#          + cold_start_overhead
#          + _TIME_BUFFER_S          # +30 min safety
#   minutes = ceil(wall_s / 60)
#   rounded = 30 * ceil(minutes / 30) # round up to next :00 or :30
#   if rounded > 24h: clip to 24h, warn
#
# Per-question cost is keyed by ``(tier, mode_key)`` where mode_key is
# built by ``_mode_key_for_time_budget`` from the variant's tier,
# sub_tier, ``--oracle-mode``, and ``--use-kg`` flags. Numbers come from
# observed run latencies (see ``benchmark_models/README.md``
# for the audit). Conservative — uses the slower observed value when
# API and vLLM differ, since oversize is harmless under the runner's
# resume default while undersize burns time.
_PER_QUESTION_COST_S: dict[tuple[str, str], int] = {
    ("plain",      "none"):                  5,
    ("plain",      "oracle-structured"):    10,
    ("plain",      "oracle-raw"):            5,
    ("plain",      "oracle-verbalized"):     5,
    ("reasoning",  "none"):                 15,
    # Tier-3 variants without ``--use-kg`` and without ``--oracle-mode``
    # exercise the smolagents loop with the mock-bridge / mock-tools
    # path — usually a smoke / sanity case rather than a headline run.
    # Budget like tier-3b (web-only) since the agent still spins up
    # but tool calls are cheap.
    ("tool_agent", "tier3a"):               70,
    ("tool_agent", "tier3a-usekg"):        250,
    ("tool_agent", "tier3a-oracle"):        30,
    ("tool_agent", "tier3b"):               70,
    ("tool_agent", "tier3c"):               70,
    ("tool_agent", "tier3c-usekg"):        300,
    ("tool_agent", "tier3c-oracle"):        30,
}

# Cold-start overhead per (cluster, is_vllm). vLLM cold-starts dominate
# the constant term — Helix Singularity exec + HF cache miss + warmup
# is observed at 5-15 min; bwUC3 native venv is ~5 min. API backends
# pay only the Python import + Spark/Sedona startup (~30 s with the
# warm JVM cache).
_COLD_START_OVERHEAD_S: dict[tuple[str, bool], int] = {
    ("helix", True):  900,
    ("bwuc3", True):  300,
    ("dws",   True):  300,
    ("helix", False):  30,
    ("bwuc3", False):  30,
    ("dws",   False):  30,
}

# Safety buffer added on top of the linear estimate. Covers: network /
# API retry storms, occasional long agent steps, vLLM kv-cache flush
# pauses on long sessions. Fixed at 30 min.
_TIME_BUFFER_S: int = 30 * 60

# Partition wall caps (minutes). Both Helix ``gpu-single`` and bwUC3
# ``gpu_h100`` are 24h; if a future partition diverges, parameterize
# per cluster. Computed budgets above this are clipped down to the
# cap with a logger.warning — the runner's resume default picks the
# job back up on the next sbatch.
_PARTITION_WALL_CAP_MIN: int = 24 * 60

# Effective concurrency per (tier, mode_key). Tier 1/2 plain/reasoning
# rely on the runner's batch_size=16 default (ThreadPoolExecutor for
# API backends, vLLM continuous-batching for local). Tier-3 cells use
# ``max_concurrent`` from the CLI; the defaults below match the
# generator's existing resolver:
#   - tier-3 oracle-t3 fans out to 8 (the oracle workflow has no
#     per-call Spark contention)
#   - tier-3 use-kg / 3b stays at 4 (Spark session contention + agent
#     loop memory)
# When the caller passes an explicit ``--max-concurrent`` value, the
# emitted SLURM uses that override directly and the budget compute
# uses it too via ``_resolve_concurrency_for_budget``.
_DEFAULT_CONCURRENCY: dict[tuple[str, str], int] = {
    ("plain",      "none"):                 16,
    ("plain",      "oracle-structured"):    16,
    ("plain",      "oracle-raw"):           16,
    ("plain",      "oracle-verbalized"):    16,
    ("reasoning",  "none"):                 16,
    ("tool_agent", "tier3a"):                4,
    ("tool_agent", "tier3a-usekg"):          4,
    ("tool_agent", "tier3a-oracle"):         8,
    ("tool_agent", "tier3b"):                4,
    ("tool_agent", "tier3c"):                4,
    ("tool_agent", "tier3c-usekg"):          4,
    ("tool_agent", "tier3c-oracle"):         8,
}

# Fallback question count when ``--n 0`` is set and the SFID's
# benchmark file isn't on disk (e.g., generating from a checkout that
# doesn't yet have ``benchmark/raw_multilingual_renumbered/<kg>/<sfid>/<lang>/``).
# Sized to a value that clips most tier-3 use-kg cells to the
# partition cap so the user gets a loud warning rather than an
# under-budget script. Headline runs with --n > 0 use the user's value
# directly and never hit this fallback.
_BENCHMARK_SIZE_FALLBACK: int = 5000


Granularity = Literal["cell", "sfid"]
Cluster = Literal["bwuc3", "helix", "dws"]
_CLUSTERS: tuple[Cluster, ...] = ("bwuc3", "helix", "dws")

# Per-cluster constants: GPU partition, GPU resource string, CPU
# partition, workspace path, and the "extra setup" snippet that needs
# to run before any cd / module-load (Helix sources $HOME/.bashrc with
# strict-mode-incompatible plumbing, so we have to deactivate conda
# before ``set -u`` kicks in).
_CLUSTER_GPU_PARTITION: dict[Cluster, str] = {
    "bwuc3": "gpu_h100",
    "helix": "gpu-single",
    "dws":   "gpu-vram-94gb",
}
_CLUSTER_GPU_GRES: dict[Cluster, str] = {
    "bwuc3": "gpu:1",
    "helix": "gpu:H200:1",
    "dws":   "gpu:1",
}
_CLUSTER_CPU_PARTITION: dict[Cluster, str] = {
    "bwuc3": "single",
    "helix": "cpu-single",
    "dws":   "cpu",
}
# Workspace root — used to anchor cache directories outside the SLURM
# scratch (which gets purged on job exit) and to make the templates
# portable across user accounts on the same cluster. Override per-user
# via ``--workspace-root`` if needed.
_CLUSTER_WORKSPACE_ROOT: dict[Cluster, str] = {
    "bwuc3": "/pfs/work9/workspace/scratch/ma_aiana-multiglobeqa",
    "helix": "/gpfs/bwfor/work/ws/ma_aiana-multiglobeqa",
    "dws":   "/work/mboeckli",
}


@dataclass(frozen=True)
class _Cell:
    """One job's worth of work — either a (lang, kg) cell or (lang, kg, sfid),
    optionally pinned to a specific random seed for multi-run experiments."""

    lang: str
    kg: str
    sfid: str  # empty string ⇒ no SFID filtering (cell granularity, or multimodal)
    seed: int | None = None
    """When set, the generated script passes ``--seed <N>`` to the runner
    and the output / log / job-name paths embed the seed so multiple
    seed-different runs don't collide on disk. ``None`` leaves the
    seed out of both the runner flags and those paths."""

    kg_dispatch: str = ""
    """The actual KG name passed to ``--kg-source`` and used to drill
    into the benchmark / output trees. For ordinary buckets this is
    left empty and the cell's ``kg`` is used directly. For the
    ``multimodal`` bucket — which is a slice of osmh3kg + worldkg
    rather than its own KG — this names the underlying KG
    (``"osmh3kg"`` or ``"worldkg"``) so tier-3a/3c can call
    ``schema_for(kg_source)`` with a real KG name. See
    :data:`_MULTIMODAL_DISPATCH_KGS` and ``_enumerate_cells``."""

    @property
    def kg_source(self) -> str:
        """KG name to pass to ``--kg-source`` at runtime — and the same
        name used to drill into ``benchmark/final/<size>/<kg>/...`` and
        ``results/runs/<variant>/.../<kg>/...`` for multimodal sub-cells.
        For non-multimodal cells this equals ``self.kg``."""
        return self.kg_dispatch or self.kg


# ----------------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------------


# SLURM templates are parameterised on:
#   {gpu_partition}    — gpu_h100 (bwuc3) or gpu-single (helix)
#   {gpu_gres}         — gpu:1 (bwuc3) or gpu:H200:1 (helix)
#   {cpu_partition}    — single (bwuc3) or cpu-single (helix)
#   {workspace_root}   — per-cluster scratch root for cache dirs
#   {time_budget}, {job_name}, {logs_dir}, {repo_root}
# plus the vLLM-serve variants additionally take:
#   {model_path}, {vllm_port}, {vllm_max_model_len}, {vllm_max_num_seqs}


# Tier-1/2/3 API-backend header for BwUniCluster (no GPU; talks to
# closed-source providers over the network).
_API_HEADER_BWUC3 = """#!/bin/bash
#SBATCH --partition={cpu_partition}
#SBATCH --nodes=1
#SBATCH --time={time_budget}
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/{job_name}_%j.out
#SBATCH --error={logs_dir}/{job_name}_%j.err

source $HOME/.bashrc
set -euo pipefail
cd {repo_root}

# MLflow imports GitPython at experiment-init time to capture the commit
# SHA. SLURM jobs run with --export=NONE → ``git`` isn't on PATH on the
# compute node, GitPython logs a multi-line warning at startup. Silence
# rather than wire git in — paper reproducibility comes from snapshot_id,
# not the MLflow git tag.
export GIT_PYTHON_REFRESH=quiet

# API backends require the closed-source provider keys. Fail-fast if
# the keys file is missing.
if [ -f scripts/set_api_keys.sh ]; then
    source scripts/set_api_keys.sh
else
    echo "ERROR: scripts/set_api_keys.sh not found." >&2
    exit 1
fi

# Tier-3a/3c variants spin up Spark/Sedona in-process for KG access,
# which needs JAVA_HOME pointing at a JDK 17 install. ~/.bashrc should
# export it (via the conda jdk17 env set up in scripts/postedit/README.md).
# Fail fast if missing — otherwise Spark crashes 30s in with an opaque
# Py4J error.
: "${{JAVA_HOME:?ERROR: JAVA_HOME not set — source ~/.bashrc with Java env}}"
"""


# Same API-backend pattern for Helix. The only differences vs bwuc3
# are the CPU partition name and the bashrc handshake (Helix's
# /etc/bashrc references unset vars that trip ``set -u``).
_API_HEADER_HELIX = """#!/bin/bash
#SBATCH --partition={cpu_partition}
#SBATCH --nodes=1
#SBATCH --time={time_budget}
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/{job_name}_%j.out
#SBATCH --error={logs_dir}/{job_name}_%j.err

# Source bashrc before strict mode — /etc/bashrc on Helix references
# unset vars (BASHRCSOURCED, conda plumbing) that would trip ``set -u``.
source $HOME/.bashrc
conda deactivate 2>/dev/null || true
set -euo pipefail
cd {repo_root}

# Silence MLflow's GitPython warning — see _API_HEADER_BWUC3 for rationale.
export GIT_PYTHON_REFRESH=quiet

if [ -f scripts/set_api_keys.sh ]; then
    source scripts/set_api_keys.sh
else
    echo "ERROR: scripts/set_api_keys.sh not found." >&2
    exit 1
fi

# Tier-3a/3c variants spin up Spark/Sedona in-process for KG access,
# which needs JAVA_HOME pointing at a JDK 17 install. ~/.bashrc should
# export it (via the conda jdk17 env set up in scripts/postedit/README.md).
# Fail fast if missing — otherwise Spark crashes 30s in with an opaque
# Py4J error.
: "${{JAVA_HOME:?ERROR: JAVA_HOME not set — source ~/.bashrc with Java env}}"
"""


# vLLM-HTTP header for BwUniCluster: 1×H100, launches ``vllm serve``
# in the background, waits for /health, runs the smoke, traps to kill
# serve on exit. Same vLLM-serve pattern as the postedit generator at
# ``pipelines/postedit/generate_slurm.py``.
_VLLM_HTTP_HEADER_BWUC3 = """#!/bin/bash
#SBATCH --partition={gpu_partition}
#SBATCH --nodes=1
#SBATCH --gres={gpu_gres}
#SBATCH --time={time_budget}
#SBATCH --mem={mem_gb}gb
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/{job_name}_%j.out
#SBATCH --error={logs_dir}/{job_name}_%j.err

source $HOME/.bashrc
set -euo pipefail
cd {repo_root}

# Silence MLflow's GitPython warning — see _API_HEADER_BWUC3 for rationale.
export GIT_PYTHON_REFRESH=quiet

# Source API keys (some vLLM variants still benefit from MLflow tokens).
if [ -f scripts/set_api_keys.sh ]; then
    source scripts/set_api_keys.sh
fi

# CUDA toolkit required by vLLM JIT. Fail-fast if absent — vLLM would
# otherwise crash 5+ minutes into cold-start with a confusing trace.
module load devel/cuda/12.8

# Tier-3a/3c variants spin up Spark/Sedona in-process for KG access,
# which needs JAVA_HOME pointing at a JDK 17 install. ~/.bashrc should
# export it (via the conda jdk17 env set up in scripts/postedit/README.md).
# Fail fast if missing — otherwise Spark crashes 30s in with an opaque
# Py4J error.
: "${{JAVA_HOME:?ERROR: JAVA_HOME not set — source ~/.bashrc with Java env}}"

# Pin torch.compile / vLLM caches to the workspace so they survive job
# boundaries. SLURM purges $TMPDIR on exit and inductor bakes absolute
# paths into its cache.
export TORCHINDUCTOR_CACHE_DIR={workspace_root}/.cache/torchinductor
export HF_HOME={workspace_root}/.cache/huggingface
export VLLM_CACHE_ROOT={workspace_root}/.cache/vllm
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

# Per-job vLLM port: derive from $SLURM_JOB_ID so two single-GPU jobs
# co-scheduled on the same multi-GPU node don't both try to bind
# localhost:8000. ``mod 1000`` keeps the port in the 8000-8999 range;
# the chance of two simultaneously-active jobs hashing to the same
# offset is ~1/1000 — acceptable. Falls back to the YAML-configured
# port ({vllm_port}) for non-SLURM contexts (interactive smokes).
if [ -n "${{SLURM_JOB_ID:-}}" ]; then
    VLLM_PORT=$((8000 + SLURM_JOB_ID % 1000))
else
    VLLM_PORT={vllm_port}
fi
echo "[$(date)] vLLM port: $VLLM_PORT"

# Start vLLM serve in background on localhost:$VLLM_PORT.
# Activate the dedicated vLLM venv only for ``vllm serve`` — the smoke
# command runs via ``uv run`` and uses the project .venv instead. The
# backgrounded process keeps its own env after ``deactivate``.
echo "[$(date)] starting vllm serve ({model_path}) on port $VLLM_PORT"
source {workspace_root}/.venv-vllm/bin/activate
vllm serve {model_path} \\
    --host 0.0.0.0 --port $VLLM_PORT \\
    --max-model-len {vllm_max_model_len} \\
    --max-num-seqs {vllm_max_num_seqs} \\
    --enable-prefix-caching \\
{vllm_extra_flags}    > {logs_dir}/vllm_{job_name}_${{SLURM_JOB_ID}}.log 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT
deactivate

# Wait for /health. First-ever run may pay a 5-10 min HF download
# cost; warm runs are typically ready in <3 min.
echo "[$(date)] waiting for vllm /health on port $VLLM_PORT (up to 10 min)"
for i in $(seq 1 600); do
    if curl -sf http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "[$(date)] vllm ready after ${{i}}s"
        break
    fi
    sleep 1
    if [ "$i" -eq 600 ]; then
        echo "ERROR: vllm did not become healthy in 10 minutes" >&2
        exit 1
    fi
done

# Expose the endpoint to the smoke command. Both the tier-1/2 vLLM
# backend (via ``VLLMHttpVariantConfig.effective_base_url``) and the
# Smolagents tier-3 path read VLLM_BASE_URL from the env when no
# explicit base_url is configured.
export VLLM_BASE_URL=http://localhost:$VLLM_PORT/v1
"""


# vLLM-HTTP header for Helix: wraps ``vllm serve`` in a Singularity
# container because the host glibc is too old for in-tree vLLM
# wheels. Same container pattern as the postedit generator.
_VLLM_HTTP_HEADER_HELIX = """#!/bin/bash
#SBATCH --partition={gpu_partition}
#SBATCH --nodes=1
#SBATCH --gres={gpu_gres}
#SBATCH --time={time_budget}
#SBATCH --mem={mem_gb}gb
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/{job_name}_%j.out
#SBATCH --error={logs_dir}/{job_name}_%j.err

# See _API_HEADER_HELIX for the bashrc / strict-mode dance rationale.
source $HOME/.bashrc
conda deactivate 2>/dev/null || true
set -euo pipefail
cd {repo_root}

# Silence MLflow's GitPython warning — see _API_HEADER_BWUC3 for rationale.
export GIT_PYTHON_REFRESH=quiet

if [ -f scripts/set_api_keys.sh ]; then
    source scripts/set_api_keys.sh
fi

# Tier-3a/3c variants spin up Spark/Sedona in-process for KG access,
# which needs JAVA_HOME pointing at a JDK 17 install. ~/.bashrc should
# export it (via the conda jdk17 env set up in scripts/postedit/README.md).
# Fail fast if missing — otherwise Spark crashes 30s in with an opaque
# Py4J error.
: "${{JAVA_HOME:?ERROR: JAVA_HOME not set — source ~/.bashrc with Java env}}"

# Required Helix env vars (set once in ~/.bashrc — see
# scripts/postedit/README.md for one-time setup).
: "${{SBR_WS:?ERROR: SBR_WS not set — source ~/.bashrc with Helix env}}"
: "${{SBR_VLLM_SIF:?ERROR: SBR_VLLM_SIF not set — source ~/.bashrc with Helix env}}"
if [ ! -f "$SBR_VLLM_SIF" ]; then
    echo "ERROR: vLLM container missing: $SBR_VLLM_SIF" >&2
    exit 1
fi

module load system/singularity/3.11.3

mkdir -p "$HF_HOME" "$SBR_WS/.cache/vllm"
VLLM_LOG={logs_dir}/vllm_{job_name}_${{SLURM_JOB_ID}}.log

# Per-job vLLM port: derive from $SLURM_JOB_ID so two single-GPU jobs
# co-scheduled on the same multi-GPU node don't both try to bind
# localhost:8000. Falls back to the YAML-configured port ({vllm_port})
# for non-SLURM contexts (interactive smokes).
if [ -n "${{SLURM_JOB_ID:-}}" ]; then
    VLLM_PORT=$((8000 + SLURM_JOB_ID % 1000))
else
    VLLM_PORT={vllm_port}
fi
echo "[$(date)] vLLM port: $VLLM_PORT"

echo "[$(date)] starting vllm serve ({model_path}) in container on port $VLLM_PORT"
singularity exec --nv \\
  --bind "$HF_HOME:/root/.cache/huggingface" \\
  "$SBR_VLLM_SIF" \\
  vllm serve {model_path} \\
    --host 0.0.0.0 --port $VLLM_PORT \\
    --max-model-len {vllm_max_model_len} \\
    --max-num-seqs {vllm_max_num_seqs} \\
    --enable-prefix-caching \\
{vllm_extra_flags}  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

# Wait for /health. First-ever run may pay 15+ min download cost.
echo "[$(date)] waiting for vllm /health on port $VLLM_PORT (up to 15 min)"
for i in $(seq 1 900); do
    if curl -sf http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "[$(date)] vllm ready after ${{i}}s"
        break
    fi
    sleep 1
    if [ "$i" -eq 900 ]; then
        echo "ERROR: vllm did not become healthy in 15 minutes" >&2
        exit 1
    fi
done

# See bwUC3 template above for why both tier-1/2 vLLM and tier-3
# Smolagents paths read VLLM_BASE_URL from the env.
export VLLM_BASE_URL=http://localhost:$VLLM_PORT/v1
"""


# API-backend header for DWS Mannheim. Mirrors _API_HEADER_BWUC3 with
# two deltas required by the DWS environment:
#
#   1. No bashrc-driven JAVA_HOME assertion. DWS has only the system
#      OpenJDK 21 at /usr/lib/jvm/java-21-openjdk; we export it inline
#      so users don't have to maintain JAVA_HOME themselves. Spark
#      3.5+ supports JDK 21. Tier-1/2 ignore the export.
#   2. (No CUDA module load — irrelevant for API jobs, but DWS has no
#      LMOD installed at all, so we couldn't ``module load`` even on
#      the GPU side.)
_API_HEADER_DWS = """#!/bin/bash
#SBATCH --partition={cpu_partition}
#SBATCH --nodes=1
#SBATCH --time={time_budget}
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/{job_name}_%j.out
#SBATCH --error={logs_dir}/{job_name}_%j.err

source $HOME/.bashrc
set -euo pipefail
cd {repo_root}

# Silence MLflow's GitPython warning — see _API_HEADER_BWUC3 for rationale.
export GIT_PYTHON_REFRESH=quiet

# API backends require the closed-source provider keys. Fail-fast if
# the keys file is missing.
if [ -f scripts/set_api_keys.sh ]; then
    source scripts/set_api_keys.sh
else
    echo "ERROR: scripts/set_api_keys.sh not found." >&2
    exit 1
fi

# DWS ships only system OpenJDK 21 (no JDK 17 install). Tier-3a/3c
# API runs spin up Spark/Sedona; tier-1/2 ignore this export. Spark
# 3.5+ supports JDK 21 so we don't need to provision a separate JDK.
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
"""


# vLLM-HTTP header for DWS Mannheim. Mirrors _VLLM_HTTP_HEADER_BWUC3
# with three deltas required by the DWS environment:
#
#   1. No ``module load devel/cuda/12.8``. DWS has no LMOD; CUDA is
#      provided by the partition node image (``/usr/local/cuda*``,
#      ``nvidia-smi`` on PATH).
#   2. ``export JAVA_HOME=/usr/lib/jvm/java-21-openjdk`` instead of
#      asserting JAVA_HOME. Same reasoning as _API_HEADER_DWS.
#   3. ``uv run --extra vllm vllm serve`` instead of activating a
#      separate .venv-vllm. The project .venv plus the ``[vllm]``
#      extra in pyproject.toml replaces the per-cluster venv dance.
_VLLM_HTTP_HEADER_DWS = """#!/bin/bash
#SBATCH --partition={gpu_partition}
#SBATCH --nodes=1
#SBATCH --gres={gpu_gres}
#SBATCH --time={time_budget}
#SBATCH --mem={mem_gb}gb
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/{job_name}_%j.out
#SBATCH --error={logs_dir}/{job_name}_%j.err

source $HOME/.bashrc
set -euo pipefail
cd {repo_root}

# Silence MLflow's GitPython warning — see _API_HEADER_BWUC3 for rationale.
export GIT_PYTHON_REFRESH=quiet

# Source API keys (some vLLM variants still benefit from MLflow tokens).
if [ -f scripts/set_api_keys.sh ]; then
    source scripts/set_api_keys.sh
fi

# DWS has no LMOD; CUDA is exposed by the partition node image
# (/usr/local/cuda*, nvidia-smi on PATH) — no LMOD invocation here.

# Tier-3a/3c spin up Spark/Sedona in-process for KG access. DWS has
# only system OpenJDK 21 — Spark 3.5+ supports JDK 21. Auto-export so
# users don't have to maintain JAVA_HOME in their bashrc.
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk

# Pin torch.compile / vLLM caches to the workspace so they survive job
# boundaries. SLURM purges $TMPDIR on exit and inductor bakes absolute
# paths into its cache.
export TORCHINDUCTOR_CACHE_DIR={workspace_root}/.cache/torchinductor
export HF_HOME={workspace_root}/.cache/huggingface
export VLLM_CACHE_ROOT={workspace_root}/.cache/vllm
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

# Per-job vLLM port: same $SLURM_JOB_ID-hash trick as bwUC3 (see that
# template for the collision-rate rationale). Falls back to the
# YAML-configured port ({vllm_port}) for non-SLURM contexts.
if [ -n "${{SLURM_JOB_ID:-}}" ]; then
    VLLM_PORT=$((8000 + SLURM_JOB_ID % 1000))
else
    VLLM_PORT={vllm_port}
fi
echo "[$(date)] vLLM port: $VLLM_PORT"

# Reuse the project .venv via uv. The [vllm] extra in pyproject.toml
# brings in vllm + CUDA wheels; ``uv run --extra vllm`` ensures the
# extra is installed before launching ``vllm serve``. First-ever run
# pays a 2-5GB install cost; subsequent runs reuse the cache under
# {workspace_root}/.cache.
echo "[$(date)] starting vllm serve ({model_path}) on port $VLLM_PORT"
uv run --extra vllm vllm serve {model_path} \\
    --host 0.0.0.0 --port $VLLM_PORT \\
    --max-model-len {vllm_max_model_len} \\
    --max-num-seqs {vllm_max_num_seqs} \\
    --enable-prefix-caching \\
{vllm_extra_flags}    > {logs_dir}/vllm_{job_name}_${{SLURM_JOB_ID}}.log 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

# Wait for /health. DWS node throughput varies dramatically: the
# 8-GPU/96-CPU boxes (dws-14..17) complete cold start in ~7 min, but
# the 2-GPU/40-CPU box (dws-10) needs 20+ min for the same work (job
# 248266 tripped a 20-min budget mid CUDA-graph capture; model load
# was 153s on dws-10 vs 38s on dws-16, warmup 210s vs 60s). 30 min
# covers the slow-node worst case; warm reruns on fast nodes still
# respond well under 10 min.
echo "[$(date)] waiting for vllm /health on port $VLLM_PORT (up to 30 min)"
for i in $(seq 1 1800); do
    if curl -sf http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "[$(date)] vllm ready after ${{i}}s"
        break
    fi
    sleep 1
    if [ "$i" -eq 1800 ]; then
        echo "ERROR: vllm did not become healthy in 30 minutes" >&2
        exit 1
    fi
done

# Expose the endpoint to the smoke command (see bwUC3 template).
export VLLM_BASE_URL=http://localhost:$VLLM_PORT/v1
"""


_BODY = """
echo "[$(date)] running benchmark_models (variant={variant}, language={lang}, kg={kg}{sfid_descr})"
uv run python -m benchmark_models smoke \\
    --variant {variant} \\
    --benchmark {benchmark_path} \\
    --language {lang} \\
    --output {output_path} \\
{extra_flags}    --batch-size {batch_size}

echo "[$(date)] benchmark_models done"
"""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _is_vllm_backend(config: VariantConfig) -> bool:
    return config.backend == "vllm_http"


def _vllm_extra_serve_flags(config: VariantConfig) -> str:
    """Build the trailing ``vllm serve`` flags from per-variant config.

    Each flag is emitted on its own line with the 4-space indent +
    trailing backslash used by the surrounding template. Flags whose
    value is ``None`` are omitted (e.g. ``--quantization`` is only
    emitted when ``quantization`` is set on the config). Flags with
    a non-None scalar value are always emitted so the SLURM template
    is explicit about ``dtype`` / ``gpu_memory_utilization`` rather
    than relying on vLLM defaults that vary across versions and GPUs.
    """
    import json

    flags: list[str] = []
    dtype = getattr(config, "dtype", None)
    if dtype:
        flags.append(f"    --dtype {dtype}")
    quantization = getattr(config, "quantization", None)
    if quantization:
        flags.append(f"    --quantization {quantization}")
    gmu = getattr(config, "gpu_memory_utilization", None)
    if gmu is not None:
        flags.append(f"    --gpu-memory-utilization {gmu}")
    # YaRN / linear / dynamic rope-scaling — passed via ``--hf-overrides``
    # rather than ``--rope-scaling``. vLLM's CLI ``--rope-scaling`` is
    # available only in some versions; ``--hf-overrides`` has been the
    # canonical, version-stable way to override HuggingFace config.json
    # values at vLLM load time since very early releases. We wrap the
    # rope-scaling dict under the ``rope_scaling`` key so vLLM treats
    # the model as if its config.json carried this entry — vLLM's
    # standard rope-scaling path then picks it up automatically.
    # Empty dicts treated as "not set" so YAML mis-edits don't emit
    # ``--hf-overrides '{"rope_scaling": {}}'``.
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling:
        hf_overrides = {"rope_scaling": rope_scaling}
        flags.append(
            f"    --hf-overrides '{json.dumps(hf_overrides, separators=(',', ':'))}'"
        )

    # Multimodal serving: allow exactly one image per prompt for
    # ``-image`` variants. vLLM defaults vary across model families and
    # vLLM versions (some default to 0, blocking all image inputs);
    # being explicit here guarantees the multimodal request format the
    # OpenAI backend emits (``image_url`` block per ``Example.images``)
    # is accepted by the server. Skip for ``multimodal_mode in {"off",
    # "name"}`` — those don't send image data.
    #
    # vLLM 0.20+ requires the JSON dict form (``{"image": 1}``); the
    # older ``image=1`` key-value form is rejected with
    # ``Value image=1 cannot be converted to <function loads at ...>``
    # because the CLI parser runs ``json.loads`` on the argument.
    multimodal_mode = getattr(config, "multimodal_mode", "off")
    if multimodal_mode == "image":
        flags.append("    --limit-mm-per-prompt '{\"image\":1}'")
    # Each flag on its own line, trailing backslash. The template's
    # next line is the redirect (``> ... 2>&1 &``), so the LAST flag
    # must also terminate with a backslash. Empty list (no extra
    # flags) returns ``""``, which slots cleanly under the
    # ``--enable-prefix-caching`` line.
    return "".join(f"{flag} \\\n" for flag in flags)


def _default_granularity(config: VariantConfig) -> Granularity:
    return "sfid" if config.tier == "tool_agent" else "cell"


# Per-(sub_tier, backend) fan-out budgets for tier-3 / oracle-t3 runs
# where we *can* batch (no shared Spark/Sedona bottleneck). Picked
# conservatively so that:
#   * API backends stay under provider RPM at ``K × max_steps``
#     concurrent in-flight requests (K=4, max_steps≤10 → ~40 RPM).
#   * vLLM serves benefit from continuous batching at ~8 concurrent
#     generations; well under the SLURM template's
#     ``--max-num-seqs 64`` ceiling.
# Tier-3a / 3c are deliberately excluded — the shared SparkSession on
# ``local[*]`` with a 4 GB driver heap serialises queries and OOMs
# under concurrent broad scans (see executor_bridge.py:166).
_TIER3_FANOUT_BATCH_SIZE_VLLM = 16
_TIER3_FANOUT_BATCH_SIZE_API = 8
_TIER3_FANOUT_MAX_CONCURRENT_VLLM = 8
_TIER3_FANOUT_MAX_CONCURRENT_API = 4


def _default_batch_size(
    config: VariantConfig, *, oracle_mode: str | None = None
) -> int:
    """Pick the runner ``--batch-size`` argument for one cell.

    Mirrors the runner-side resolver in
    ``benchmark_models.runner.resolve_default_batch_size`` for tier-1
    and tier-2, but discriminates further on sub-tier and oracle mode
    for tier-3:

    * **Oracle-t3** (any sub-tier): the agent loses every tool and
      becomes a pure LLM → Python sandbox loop. No KG / web I/O →
      safe to fan out. ``16`` on vLLM (continuous batching), ``8`` on
      API backends (provider RPM).
    * **Tier-3b** (web + Python, no KG): HTTP-bound, no shared
      compute resource → same fan-out budget as oracle-t3.
    * **Tier-3a / 3c** (KG access via Spark/Sedona): shared
      SparkSession on ``local[*]`` master with a 4 GB driver heap.
      Concurrent queries contend for JVM cores and risk OOM on broad
      polygon scans (e.g. OSMH3KG). Stays at ``1`` until the bridge
      grows per-thread Spark sessions.
    * **Tier 1 / 2 vLLM**: honour ``config.batch_size`` (per-variant
      YAML, default 32 on ``VLLMHttpVariantConfig``).
    * **Tier 1 / 2 API**: ``16`` activates ThreadPoolExecutor
      batching in openai / anthropic / google / deepseek backends.
    """
    # Oracle-t3 collapses every sub-tier into "tool-less LLM loop",
    # so even tier-3a / 3c are safe to batch under this mode — the
    # KG bridge is never touched.
    if oracle_mode == "t3":
        return _TIER3_FANOUT_BATCH_SIZE_VLLM if _is_vllm_backend(config) else _TIER3_FANOUT_BATCH_SIZE_API
    if config.tier == "tool_agent":
        if getattr(config, "sub_tier", None) == "3b":
            return _TIER3_FANOUT_BATCH_SIZE_VLLM if _is_vllm_backend(config) else _TIER3_FANOUT_BATCH_SIZE_API
        return 1
    if _is_vllm_backend(config):
        return int(getattr(config, "batch_size", 32) or 32)
    return 16


def _default_max_concurrent(
    config: VariantConfig,
    batch_size: int,
    *,
    oracle_mode: str | None = None,
) -> int | None:
    """Pick the runner ``--max-concurrent`` argument for one cell.

    Returns ``None`` when the flag should be **omitted** from the
    emitted script:

    * Tier 1 / 2: the runner rejects ``--max-concurrent`` (cli.py:480-481).
    * Tier-3 with ``batch_size <= 1``: the runner takes the
      sequential branch (runner.py:430-433) and never enters
      ``predict_batch``, where ``max_concurrent`` gates the
      ThreadPoolExecutor. Emitting the flag would silently have no
      effect.

    Otherwise returns the same backend-keyed budget used by
    :func:`_default_batch_size` so the resolved values agree by
    default (``min(K, B)`` workers in ``predict_batch``).
    """
    is_tier3_like = config.tier == "tool_agent" or oracle_mode == "t3"
    if not is_tier3_like:
        return None
    if batch_size <= 1:
        return None
    return (
        _TIER3_FANOUT_MAX_CONCURRENT_VLLM
        if _is_vllm_backend(config)
        else _TIER3_FANOUT_MAX_CONCURRENT_API
    )


def _enumerate_cells(
    languages: list[str],
    kgs: list[str],
    granularity: Granularity,
    seeds: list[int] | None = None,
) -> list[_Cell]:
    """Expand (lang, kg, [sfid]) × seeds into one cell per job.

    When ``seeds`` is ``None`` or empty, each (lang, kg, [sfid]) produces a
    single cell with ``seed=None`` and no seed segment in the paths. When
    ``seeds`` has one or more values, each (lang, kg, [sfid]) is
    multiplied by the seed list — i.e., ``--seeds 42,43,44`` triples the
    script count and embeds each seed in the output path.
    """
    seed_iter: list[int | None] = list(seeds) if seeds else [None]  # type: ignore[list-item]
    cells: list[_Cell] = []
    for seed in seed_iter:
        for lang in languages:
            for kg in kgs:
                # ``multimodal`` is a slice spanning two real KGs; fan out
                # into one sub-cell per underlying KG so tier-3a/3c jobs
                # receive a valid ``--kg-source``. For every other bucket
                # the dispatch IS the bucket — kg_dispatch stays empty and
                # ``_Cell.kg_source`` falls back to ``kg``.
                dispatch_kgs: tuple[str, ...] = (
                    _MULTIMODAL_DISPATCH_KGS if kg == "multimodal" else ("",)
                )
                for dispatch_kg in dispatch_kgs:
                    if granularity == "cell":
                        cells.append(
                            _Cell(
                                lang=lang, kg=kg, sfid="",
                                seed=seed, kg_dispatch=dispatch_kg,
                            )
                        )
                    else:
                        for sfid in SFIDS_BY_KG[kg]:
                            cells.append(
                                _Cell(
                                    lang=lang, kg=kg, sfid=sfid,
                                    seed=seed, kg_dispatch=dispatch_kg,
                                )
                            )
    return cells


def _benchmark_path(cell: _Cell, size: str = "large") -> str:
    parts = ["benchmark/final", size, cell.kg]
    # Multimodal sub-cells live under ``multimodal/<underlying_kg>/`` —
    # drill in so the runner only loads the rows for this dispatch's
    # KG, not the combined osmh3kg+worldkg slice at ``multimodal/``.
    if cell.kg == "multimodal" and cell.kg_dispatch:
        parts.append(cell.kg_dispatch)
    if cell.sfid:
        parts.append(cell.sfid)
    return "/".join(parts) + "/"


def _mode_tag(
    config: VariantConfig,
    *,
    oracle_mode: str | None,
    use_kg: bool,
) -> str:
    """One-token disambiguator for runs that share (variant, lang, kg, sfid)
    but differ in evaluation regime.

    The bare variant invocation has tag ``""`` (no extra path segment).
    ``--oracle-mode <m>`` tags as ``oracle-<m>``; ``--use-kg`` on a
    tier-3a/3c variant tags as ``usekg``.

    Oracle wins over use-kg when both are passed — the runner short-circuits
    ``use_kg`` whenever ``oracle_mode`` is set (cli.py:387-393), so the two
    can never co-vary on the same invocation, and the tag reflects that.
    """
    if oracle_mode is not None:
        return f"oracle-{oracle_mode}"
    if use_kg and config.tier == "tool_agent" and config.sub_tier in ("3a", "3c"):
        return "usekg"
    return ""


def _output_path(variant: str, cell: _Cell, size: str = "large", mode_tag: str = "") -> str:
    parts = ["results/runs", size, variant]
    if mode_tag:
        parts.append(mode_tag)
    if cell.seed is not None:
        parts.append(f"seed{cell.seed}")
    parts.extend([cell.lang, cell.kg])
    # Mirror the data tree: multimodal sub-cells write under
    # ``<lang>/multimodal/<underlying_kg>/predictions.jsonl`` so the
    # two underlying KGs' outputs stay separated and one job doesn't
    # interleave with the other on disk.
    if cell.kg == "multimodal" and cell.kg_dispatch:
        parts.append(cell.kg_dispatch)
    if cell.sfid:
        parts.append(cell.sfid)
    parts.append("predictions.jsonl")
    return "/".join(parts)


def _multimodal_dispatch_suffix(cell: _Cell) -> str:
    """Filename/job-name segment that disambiguates multimodal sub-cells.

    Returns ``"_<underlying_kg>"`` for multimodal cells (so
    ``eng_multimodal_osmh3kg_usekg.sh`` differs from
    ``eng_multimodal_worldkg_usekg.sh``), empty string otherwise."""
    if cell.kg == "multimodal" and cell.kg_dispatch:
        return f"_{cell.kg_dispatch}"
    return ""


def _job_name(variant: str, cell: _Cell, mode_tag: str = "") -> str:
    mode_part = f"_{mode_tag}" if mode_tag else ""
    seed_part = f"_seed{cell.seed}" if cell.seed is not None else ""
    suffix = f"_{cell.sfid}" if cell.sfid else ""
    dispatch_part = _multimodal_dispatch_suffix(cell)
    return (
        f"bm_{variant}{mode_part}{seed_part}_{cell.lang}_{cell.kg}"
        f"{dispatch_part}{suffix}"
    )


def _script_filename(cell: _Cell, mode_tag: str = "") -> str:
    # The seed is encoded as a parent directory in :func:`write_script`,
    # not as a filename prefix — so a single ``seed42/`` subtree groups
    # all of one seed's scripts together and can be sbatched with a
    # single glob (``jobs/.../<size>/<variant>/seed42/*.sh``).
    suffix = f"_{cell.sfid}" if cell.sfid else ""
    mode_suffix = f"_{mode_tag}" if mode_tag else ""
    dispatch_part = _multimodal_dispatch_suffix(cell)
    return f"{cell.lang}_{cell.kg}{dispatch_part}{suffix}{mode_suffix}.sh"


def _mode_key_for_time_budget(
    config: VariantConfig,
    *,
    oracle_mode: str | None,
    use_kg: bool,
) -> str:
    """Build the ``(tier, mode_key)`` lookup tail used by
    :data:`_PER_QUESTION_COST_S` and :data:`_DEFAULT_CONCURRENCY`.

    The mode_key encodes the load profile of the run rather than the
    variant name — tier-3a with --use-kg and tier-3c with --use-kg
    have similar Spark/Sedona pressure regardless of which open-weights
    or API model sits on top.
    """
    if config.tier == "plain":
        if oracle_mode == "structured":
            return "oracle-structured"
        if oracle_mode == "raw":
            return "oracle-raw"
        if oracle_mode == "verbalized":
            return "oracle-verbalized"
        return "none"
    if config.tier == "reasoning":
        return "none"
    # tier == "tool_agent"
    sub_tier = getattr(config, "sub_tier", None) or "3b"
    if oracle_mode == "t3":
        return f"tier{sub_tier}-oracle"
    if use_kg:
        return f"tier{sub_tier}-usekg"
    return f"tier{sub_tier}"


def _benchmark_size_for_cell(
    cell: _Cell, *, repo_root: Path = REPO_ROOT,
) -> int:
    """Count rows in ``benchmark/raw_multilingual_renumbered/<kg>/<sfid>/<lang>/benchmark.jsonl/``.

    The benchmark.jsonl path is actually a *directory* of Spark-emitted
    ``part-*.json`` shards; one row per JSON object across all shards.

    Returns :data:`_BENCHMARK_SIZE_FALLBACK` when the file isn't on
    disk (e.g., generating from a checkout that hasn't run the
    benchmark build pipeline). Used only when ``--n 0`` is the runner
    sentinel for 'all questions'; positive ``--n`` values bypass this
    entirely.
    """
    if not cell.sfid:
        # Cell-granularity (no SFID filter) — sum across every SFID for
        # this (lang, kg). Slow only if hundreds of SFIDs exist; in
        # practice each KG has <20.
        kg_root = repo_root / "benchmark" / "raw_multilingual_renumbered" / cell.kg
        if not kg_root.is_dir():
            return _BENCHMARK_SIZE_FALLBACK
        total = 0
        for sfid_dir in kg_root.glob("sfid_*"):
            shard_dir = sfid_dir / cell.lang / "benchmark.jsonl"
            if not shard_dir.is_dir():
                continue
            for shard in shard_dir.glob("part-*.json"):
                with shard.open() as f:
                    total += sum(1 for _ in f)
        return total or _BENCHMARK_SIZE_FALLBACK

    shard_dir = (
        repo_root
        / "benchmark"
        / "raw_multilingual_renumbered"
        / cell.kg
        / cell.sfid
        / cell.lang
        / "benchmark.jsonl"
    )
    if not shard_dir.is_dir():
        return _BENCHMARK_SIZE_FALLBACK
    total = 0
    for shard in shard_dir.glob("part-*.json"):
        with shard.open() as f:
            total += sum(1 for _ in f)
    return total or _BENCHMARK_SIZE_FALLBACK


def _round_up_to_half_hour(seconds: float) -> int:
    """Round seconds up to the next 30-minute boundary, returning
    minutes. Minimum is 30 min — even a 1-minute job should reserve
    a half-hour SLURM slot to absorb startup variance.
    """
    minutes = math.ceil(seconds / 60)
    rounded = 30 * math.ceil(minutes / 30)
    return max(30, rounded)


def _format_hms(total_minutes: int) -> str:
    """Format minutes as ``HH:MM:SS`` for SLURM ``--time``."""
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:02d}:{minutes:02d}:00"


def _compute_time_budget(
    config: VariantConfig,
    cell: _Cell,
    *,
    variant: str,
    n: int,
    cluster: Cluster,
    is_vllm: bool,
    oracle_mode: str | None,
    use_kg: bool,
    max_concurrent: int | None,
) -> str:
    """Compute SLURM ``--time`` from per-question cost + cold-start.

    Algorithm::

        wall_s = (n / concurrency) * per_q_cost + cold_start_overhead + 30 min
        minutes = ceil(wall_s / 60)
        rounded = next multiple of 30 minutes above ``minutes`` (floor 30)
        if rounded > 24 h: clip to 24 h, logger.warning

    Returns the ``HH:MM:SS`` string used directly by the SLURM
    ``--time`` directive in the header templates.
    """
    if n <= 0:
        n = _benchmark_size_for_cell(cell)

    mode_key = _mode_key_for_time_budget(
        config, oracle_mode=oracle_mode, use_kg=use_kg,
    )
    per_q = _PER_QUESTION_COST_S.get((config.tier, mode_key))
    if per_q is None:
        # Unknown mode for the cost table — fall back to a generous
        # per-question estimate that won't undersize but will likely
        # over-budget. Better to clip than to die at 30 min.
        logger.warning(
            "No per-question cost for (tier={!r}, mode={!r}); using "
            "60 s/q fallback. Consider extending _PER_QUESTION_COST_S.",
            config.tier, mode_key,
        )
        per_q = 60

    concurrency = (
        max_concurrent
        if max_concurrent is not None and max_concurrent > 0
        else _DEFAULT_CONCURRENCY.get((config.tier, mode_key), 4)
    )

    cold_start = _COLD_START_OVERHEAD_S[(cluster, is_vllm)]
    wall_s = (n / concurrency) * per_q + cold_start + _TIME_BUFFER_S
    rounded_min = _round_up_to_half_hour(wall_s)

    if rounded_min > _PARTITION_WALL_CAP_MIN:
        cell_id = f"{variant}/{cell.lang}/{cell.kg}/{cell.sfid or 'cell'}"
        logger.warning(
            "Time budget for {} would be {:d}h{:02d}m at --n={} but partition "
            "wall cap is {}h. Clipping to cap. Rely on resume (default in "
            "runner.run) and re-submit the same script after the wall — the "
            "second run will pick up only the remaining questions. To avoid "
            "the resubmission, lower --n or override with --time-budget.",
            cell_id,
            rounded_min // 60, rounded_min % 60, n,
            _PARTITION_WALL_CAP_MIN // 60,
        )
        rounded_min = _PARTITION_WALL_CAP_MIN

    return _format_hms(rounded_min)


def _extra_flags(
    config: VariantConfig,
    *,
    cell: "_Cell",
    oracle_mode: str | None,
    use_kg: bool,
    no_mlflow: bool,
    retry_errors: bool = False,
    max_concurrent: int | None = None,
    n: int = 0,
    max_steps: int | None = None,
    wkt_simplify_tolerance: float | None = None,
    gold_triples_path: str | None = None,
) -> str:
    flags: list[str] = []
    # ``--n 0`` is the runner's sentinel for "process every row in the
    # shard" (the right default for full eval runs). A positive integer
    # caps rows per cell — useful for limited-row smoke tests via
    # SLURM, since vLLM serves are only reachable inside a job
    # allocation and a direct ``python -m benchmark_models smoke``
    # against vLLM needs the serve running anyway.
    flags.append(f"    --n {n}")
    if oracle_mode is not None:
        flags.append(f"    --oracle-mode {oracle_mode}")
        # Scope gold triples to THIS cell's size+path so the run is
        # self-contained (a ``small`` oracle job reads ``small`` gold,
        # not the whole ``large`` tree). Only read by the runner when
        # --oracle-mode is set; omitted otherwise.
        if gold_triples_path is not None:
            flags.append(f"    --gold-triples {gold_triples_path}")
    if use_kg and config.tier == "tool_agent" and config.sub_tier in ("3a", "3c"):
        flags.append("    --use-kg")
        # Pass --kg-source so the bridge loads the right
        # benchmark/kgs/<kg>/ Delta table for this cell. For multimodal
        # sub-cells ``cell.kg_source`` resolves to the underlying KG
        # (``osmh3kg`` / ``worldkg``) rather than the slice name
        # ``"multimodal"`` — which would otherwise raise
        # ``ValueError: Unknown KG`` in ``schema_for``.
        flags.append(f"    --kg-source {cell.kg_source}")
    if max_concurrent is not None:
        # Only relevant for tier-3 + oracle-t3 runs with batch_size > 1
        # (see :func:`_default_max_concurrent`). The runner rejects this
        # flag on tier-1 / 2; the caller is responsible for not passing
        # it through in those cases.
        flags.append(f"    --max-concurrent {max_concurrent}")
    if max_steps is not None:
        # Only relevant for tier-3 + oracle-t3 runs (the runner rejects
        # the flag on tier-1 / 2). When set, propagated both into the
        # agent loop's step cap and the prompt's ``$max_tool_calls``
        # slot so prompt + agent never disagree. Emitting it
        # explicitly even when it matches the catalogue default makes
        # the SLURM script self-documenting about the headline-run
        # cap rather than relying on the YAML default to track over
        # time.
        flags.append(f"    --max-steps {max_steps}")
    if oracle_mode is not None:
        # Always emit --wkt-simplify-tolerance for oracle modes so the
        # SLURM body documents the value used; uniform across all
        # backends so cross-model oracle comparison stays
        # apples-to-apples. When the caller didn't pass an explicit
        # tolerance, use 0.001 — the same default as
        # ``OracleContext.wkt_simplify_tolerance`` and
        # ``EntityService.wkt_simplify_tolerance``, so simplification
        # is on by default but the SLURM body still shows it.
        # Pass ``--wkt-simplify-tolerance 0`` to disable for an
        # ablation that quantifies the simplifier's accuracy impact.
        effective = (
            wkt_simplify_tolerance
            if wkt_simplify_tolerance is not None
            else 0.001
        )
        flags.append(f"    --wkt-simplify-tolerance {effective}")
    if cell.seed is not None:
        # ``--seed`` is recorded on every Prediction. Honored by OpenAI /
        # DeepSeek / vLLM at the API level; Anthropic and Google ignore
        # the parameter but the per-call sampling is still stochastic
        # under ``temperature > 0``, so multi-seed runs give independent
        # samples regardless. Storing the seed in the path makes the
        # provenance unambiguous on disk either way.
        flags.append(f"    --seed {cell.seed}")
    if no_mlflow:
        flags.append("    --no-mlflow")
    if retry_errors:
        # Re-run only rows whose prior prediction had final_state=='error'
        # (transient API failures, e.g. 429 RESOURCE_EXHAUSTED quota). The
        # runner drops those rows from the existing predictions.jsonl, then
        # the question_id-based resume re-runs just their ids; non-errored
        # rows are kept untouched. No-op against a clean (error-free) cell.
        flags.append("    --retry-errors")
    # Each flag on its own line, trailing backslash. Caller's template
    # already adds ``--batch-size`` on the next line, so don't terminate
    # the last extra flag with a backslash.
    return "".join(f"{flag} \\\n" for flag in flags)


# Header template registry: (cluster, is_vllm) → template string.
_HEADER_TEMPLATES: dict[tuple[Cluster, bool], str] = {
    ("bwuc3", False): _API_HEADER_BWUC3,
    ("bwuc3", True): _VLLM_HTTP_HEADER_BWUC3,
    ("helix", False): _API_HEADER_HELIX,
    ("helix", True): _VLLM_HTTP_HEADER_HELIX,
    ("dws", False): _API_HEADER_DWS,
    ("dws", True): _VLLM_HTTP_HEADER_DWS,
}


def render_script(
    variant: str,
    config: VariantConfig,
    cell: _Cell,
    *,
    granularity: Granularity,
    repo_root: Path = REPO_ROOT,
    logs_dir: Path | None = None,
    oracle_mode: str | None = None,
    use_kg: bool = False,
    no_mlflow: bool = False,
    retry_errors: bool = False,
    cluster: Cluster = "bwuc3",
    workspace_root: str | None = None,
    batch_size: int | None = None,
    max_concurrent: int | None = None,
    n: int = 0,
    max_steps: int | None = None,
    wkt_simplify_tolerance: float | None = None,
    time_budget_override: str | None = None,
    size: str = "large",
) -> str:
    """Render the full SLURM script body for one cell.

    ``cluster`` picks the partition / GPU type / workspace path bundle.
    Default ``bwuc3`` emits bwUniCluster 3.0 scripts; pass ``"helix"``
    to emit Helix-compatible scripts (Singularity-wrapped vllm serve,
    H200 GPU type, cpu-single partition for API runs).

    ``workspace_root`` overrides the per-cluster default scratch root
    (used to anchor ``.cache/torchinductor``, ``.cache/vllm`` and the
    ``.venv-vllm`` activation path). When ``None`` the cluster default
    from ``_CLUSTER_WORKSPACE_ROOT`` is used.

    ``batch_size`` / ``max_concurrent``, when set, override the
    resolver. ``max_concurrent`` is silently dropped for non-tier-3 /
    non-oracle-t3 variants (the runner rejects it in those cases).
    """
    if logs_dir is None:
        logs_dir = JOBS_ROOT / variant / "logs"
    is_vllm = _is_vllm_backend(config)
    header_template = _HEADER_TEMPLATES[(cluster, is_vllm)]
    if time_budget_override is not None:
        time_budget = time_budget_override
    else:
        time_budget = _compute_time_budget(
            config, cell,
            variant=variant,
            n=n,
            cluster=cluster,
            is_vllm=is_vllm,
            oracle_mode=oracle_mode,
            use_kg=use_kg,
            max_concurrent=max_concurrent,
        )
    mode_tag = _mode_tag(config, oracle_mode=oracle_mode, use_kg=use_kg)
    job_name = _job_name(variant, cell, mode_tag)
    header_kwargs: dict[str, Any] = {
        "time_budget": time_budget,
        "job_name": job_name,
        "logs_dir": logs_dir,
        "repo_root": repo_root,
        "cpu_partition": _CLUSTER_CPU_PARTITION[cluster],
        "gpu_partition": _CLUSTER_GPU_PARTITION[cluster],
        "gpu_gres": _CLUSTER_GPU_GRES[cluster],
        "workspace_root": workspace_root or _CLUSTER_WORKSPACE_ROOT[cluster],
    }
    if is_vllm:
        # vLLM-HTTP templates need the model path + serve port etc.
        # ``model_path`` and friends are documented on
        # ``VLLMHttpVariantConfig`` in benchmark_models/config.py.
        # ``slurm_mem_gb`` defaults to 80 on the config; YaRN-extended
        # variants raise to 128 to absorb the larger KV-cache CPU
        # shadow (see ``_qwen35_27b_yarn_64k`` anchor in
        # ``models.yaml``).
        header_kwargs.update(
            model_path=getattr(config, "model_path"),
            vllm_port=getattr(config, "vllm_port", 8000),
            vllm_max_model_len=getattr(config, "max_model_len", 8192),
            vllm_max_num_seqs=getattr(config, "max_num_seqs", 64),
            vllm_extra_flags=_vllm_extra_serve_flags(config),
            mem_gb=getattr(config, "slurm_mem_gb", 80),
        )
    header = header_template.format(**header_kwargs)
    sfid_descr = f", sfid={cell.sfid}" if cell.sfid else ""
    resolved_batch_size = (
        batch_size
        if batch_size is not None
        else _default_batch_size(config, oracle_mode=oracle_mode)
    )
    # ``max_concurrent`` only meaningful for tier-3-like runs *and* when
    # the resolved batch_size > 1 (the runner sequential branch at
    # runner.py:430-433 never enters ``predict_batch`` otherwise). Drop
    # silently when it isn't applicable so callers can pass the same
    # override across variants without per-tier guards.
    is_tier3_like = config.tier == "tool_agent" or oracle_mode == "t3"
    if max_concurrent is not None and (not is_tier3_like or resolved_batch_size <= 1):
        resolved_max_concurrent: int | None = None
    elif max_concurrent is not None:
        resolved_max_concurrent = max_concurrent
    else:
        resolved_max_concurrent = _default_max_concurrent(
            config, resolved_batch_size, oracle_mode=oracle_mode
        )
    # ``max_steps`` is also tier-3-only. The runner rejects the flag
    # on tier-1 / 2 (cli.py:480-481), so suppress an override that
    # would land on a non-tier-3 variant rather than baking a flag
    # the job would reject at startup.
    if max_steps is not None and not is_tier3_like:
        resolved_max_steps: int | None = None
    else:
        resolved_max_steps = max_steps
    benchmark_path = _benchmark_path(cell, size)
    body = _BODY.format(
        variant=variant,
        lang=cell.lang,
        kg=cell.kg,
        sfid_descr=sfid_descr,
        benchmark_path=benchmark_path,
        output_path=_output_path(variant, cell, size=size, mode_tag=mode_tag),
        batch_size=resolved_batch_size,
        extra_flags=_extra_flags(
            config,
            cell=cell,
            oracle_mode=oracle_mode,
            use_kg=use_kg,
            no_mlflow=no_mlflow,
            retry_errors=retry_errors,
            max_concurrent=resolved_max_concurrent,
            n=n,
            max_steps=resolved_max_steps,
            wkt_simplify_tolerance=wkt_simplify_tolerance,
            gold_triples_path=benchmark_path,
        ),
    )
    return header + body


def write_script(
    variant: str,
    config: VariantConfig,
    cell: _Cell,
    *,
    granularity: Granularity,
    out_root: Path = JOBS_ROOT,
    oracle_mode: str | None = None,
    use_kg: bool = False,
    no_mlflow: bool = False,
    retry_errors: bool = False,
    cluster: Cluster = "bwuc3",
    workspace_root: str | None = None,
    batch_size: int | None = None,
    max_concurrent: int | None = None,
    n: int = 0,
    max_steps: int | None = None,
    wkt_simplify_tolerance: float | None = None,
    time_budget_override: str | None = None,
    size: str = "large",
) -> Path:
    """Write one SLURM script for ``cell`` under ``out_root/<size>/<variant>/``.

    When ``cell.seed`` is set, the script lives one level deeper under
    ``<size>/<variant>/seed<N>/`` so each seed's scripts group together and
    can be submitted with a single glob:

        for s in jobs/benchmark_models/<size>/<variant>/seed42/*.sh; do
            sbatch "$s"
        done

    Logs stay shared at ``<size>/<variant>/logs/`` regardless of seed — the
    SLURM job name already embeds the seed so log filenames don't
    collide across seeds.
    """
    variant_dir = out_root / size / variant
    logs_dir = variant_dir / "logs"
    script_dir = variant_dir if cell.seed is None else variant_dir / f"seed{cell.seed}"
    script_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = _mode_tag(config, oracle_mode=oracle_mode, use_kg=use_kg)
    script_path = script_dir / _script_filename(cell, mode_tag)
    text = render_script(
        variant,
        config,
        cell,
        granularity=granularity,
        logs_dir=logs_dir,
        oracle_mode=oracle_mode,
        use_kg=use_kg,
        no_mlflow=no_mlflow,
        retry_errors=retry_errors,
        cluster=cluster,
        workspace_root=workspace_root,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        n=n,
        max_steps=max_steps,
        wkt_simplify_tolerance=wkt_simplify_tolerance,
        time_budget_override=time_budget_override,
        size=size,
    )
    script_path.write_text(text, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", required=True,
        help="Variant name from models.yaml (e.g., 'gemini-3-flash-plain').",
    )
    parser.add_argument(
        "--languages", default=",".join(TARGET_LANGUAGES),
        help=f"Comma-separated ISO 639-3 language codes "
             f"(default: all {len(TARGET_LANGUAGES)} target languages).",
    )
    parser.add_argument(
        "--kgs", default=",".join(KG_BUCKETS),
        help=f"Comma-separated KG buckets (default: {','.join(KG_BUCKETS)}).",
    )
    parser.add_argument(
        "--size", choices=("large", "small"), default="large",
        help="Benchmark scope to target. 'large' (default) = the full "
             "benchmark/final/large/ tree; 'small' = the canonical "
             "benchmark/final/small/ sub-sample. Selects both the "
             "--benchmark / --gold-triples data root and the "
             "results/runs/<size>/ output sub-tree.",
    )
    parser.add_argument(
        "--granularity", choices=("cell", "sfid"), default=None,
        help="Job granularity. Default depends on tier: 'cell' for "
             "tier 1/2, 'sfid' for tier 3.",
    )
    parser.add_argument(
        "--oracle-mode", default=None,
        choices=("structured", "raw", "verbalized", "t3"),
        help="Add --oracle-mode to each emitted job.",
    )
    parser.add_argument(
        "--use-kg", action="store_true",
        help="Add --use-kg to tier-3a/3c jobs (real Spark/Sedona KG access).",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Add --no-mlflow to each emitted job.",
    )
    parser.add_argument(
        "--retry-errors", action="store_true",
        help="Add --retry-errors to each emitted job: on resume, drop "
             "rows whose prior prediction had final_state=='error' "
             "(transient API failures, e.g. 429 quota) and re-run only "
             "those question_ids. No-op against a clean cell. Use to "
             "recover infra-errored rows without re-running the whole cell.",
    )
    parser.add_argument(
        "--n", type=int, default=0,
        help="Row limit per emitted script (forwarded to the runner's "
             "``--n`` flag). ``0`` (default) is the runner's sentinel "
             "for 'process every row in the shard' — the right value "
             "for a full eval. A positive integer caps rows per cell, "
             "useful for limited-row smokes via SLURM (vLLM serves are "
             "only reachable inside a job allocation, so the smoke "
             "needs to go through sbatch).",
    )
    parser.add_argument(
        "--out-root", type=Path, default=JOBS_ROOT,
        help="Override the output directory (default: jobs/benchmark_models/).",
    )
    parser.add_argument(
        "--seeds", default=None,
        help="Comma-separated list of integer seeds for multi-run "
             "experiments (e.g. '42,43,44'). When given, each "
             "(lang, kg[, sfid]) cell is multiplied by the seed list: "
             "3 seeds × 17 langs × N cells = 51N scripts. Output paths "
             "embed the seed (``results/runs/<size>/<variant>/seed<N>/...``) "
             "so seed-different runs don't collide. The runner is "
             "invoked with ``--seed <N>`` per script. When omitted, a "
             "single script per cell is emitted (current behaviour) "
             "and no seed flag is passed.",
    )
    parser.add_argument(
        "--cluster", choices=_CLUSTERS, default="bwuc3",
        help="Target SLURM cluster. ``bwuc3`` (default) emits scripts "
             "for bwUniCluster 3.0 (partitions: gpu_h100 / single, H100 "
             "GPUs). ``helix`` emits Helix scripts (partitions: "
             "gpu-single / cpu-single, H200 GPUs, vLLM wrapped in "
             "Singularity to bypass the glibc 2.28 host constraint).",
    )
    parser.add_argument(
        "--workspace-root", default=None,
        help="Override the per-cluster scratch root used to anchor "
             "``.cache/torchinductor``, ``.cache/vllm`` and the "
             "``.venv-vllm`` activation path. Defaults: "
             f"{', '.join(f'{c}={p}' for c, p in _CLUSTER_WORKSPACE_ROOT.items())}.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override the runner's ``--batch-size``. When omitted, "
             "the resolver picks a sub-tier-aware default: tier 1/2 "
             "vLLM honors models.yaml (32 by default, 64 for "
             "qwen35-a3b); tier 1/2 API uses 16; tier-3a/3c uses 1 "
             "(Spark contention); tier-3b and oracle-t3 use "
             f"{_TIER3_FANOUT_BATCH_SIZE_VLLM} on vLLM / "
             f"{_TIER3_FANOUT_BATCH_SIZE_API} on API to fan out under "
             "predict_batch.",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=None,
        help="Override the runner's tier-3 ``--max-concurrent``. "
             "Silently dropped on non-tier-3 / non-oracle-t3 variants "
             "(the runner rejects it there) and on tier-3 cells whose "
             "resolved batch_size is 1 (predict_batch is never "
             "invoked at runner.py:430-433 in that case, so the flag "
             f"would no-op). Default when emitted: "
             f"{_TIER3_FANOUT_MAX_CONCURRENT_VLLM} on vLLM / "
             f"{_TIER3_FANOUT_MAX_CONCURRENT_API} on API.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override the runner's tier-3 ``--max-steps`` (a.k.a. "
             "``max_tool_calls``). When set, the emitted script "
             "carries ``--max-steps N`` explicitly so the headline "
             "cap is self-documenting in the SLURM body rather than "
             "drifting with the catalogue's DEFAULT_MAX_TOOL_CALLS "
             "over time. Silently dropped on non-tier-3 / "
             "non-oracle-t3 variants (the runner rejects it there).",
    )
    parser.add_argument(
        "--wkt-simplify-tolerance", type=float, default=None,
        help="Douglas-Peucker tolerance (lat/lon degrees) applied to "
             "WKT geometries in oracle gold contexts. When set with "
             "an --oracle-mode, the SLURM body carries the value so "
             "the simplification is uniform across the sweep and "
             "auditable from the script. Only emitted when "
             "--oracle-mode is also set; ignored otherwise.",
    )
    parser.add_argument(
        "--time-budget", default=None,
        help="Override the computed SLURM #SBATCH --time value for "
             "every emitted script. Format ``HH:MM:SS``. When omitted, "
             "the generator computes a per-cell budget from observed "
             "per-question costs scaled by --n and concurrency, plus "
             "cluster cold-start and a 30-min safety buffer, rounded "
             "up to the next 30-min boundary, clipped at 24 h with a "
             "logger.warning (resume picks up after the wall). Pass "
             "this flag only when the heuristic mis-sizes a specific "
             "sweep (e.g., a known network-slow cluster or a model "
             "with unusually long warmup).",
    )
    args = parser.parse_args()

    catalogue = load_models_catalogue()
    if args.variant not in catalogue:
        logger.error("Unknown variant: {}", args.variant)
        return 1
    config = catalogue.get(args.variant)
    granularity: Granularity = args.granularity or _default_granularity(config)

    languages = [s.strip() for s in args.languages.split(",") if s.strip()]
    kgs = [s.strip() for s in args.kgs.split(",") if s.strip()]
    for lang in languages:
        if lang not in TARGET_LANGUAGES:
            logger.warning("Unknown language: {} (will still be emitted)", lang)
    for kg in kgs:
        if kg not in KG_BUCKETS:
            logger.error("Unknown kg: {}", kg)
            return 1

    seeds: list[int] | None = None
    if args.seeds:
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        except ValueError as exc:
            logger.error("--seeds must be a comma-separated list of integers: {}", exc)
            return 1
        if not seeds:
            logger.error("--seeds was passed but contained no integers")
            return 1

    cells = _enumerate_cells(languages, kgs, granularity, seeds=seeds)
    n_written = 0
    for cell in cells:
        script = write_script(
            args.variant, config, cell,
            granularity=granularity,
            out_root=args.out_root,
            workspace_root=args.workspace_root,
            oracle_mode=args.oracle_mode,
            use_kg=args.use_kg,
            no_mlflow=args.no_mlflow,
            retry_errors=args.retry_errors,
            cluster=args.cluster,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            max_steps=args.max_steps,
            wkt_simplify_tolerance=args.wkt_simplify_tolerance,
            n=args.n,
            time_budget_override=args.time_budget,
            size=args.size,
        )
        logger.info("Wrote {}", script)
        n_written += 1

    logger.info(
        "Done. {} script(s) under {} (granularity={}, backend={}).",
        n_written, args.out_root / args.variant, granularity, config.backend,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
