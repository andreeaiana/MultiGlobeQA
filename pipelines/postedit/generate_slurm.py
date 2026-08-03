"""Emit per-(language, kg) SLURM scripts for the post-edit pipeline.

Produces one script per (lang, kg) combo. Output directory depends on
``--cluster``:

* ``--cluster bwuc3`` (default) — bwUniCluster 3.0 H100. Scripts go to
  ``jobs/postedit/<lang>_<kg>.sh``. vLLM runs from a dedicated
  ``.venv-vllm`` on the workspace, with a system CUDA module loaded.
* ``--cluster helix`` — Helix H200. Scripts go to
  ``jobs/postedit_helix/<lang>_<kg>.sh``. vLLM runs inside a
  Singularity container (Helix is RHEL 8.10 / glibc 2.28, too old for
  any vLLM PyPI wheel — all require glibc >=2.31). Requires ``SBR_WS``
  and ``SBR_VLLM_SIF`` exported in ``~/.bashrc`` — see
  ``scripts/postedit/README.md`` for the one-time setup.

Each script runs vLLM + the postedit pipeline on one GPU, processing
all SFIDs of the chosen (lang, kg). The user submits whichever subset
they want; this generator does **not** ship a "submit-everything"
helper — intentional, so nothing floods the scheduler or the upstream
LLM rate limits.

Usage::

    uv run python scripts/postedit/generate_slurm.py
    uv run python scripts/postedit/generate_slurm.py --cluster helix
    uv run python scripts/postedit/generate_slurm.py --cluster helix --partition gpu-single
    uv run python scripts/postedit/generate_slurm.py --languages deu,fra,ita
    uv run python scripts/postedit/generate_slurm.py --kgs kwg,worldkg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from pipelines.postedit.config import KG_BUCKETS, TARGET_LANGUAGES


REPO_ROOT = Path(__file__).resolve().parents[2]


# Per-KG SLURM time budget. Empirically: ~10 questions/sec at
# concurrency=16 on one H100, plus ~5 min vLLM cold-start (longer on
# first run when the model has to download).
_KG_TIME_BUDGET = {
    "kwg": "03:00:00",
    "osmh3kg": "06:00:00",     # largest bucket
    "worldkg": "04:00:00",
    "multimodal": "00:30:00",  # smallest
}


_CLUSTERS = ("bwuc3", "helix")

_DEFAULT_PARTITION = {
    "bwuc3": "gpu_h100",
    "helix": "gpu-single",
}

# Helix exposes multiple GPU types on the same partition, so the type
# must be pinned explicitly. bwUC3's gpu_h100 is single-type, so plain
# gpu:1 is sufficient there.
_GRES = {
    "bwuc3": "gpu:1",
    "helix": "gpu:H200:1",
}

_JOBS_SUBDIR = {
    "bwuc3": "postedit",
    "helix": "postedit_helix",
}


_BWUC3_TEMPLATE = """#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --gres={gres}
#SBATCH --time={time_budget}
#SBATCH --mem=80gb
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --job-name=postedit_{lang}_{kg}
#SBATCH --output={logs_dir}/postedit_{lang}_{kg}_%j.out
#SBATCH --error={logs_dir}/postedit_{lang}_{kg}_%j.err

source $HOME/.bashrc
set -euo pipefail
cd {repo_root}

# Source API keys (file is gitignored; create from scripts/set_api_keys.sh.example).
if [ -f scripts/set_api_keys.sh ]; then
    source scripts/set_api_keys.sh
else
    echo "ERROR: scripts/set_api_keys.sh not found. Copy from .example and fill in keys." >&2
    exit 1
fi

# CUDA toolkit required by vLLM JIT (Qwen 27B FP8). Fail-fast if the
# module isn't available — vLLM would otherwise crash with a confusing
# error 5+ minutes into the cold-start.
module load devel/cuda/12.8

# Pin torch.compile / vLLM caches to the workspace so they survive job
# boundaries. SLURM purges $TMPDIR (/scratch/slurm_tmpdir/job_<id>) on job
# exit, and inductor bakes absolute paths into its cache — leaving stale
# references that PermissionError on the next run.
export TORCHINDUCTOR_CACHE_DIR=/pfs/work9/workspace/scratch/ma_aiana-multiglobeqa/.cache/torchinductor
export VLLM_CACHE_ROOT=/pfs/work9/workspace/scratch/ma_aiana-multiglobeqa/.cache/vllm
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

# Start vLLM in background.
echo "[$(date)] starting vLLM (Qwen3.5-27B-FP8) on port 8000"
source /pfs/work9/workspace/scratch/ma_aiana-multiglobeqa/.venv-vllm/bin/activate
vllm serve Qwen/Qwen3.5-27B-FP8 \\
    --host 0.0.0.0 --port 8000 \\
    --max-model-len 8192 \\
    --max-num-seqs 64 \\
    --enable-prefix-caching \\
    > {logs_dir}/vllm_{lang}_{kg}_${{SLURM_JOB_ID}}.log 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT
deactivate

# Wait for vLLM /health. First-ever run may pay a 5-10 min HF download
# cost; warm runs are typically ready in <3 min.
echo "[$(date)] waiting for vLLM /health (up to 10 min)"
for i in $(seq 1 600); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "[$(date)] vLLM ready after ${{i}}s"
        break
    fi
    sleep 1
    if [ "$i" -eq 600 ]; then
        echo "ERROR: vLLM did not become healthy in 10 minutes" >&2
        exit 1
    fi
done

# Run the post-edit pipeline.
echo "[$(date)] running postedit pipeline (language={lang}, kg={kg})"
uv run python scripts/postedit/postedit_pipeline.py \\
    --language {lang} --kg {kg} \\
    --benchmark-root benchmark/sampled_multilingual \\
    --question-concurrency 16 \\
    --resume skip-done

echo "[$(date)] postedit done"
"""


_HELIX_TEMPLATE = """#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --gres={gres}
#SBATCH --time={time_budget}
#SBATCH --mem=80gb
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --job-name=postedit_{lang}_{kg}
#SBATCH --output={logs_dir}/postedit_{lang}_{kg}_%j.out
#SBATCH --error={logs_dir}/postedit_{lang}_{kg}_%j.err

# Source bashrc first — /etc/bashrc references unset vars (e.g. BASHRCSOURCED)
# that would trip ``set -u`` if strict mode were already on. Same for conda's
# deactivate hook, which can touch unset variables in its plumbing.
source $HOME/.bashrc
conda deactivate 2>/dev/null || true
set -euo pipefail
cd {repo_root}

# Source API keys (gitignored; create from scripts/set_api_keys.sh.example).
if [ ! -f scripts/set_api_keys.sh ]; then
    echo "ERROR: scripts/set_api_keys.sh not found. Copy from .example and fill in keys." >&2
    exit 1
fi
source scripts/set_api_keys.sh

# Sanity-check the Helix env vars (set once in ~/.bashrc — see
# scripts/postedit/README.md for one-time setup).
: "${{SBR_WS:?ERROR: SBR_WS not set — source ~/.bashrc with Helix env}}"
: "${{SBR_VLLM_SIF:?ERROR: SBR_VLLM_SIF not set — source ~/.bashrc with Helix env}}"
if [ ! -f "$SBR_VLLM_SIF" ]; then
    echo "ERROR: vLLM container missing: $SBR_VLLM_SIF" >&2
    exit 1
fi

# vLLM runs inside a Singularity container — Helix is RHEL 8.10
# (glibc 2.28), too old for any vLLM PyPI/GitHub wheel (all require
# glibc >=2.31). Container ships its own glibc, CUDA toolkit, and Python.
module load system/singularity/3.11.3

mkdir -p "$HF_HOME" "$SBR_WS/.cache/vllm"

VLLM_LOG="{logs_dir}/vllm_{lang}_{kg}_${{SLURM_JOB_ID}}.log"

echo "[$(date)] starting vLLM (Qwen3.5-27B-FP8) in container on $(hostname):8000"
singularity exec --nv \\
  --bind "$HF_HOME:/root/.cache/huggingface" \\
  "$SBR_VLLM_SIF" \\
  vllm serve Qwen/Qwen3.5-27B-FP8 \\
    --host 0.0.0.0 --port 8000 \\
    --max-model-len 8192 \\
    --max-num-seqs 64 \\
    --enable-prefix-caching \\
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

# Wait for vLLM /health. First-ever run may pay a 15 min HF download
# cost (model is ~27 GB); warm runs are typically ready in <3 min.
echo "[$(date)] waiting for vLLM /health (up to 15 min)"
for i in $(seq 1 900); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "[$(date)] vLLM ready after ${{i}}s"
        break
    fi
    sleep 1
    if [ "$i" -eq 900 ]; then
        echo "ERROR: vLLM did not become healthy in 15 minutes" >&2
        tail -100 "$VLLM_LOG" >&2
        exit 1
    fi
done

# Pipeline runs on host under project .venv (uv run); talks HTTP to
# localhost:8000 of the in-container vLLM.
echo "[$(date)] running postedit pipeline (language={lang}, kg={kg})"
uv run python scripts/postedit/postedit_pipeline.py \\
    --language {lang} --kg {kg} \\
    --benchmark-root benchmark/sampled_multilingual \\
    --question-concurrency 16 \\
    --resume skip-done

echo "[$(date)] postedit done"
"""


_TEMPLATES = {
    "bwuc3": _BWUC3_TEMPLATE,
    "helix": _HELIX_TEMPLATE,
}


def _jobs_dir(cluster: str) -> Path:
    return REPO_ROOT / "jobs" / _JOBS_SUBDIR[cluster]


def write_script(
    lang: str,
    kg: str,
    *,
    cluster: str,
    partition: str,
    repo_root: Path,
) -> Path:
    """Write one SLURM script for (lang, kg). Returns the script path."""
    jobs_dir = _jobs_dir(cluster)
    logs_dir = jobs_dir / "logs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    script_path = jobs_dir / f"{lang}_{kg}.sh"
    text = _TEMPLATES[cluster].format(
        lang=lang,
        kg=kg,
        time_budget=_KG_TIME_BUDGET.get(kg, "04:00:00"),
        repo_root=repo_root,
        logs_dir=logs_dir,
        partition=partition,
        gres=_GRES[cluster],
    )
    script_path.write_text(text, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cluster", choices=_CLUSTERS, default="bwuc3",
        help=f"Target cluster (default: bwuc3). One of: {', '.join(_CLUSTERS)}",
    )
    parser.add_argument(
        "--partition", type=str, default=None,
        help="SLURM partition (default: cluster-specific — "
             f"{', '.join(f'{c}={p}' for c, p in _DEFAULT_PARTITION.items())})",
    )
    parser.add_argument(
        "--languages", type=str, default=",".join(TARGET_LANGUAGES),
        help=f"Comma-separated ISO 639-3 language codes "
             f"(default: all {len(TARGET_LANGUAGES)} target languages)",
    )
    parser.add_argument(
        "--kgs", type=str, default=",".join(KG_BUCKETS),
        help=f"Comma-separated KG buckets (default: {','.join(KG_BUCKETS)})",
    )
    args = parser.parse_args()

    partition = args.partition or _DEFAULT_PARTITION[args.cluster]

    languages = [s.strip() for s in args.languages.split(",") if s.strip()]
    kgs = [s.strip() for s in args.kgs.split(",") if s.strip()]
    for lang in languages:
        if lang not in TARGET_LANGUAGES:
            logger.warning("Unknown language: {} (no rider shipped)", lang)
    for kg in kgs:
        if kg not in KG_BUCKETS:
            logger.error("Unknown kg: {}", kg)
            return 1

    n_written = 0
    for lang in languages:
        for kg in kgs:
            script = write_script(
                lang, kg,
                cluster=args.cluster,
                partition=partition,
                repo_root=REPO_ROOT,
            )
            logger.info("Wrote {}", script)
            n_written += 1

    logger.info(
        "Done. {} script(s) under {} (cluster={}, partition={})",
        n_written, _jobs_dir(args.cluster), args.cluster, partition,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
