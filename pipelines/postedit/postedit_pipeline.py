"""LLM ensemble post-editing pipeline.

Reads instantiated benchmark questions from the multilingual
benchmark layout, runs an ensemble of three LLMs to fix minor
grammar/syntax issues caused by entity substitution, votes on the
best correction, and writes structured output JSONL — one file per
(language, kg, sfid) cell.

Usage:
    # Walk every cell of one language under benchmark/sampled_multilingual
    uv run python -m pipelines.postedit.postedit_pipeline --language deu

    # One language + one KG (typical SLURM-job granularity)
    uv run python -m pipelines.postedit.postedit_pipeline \\
        --language deu --kg kwg

    # Just one cell
    uv run python -m pipelines.postedit.postedit_pipeline \\
        --language deu --kg kwg --sfid sfid_a

    # Resume after a kill: skip qids that already have a row
    uv run python -m pipelines.postedit.postedit_pipeline \\
        --language deu --kg kwg --resume skip-done

    # Re-run only rows where any model failed previously
    uv run python -m pipelines.postedit.postedit_pipeline \\
        --language deu --kg kwg --resume retry-failed

    # Single-file mode (tests + ad-hoc runs)
    uv run python -m pipelines.postedit.postedit_pipeline \\
        --input /tmp/sample.jsonl --output /tmp/results.jsonl --language eng

Environment variables:
    GEMINI_API_KEY    - required if Gemini is in --models
    GEMINI_MODEL      - default: gemini-3-flash-preview
    DEEPSEEK_API_KEY  - required if DeepSeek is in --models
                        (get one at https://platform.deepseek.com/api_keys)
    DEEPSEEK_MODEL    - default: deepseek-v4-flash
    DEEPSEEK_BASE_URL - default: https://api.deepseek.com
    VLLM_BASE_URL     - default: http://localhost:8000
    VLLM_MODEL        - default: Qwen/Qwen3.5-27B-FP8
    VLLM_API_KEY      - optional
    POSTEDIT_TIMEOUT  - per-LLM-call HTTP timeout, default 60s
"""

import argparse
import json
import os
import sys
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger

from pipelines.postedit.cell_walker import Cell, iter_cells, iter_records, output_path_for
from pipelines.postedit.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_LANGUAGE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PER_QUESTION_TIMEOUT_S,
    DEFAULT_QUESTION_CONCURRENCY,
    DEFAULT_TEMPLATES_DIR,
    KG_BUCKETS,
    MODEL_NAMES,
    RESUME_MODES,
    RESUME_OFF,
    RESUME_RETRY_FAILED,
    RESUME_SKIP_DONE,
)
from pipelines.postedit.edit_apply import strip_entity_braces, validate_and_strip
from pipelines.postedit.ensemble import decide
from pipelines.postedit.llm_clients import BaseLLMClient, LLMError, VLLMClient, build_clients
from pipelines.postedit.logging_setup import setup_logging
from pipelines.postedit import mlflow_logging
from pipelines.postedit.normalize import normalize_record
from pipelines.postedit.prompts import build_user_prompt, get_system_prompt
from pipelines.postedit.resume import (
    load_existing_qids,
    rewrite_dropping_qids,
    split_done_failed,
)
from pipelines.postedit.schema import EnsembleRecord, InputQuestion, ModelResult, parse_model_json
from pipelines.postedit.templates_lookup import TemplateRegistry

DEFAULT_FINAL_ROOT = Path(__file__).resolve().parents[2] / "benchmark" / "final" / "large"


# ---------------------------------------------------------------------------
# Benchmark/final/ writer
# ---------------------------------------------------------------------------


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write ``rows`` to ``path`` via ``.tmp`` + ``os.replace``.

    The rename is atomic, so an interrupted run never leaves a
    half-written file in place of a good one. Implemented locally to
    keep postedit free of cross-package imports.
    """
    import os as _os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    _os.replace(tmp, path)


def _write_final_benchmark(
    cell: "Cell",
    postedit_jsonl: Path,
    final_root: Path,
) -> Path | None:
    """Emit the post-edited final questions for ``cell`` to
    ``benchmark/final/large/<kg>/<sfid_dir>/<lang>/benchmark.jsonl/part-00000.json``.

    For every row in the source cell's ``benchmark.jsonl/part-*`` shards:
    * If the postedit output has a ``final_question`` for this qid →
      replace the row's ``question`` field with that text. The row is
      otherwise verbatim (entity_*, ground_truth, parameters, …).
    * If the postedit output has no record for the qid (e.g. multimodal
      rows that ``normalize.py`` skips) → keep the original ``question``.

    Every input row lands in the output — questions are never dropped.
    Returns the written path, or ``None`` if there were no input rows
    or the postedit JSONL is missing.
    """
    if not postedit_jsonl.is_file():
        logger.debug("Final writer: postedit JSONL missing at {} — skipping",
                     postedit_jsonl)
        return None

    # Build qid → final_question map from the postedit output.
    final_map: dict[str, str] = {}
    with postedit_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("question_id")
            fq = row.get("final_question")
            if qid and fq:
                final_map[qid] = fq

    # Walk source rows and merge.
    out_rows: list[dict] = []
    for record in iter_records(cell):
        merged = dict(record)
        qid = record.get("question_id")
        if qid and qid in final_map:
            merged["question"] = final_map[qid]
        out_rows.append(merged)

    if not out_rows:
        return None

    out_path = (
        final_root / cell.kg / cell.sfid_dir / cell.language
        / "benchmark.jsonl" / "part-00000.json"
    )
    _atomic_write_jsonl(out_path, out_rows)
    return out_path


def run_one_model(
    client: BaseLLMClient,
    system: str,
    user: str,
    marked_question: str,
    instantiations: dict[str, str],
    fallback_text: str,
) -> ModelResult:
    """Call one LLM, parse the response, validate, and produce a ModelResult.

    All exceptions from the client are caught here so a single LLM
    failure can't kill the surrounding ``process_question`` call.
    Real clients wrap transient errors as :class:`LLMError`; we also
    catch the unexpected case where something else slips through.
    """
    try:
        gen = client.generate(system, user)
    except LLMError as e:
        logger.error(f"{client.name}: API call failed after retries: {e}")
        return ModelResult(ok=False, error=str(e))
    except Exception as e:  # noqa: BLE001
        logger.error(f"{client.name}: unexpected client exception: {e!r}")
        return ModelResult(ok=False, error=f"unexpected: {e!r}")

    try:
        raw = parse_model_json(gen.raw_text)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"{client.name}: JSON parse failure ({e}). Raw response (truncated): "
            f"{gen.raw_text[:200]}"
        )
        return ModelResult(
            ok=False,
            error=f"json_parse: {e}",
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
        )

    raw_dict = raw.model_dump()
    ops = [e.op for e in raw.edits] if raw.needs_edit else []

    if not raw.needs_edit:
        logger.debug(f"{client.name}: NO_CHANGE")
        return ModelResult(
            raw=raw_dict,
            ops=ops,
            final_text=fallback_text,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            ok=True,
        )

    if not raw.corrected_question:
        logger.warning(
            f"{client.name}: needs_edit=true but corrected_question missing"
        )
        return ModelResult(
            raw=raw_dict,
            ops=ops,
            final_text=fallback_text,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            ok=False,
            error="needs_edit=true but corrected_question missing",
        )

    final_text, ok, err, entity_edits = validate_and_strip(
        raw.corrected_question,
        instantiations,
        marked_question,
    )
    if not ok:
        logger.warning(f"{client.name}: validation failed ({err})")
    else:
        logger.debug(f"{client.name}: edit ops={ops}")
        if entity_edits:
            logger.debug(
                f"{client.name}: entity polish edits={entity_edits}"
            )
    return ModelResult(
        raw=raw_dict,
        ops=ops,
        final_text=final_text if ok else fallback_text,
        prompt_tokens=gen.prompt_tokens,
        completion_tokens=gen.completion_tokens,
        ok=ok,
        error=err if not ok else None,
        entity_edits=entity_edits if ok else [],
    )


def process_question(
    question: InputQuestion,
    clients: dict[str, BaseLLMClient],
    *,
    system_prompt: str | None = None,
    template_text: str | None = None,
) -> EnsembleRecord:
    """Run all clients in parallel for one question and return the ensemble record.

    ``system_prompt`` defaults to the language-agnostic base prompt if
    not given. Production callers should pass the riderized prompt
    from :func:`prompts.get_system_prompt`.

    ``template_text`` defaults to ``question.base_template`` (English)
    if not given. Production callers should pass the target-language
    template from :class:`templates_lookup.TemplateRegistry`.
    """
    if system_prompt is None:
        system_prompt = get_system_prompt(question.language)
    if template_text is None:
        template_text = question.base_template

    user = build_user_prompt(
        question.language,
        template_text,
        question.marked_question,
    )
    fallback_text = strip_entity_braces(
        question.marked_question, list(question.instantiations.values()),
    )

    results: dict[str, ModelResult] = {}
    with ThreadPoolExecutor(max_workers=len(clients)) as ex:
        future_to_name = {
            ex.submit(
                run_one_model,
                client,
                system_prompt,
                user,
                question.marked_question,
                question.instantiations,
                fallback_text,
            ): name
            for name, client in clients.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            results[name] = future.result()

    decision = decide(results, fallback_text)

    return EnsembleRecord(
        question_id=question.question_id,
        language=question.language,
        original_question=question.question,
        base_template=question.base_template,
        marked_question=question.marked_question,
        instantiations=question.instantiations,
        source_metadata=question.source_metadata,
        model_outputs=results,
        decision=decision.decision,
        final_question=decision.final_text,
        selected_model=decision.selected_model,
    )


# ---------------------------------------------------------------------------
# Cell-driven processing helpers
# ---------------------------------------------------------------------------


def _normalize_cell_questions(
    cell: Cell,
    *,
    skip_qids: set[str],
) -> list[InputQuestion]:
    """Read every record in ``cell``, normalise it, and skip qids
    already in ``skip_qids``."""
    out: list[InputQuestion] = []
    n_skipped_done = 0
    for record in iter_records(cell):
        qid = record.get("question_id")
        if qid in skip_qids:
            n_skipped_done += 1
            continue
        normalized = normalize_record(record, language=cell.language)
        if normalized is not None:
            out.append(normalized)
    if n_skipped_done:
        logger.info("Cell {}: skipping {} already-done qid(s)",
                    f"{cell.kg}/{cell.sfid_dir}", n_skipped_done)
    return out


def _process_questions_concurrent(
    questions: list[InputQuestion],
    clients: dict[str, BaseLLMClient],
    *,
    template_registry: TemplateRegistry | None,
    concurrency: int,
    per_question_timeout_s: float,
    out_path: Path,
    batch_size: int,
) -> tuple[int, int, dict[str, int]]:
    """Drive the question loop with question-level concurrency.

    Each worker pulls one question, runs the 3-LLM ensemble (which has
    its own per-question 3-thread pool), and writes the result line.
    Results are written in *completion order*, not input order — that's
    fine because the output is keyed by ``question_id``.
    """
    decision_counts: dict[str, int] = {}
    n_done = 0
    n_errors = 0

    def _resolve_template(q: InputQuestion) -> str:
        if template_registry is None:
            return q.base_template
        vid = q.source_metadata.get("vid") or ""
        return template_registry.lookup(
            vid, q.language, eng_fallback=q.base_template,
        )

    def _do_one(q: InputQuestion) -> EnsembleRecord:
        return process_question(
            q,
            clients,
            system_prompt=get_system_prompt(q.language),
            template_text=_resolve_template(q),
        )

    out_f = out_path.open("a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(_do_one, q): q for q in questions}
            total = len(futures)
            for fut in as_completed(futures):
                q = futures[fut]
                try:
                    record = fut.result(timeout=per_question_timeout_s)
                except FuturesTimeoutError:
                    n_errors += 1
                    logger.warning(
                        "Question {} hit per-question timeout ({}s); "
                        "will be retryable on next --resume retry-failed",
                        q.question_id, per_question_timeout_s,
                    )
                    continue
                except Exception as e:  # noqa: BLE001
                    n_errors += 1
                    logger.exception(
                        "Question {}: pipeline crash: {}", q.question_id, e,
                    )
                    continue

                out_f.write(record.model_dump_json() + "\n")
                out_f.flush()
                n_done += 1
                decision_counts[record.decision] = (
                    decision_counts.get(record.decision, 0) + 1
                )
                if n_done % batch_size == 0:
                    logger.info(
                        "Progress: {}/{} done (last decision: {})",
                        n_done, total, record.decision,
                    )
    finally:
        out_f.close()

    return n_done, n_errors, decision_counts


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def preflight_clients(selected_models: Iterable[str], *,
                      skip_vllm_health_check: bool) -> dict[str, BaseLLMClient]:
    """Build clients and probe their endpoints / API keys.

    * Gemini & DeepSeek — refuse to start if their API keys aren't set.
    * vLLM — call ``/health`` once unless skipped.

    Returns the built client map. Calls ``sys.exit(1)`` on failure
    rather than raising so the caller doesn't need to repeat this
    boilerplate.
    """
    selected = list(selected_models)
    if "gemini" in selected:
        use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
        if use_vertex:
            if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
                logger.error(
                    "Vertex mode requested (GOOGLE_GENAI_USE_VERTEXAI=true) but "
                    "GOOGLE_CLOUD_PROJECT is not set."
                )
                sys.exit(1)
        elif not os.environ.get("GEMINI_API_KEY"):
            logger.error(
                "GEMINI_API_KEY is not set (or set GOOGLE_GENAI_USE_VERTEXAI=true "
                "for Vertex mode)."
            )
            sys.exit(1)
    if "deepseek" in selected and not os.environ.get("DEEPSEEK_API_KEY"):
        logger.error(
            "DEEPSEEK_API_KEY is not set in the environment "
            "(get one at https://platform.deepseek.com/api_keys)"
        )
        sys.exit(1)

    clients = build_clients(selected)

    if "qwen" in clients and not skip_vllm_health_check:
        vllm: VLLMClient = clients["qwen"]  # type: ignore[assignment]
        logger.info("Checking vLLM endpoint at {}", vllm._base_url)
        if not vllm.health_check():
            logger.error("vLLM endpoint at {} is not responding.", vllm._base_url)
            logger.error(
                "Start vLLM first, or pass --skip-vllm-health-check, "
                "or remove qwen from --models."
            )
            sys.exit(1)
    return clients


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Cell-walking inputs (preferred mode)
    parser.add_argument(
        "--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT,
        help=f"Root of the multilingual benchmark "
             f"(default: {DEFAULT_BENCHMARK_ROOT})",
    )
    parser.add_argument(
        "--language", type=str, default=DEFAULT_LANGUAGE,
        help=f"ISO 639-3 language code (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--kg", type=str, default=None, choices=[*KG_BUCKETS, None],
        help="Restrict to one KG bucket. Default: walk all KGs.",
    )
    parser.add_argument(
        "--sfid", type=str, default=None,
        help="Restrict to one SFID dir under --kg (e.g. sfid_a). "
             "No effect on multimodal.",
    )
    parser.add_argument(
        "--templates-dir", type=Path, default=DEFAULT_TEMPLATES_DIR,
        help=f"Path to the per-language templates registry CSV "
             f"(default: {DEFAULT_TEMPLATES_DIR})",
    )

    # Output
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output root for cell results (default: {DEFAULT_OUTPUT_DIR})",
    )

    # Single-file mode (tests / ad-hoc runs)
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Single-file input JSONL. If set, --benchmark-root "
             "/ --kg / --sfid are ignored.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Single-file output JSONL (paired with --input).",
    )

    # Resume + concurrency
    parser.add_argument(
        "--resume", type=str, default=RESUME_OFF, choices=RESUME_MODES,
        help=f"Resume mode (default: {RESUME_OFF}).",
    )
    parser.add_argument(
        "--question-concurrency", type=int, default=DEFAULT_QUESTION_CONCURRENCY,
        help=f"Number of questions processed in parallel "
             f"(default: {DEFAULT_QUESTION_CONCURRENCY})",
    )
    parser.add_argument(
        "--per-question-timeout-s", type=float,
        default=DEFAULT_PER_QUESTION_TIMEOUT_S,
        help=f"Hard timeout per question, including all 3 model calls "
             f"(default: {DEFAULT_PER_QUESTION_TIMEOUT_S}s)",
    )

    # Models
    parser.add_argument(
        "--models", type=str, default=",".join(MODEL_NAMES),
        help=f"Comma-separated model names (default: {','.join(MODEL_NAMES)})",
    )

    # Misc
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N questions per cell")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Progress-report interval (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate cell-counts/calls without making API calls")
    parser.add_argument("--skip-vllm-health-check", action="store_true",
                        help="Skip the vLLM startup health check")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose console output (DEBUG level)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Quiet console output (WARNING level)")
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="Directory for log files (default: output/postedit/logs)")

    # MLflow tracking (opt-out only).
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Disable MLflow tracking (default: enabled). Useful for "
             "smoke tests or when mlflow isn't installed.",
    )
    parser.add_argument(
        "--mlflow-experiment", type=str, default=None,
        help="MLflow experiment name (default from env "
             "MLFLOW_EXPERIMENT_NAME or 'Postedit')",
    )

    # benchmark/final/large/ writer (post-edited questions in the benchmark
    # tree layout; complements the per-question audit JSONL under
    # output/postedit/).
    parser.add_argument(
        "--final-root", type=Path, default=DEFAULT_FINAL_ROOT,
        help=f"Root dir for the post-edited benchmark JSONL "
             f"(mirrors the benchmark tree layout). Default: "
             f"{DEFAULT_FINAL_ROOT}",
    )
    parser.add_argument(
        "--no-final", action="store_true",
        help="Skip writing the post-edited questions to "
             "benchmark/final/large/. Useful for smoke runs where you only "
             "want the audit metadata under output/postedit/.",
    )
    return parser.parse_args()


def _validate_models(args_models: str) -> list[str]:
    selected = [m.strip() for m in args_models.split(",") if m.strip()]
    if not selected:
        logger.error("--models must specify at least one model "
                     "(allowed: {})", ", ".join(MODEL_NAMES))
        sys.exit(1)
    for m in selected:
        if m not in MODEL_NAMES:
            logger.error("Unknown model {!r} (allowed: {})",
                         m, ", ".join(MODEL_NAMES))
            sys.exit(1)
    return selected


# ---------------------------------------------------------------------------
# Single-cell driver
# ---------------------------------------------------------------------------


def _run_cell(
    cell: Cell,
    *,
    output_root: Path,
    clients: dict[str, BaseLLMClient],
    template_registry: TemplateRegistry | None,
    resume_mode: str,
    question_concurrency: int,
    per_question_timeout_s: float,
    batch_size: int,
    limit: Optional[int],
    final_root: Path | None = None,
) -> tuple[int, int, dict[str, int], Path]:
    """Process one cell end-to-end.

    Returns ``(n_done, n_errors, decisions, out_path)``. The
    ``out_path`` is included so the caller can hand it to MLflow.
    """
    out_path = output_path_for(output_root, cell)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skip_qids: set[str] = set()
    if resume_mode == RESUME_OFF and out_path.exists():
        # Truncate semantics — but be safe about it.
        logger.info("Cell {}: --resume off, truncating existing output {}",
                    f"{cell.kg}/{cell.sfid_dir}", out_path)
        out_path.unlink()
    elif resume_mode == RESUME_SKIP_DONE:
        skip_qids = load_existing_qids(out_path)
        if skip_qids:
            logger.info("Cell {}: skip-done — found {} done qids",
                        f"{cell.kg}/{cell.sfid_dir}", len(skip_qids))
    elif resume_mode == RESUME_RETRY_FAILED:
        done, failed = split_done_failed(out_path)
        rewrite_dropping_qids(out_path, failed)
        skip_qids = done - failed
        logger.info("Cell {}: retry-failed — keeping {} good rows, "
                    "re-running {} failed qids",
                    f"{cell.kg}/{cell.sfid_dir}",
                    len(done) - len(failed), len(failed))

    questions = _normalize_cell_questions(cell, skip_qids=skip_qids)
    if limit is not None:
        questions = questions[:limit]
    if not questions:
        logger.info("Cell {}: nothing to do (empty / fully resumed)",
                    f"{cell.kg}/{cell.sfid_dir}")
        # Refresh the final/ output anyway in case the postedit JSONL
        # changed under retry-failed / skip-done.
        if final_root is not None:
            final_path = _write_final_benchmark(cell, out_path, final_root)
            if final_path is not None:
                logger.info("Cell {}: final/ refreshed at {}",
                            f"{cell.kg}/{cell.sfid_dir}", final_path)
        return 0, 0, {}, out_path

    logger.info("Cell {}: processing {} questions → {}",
                f"{cell.kg}/{cell.sfid_dir}", len(questions), out_path)
    n_done, n_err, decisions = _process_questions_concurrent(
        questions, clients,
        template_registry=template_registry,
        concurrency=question_concurrency,
        per_question_timeout_s=per_question_timeout_s,
        out_path=out_path,
        batch_size=batch_size,
    )
    if final_root is not None:
        final_path = _write_final_benchmark(cell, out_path, final_root)
        if final_path is not None:
            logger.info("Cell {}: final/ written to {}",
                        f"{cell.kg}/{cell.sfid_dir}", final_path)
    return n_done, n_err, decisions, out_path


def _legacy_single_file_main(args: argparse.Namespace) -> int:
    """Single-file mode driven by ``--input`` (fixtures + ad-hoc runs).

    Loads the templates registry if ``--templates-dir`` points to an
    existing file; otherwise the prompt's TEMPLATE field is the row's
    English ``base_template``.
    """
    if not args.input.exists():
        logger.error("Input file not found: {}", args.input)
        return 1
    out_path = args.output or (DEFAULT_OUTPUT_DIR / f"results_{args.language}.jsonl")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected = _validate_models(args.models)
    if args.dry_run:
        from normalize import normalize_input
        n = sum(1 for _ in normalize_input(args.input, language=args.language))
        logger.info("Dry run: {} questions × {} models = {} calls",
                    n, len(selected), n * len(selected))
        return 0

    template_registry: TemplateRegistry | None = None
    if args.templates_dir and args.templates_dir.is_dir():
        template_registry = TemplateRegistry.load(args.templates_dir)
    else:
        logger.info(
            "Single-file --input mode: templates registry not loaded "
            "(--templates-dir missing or path doesn't exist); "
            "prompts will use the row's base_template as TEMPLATE."
        )

    clients = preflight_clients(
        selected, skip_vllm_health_check=args.skip_vllm_health_check,
    )
    from normalize import normalize_input
    questions = list(normalize_input(args.input, language=args.language))
    if args.limit is not None:
        questions = questions[: args.limit]

    skip_qids: set[str] = set()
    if args.resume == RESUME_OFF and out_path.exists():
        out_path.unlink()
    elif args.resume == RESUME_SKIP_DONE:
        skip_qids = load_existing_qids(out_path)
    elif args.resume == RESUME_RETRY_FAILED:
        done, failed = split_done_failed(out_path)
        rewrite_dropping_qids(out_path, failed)
        skip_qids = done - failed
    questions = [q for q in questions if q.question_id not in skip_qids]

    n_done, n_err, decisions = _process_questions_concurrent(
        questions, clients,
        template_registry=template_registry,
        concurrency=args.question_concurrency,
        per_question_timeout_s=args.per_question_timeout_s,
        out_path=out_path,
        batch_size=args.batch_size,
    )
    logger.info("Single-file run: {} done, {} errors. Decisions: {}",
                n_done, n_err, decisions)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    console_level = (
        "DEBUG" if args.verbose
        else ("WARNING" if args.quiet else "INFO")
    )
    setup_logging(args.language, log_dir=args.log_dir, console_level=console_level)

    if args.input is not None:
        return _legacy_single_file_main(args)

    selected = _validate_models(args.models)

    cells = list(iter_cells(
        args.benchmark_root, args.language,
        kg=args.kg, sfid=args.sfid,
    ))
    if not cells:
        logger.error("No cells matched: root={} lang={} kg={} sfid={}",
                     args.benchmark_root, args.language, args.kg, args.sfid)
        return 1

    if args.dry_run:
        total_rows = 0
        for cell in cells:
            n = sum(1 for _ in iter_records(cell))
            total_rows += n
            logger.info("  {}: {} rows", f"{cell.kg}/{cell.sfid_dir}", n)
        logger.info("Dry run: {} cells, {} total rows × {} models = {} calls",
                    len(cells), total_rows, len(selected),
                    total_rows * len(selected))
        return 0

    template_registry = TemplateRegistry.load(args.templates_dir)

    clients = preflight_clients(
        selected, skip_vllm_health_check=args.skip_vllm_health_check,
    )

    mlflow_logging.init_experiment(
        enabled=not args.no_mlflow,
        experiment_name=args.mlflow_experiment,
    )
    mlflow_logging.start_parent_run(
        language=args.language,
        kg=args.kg,
        sfid=args.sfid,
        models=selected,
        question_concurrency=args.question_concurrency,
        per_question_timeout_s=args.per_question_timeout_s,
        resume_mode=args.resume,
        benchmark_root=args.benchmark_root,
    )

    grand_done = 0
    grand_errors = 0
    grand_decisions: dict[str, int] = {}
    parent_status = "FINISHED"
    final_root = None if args.no_final else args.final_root
    logger.info("Will process {} cell(s) for language={}",
                len(cells), args.language)
    if final_root is not None:
        logger.info("Post-edited questions will land under {}", final_root)
    try:
        for cell in cells:
            n_done, n_err, decisions, out_path = _run_cell(
                cell,
                output_root=args.output_root,
                clients=clients,
                template_registry=template_registry,
                resume_mode=args.resume,
                question_concurrency=args.question_concurrency,
                per_question_timeout_s=args.per_question_timeout_s,
                batch_size=args.batch_size,
                limit=args.limit,
                final_root=final_root,
            )
            grand_done += n_done
            grand_errors += n_err
            for k, v in decisions.items():
                grand_decisions[k] = grand_decisions.get(k, 0) + v
            mlflow_logging.log_cell_run(
                language=cell.language,
                kg=cell.kg,
                sfid_dir=cell.sfid_dir,
                n_done=n_done,
                n_errors=n_err,
                decisions=decisions,
                out_path=out_path,
            )
    except KeyboardInterrupt:
        parent_status = "FAILED"
        raise
    except Exception:
        parent_status = "FAILED"
        raise
    finally:
        mlflow_logging.end_parent_run(status=parent_status)

    logger.info("=== Done ===")
    logger.info("Total: {} rows written, {} errors, decisions: {}",
                grand_done, grand_errors, grand_decisions)
    # Always exit 0: per-question errors are recoverable via
    # ``--resume retry-failed``; surfacing a non-zero exit here would
    # cause SLURM to retry the whole job needlessly.
    return 0


if __name__ == "__main__":
    sys.exit(main())
