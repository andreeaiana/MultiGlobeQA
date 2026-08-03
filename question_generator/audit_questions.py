"""Quality audit for generated benchmark questions.

Scans `data/benchmark_output/full/<kg>/<sfid>/benchmark.parquet` and
reports per-TID quality issues:

  - Heterogeneous-category labels in entity_a/b/c_name (e.g. "POIs",
    "natural_feature_category") that should be specific subcategories.
  - Self-reference (entity_a_name == entity_b_name — typical "Which X
    in X minimises ..." degeneracy).
  - Ground-truth degeneracy (% of rows with GT == 0, GT NULL, single
    dominant value, very low cardinality).
  - Subcategory diversity (distinct entity_a_name values per TID —
    a healthy benchmark covers many subcategories per TID).

Run via srun (Spark + Sedona init):

    srun --mem=32G --cpus-per-task=4 --partition=cpu \\
      uv run python -m question_generator.audit_questions \\
        --output reports/question_audit.jsonl

Optional filters: ``--kg``, ``--sfid``, ``--tid``, ``--limit-samples``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import functions as F

from question_generator.main import init_spark

# Labels that indicate a TID is using a heterogeneous category in place
# of a specific subcategory. Match against entity_a/b/c_name. The
# benchmark target is to drive these toward zero in favour of raw
# node_type values like "Pharmacy", "school", "AmenityHospital", etc.
HETEROGENEOUS_LABELS = {
    "POIs",
    "POI_category",
    "regions",
    "region_category",
    "natural_features",
    "natural_feature_category",
    "events",
    "hazard_event_category",
    "road_network_features",
    "road_network_feature_category",
}


def audit_kg_sfid(spark, kg: str, sfid: str, output_dir: Path,
                  tid_filter: str | None, sample_count: int):
    """Run the audit pipeline for a single (kg, sfid) pair."""
    base = Path("data/benchmark_output/full") / kg / sfid / "benchmark.parquet"
    if not base.exists():
        print(f"[SKIP] missing: {base}", flush=True)
        return None
    try:
        df = spark.read.parquet(str(base))
    except Exception as exc:
        print(f"[ERROR] read {base}: {exc}", flush=True)
        return None
    n_total = df.count()
    if n_total == 0:
        print(f"[SKIP] empty: {kg}/{sfid}", flush=True)
        return None

    # Add per-row diagnostic flags before grouping by TID.
    het_check = F.lit(False)
    for col in ("entity_a_name", "entity_b_name", "entity_c_name"):
        if col in df.columns:
            het_check = het_check | F.col(col).isin(*HETEROGENEOUS_LABELS)
    df = df.withColumn("_uses_heterogeneous_label", het_check)

    if "entity_a_name" in df.columns and "entity_b_name" in df.columns:
        df = df.withColumn(
            "_self_reference_a_b",
            F.col("entity_a_name") == F.col("entity_b_name"),
        )
    else:
        df = df.withColumn("_self_reference_a_b", F.lit(False))

    if "ground_truth" in df.columns:
        df = df.withColumn(
            "_gt_double",
            F.expr("try_cast(ground_truth as double)"),
        )
        df = df.withColumn(
            "_gt_is_zero",
            F.col("_gt_double").isNotNull() & (F.col("_gt_double") == 0.0),
        )
        df = df.withColumn("_gt_is_null", F.col("ground_truth").isNull())
    else:
        df = df.withColumn("_gt_is_zero", F.lit(False))
        df = df.withColumn("_gt_is_null", F.lit(False))

    if tid_filter:
        df = df.filter(F.col("tid") == tid_filter)
    df.cache()

    # Per-TID rollup.
    by_tid = df.groupBy("tid").agg(
        F.count(F.lit(1)).alias("n_rows"),
        F.sum(F.col("_uses_heterogeneous_label").cast("int")).alias("n_heterogeneous"),
        F.sum(F.col("_self_reference_a_b").cast("int")).alias("n_self_ref"),
        F.sum(F.col("_gt_is_zero").cast("int")).alias("n_gt_zero"),
        F.sum(F.col("_gt_is_null").cast("int")).alias("n_gt_null"),
        F.countDistinct("entity_a_name").alias("n_distinct_entity_a")
            if "entity_a_name" in df.columns else F.lit(0).alias("n_distinct_entity_a"),
        F.countDistinct("ground_truth").alias("n_distinct_gt")
            if "ground_truth" in df.columns else F.lit(0).alias("n_distinct_gt"),
    ).collect()

    output = []
    for row in sorted(by_tid, key=lambda r: r["tid"]):
        tid = row["tid"]
        n = row["n_rows"]
        het_pct = row["n_heterogeneous"] / n * 100 if n else 0
        self_pct = row["n_self_ref"] / n * 100 if n else 0
        zero_pct = row["n_gt_zero"] / n * 100 if n else 0
        rec = {
            "kg": kg,
            "sfid": sfid,
            "tid": tid,
            "n_rows": n,
            "heterogeneous_label_pct": round(het_pct, 1),
            "self_reference_pct": round(self_pct, 1),
            "gt_zero_pct": round(zero_pct, 1),
            "gt_null_pct": round(row["n_gt_null"] / n * 100, 1) if n else 0,
            "distinct_entity_a": row["n_distinct_entity_a"],
            "distinct_gt": row["n_distinct_gt"],
            "flags": [],
        }
        if rec["heterogeneous_label_pct"] >= 50:
            rec["flags"].append("MOSTLY_HETEROGENEOUS_LABEL")
        if rec["self_reference_pct"] >= 10:
            rec["flags"].append("HIGH_SELF_REFERENCE")
        if rec["gt_zero_pct"] >= 25:
            rec["flags"].append("HIGH_ZERO_GROUND_TRUTH")
        if rec["distinct_entity_a"] <= 3 and n > 50:
            rec["flags"].append("LOW_SUBCATEGORY_DIVERSITY")
        if rec["distinct_gt"] <= 5 and n > 50:
            rec["flags"].append("LOW_GT_DIVERSITY")

        # Sample a handful of flagged rows for human review.
        if rec["flags"]:
            samples_df = df.filter(F.col("tid") == tid)
            if "MOSTLY_HETEROGENEOUS_LABEL" in rec["flags"]:
                samples_df = samples_df.filter(F.col("_uses_heterogeneous_label"))
            elif "HIGH_SELF_REFERENCE" in rec["flags"]:
                samples_df = samples_df.filter(F.col("_self_reference_a_b"))
            elif "HIGH_ZERO_GROUND_TRUTH" in rec["flags"]:
                samples_df = samples_df.filter(F.col("_gt_is_zero"))
            sample_rows = samples_df.select(
                "question", "ground_truth", "entity_a_name",
                "entity_b_name",
            ).limit(sample_count).collect()
            rec["samples"] = [r.asDict() for r in sample_rows]
        output.append(rec)

    df.unpersist()

    # Append to JSONL.
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "question_audit.jsonl", "a") as f:
        for rec in output:
            f.write(json.dumps(rec) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", default=None,
                        help="Restrict to one KG (kwg, osmh3kg, worldkg).")
    parser.add_argument("--sfid", default=None,
                        help="Restrict to one sfid (e.g. sfid_a).")
    parser.add_argument("--tid", default=None,
                        help="Restrict to one TID (e.g. 8.1).")
    parser.add_argument("--output-dir", default="reports",
                        help="Where to write question_audit.jsonl.")
    parser.add_argument("--limit-samples", type=int, default=5,
                        help="Sample rows per flagged TID.")
    args = parser.parse_args()

    spark = init_spark()
    spark.sparkContext.setLogLevel("WARN")
    output_dir = Path(args.output_dir)
    # Truncate previous audit.
    out_file = output_dir / "question_audit.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()

    base = Path("data/benchmark_output/full")
    summary = {"flagged_tids": 0, "total_tids": 0}
    for kg_dir in sorted(base.iterdir()):
        if not kg_dir.is_dir():
            continue
        if args.kg and kg_dir.name != args.kg:
            continue
        for sfid_dir in sorted(kg_dir.iterdir()):
            if not sfid_dir.is_dir():
                continue
            if args.sfid and sfid_dir.name != args.sfid:
                continue
            recs = audit_kg_sfid(
                spark, kg_dir.name, sfid_dir.name, output_dir,
                args.tid, args.limit_samples,
            )
            if recs:
                summary["total_tids"] += len(recs)
                summary["flagged_tids"] += sum(1 for r in recs if r["flags"])
                for r in recs:
                    flag_str = ",".join(r["flags"]) if r["flags"] else "OK"
                    print(
                        f"  {kg_dir.name:10}{sfid_dir.name:10} TID {r['tid']:8} "
                        f"n={r['n_rows']:>7d} het={r['heterogeneous_label_pct']:>5.1f}% "
                        f"self={r['self_reference_pct']:>5.1f}% zero={r['gt_zero_pct']:>5.1f}% "
                        f"a_div={r['distinct_entity_a']:>4d} gt_div={r['distinct_gt']:>4d}  "
                        f"[{flag_str}]",
                        flush=True,
                    )

    print(
        f"\nSummary: {summary['flagged_tids']}/{summary['total_tids']} TIDs flagged. "
        f"Full report: {out_file}",
        flush=True,
    )
    spark.stop()


if __name__ == "__main__":
    main()
