"""
KnowWhereGraph REST Crawler.

Streams RDF triples from the KnowWhereGraph GraphDB REST API,
partitioned by named graph for resumability and parallelism.
"""

import json
import logging
import sys
import time
from pathlib import Path

import requests

# Configuration
BASE_URL = "https://stko-kwg.geog.ucsb.edu/graphdb/repositories/KWG"
OUTPUT_DIR = "data/kwg"
OUTPUT_FILE = "data/kwg/kwg_dump.nt"

# Request settings
CHUNK_SIZE = 64 * 1024  # 64KB read chunks for streaming
REQUEST_TIMEOUT = (30, 300)  # (connect, read) timeouts in seconds
MAX_RETRIES = 3
RETRY_BASE_DELAY = 60  # base delay in seconds (1 minute)

# Rotate output file every 100M triples (~20 GB per part file)
SPLIT_THRESHOLD = 100_000_000

# Subgraphs to include in the benchmark crawl.
# Excludes: air-quality-epa, drought-ndmc, admin-regions-gadm-new,
#           cropland-usda, soil-ssurgo-usda (TB-scale, not needed for benchmark).
BENCHMARK_SUBGRAPHS: set[str] = {
    "subgraph.ontologies",
    "subgraph.metadata-model",
    "subgraph.topic-hierarchies",
    "subgraph.spoke-mappings",
    "subgraph.time",
    "subgraph.admin-regions-wikidata",
    "subgraph.public-health",
    "subgraph.fqhc-hrsa",
    "subgraph.relief-web",
    "subgraph.bluesky-forecast-airfire",
    "subgraph.national-weather-zone-noaa",
    "subgraph.wildfire-nifc",
    "subgraph.hurricane-tracks-noaa",
    "subgraph.disaster-declaration-fema",
    "subgraph.historical-fires-mtbs",
    "subgraph.federal-judicial-district-doj",
    "subgraph.zipcode-uscb",
    "subgraph.transportation-usdot",
    "subgraph.census-uscb",
    "subgraph.cross-place-integration",
    "subgraph.admin-regions-uscb",
    "subgraph.gnis-ld-usgs",
    "subgraph.storm-impacts-noaa",
    "subgraph.smoke-plumes",
    "subgraph.climate-divisions-observations-noaa",
    "subgraph.earthquake-usgs",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _progress_path(output_dir: str) -> Path:
    """Return the progress file path."""
    return Path(output_dir) / "crawl_progress.json"


def _load_progress(output_dir: str) -> dict:
    """Load crawl progress: which graphs have been completed."""
    path = _progress_path(output_dir)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.info(f"Loaded progress: {len(data.get('completed', []))} graphs done")
            return data
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Invalid progress file, starting fresh: {e}")
    return {"completed": [], "total_triples": 0}


def _save_progress(output_dir: str, progress: dict) -> None:
    """Persist crawl progress."""
    path = _progress_path(output_dir)
    path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def fetch_contexts(base_url: str = BASE_URL) -> list[str]:
    """
    Fetch the list of named graphs from the repository.

    Returns:
        List of named graph URIs.
    """
    url = f"{base_url}/contexts"
    resp = requests.get(url, headers={"Accept": "application/json"}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    graphs = [b["contextID"]["value"] for b in data["results"]["bindings"]]
    logger.info(f"Found {len(graphs)} named graphs")
    return graphs


def stream_graph(
    graph_uri: str | None,
    output_dir: Path,
    base_name: str,
    base_url: str = BASE_URL,
) -> int:
    """
    Stream all triples from a single named graph to one or more part files.

    Output files are named ``{base_name}_part{N:04d}.nt`` and rotated every
    ``SPLIT_THRESHOLD`` triples.  Before streaming, any leftover part files
    from a previous incomplete attempt are deleted so the graph is always
    written from scratch.

    Args:
        graph_uri: The named graph URI, or None for the default graph.
        output_dir: Directory to write part files into.
        base_name: Safe base filename (no extension).
        base_url: GraphDB repository URL.

    Returns:
        Number of triples (lines) written across all parts.
    """
    url = f"{base_url}/statements"
    params = {}
    if graph_uri:
        params["context"] = f"<{graph_uri}>"

    headers = {"Accept": "application/n-triples"}

    # Clean up stale part files from a previous incomplete run
    for old_part in output_dir.glob(f"{base_name}_part*.nt"):
        old_part.unlink()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                params=params,
                headers=headers,
                stream=True,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()

            total_lines = 0
            part_lines = 0
            part_num = 1
            leftover = ""

            def _part_path(n: int) -> Path:
                return output_dir / f"{base_name}_part{n:04d}.nt"

            f = open(_part_path(part_num), "w", encoding="utf-8")  # noqa: SIM115

            try:
                for raw_chunk in resp.iter_content(chunk_size=CHUNK_SIZE, decode_unicode=True):
                    if not raw_chunk:
                        continue

                    chunk = leftover + raw_chunk
                    # Split into complete lines; last element is either "" or
                    # an incomplete trailing line to carry over.
                    parts = chunk.split("\n")
                    leftover = parts.pop()

                    for line in parts:
                        if not line:
                            continue
                        f.write(line)
                        f.write("\n")
                        total_lines += 1
                        part_lines += 1

                        if part_lines >= SPLIT_THRESHOLD:
                            f.close()
                            logger.info(
                                f"  Part {part_num}: {part_lines:,} triples -> "
                                f"{_part_path(part_num).name}"
                            )
                            part_num += 1
                            part_lines = 0
                            f = open(  # noqa: SIM115
                                _part_path(part_num), "w", encoding="utf-8"
                            )

                # Flush any remaining buffered text
                if leftover.strip():
                    f.write(leftover)
                    f.write("\n")
                    total_lines += 1
                    part_lines += 1

            finally:
                f.close()

            if part_lines > 0:
                logger.info(
                    f"  Part {part_num}: {part_lines:,} triples -> {_part_path(part_num).name}"
                )
            elif part_num > 1:
                # Last part file was empty after rotation — remove it
                _part_path(part_num).unlink(missing_ok=True)

            return total_lines

        except (requests.RequestException, OSError) as e:
            label = graph_uri or "default"
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {label}: {e}")
            # Clean up partial files before retry
            for old_part in output_dir.glob(f"{base_name}_part*.nt"):
                old_part.unlink()
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))
            else:
                raise
    return 0  # unreachable, but keeps mypy happy


def _graph_base_name(graph_uri: str | None) -> str:
    """Derive a safe base filename (without extension) from a graph URI."""
    if graph_uri is None:
        return "default_graph"
    # Extract last path segment as name
    name = graph_uri.rstrip("/").rsplit("/", 1)[-1]
    # Sanitise for filesystem
    return name.replace(".", "_").replace(":", "_").replace(" ", "_")


def crawl(
    output_dir: str = OUTPUT_DIR,
    base_url: str = BASE_URL,
    include_default_graph: bool = True,
) -> int:
    """
    Crawl the KnowWhereGraph by streaming each named graph via the REST API.

    Only graphs listed in ``BENCHMARK_SUBGRAPHS`` are fetched.  Large graphs
    are automatically split into part files of ~100 M triples each.

    Args:
        output_dir: Directory to write per-graph NT files.
        base_url: GraphDB repository URL.
        include_default_graph: Whether to also dump the default (unnamed) graph.

    Returns:
        Total number of triples fetched.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    progress = _load_progress(output_dir)
    completed = set(progress["completed"])
    total_triples = progress["total_triples"]
    start_time = time.monotonic()

    # Build list of graphs to fetch, filtered to benchmark subgraphs
    graphs: list[str | None] = [
        g for g in fetch_contexts(base_url) if g.rsplit("/", 1)[-1] in BENCHMARK_SUBGRAPHS
    ]
    if include_default_graph:
        graphs.append(None)

    remaining = [g for g in graphs if (g or "default") not in completed]
    logger.info(
        f"{len(graphs)} benchmark graphs, {len(completed)} already done, {len(remaining)} remaining"
    )

    for i, graph_uri in enumerate(remaining, 1):
        label = graph_uri or "default"
        base_name = _graph_base_name(graph_uri)

        logger.info(f"[{i}/{len(remaining)}] Streaming {label} -> {base_name}_part*.nt")

        try:
            count = stream_graph(graph_uri, out, base_name, base_url)
        except Exception as e:
            logger.error(f"Failed to stream {label} after {MAX_RETRIES} attempts: {e}")
            continue

        total_triples += count

        # Mark graph as completed
        completed.add(label)
        progress["completed"] = list(completed)
        progress["total_triples"] = total_triples
        _save_progress(output_dir, progress)

        time.sleep(30)  # cooldown between graphs

        elapsed = time.monotonic() - start_time
        rate = total_triples / elapsed if elapsed > 0 else 0
        logger.info(
            f"[{i}/{len(remaining)}] {label}: {count:,} triples "
            f"(total: {total_triples:,}, rate: {rate:,.0f} triples/s, "
            f"elapsed: {elapsed:.0f}s)"
        )

    return total_triples


def merge_files(output_dir: str = OUTPUT_DIR, output_file: str = OUTPUT_FILE) -> int:
    """
    Merge all per-graph NT files into a single output file.

    Args:
        output_dir: Directory containing per-graph NT files.
        output_file: Path to the merged output file.

    Returns:
        Total number of lines written.
    """
    out = Path(output_dir)
    nt_files = sorted(out.glob("*.nt"))
    merged = Path(output_file)
    total_lines = 0

    logger.info(f"Merging {len(nt_files)} files -> {merged}")

    with open(merged, "w", encoding="utf-8") as out_f:
        for nt_file in nt_files:
            if nt_file.resolve() == merged.resolve():
                continue
            with open(nt_file, encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
                    total_lines += 1

    logger.info(f"Merged {total_lines:,} triples into {merged}")
    return total_lines


def main() -> int:
    """
    Main entry point for KWG crawler.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        total = crawl()
        logger.info(f"Crawl completed. Total triples: {total:,}")

        if "--merge" in sys.argv:
            merge_files()

        return 0
    except KeyboardInterrupt:
        logger.info("Crawl interrupted by user (progress saved)")
        return 0
    except Exception as e:
        logger.error(f"Crawl failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
