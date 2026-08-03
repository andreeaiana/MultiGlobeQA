"""Download and cache reference data for hierarchical sampling.

Two datasets are managed here:

1. **Natural Earth 110m Admin-0 Countries** — polygon boundaries for every
   country, used to assign entities to a country via a spatial join
   (ST_Contains).  Cached to ``data/reference/countries.geojson``.

2. **World Bank income groups** — Low / Lower-Middle / Upper-Middle / High
   classifications fetched from the World Bank API.
   Cached to ``data/reference/wb_income_groups.json``.

Both files are written once and reused on subsequent runs.  Delete a file
to force a fresh download.

Usage (standalone):
    python -m data_extraction.reference_data.downloader

Usage (programmatic):
    from data_extraction.reference_data import ensure_reference_data
    ensure_reference_data()          # download if missing, no-op otherwise

    from data_extraction.reference_data import load_countries_geodataframe
    gdf = load_countries_geodataframe()   # returns a GeoDataFrame

    from data_extraction.reference_data import load_income_groups
    groups = load_income_groups()    # returns dict[iso3 -> income_level_id]
"""

import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Download URLs
# ---------------------------------------------------------------------------

# Natural Earth 10m (high-res). 110m (coarsest) was dropping small islands —
# Okinawa, Hokkaido detail, Philippine archipelago, Norway fjord coastline —
# producing ~322K false UNKNOWN entities in osmh3kg strata. 10m preserves
# ~10× vertex detail and keeps small islands polygonised.
_NATURAL_EARTH_URL = (
    "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
)

# World Bank country API — returns all countries with income classification
# per_page=500 is more than enough for all ~250 countries
_WORLD_BANK_URL = "https://api.worldbank.org/v2/country?format=json&per_page=500"

# World Bank income level IDs (as returned by the API)
_WB_INCOME_LEVEL_IDS = frozenset(
    {
        "LIC",  # Low income
        "LMC",  # Lower middle income
        "UMC",  # Upper middle income
        "HIC",  # High income
    }
)

# Fallback for countries currently INX (Not classified) at WB. Values = last
# published classification. Keeps stratified sampling workable for active
# countries with suspended WB classification (e.g. ETH, VEN).
_WB_INX_FALLBACK: dict[str, str] = {
    "ETH": "LIC",
    "VEN": "UMC",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _http_get(url: str, timeout: int = 60) -> bytes:
    """Fetch a URL and return its content as bytes."""
    logger.debug("Fetching %s", url)
    req = Request(url, headers={"User-Agent": "SpatialBenchRAG/0.1"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _download_countries_geojson(dest: Path) -> None:
    """Download Natural Earth 110m countries shapefile and convert to GeoJSON.

    The Natural Earth source is a ZIP containing an ESRI Shapefile.  We
    extract it in-memory using geopandas and write out a GeoJSON file.

    Args:
        dest: Target file path (e.g. data/reference/countries.geojson).
    """
    logger.info("Downloading Natural Earth country boundaries …")
    raw = _http_get(_NATURAL_EARTH_URL)

    # Unzip in memory and read with geopandas
    import geopandas as gpd  # type: ignore[import-untyped]

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        # Find the .shp member
        shp_names = [n for n in zf.namelist() if n.endswith(".shp")]
        if not shp_names:
            raise RuntimeError("Natural Earth ZIP contains no .shp file")

        # Extract to a temporary in-memory VFS via fiona/pyogrio
        # We write all members to a temp dir then read the shapefile
        import tempfile, shutil, os

        tmp_dir = tempfile.mkdtemp(prefix="ne_countries_")
        try:
            zf.extractall(tmp_dir)
            shp_path = os.path.join(tmp_dir, shp_names[0])
            gdf = gpd.read_file(shp_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Keep only essential columns
    keep_cols = ["ISO_A3", "ISO_A3_EH", "NAME", "NAME_EN", "ADMIN", "geometry"]
    available = [c for c in keep_cols if c in gdf.columns]
    gdf = gdf[available].copy()

    # Normalise ISO3 — prefer ISO_A3, fall back to ISO_A3_EH.
    # Natural Earth encodes "-99" as a 3-char placeholder in ISO_A3 for
    # several real countries (Norway, France, Kosovo, ...) — the actual
    # ISO3 lives in ISO_A3_EH. A plain `.str.len() == 3` check would accept
    # "-99" literally and miss Norway/France/Kosovo, driving them into the
    # -99 bucket at enrichment time.
    import re as _re
    _iso3_re = _re.compile(r"^[A-Z]{3}$")
    if "ISO_A3" in gdf.columns and "ISO_A3_EH" in gdf.columns:
        valid = gdf["ISO_A3"].astype(str).str.match(_iso3_re)
        gdf["iso3"] = gdf["ISO_A3"].where(valid, gdf["ISO_A3_EH"])
    elif "ISO_A3" in gdf.columns:
        gdf["iso3"] = gdf["ISO_A3"]
    else:
        gdf["iso3"] = gdf.get("ISO_A3_EH", "")

    gdf = gdf.to_crs(epsg=4326)  # ensure WGS-84

    dest.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(dest, driver="GeoJSON")
    logger.info("Saved %d country boundaries to %s", len(gdf), dest)


def _download_wb_income_groups(dest: Path) -> None:
    """Download World Bank country income classifications.

    The World Bank API returns a list of all countries.  We extract only
    those with a recognised income-level ID (ignoring regional aggregates
    which have level id "NA" or "")).

    The result is a dict mapping ISO3 → income_level_id, e.g.::

        {"DEU": "HIC", "BGD": "LMC", ...}

    Args:
        dest: Target JSON file path.
    """
    logger.info("Downloading World Bank income group data …")
    raw = _http_get(_WORLD_BANK_URL)
    payload = json.loads(raw)

    # World Bank API response is a two-element list: [metadata, data]
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(f"Unexpected World Bank API response structure: {type(payload)}")

    data = payload[1]
    income_map: dict[str, str] = {}

    for entry in data:
        iso3: str = entry.get("id", "")
        income_info = entry.get("incomeLevel", {}) or {}
        level_id: str = income_info.get("id", "")

        if len(iso3) != 3:
            continue
        key = iso3.upper()
        if level_id in _WB_INCOME_LEVEL_IDS:
            income_map[key] = level_id
        elif level_id == "INX" and key in _WB_INX_FALLBACK:
            income_map[key] = _WB_INX_FALLBACK[key]
            logger.info("Applied INX fallback: %s -> %s", key, income_map[key])

    if not income_map:
        raise RuntimeError(
            "World Bank API returned no usable income group entries.  "
            "Check the URL or API availability."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        json.dump(income_map, f, indent=2, sort_keys=True)

    logger.info("Saved %d country income groups to %s", len(income_map), dest)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_reference_data(
    countries_path: Optional[Path] = None,
    income_groups_path: Optional[Path] = None,
) -> None:
    """Download reference data files if they do not already exist.

    This is idempotent — existing files are left untouched.  Delete a file
    manually to force a fresh download.

    Args:
        countries_path: Override for the countries GeoJSON output path.
            Defaults to the path from config.py.
        income_groups_path: Override for the income-groups JSON output path.
            Defaults to the path from config.py.
    """
    # Lazy import to allow the module to load without config being evaluated
    if countries_path is None or income_groups_path is None:
        from question_generator.config import (
            COUNTRIES_GEOJSON_PATH,
            WB_INCOME_GROUPS_JSON_PATH,
        )

        countries_path = countries_path or COUNTRIES_GEOJSON_PATH
        income_groups_path = income_groups_path or WB_INCOME_GROUPS_JSON_PATH

    countries_path = Path(countries_path)
    income_groups_path = Path(income_groups_path)

    if not countries_path.exists():
        try:
            _download_countries_geojson(countries_path)
        except (URLError, Exception) as exc:
            logger.error("Failed to download country boundaries: %s", exc)
            raise
    else:
        logger.info("Country boundaries already cached at %s", countries_path)

    if not income_groups_path.exists():
        try:
            _download_wb_income_groups(income_groups_path)
        except (URLError, Exception) as exc:
            logger.error("Failed to download income groups: %s", exc)
            raise
    else:
        logger.info("Income groups already cached at %s", income_groups_path)


def load_countries_geodataframe(countries_path: Optional[Path] = None):
    """Load the cached countries GeoDataFrame.

    Calls ``ensure_reference_data()`` first so the file is downloaded if
    it does not exist yet.

    Args:
        countries_path: Override path.  Defaults to config value.

    Returns:
        geopandas.GeoDataFrame with at least columns:
        ``iso3``, ``NAME``, ``geometry``.
    """
    import geopandas as gpd  # type: ignore[import-untyped]

    if countries_path is None:
        from question_generator.config import COUNTRIES_GEOJSON_PATH

        countries_path = COUNTRIES_GEOJSON_PATH

    countries_path = Path(countries_path)
    ensure_reference_data(countries_path=countries_path)
    return gpd.read_file(countries_path)


def load_eez_geodataframe(eez_path: Optional[Path] = None):
    """Load the Marine Regions World EEZ v12 shapefile.

    Not auto-downloaded — Marine Regions requires a free academic
    registration form. Expected placement:
        data/reference/eez/World_EEZ_v12_20231025/eez_v12.shp

    Used as Layer-2 fallback in country_enrichment when an entity's
    geometry falls outside all country polygons (typical case: marine
    features like USGS earthquakes or NOAA hurricane segments).

    Returns:
        geopandas.GeoDataFrame with ``ISO_SOV1`` (sovereign ISO3) and
        ``geometry`` columns retained. Non-``200NM``-type polygons
        (joint regimes, overlapping claims) are dropped so a single
        sovereign is attributable per geometry.
    """
    import geopandas as gpd  # type: ignore[import-untyped]

    if eez_path is None:
        from question_generator.config import REFERENCE_DATA_PATH

        eez_path = REFERENCE_DATA_PATH / "eez" / "World_EEZ_v12_20231025" / "eez_v12.shp"

    eez_path = Path(eez_path)
    if not eez_path.exists():
        raise FileNotFoundError(
            f"EEZ shapefile not found at {eez_path}. Download World EEZ v12 from "
            "https://www.marineregions.org/downloads.php (registration required) "
            "and extract into data/reference/eez/World_EEZ_v12_20231025/."
        )

    gdf = gpd.read_file(eez_path)
    # Keep only unambiguous 200NM territorial EEZs — joint regimes and
    # overlapping claims would assign multiple sovereigns and confuse
    # the tie-break at country_enrichment's Layer-2 dedup step.
    if "POL_TYPE" in gdf.columns:
        gdf = gdf[gdf["POL_TYPE"] == "200NM"].copy()
    gdf = gdf[["ISO_SOV1", "geometry"]].dropna(subset=["ISO_SOV1"])
    gdf = gdf.to_crs(epsg=4326)
    return gdf


def load_income_groups(
    income_groups_path: Optional[Path] = None,
) -> dict[str, str]:
    """Load the cached World Bank income group mapping.

    Calls ``ensure_reference_data()`` first so the file is downloaded if
    it does not exist yet.

    Args:
        income_groups_path: Override path.  Defaults to config value.

    Returns:
        dict mapping ISO3 country code → World Bank income level ID
        (one of ``"LIC"``, ``"LMC"``, ``"UMC"``, ``"HIC"``).
    """
    if income_groups_path is None:
        from question_generator.config import WB_INCOME_GROUPS_JSON_PATH

        income_groups_path = WB_INCOME_GROUPS_JSON_PATH

    income_groups_path = Path(income_groups_path)
    ensure_reference_data(income_groups_path=income_groups_path)

    with income_groups_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_reference_data()
    print("Reference data is ready.")
