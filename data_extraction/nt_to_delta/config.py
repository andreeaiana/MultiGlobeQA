"""Configuration for N-Triples to Delta conversion."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NtToDeltaConfig:
    """Configuration for the N-Triples to Delta conversion pipeline.

    Attributes:
        input_path: Path to the input N-Triples file.
        triples_delta_path: Path for intermediate triples Delta table.
        entities_delta_path: Path for final entities Delta table.
        h3_resolution: H3 resolution for spatial indexing (0-15, default 4).
        batch_size: Batch size for Rust parser (default 100000).
        rust_binary_path: Path to the nt_parser Rust binary.
        overwrite: Whether to overwrite existing Delta tables.
    """

    input_path: Path
    triples_delta_path: Path
    entities_delta_path: Path
    kg_source: str = "worldkg"  # "worldkg", "osmh3kg", "kwg"
    h3_resolution: int = 7
    batch_size: int = 100_000
    rust_binary_path: Path = field(default_factory=lambda: Path("nt_parser"))
    overwrite: bool = False
    class_lookup_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "entity_constraints" / "class_lookup.csv"
    )

    # RDF predicates for entity extraction
    # WorldKG-specific English name predicate takes priority over rdfs:label
    name_en_predicate: str = "http://www.worldkg.org/schema/nameEn"
    label_predicates: list[str] = field(
        default_factory=lambda: [
            "http://www.w3.org/2000/01/rdf-schema#label",
            "http://schema.org/name",
            "https://schema.org/name",
        ]
    )
    # Predicates linking an entity to its geometry node (intermediate hop)
    geometry_link_predicates: list[str] = field(
        default_factory=lambda: [
            "http://www.worldkg.org/schema/spatialObject",
            "http://www.opengis.net/ont/geosparql#hasGeometry",
        ]
    )
    # Predicates on geometry nodes that yield WKT literals
    geometry_predicates: list[str] = field(
        default_factory=lambda: [
            "http://www.opengis.net/ont/geosparql#asWKT",
        ]
    )
    type_predicates: list[str] = field(
        default_factory=lambda: [
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        ]
    )

    # Address predicates for name disambiguation (OSMH3KG).
    # When set, builds "Name, Street Nr" or "Name (Suburb)" display names.
    address_street_predicate: str | None = None
    address_housenumber_predicate: str | None = None
    address_suburb_predicate: str | None = None

    # Temporal extraction mode:
    #   "column"    — OSMH3KG: triples have a 4th `date` column,
    #                 start_date = MIN(date), end_date = MAX(date) per subject
    #   "predicate" — KWG: dates are stored as triple objects on temporal predicates,
    #                 start_date from start predicates, end_date from end predicates
    #   None        — WorldKG: no temporal data available
    temporal_mode: str | None = None

    # Predicates whose object contains a start date literal (used when temporal_mode="predicate")
    start_date_predicates: list[str] = field(
        default_factory=lambda: [
            "http://www.w3.org/ns/prov#generatedAtTime",
            "http://stko-kwg.geog.ucsb.edu/lod/ontology/fireDiscoveryTime",
        ]
    )
    # Predicates whose object contains an end date literal (used when temporal_mode="predicate")
    end_date_predicates: list[str] = field(
        default_factory=lambda: [
            "http://www.w3.org/ns/prov#invalidatedAtTime",
            "http://stko-kwg.geog.ucsb.edu/lod/ontology/fireOutTime",
        ]
    )

    def __post_init__(self) -> None:
        """Convert string paths to Path objects and apply KG-specific predicate presets."""
        if isinstance(self.input_path, str):
            self.input_path = Path(self.input_path)
        if isinstance(self.triples_delta_path, str):
            self.triples_delta_path = Path(self.triples_delta_path)
        if isinstance(self.entities_delta_path, str):
            self.entities_delta_path = Path(self.entities_delta_path)
        if isinstance(self.rust_binary_path, str):
            self.rust_binary_path = Path(self.rust_binary_path)

        # Predicate presets per KG source. Defaults above target worldkg/kwg
        # (full URIs). The current osmh3kg pipeline (kg_creation in
        # data_extraction/osmh3kg/data_preparation/osm_kg_data_preparation.py)
        # writes full URIs for rdf:type / geo:hasGeometry / geo:asWKT, but
        # bare OSM tag keys for labels (name, name:en). Older planet triples
        # at /ceph/mboeckli/stkg/gold/knowledge_graph used short prefixes
        # (rdf:type, geo:asWKT, ...) — that format is deprecated here; the
        # 12-country subgraph at data/osmh3kg/osmh3kg_triples_subgraph is
        # canonical. Without matching predicates every filter matches zero
        # rows and the transformer silently writes an empty entity table.
        if self.kg_source == "osmh3kg":
            self.name_en_predicate = "name:en"
            self.label_predicates = ["name"]
            self.start_date_predicates = []
            self.end_date_predicates = []

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        if not 0 <= self.h3_resolution <= 15:
            raise ValueError(f"H3 resolution must be between 0 and 15, got {self.h3_resolution}")

        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
