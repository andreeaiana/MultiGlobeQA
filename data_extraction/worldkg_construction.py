"""
WorldKG Knowledge Graph Construction from OpenStreetMap PBF files.

This module converts OpenStreetMap data into RDF triples following the WorldKG schema.
Code adapted from: https://github.com/alishiba14/WorldKG-Knowledge-Graph.git
"""

import logging
import sys
import time
import urllib.parse
from typing import Any

import osmium
import pandas as pd
from rdflib import Literal, Namespace, URIRef
from rdflib.namespace import RDF

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OSM2RDFHandler(osmium.SimpleHandler):
    """
    Handler for converting OpenStreetMap nodes to RDF triples.

    Processes OSM PBF files and outputs N-Triples format RDF data
    following the WorldKG ontology.
    """

    # Namespaces
    WD = Namespace("http://www.wikidata.org/wiki/")
    WDT = Namespace("http://www.wikidata.org/prop/direct/")
    WKG = Namespace("http://www.worldkg.org/resource/")
    WKGS = Namespace("http://www.worldkg.org/schema/")
    GEO = Namespace("http://www.opengis.net/ont/geosparql#")
    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    SF = Namespace("http://www.opengis.net/ont/sf#")
    OSMN = Namespace("https://www.openstreetmap.org/node/")

    def __init__(
        self,
        output_file_path: str,
        ontology_csv_path: str = "data/worldkg/OSM_Ontology_map_features.csv",
        key_list_csv_path: str = "data/worldkg/Key_List.csv",
    ) -> None:
        """
        Initialize the OSM to RDF handler.

        Args:
            output_file_path: Path for the output N-Triples file.
            ontology_csv_path: Path to the OSM ontology mapping CSV.
            key_list_csv_path: Path to the allowed keys CSV.
        """
        logger.info(
            "Code adapted from: https://github.com/alishiba14/WorldKG-Knowledge-Graph.git"
        )
        osmium.SimpleHandler.__init__(self)

        self.counts: int = 0
        self.output_file = open(output_file_path, "w", encoding="utf-8")

        # Load ontology and mapping data
        self.supersub = self._load_ontology(ontology_csv_path)
        self.key_list = self._load_key_list(key_list_csv_path)
        self.dict_class = self._build_class_dict()

    def _load_ontology(self, path: str) -> pd.DataFrame:
        """Load and prepare the OSM ontology mapping."""
        try:
            df = pd.read_csv(path, sep="\t", encoding="utf-8")
            return df.drop_duplicates()
        except FileNotFoundError:
            logger.warning(f"Ontology file not found: {path}")
            return pd.DataFrame(columns=["key", "value", "appendedClass"])

    def _load_key_list(self, path: str) -> list[str]:
        """Load the list of allowed OSM keys."""
        try:
            df = pd.read_csv(path, sep="\t", encoding="utf-8")
            return list(df["key"])
        except FileNotFoundError:
            logger.warning(f"Key list file not found: {path}")
            return []

    def _build_class_dict(self) -> dict[str, list[str]]:
        """Build a dictionary mapping keys to their subclass values."""
        if self.supersub.empty:
            return {}
        return (
            self.supersub.groupby("key")["value"]
            .apply(list)
            .reset_index(name="subclasses")
            .set_index("key")
            .to_dict()["subclasses"]
        )

    def write_triple(self, s: URIRef, p: URIRef, o: Any) -> None:
        """
        Write a single triple in N-Triples format.

        Args:
            s: Subject URI.
            p: Predicate URI.
            o: Object (URI or Literal).
        """
        try:
            self.output_file.write(f"{s.n3()} {p.n3()} {o.n3()} .\n")
            self.counts += 1
            if self.counts % 1_000_000 == 0:
                logger.info(f"Processed {self.counts:,} triples...")
        except (AttributeError, UnicodeEncodeError) as e:
            logger.debug(f"Could not write triple: {e}")

    def to_camel_case_class(self, word: str) -> str:
        """Convert a string to CamelCase for class names."""
        word = word.replace(":", "_")
        return "".join(x.capitalize() or "_" for x in word.split("_"))

    def to_camel_case_class_append(self, key: str, val: str) -> str | None:
        """Get the appended class name from the ontology mapping."""
        try:
            result = self.supersub.loc[
                (self.supersub["value"] == val) & (self.supersub["key"] == key)
            ]["appendedClass"].values
            return result[0] if len(result) > 0 else None
        except (KeyError, IndexError):
            return None

    def to_camel_case_key(self, input_str: str) -> str:
        """Convert a string to camelCase for property names."""
        input_str = input_str.replace(":", "_")
        words = input_str.split("_")
        return words[0] + "".join(x.title() for x in words[1:])

    def print_triple(self, s_id: str, p: str, o: str) -> None:
        """
        Process and write a triple based on the property type.

        Args:
            s_id: Subject node ID.
            p: Property/predicate name.
            o: Object value.
        """
        if p in self.dict_class:
            if o in self.dict_class[p]:
                appended_class = self.to_camel_case_class_append(p, o)
                if appended_class:
                    rel = URIRef(f"http://www.worldkg.org/resource/{s_id}")
                    instance_of = URIRef(RDF.type)
                    res = URIRef(f"http://www.worldkg.org/schema/{appended_class}")
                    self.write_triple(rel, instance_of, res)

            if o == "Yes":
                rel = URIRef(f"http://www.worldkg.org/resource/{s_id}")
                instance_of = URIRef(RDF.type)
                res = URIRef(
                    f"http://www.worldkg.org/schema/{self.to_camel_case_class(p)}"
                )
                self.write_triple(rel, instance_of, res)
        else:
            self._handle_special_properties(s_id, p, o)

    def _handle_special_properties(self, s_id: str, p: str, o: str) -> None:
        """Handle special property types like Point, osmLink, name, etc."""
        if p == "Point":
            sub = URIRef(f"http://www.worldkg.org/resource/{s_id}")
            geoprop = URIRef("http://www.worldkg.org/schema/spatialObject")
            geoobj = URIRef(f"http://www.worldkg.org/resource/geo{s_id}")
            prop = URIRef("http://www.opengis.net/ont/sf#Point")
            typ = URIRef(RDF.type)

            self.write_triple(sub, geoprop, geoobj)
            self.write_triple(geoobj, typ, prop)
            self.write_triple(
                geoobj,
                self.GEO["asWKT"],
                Literal(o, datatype=self.GEO.wktLiteral),
            )

        elif p == "osmLink":
            sub = URIRef(f"http://www.worldkg.org/resource/{s_id}")
            prop = URIRef("http://www.worldkg.org/schema/osmLink")
            obj = URIRef(f"https://www.openstreetmap.org/node/{o}")
            self.write_triple(sub, prop, obj)

        elif p == "name":
            sub = URIRef(f"http://www.worldkg.org/resource/{s_id}")
            prop = URIRef("http://www.w3.org/2000/01/rdf-schema#label")
            self.write_triple(sub, prop, Literal(o))

        elif p == "wikidata":
            sub = URIRef(f"http://www.worldkg.org/resource/{s_id}")
            prop = URIRef(f"http://www.worldkg.org/schema/{p}")
            if o.startswith("Q"):
                obj: Any = URIRef(f"http://www.wikidata.org/wiki/{o}")
            else:
                obj = Literal(o)
            self.write_triple(sub, prop, obj)

        elif p == "wikipedia":
            sub = URIRef(f"http://www.worldkg.org/resource/{s_id}")
            prop = URIRef("http://www.worldkg.org/schema/wikipedia")
            try:
                parts = o.split(":", 1)
                country = parts[0] if len(parts) > 1 else ""
                ids = parts[1] if len(parts) > 1 else o
                url = f"https://{country}.wikipedia.org/wiki/{country}:{ids}"
                url = urllib.parse.quote(url, safe=":/")
                obj = URIRef(url)
                self.write_triple(sub, prop, obj)
            except (ValueError, IndexError):
                pass

        elif p in self.key_list:
            sub = URIRef(f"http://www.worldkg.org/resource/{s_id}")
            prop = URIRef(
                f"http://www.worldkg.org/schema/{self.to_camel_case_key(p)}"
            )
            self.write_triple(sub, prop, Literal(o))

    def close(self) -> None:
        """Close the output file and log final statistics."""
        logger.info(f"Total triple count: {self.counts:,}")
        self.output_file.close()

    def node(self, n: osmium.osm.Node) -> None:
        """
        Process an OSM node.

        Args:
            n: The OSM node to process.
        """
        if len(n.tags) > 1:
            node_id = str(n.id)
            point = f"Point({n.location.lon} {n.location.lat})"

            self.print_triple(node_id, "Point", point)
            self.print_triple(node_id, "osmLink", node_id)

            for k, v in n.tags:
                val = str(v)
                # Escape special characters
                val = val.replace("\\", "\\\\")
                val = val.replace('"', '\\"')
                val = val.replace("\n", " ")

                k = k.replace(" ", "")
                self.print_triple(node_id, k, val)


def main() -> int:
    """
    Main entry point for WorldKG construction.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    if len(sys.argv) < 3:
        print("Usage: python worldkg_construction.py <input.osm.pbf> <output.nt>")
        return 1

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    start_time = time.time()

    handler = OSM2RDFHandler(output_file)
    handler.apply_file(input_file)
    handler.close()

    elapsed = time.time() - start_time
    logger.info(f"Processing completed in {elapsed:.2f} seconds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
