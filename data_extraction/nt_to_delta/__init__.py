"""N-Triples to Delta Table Converter.

This package provides a two-stage pipeline for converting N-Triples (.nt) files
produced by knowledge graph crawlers into Delta tables suitable for the
benchmark question generator.

Stage 1: Rust parser (nt_parser binary) - Fast streaming N-Triples parsing
Stage 2: PySpark transformer - Pivot triples to entity-centric format with H3

Usage:
    python -m data_extraction.nt_to_delta --input data.nt --output /path/to/delta
"""

from data_extraction.nt_to_delta.config import NtToDeltaConfig
from data_extraction.nt_to_delta.transformer import TriplesTransformer

__all__ = ["NtToDeltaConfig", "TriplesTransformer"]
