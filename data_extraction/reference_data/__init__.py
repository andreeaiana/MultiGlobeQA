"""Reference data package for hierarchical sampling.

Provides helpers to download and cache country boundary and World Bank
income-group data that are required by the HierarchicalSampler.
"""

from data_extraction.reference_data.downloader import (
    ensure_reference_data,
    load_countries_geodataframe,
    load_income_groups,
)

__all__ = [
    "ensure_reference_data",
    "load_countries_geodataframe",
    "load_income_groups",
]
