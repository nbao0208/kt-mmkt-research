"""
Data Processing and Loading Package for Knowledge Tracing.
"""

from src.data.xes3g5m_dataset import (
    XES3G5MDataset,
    collate_xes3g5m,
    get_xes3g5m_dataloaders,
    build_test_strided_windows,
    create_mg_static_item_bank,
)

try:
    from src.data.pykt_dataset import (
        get_pykt_dataloaders,
        load_test_grouped_data,
        load_stratification_metadata,
    )
except ImportError:
    get_pykt_dataloaders = None
    load_test_grouped_data = None
    load_stratification_metadata = None

__all__ = [
    "XES3G5MDataset",
    "collate_xes3g5m",
    "get_xes3g5m_dataloaders",
    "build_test_strided_windows",
    "create_mg_static_item_bank",
    "get_pykt_dataloaders",
    "load_test_grouped_data",
    "load_stratification_metadata",
]
