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
from src.data.eedi_dataset import (
    EediDataset,
    collate_eedi,
    get_eedi_dataloaders,
    build_eedi_test_strided_windows,
    create_eedi_mg_static_item_bank,
    load_eedi_static_item_bank,
)
from src.data.eedi_preprocessor import EediTask34Preprocessor
from src.data.dataset_router import (
    resolve_dataset_key,
    resolve_dataset_paths,
    resolve_split_tag,
    load_static_item_bank_for_dataset,
    get_dataloaders_for_dataset,
    load_acd_v3_fold_metadata,
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
    "EediDataset",
    "collate_eedi",
    "get_eedi_dataloaders",
    "build_eedi_test_strided_windows",
    "create_eedi_mg_static_item_bank",
    "load_eedi_static_item_bank",
    "EediTask34Preprocessor",
    "resolve_dataset_key",
    "resolve_dataset_paths",
    "resolve_split_tag",
    "load_static_item_bank_for_dataset",
    "get_dataloaders_for_dataset",
    "load_acd_v3_fold_metadata",
    "get_pykt_dataloaders",
    "load_test_grouped_data",
    "load_stratification_metadata",
]
