"""
Baseline models for Knowledge Tracing.
"""

from src.models.baselines.sakt import IDOnlySAKT

try:
    from src.models.baselines.pykt_models import init_pykt_baseline, sync_pykt_device
except ImportError:
    init_pykt_baseline = None
    sync_pykt_device = None

__all__ = [
    "IDOnlySAKT",
    "init_pykt_baseline",
    "sync_pykt_device",
]
