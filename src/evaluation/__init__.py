"""
Evaluation Package for Knowledge Tracing.
"""

from src.evaluation.metrics import (
    calculate_metrics,
    multimodal_stratified_evaluation,
)
from src.evaluation.explainability import FirstOrderExplainer
from src.evaluation.explainability_v2 import MultiLevelExplainerV2
from src.evaluation.acd_explainability import ACDExplainabilityEngine
from src.evaluation.acd_v2_explainability import ACDv2ExplainabilityEngine
from src.evaluation.acd_v3_explainability import ACDv3ExplainabilityEngine

try:
    from src.evaluation.pykt_evaluator import PyKTEvaluator
except ImportError:
    PyKTEvaluator = None

__all__ = [
    "calculate_metrics",
    "multimodal_stratified_evaluation",
    "FirstOrderExplainer",
    "MultiLevelExplainerV2",
    "ACDExplainabilityEngine",
    "ACDv2ExplainabilityEngine",
    "ACDv3ExplainabilityEngine",
    "PyKTEvaluator",
]
