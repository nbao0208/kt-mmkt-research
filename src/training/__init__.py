"""
Training Package for Knowledge Tracing.
"""

from src.training.losses import (
    MaskedBCELoss,
    SolutionDistillationLoss,
    CTTDifficultyCalibrationLoss,
    OrthogonalDisentanglementLoss,
    DeepIRTResidualRegLoss,
    ACDMMKTLoss,
    PrivilegedSolutionDistillationLoss,
    ModalitySuppressionRegularizationLoss,
    MGMMKTv2Loss,
    BayesianWeightedDifficultyLoss,
    ACDMMKTv2Loss,
    ACDMMKTv3Loss,
)
from src.training.trainer import MGMMKTTrainer
from src.training.mg_v2_trainer import MGMMKTv2Trainer
from src.training.acd_trainer import ACDMMKTTrainer
from src.training.acd_v2_trainer import ACDMMKTv2Trainer
from src.training.acd_v3_trainer import ACDMMKTv3Trainer

try:
    from src.training.pykt_trainer import PyKTTrainer
except ImportError:
    PyKTTrainer = None

__all__ = [
    "MaskedBCELoss",
    "SolutionDistillationLoss",
    "CTTDifficultyCalibrationLoss",
    "OrthogonalDisentanglementLoss",
    "DeepIRTResidualRegLoss",
    "ACDMMKTLoss",
    "PrivilegedSolutionDistillationLoss",
    "ModalitySuppressionRegularizationLoss",
    "MGMMKTv2Loss",
    "BayesianWeightedDifficultyLoss",
    "ACDMMKTv2Loss",
    "ACDMMKTv3Loss",
    "MGMMKTTrainer",
    "MGMMKTv2Trainer",
    "ACDMMKTTrainer",
    "ACDMMKTv2Trainer",
    "ACDMMKTv3Trainer",
    "PyKTTrainer",
]
