"""
Unified Trainer for pyKT Baseline Models (DKT, DKVMN, SAKT) on XES3G5M.

Implements the official training loop, loss calculation with masked binary cross-entropy,
validation monitoring, early stopping, and checkpoint management.
Compatible with Apple Silicon MPS, NVIDIA CUDA, and CPU.
"""

import os
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from src.models.baselines.pykt_models import init_pykt_baseline, sync_pykt_device
from src.utils.logger import get_logger


class PyKTTrainer:
    """
    Trainer executing 5-fold cross-validation for pyKT baseline models.
    """

    def __init__(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        train_config: Dict[str, Any],
        fold: int = 0,
        device: Optional[torch.device] = None,
        logger: Optional[Any] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model_name: Baseline model identifier ('dkt', 'dkvmn', 'sakt').
            model_config: Architecture hyperparameters dictionary.
            train_config: Training configuration dictionary.
            fold: Active cross-validation fold index (0..4).
            device: Execution device (auto-detected if None).
            logger: Custom logger instance.
        """
        self.model_name = model_name.lower().strip()
        self.model_config = model_config
        self.train_config = train_config
        self.fold = fold

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        sync_pykt_device(self.device)

        # Logger
        self.logger = logger or get_logger(f"PyKTTrainer_{self.model_name}_Fold{self.fold}")

        # Training hyperparameters
        self.num_epochs = self.train_config.get("num_epochs", 30)
        self.lr = float(self.train_config.get("lr", 1e-3))
        self.weight_decay = float(self.train_config.get("weight_decay", 0.0))
        self.patience = self.train_config.get("patience", 10)
        self.max_grad_norm = float(self.train_config.get("max_grad_norm", 1.0))

        # Output paths
        base_save_dir = self.train_config.get("save_dir", "outputs/checkpoints")
        self.save_dir = os.path.join(base_save_dir, f"{self.model_name}_fold{self.fold}")
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize baseline model directly from pyKT
        self.model = init_pykt_baseline(
            model_name=self.model_name,
            model_config=self.model_config,
            device=self.device,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _forward_step(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute forward propagation for a single batch based on model type.

        Returns:
            Tuple of (predictions [B, L-1], target responses [B, L-1], select mask [B, L-1]).
        """
        c = data["cseqs"].to(self.device)
        r = data["rseqs"].to(self.device)
        cshft = data["shft_cseqs"].to(self.device)
        rshft = data["shft_rseqs"].to(self.device)
        sm = data["smasks"].to(self.device)

        if self.model_name in ["dkt", "dkt+"]:
            y = self.model(c.long(), r.long())
            # Gather predictions corresponding to shifted concept indices
            num_c = getattr(self.model, "num_c", self.model_config.get("num_c", 865))
            y = (y * one_hot(cshft.long(), num_c)).sum(-1)
        elif self.model_name in ["dkvmn", "deep_irt"]:
            cc = torch.cat((c[:, 0:1], cshft), dim=1)
            cr = torch.cat((r[:, 0:1], rshft), dim=1)
            y = self.model(cc.long(), cr.long())
            y = y[:, 1:]
        elif self.model_name in ["sakt"]:
            y = self.model(c.long(), r.long(), cshft.long())
        else:
            raise NotImplementedError(f"Forward step not implemented for {self.model_name}")

        return y, rshft, sm

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train the model for one epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_data in train_loader:
            self.optimizer.zero_grad()
            y, rshft, sm = self._forward_step(batch_data)

            active_y = torch.masked_select(y, sm)
            active_t = torch.masked_select(rshft, sm)

            if active_y.numel() == 0:
                continue

            loss = binary_cross_entropy(active_y.float(), active_t.float())
            loss.backward()

            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate concept-level prediction performance on the validation set.

        Returns:
            Dictionary with 'auc', 'acc', and 'loss'.
        """
        self.model.eval()
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        total_loss = 0.0
        num_batches = 0

        for batch_data in val_loader:
            y, rshft, sm = self._forward_step(batch_data)

            active_y = torch.masked_select(y, sm)
            active_t = torch.masked_select(rshft, sm)

            if active_y.numel() == 0:
                continue

            loss = binary_cross_entropy(active_y.float(), active_t.float())
            total_loss += loss.item()
            num_batches += 1

            all_preds.append(active_y.detach().cpu().numpy())
            all_targets.append(active_t.detach().cpu().numpy())

        if len(all_targets) == 0:
            return {"auc": 0.5, "acc": 0.0, "loss": 0.0}

        y_true = np.concatenate(all_targets, axis=0)
        y_score = np.concatenate(all_preds, axis=0)

        auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.5
        acc = float(accuracy_score(y_true, (y_score >= 0.5).astype(int)))
        mean_loss = total_loss / max(num_batches, 1)

        return {"auc": auc, "acc": acc, "loss": mean_loss}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Run the complete training loop with validation monitoring and early stopping.

        Returns:
            Dictionary of training history, best metrics, and checkpoint file path.
        """
        self.logger.info(
            f"Starting training for {self.model_name.upper()} Fold {self.fold} "
            f"on {self.device} (Max Epochs: {self.num_epochs}, Patience: {self.patience})"
        )

        best_auc = 0.0
        best_acc = 0.0
        best_epoch = -1
        history: List[Dict[str, Any]] = []

        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            t_epoch_start = time.time()
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            epoch_duration = time.time() - t_epoch_start

            val_auc = val_metrics["auc"]
            val_acc = val_metrics["acc"]
            val_loss = val_metrics["loss"]

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_auc": round(val_auc, 4),
                "val_acc": round(val_acc, 4),
                "epoch_time_s": round(epoch_duration, 2),
            }
            history.append(record)

            self.logger.info(
                f"Epoch {epoch:02d}/{self.num_epochs:02d} - "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | Val ACC: {val_acc:.4f} | "
                f"Time: {epoch_duration:.1f}s"
            )

            # Checkpoint saving if improved
            if val_auc > best_auc + 1e-4:
                best_auc = val_auc
                best_acc = val_acc
                best_epoch = epoch

                ckpt_path = os.path.join(self.save_dir, "best_model.pt")
                torch.save(
                    {
                        "model_name": self.model_name,
                        "model_config": self.model_config,
                        "fold": self.fold,
                        "epoch": epoch,
                        "val_auc": val_auc,
                        "val_acc": val_acc,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                self.logger.info(
                    f"--> Best checkpoint saved (Epoch {epoch:02d}, Val AUC: {val_auc:.4f})"
                )

            # Early stopping check
            if epoch - best_epoch >= self.patience:
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch} (No improvement since epoch {best_epoch})."
                )
                break

        total_time = time.time() - start_time
        self.logger.info(
            f"Finished training {self.model_name.upper()} Fold {self.fold} in {total_time:.1f}s. "
            f"Best Val AUC: {best_auc:.4f} (Epoch {best_epoch})"
        )

        return {
            "model_name": self.model_name,
            "fold": self.fold,
            "best_epoch": best_epoch,
            "best_val_auc": best_auc,
            "best_val_acc": best_acc,
            "total_time_s": total_time,
            "checkpoint_path": os.path.join(self.save_dir, "best_model.pt"),
            "history": history,
        }

