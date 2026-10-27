"""
Training and Validation Pipeline for MG-MMKT.

Manages the end-to-end training loop:
- Optimizer configuration (AdamW with weight decay).
- Learning rate scheduling (Cosine Annealing / ReduceLROnPlateau).
- Gradient norm clipping to prevent exploding gradients.
- Epoch-wise validation with pyKT global micro ROC-AUC and Accuracy.
- Early Stopping based on validation AUC and automated checkpoint serialization.
"""

from typing import Dict, Any, Optional, List, Union
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.training.losses import MaskedBCELoss
from src.evaluation.metrics import calculate_metrics
from src.utils.logger import get_logger


class MGMMKTTrainer:
    """
    Trainer for Masked-Gated Multimodal Knowledge Tracing models.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        patience: int = 10,
        device: Optional[Union[str, torch.device]] = None,
        save_dir: str = "outputs/checkpoints",
        model_cfg: Optional[Dict[str, Any]] = None,
        warmup_epochs: int = 2,
        eta_min: float = 1e-5,
        logger: Optional[Any] = None,
    ):
        """
        Initialize MGMMKTTrainer.

        Args:
            model: PyTorch model instance (MGMMKT).
            train_loader: DataLoader for the training set (4 folds merged).
            val_loader: DataLoader for the validation set (1 fold held-out).
            test_loader: Optional DataLoader for the independent test set.
            lr: Learning rate for AdamW optimizer (default: 1e-3).
            weight_decay: L2 regularization coefficient (default: 1e-4).
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0).
            patience: Number of epochs without validation AUC improvement before early stopping (default: 10).
            device: Compute device ('cuda', 'mps', or 'cpu').
            save_dir: Directory where checkpoints and training history are saved.
            model_cfg: Optional dictionary specifying model hyperparameters to persist in checkpoint.
            warmup_epochs: Number of linear warmup epochs (default: 2).
            eta_min: Minimum learning rate for cosine annealing (default: 1e-5).
            logger: Optional logger instance.
        """
        self.device = device or (
            torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.save_dir = save_dir
        self.model_cfg = model_cfg or getattr(model, "config", None)
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        self.logger = logger or get_logger("MGMMKTTrainer")

        os.makedirs(self.save_dir, exist_ok=True)

        # Loss function
        self.loss_fn = MaskedBCELoss()

        # Optimizer: Separate parameters to prevent decaying e_null, y_0, LayerNorm, and biases
        # (Fixes review issue H1: prevents norm shrinking of learnable canonical missing tokens)
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                param.ndim <= 1
                or "bias" in name
                or "ln" in name
                or "LayerNorm" in name
                or "e_null" in name
                or "y_0" in name
                or "raw_delta" in name
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)

        # Initial scheduler placeholder (will be configured with actual num_epochs in fit())
        self._setup_scheduler(num_epochs=50)

        # Training history state
        self.best_val_auc = -1.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.history: List[Dict[str, Any]] = []

    def _setup_scheduler(self, num_epochs: int) -> None:
        """
        Configure learning rate scheduler with Linear Warmup + Cosine Annealing (Plan Step 5.1).
        """
        if self.warmup_epochs > 0 and num_epochs > self.warmup_epochs:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, total_iters=self.warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, num_epochs - self.warmup_epochs),
                eta_min=self.eta_min,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs],
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, num_epochs),
                eta_min=self.eta_min,
            )

    def train_epoch(self, epoch: int) -> float:
        """
        Execute one complete training epoch across train_loader.

        Args:
            epoch: Current epoch index.

        Returns:
            Average training loss across the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch)
            logits = outputs["logits"]
            targets = batch["r_seq"]
            loss_mask = batch["loss_mask"]

            # Compute masked BCE loss normalized by valid step count
            loss = self.loss_fn(logits, targets, loss_mask)

            # Backward pass & gradient clipping
            loss.backward()
            if self.max_grad_norm > 0:
                clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        elapsed = time.time() - start_time

        # Track gate deltas if supported by model (Issue M2)
        delta_str = ""
        if hasattr(self.model, "get_gate_deltas"):
            deltas = self.model.get_gate_deltas()
            if deltas:
                delta_str = " | Deltas: " + ", ".join(f"{k}={v:.2f}" for k, v in deltas.items())

        self.logger.info(
            f"Epoch {epoch:03d} [Train] Loss: {avg_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}{delta_str} | Time: {elapsed:.1f}s"
        )
        return avg_loss

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        split_name: str = "Val",
    ) -> Dict[str, float]:
        """
        Evaluate model predictions using pyKT global micro ROC-AUC and Accuracy.

        Args:
            loader: DataLoader to evaluate on.
            split_name: Name of evaluation split (e.g. 'Val' or 'Test').

        Returns:
            Dictionary containing 'auc', 'acc', and 'num_samples'.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            outputs = self.model(batch)
            preds = outputs["preds"]  # [B, L]
            targets = batch["r_seq"]  # [B, L]
            eval_mask = batch["eval_mask"]  # [B, L]

            # Filter only active evaluation steps
            active_preds = preds[eval_mask].detach().cpu().numpy()
            active_targets = targets[eval_mask].detach().cpu().numpy()

            all_preds.extend(active_preds.tolist())
            all_targets.extend(active_targets.tolist())

        metrics = calculate_metrics(all_preds, all_targets)
        self.logger.info(
            f"[{split_name}] Global AUC: {metrics['auc']:.4f} | ACC: {metrics['acc']:.4f} | Evaluated Interactions: {metrics['num_samples']:,}"
        )
        return metrics

    def fit(self, num_epochs: int = 30) -> Dict[str, Any]:
        """
        Execute full training loop with validation, early stopping, and checkpointing.

        Args:
            num_epochs: Maximum number of epochs to train.

        Returns:
            Dictionary containing training summary and best performance metrics.
        """
        self.logger.info(
            f"Starting MG-MMKT training on {self.device} for {num_epochs} epochs (Early stopping patience: {self.patience})..."
        )

        # Configure scheduler dynamically based on actual num_epochs and warmup_epochs (Issue M3)
        self._setup_scheduler(num_epochs=num_epochs)

        model_config = getattr(self.model, "config", None) or self.model_cfg

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, split_name=f"Val-Epoch-{epoch:02d}")

            # Step learning rate scheduler
            self.scheduler.step()

            # Record history
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auc": val_metrics["auc"],
                "val_acc": val_metrics["acc"],
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(epoch_record)

            # Check for improvement in validation AUC
            val_auc = val_metrics["auc"]
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                self.epochs_no_improve = 0

                best_ckpt_path = os.path.join(self.save_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_val_auc": self.best_val_auc,
                        "model_cfg": model_config,
                        "model_class": self.model.__class__.__name__,
                    },
                    best_ckpt_path,
                )
                self.logger.info(f"==> Saved New Best Checkpoint to {best_ckpt_path} (Val AUC: {self.best_val_auc:.4f})")
            else:
                self.epochs_no_improve += 1
                self.logger.info(
                    f"No improvement in Val AUC for {self.epochs_no_improve}/{self.patience} epochs (Best: {self.best_val_auc:.4f} at epoch {self.best_epoch})"
                )

                if self.epochs_no_improve >= self.patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}. Stopping training.")
                    break

        # Save last model checkpoint
        last_ckpt_path = os.path.join(self.save_dir, "last_model.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_auc": self.best_val_auc,
                "model_cfg": model_config,
                "model_class": self.model.__class__.__name__,
            },
            last_ckpt_path,
        )

        # Final evaluation on Test Set if available
        test_metrics = None
        if self.test_loader is not None:
            self.logger.info("Evaluating Best Model on Independent Test Set...")
            # Load best checkpoint weights
            best_ckpt = torch.load(os.path.join(self.save_dir, "best_model.pt"), map_location=self.device)
            self.model.load_state_dict(best_ckpt["model_state_dict"])
            test_metrics = self.evaluate(self.test_loader, split_name="Test (Best Model)")

        return {
            "best_epoch": self.best_epoch,
            "best_val_auc": self.best_val_auc,
            "test_metrics": test_metrics,
            "history": self.history,
        }
