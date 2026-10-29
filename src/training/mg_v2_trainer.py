"""
Dedicated Training and Validation Pipeline for MG-MMKT v2.

Manages:
- Parameter grouping for AdamW (separating decay and no_decay parameters to protect e_null, h_0, delta, theta).
- Multi-task objective optimization: L_total = L_BCE + lambda_priv * L_priv + lambda_gate * L_gate.
- Dynamic learning rate schedule with Linear Warmup and Cosine Annealing.
- Global Micro ROC-AUC / Accuracy validation per pyKT protocol.
- Early Stopping on validation AUC with self-describing checkpoint serialization.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.training.losses import MGMMKTv2Loss
from src.evaluation.metrics import calculate_metrics
from src.utils.logger import get_logger


class MGMMKTv2Trainer:
    """
    Dedicated Trainer for MG-MMKT v2 with Multi-Task Optimization and pyKT Protocol.
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
        lambda_priv: float = 0.10,
        lambda_gate: float = 0.05,
        device: Optional[Union[str, torch.device]] = None,
        save_dir: str = "outputs/checkpoints",
        model_cfg: Optional[Dict[str, Any]] = None,
        warmup_epochs: int = 2,
        eta_min: float = 1e-5,
        logger: Optional[Any] = None,
    ):
        """
        Args:
            model: MGMMKTv2 model instance.
            train_loader: DataLoader for training set (4 folds merged).
            val_loader: DataLoader for validation fold.
            test_loader: Optional DataLoader for independent test set.
            lr: Learning rate for AdamW (default: 1e-3).
            weight_decay: L2 penalty for decay parameters (default: 1e-4).
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0).
            patience: Early stopping patience epochs (default: 10).
            lambda_priv: Weight for privileged distillation loss (default: 0.10).
            lambda_gate: Weight for modality gate regularization (default: 0.05).
            device: Compute device ('cuda', 'mps', or 'cpu').
            save_dir: Directory to save checkpoints.
            model_cfg: Architecture configuration dictionary for checkpoint reconstruction.
            warmup_epochs: Number of linear warmup epochs (default: 2).
            eta_min: Minimum learning rate for cosine scheduler (default: 1e-5).
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
        self.logger = logger or get_logger("MGMMKTv2Trainer")

        os.makedirs(self.save_dir, exist_ok=True)

        # Auto-adjust auxiliary loss weights if ablation disables components
        if hasattr(model, "use_lupi") and not model.use_lupi:
            lambda_priv = 0.0
        if hasattr(model, "use_kc_imputation") and not model.use_kc_imputation:
            lambda_gate = 0.0

        self.lambda_priv = lambda_priv
        self.lambda_gate = lambda_gate

        # Multi-task loss module
        self.loss_fn = MGMMKTv2Loss(
            lambda_priv=self.lambda_priv,
            lambda_gate=self.lambda_gate,
        )

        # Parameter Grouping: Separate parameters to prevent weight decay on
        # e_null, h_0, raw_delta, raw_theta, biases, and LayerNorm affine parameters
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
                or "h_0" in name
                or "raw_delta" in name
                or "raw_theta" in name
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)

        # Scheduler placeholder
        self._setup_scheduler(num_epochs=50)

        # State tracking
        self.best_val_auc = -1.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.history: List[Dict[str, Any]] = []

    def _setup_scheduler(self, num_epochs: int) -> None:
        """
        Configure learning rate scheduler with Linear Warmup and Cosine Annealing.
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

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Execute one complete training epoch over the 4 merged training folds.

        Returns:
            Dictionary of epoch average training losses.
        """
        self.model.train()
        total_loss_accum = 0.0
        bce_loss_accum = 0.0
        priv_loss_accum = 0.0
        gate_loss_accum = 0.0
        num_batches = 0

        for batch in self.train_loader:
            q_seq = batch["q_seq"].to(self.device)
            r_seq = batch["r_seq"].to(self.device)
            seq_mask = batch["seq_mask"].to(self.device)
            loss_mask = batch["loss_mask"].to(self.device)
            mod_masks = batch.get("modality_masks")
            if mod_masks is not None:
                mod_masks = mod_masks.to(self.device)
            raw_mod_masks = batch.get("raw_modality_masks")
            if raw_mod_masks is not None:
                raw_mod_masks = raw_mod_masks.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                q_seq=q_seq,
                r_seq=r_seq,
                seq_mask=seq_mask,
                modality_masks=mod_masks,
                raw_modality_masks=raw_mod_masks,
            )

            loss, metrics = self.loss_fn(
                model_outputs=outputs,
                targets=r_seq,
                loss_mask=loss_mask,
                seq_mask=seq_mask,
            )

            loss.backward()

            if self.max_grad_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            total_loss_accum += metrics["loss_total"]
            bce_loss_accum += metrics["loss_bce"]
            priv_loss_accum += metrics["loss_priv"]
            gate_loss_accum += metrics["loss_gate"]
            num_batches += 1

        self.scheduler.step()

        return {
            "loss_total": total_loss_accum / max(1, num_batches),
            "loss_bce": bce_loss_accum / max(1, num_batches),
            "loss_priv": priv_loss_accum / max(1, num_batches),
            "loss_gate": gate_loss_accum / max(1, num_batches),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation or test DataLoader.
        Aggregates all valid predictions into flat arrays per pyKT protocol before calculating metrics.

        Returns:
            Dictionary with global micro ROC-AUC, Accuracy, and loss.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss_accum = 0.0
        num_batches = 0

        for batch in loader:
            q_seq = batch["q_seq"].to(self.device)
            r_seq = batch["r_seq"].to(self.device)
            seq_mask = batch["seq_mask"].to(self.device)
            eval_mask = batch["eval_mask"].to(self.device)
            mod_masks = batch.get("modality_masks")
            if mod_masks is not None:
                mod_masks = mod_masks.to(self.device)
            raw_mod_masks = batch.get("raw_modality_masks")
            if raw_mod_masks is not None:
                raw_mod_masks = raw_mod_masks.to(self.device)

            outputs = self.model(
                q_seq=q_seq,
                r_seq=r_seq,
                seq_mask=seq_mask,
                modality_masks=mod_masks,
                raw_modality_masks=raw_mod_masks,
            )

            loss, _ = self.loss_fn(
                model_outputs=outputs,
                targets=r_seq,
                loss_mask=eval_mask,
                seq_mask=seq_mask,
            )
            total_loss_accum += loss.item()
            num_batches += 1

            probs = outputs["probs"]  # [B, L]
            mask_bool = eval_mask.bool()

            valid_probs = probs[mask_bool].cpu().numpy()
            valid_targets = r_seq[mask_bool].cpu().numpy()

            all_preds.append(valid_probs)
            all_targets.append(valid_targets)

        if len(all_preds) == 0:
            return {"auc": 0.5, "acc": 0.5, "loss": total_loss_accum / max(1, num_batches)}

        flat_preds = np.concatenate(all_preds)
        flat_targets = np.concatenate(all_targets)

        metrics = calculate_metrics(flat_preds, flat_targets)
        metrics["loss"] = total_loss_accum / max(1, num_batches)
        return metrics

    def save_checkpoint(
        self,
        epoch: int,
        val_auc: float,
        val_acc: float,
        filename: str = "best_model.pt",
    ) -> str:
        """
        Serialize model checkpoint including weights, optimizer state, and self-describing config.
        """
        ckpt_path = os.path.join(self.save_dir, filename)
        checkpoint = {
            "epoch": epoch,
            "val_auc": val_auc,
            "val_acc": val_acc,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_cfg": self.model_cfg,
            "history": self.history,
            "lambda_priv": self.lambda_priv,
            "lambda_gate": self.lambda_gate,
        }
        torch.save(checkpoint, ckpt_path)
        if filename == "best_model.pt":
            self.logger.info(
                f"Best checkpoint saved: {ckpt_path} (Epoch {epoch:02d} | Val AUC: {val_auc:.4f} | Val ACC: {val_acc:.4f})"
            )
        return ckpt_path

    def fit(self, num_epochs: int) -> Dict[str, Any]:
        """
        Run end-to-end training loop with early stopping on validation AUC.

        Args:
            num_epochs: Maximum number of epochs.

        Returns:
            Dictionary with training history and best metrics.
        """
        self._setup_scheduler(num_epochs=num_epochs)
        self.logger.info(f"Starting MG-MMKT v2 training for {num_epochs} epochs on device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)
            elapsed = time.time() - start_time

            val_auc = val_metrics["auc"]
            val_acc = val_metrics["acc"]
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Monitor convergence of gating penalties and decay rates
            gating_params = {}
            if hasattr(self.model, "get_gating_and_decay_rates"):
                gating_params = self.model.get_gating_and_decay_rates()
                thetas_list = gating_params.get("thetas", [])
                thetas_str = ", ".join([f"{th:.3f}" for th in thetas_list]) if thetas_list else "N/A"
                self.logger.info(
                    f"Epoch {epoch:02d} Parameters -> delta_q: {gating_params.get('delta_q', 0.0):.4f} | "
                    f"delta_s_txt: {gating_params.get('delta_s_txt', 0.0):.4f} | "
                    f"delta_s_img: {gating_params.get('delta_s_img', 0.0):.4f} | "
                    f"theta: [{thetas_str}]"
                )

            # Log progress
            self.logger.info(
                f"Epoch {epoch:02d}/{num_epochs:02d} [{elapsed:.1f}s] - "
                f"Train Loss: {train_metrics['loss_total']:.4f} "
                f"(BCE: {train_metrics['loss_bce']:.4f}, Priv: {train_metrics['loss_priv']:.4f}, Gate: {train_metrics['loss_gate']:.4f}) | "
                f"Val AUC: {val_auc:.4f} | Val ACC: {val_acc:.4f} | LR: {current_lr:.6f}"
            )

            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss_total"],
                "train_bce": train_metrics["loss_bce"],
                "train_priv": train_metrics["loss_priv"],
                "train_gate": train_metrics["loss_gate"],
                "val_auc": val_auc,
                "val_acc": val_acc,
                "val_loss": val_metrics["loss"],
                "lr": current_lr,
                "time_sec": elapsed,
                "delta_q": gating_params.get("delta_q"),
                "delta_s_txt": gating_params.get("delta_s_txt"),
                "delta_s_img": gating_params.get("delta_s_img"),
                "thetas": gating_params.get("thetas"),
            }
            self.history.append(record)

            # Always save last model checkpoint for long training runs
            self.save_checkpoint(epoch, val_auc, val_acc, filename="last_model.pt")

            # Check for improvement
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, val_auc, val_acc, filename="best_model.pt")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch:02d}. "
                        f"Best Val AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch:02d}."
                    )
                    break

        # Optional test evaluation
        test_metrics = None
        if self.test_loader is not None:
            self.logger.info("Loading best model weights for independent test evaluation...")
            best_ckpt = torch.load(os.path.join(self.save_dir, "best_model.pt"), map_location=self.device)
            self.model.load_state_dict(best_ckpt["model_state_dict"])
            test_metrics = self.evaluate(self.test_loader)
            self.logger.info(f"Test Set Evaluation -> AUC: {test_metrics['auc']:.4f} | ACC: {test_metrics['acc']:.4f}")

        return {
            "best_epoch": self.best_epoch,
            "best_val_auc": self.best_val_auc,
            "test_metrics": test_metrics,
            "history": self.history,
        }

