"""
Adaptive End-to-End Trainer for ACD-MMKTv3.
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v3.

Key features:
1. One-Stage End-to-End Optimization:
   - Eliminates disjoint Stage 1 item-bank pre-training; concept difficulty is anchored to Empirical Bayes prior from epoch 1.
   - Optimizes L_ACD-MMKTv3 = L_BCE + beta_D * L_D^weighted + lambda_sparse * L_sparse directly.
2. Decoupled Parameter Groups in AdamW:
   - Zero weight decay on biases (b_theta, b_a, b_z, b_gate), LayerNorms, pooling query (q_pool_txt),
     response embeddings (v_correct, v_wrong, f_correct, f_wrong), initial history (h_init),
     and decay parameters (omega_decay, omega_time, omega_kc).
   - Standard weight decay (1e-4) on 2D linear and HGNN weight matrices.
3. Hardware-Adaptive Execution:
   - Auto device selection (MPS on Apple Silicon, CUDA with AMP fp16, CPU fallback).
   - Offline item-bank precomputation cache during validation and testing for ultra-fast evaluation.
4. Early stopping on Validation Global Micro ROC-AUC with atomic checkpoint saving.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.training.losses import ACDMMKTv3Loss
from src.evaluation.metrics import calculate_metrics
from src.utils.logger import get_logger


class ACDMMKTv3Trainer:
    """
    Adaptive End-to-End Trainer for ACD-MMKTv3.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        n_q: Optional[torch.Tensor] = None,
        d_q: Optional[torch.Tensor] = None,
        omega_q: Optional[torch.Tensor] = None,
        mu_concept_prior: Optional[torch.Tensor] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        patience: int = 10,
        device: Optional[Union[str, torch.device]] = None,
        precision: str = "fp32",
        grad_accum_steps: int = 1,
        save_dir: str = "outputs/checkpoints/acd_mmkt_v3_fold0",
        val_fold: int = 0,
        model_cfg: Optional[Dict[str, Any]] = None,
        warmup_epochs: int = 2,
        eta_min: float = 1e-5,
        beta_d: float = 0.20,
        lambda_sparse: float = 0.01,
        kappa: float = 10.0,
        logger: Optional[Any] = None,
    ):
        """
        Initialize ACDMMKTv3Trainer.

        Args:
            model: ACDMMKTv3 model instance.
            train_loader: DataLoader for train split (4 merged folds).
            val_loader: DataLoader for validation split (1 designated fold).
            test_loader: Optional DataLoader for test split.
            n_q: Fold-specific interaction counts array [7652].
            d_q: Fold-specific empirical error rate array [7652].
            omega_q: Fold-specific Bayesian confidence weights [7652].
            mu_concept_prior: Fold-specific concept difficulty logit prior [7652].
            lr: Peak learning rate (default: 1e-3).
            weight_decay: L2 penalty for weight matrices (default: 1e-4).
            max_grad_norm: Gradient clipping norm threshold (default: 1.0).
            patience: Early stopping patience epochs (default: 10).
            device: Computing device ('cuda:0', 'mps', or 'cpu').
            precision: Precision mode ('fp32', 'amp_fp16', 'amp_bf16').
            grad_accum_steps: Gradient accumulation steps (default: 1).
            save_dir: Destination path for model checkpoints.
            val_fold: Designated validation fold index (0..4).
            model_cfg: Architecture configuration dictionary.
            warmup_epochs: Linear warmup epochs (default: 2).
            eta_min: Minimum learning rate for cosine annealing (default: 1e-5).
            beta_d: Weight for difficulty calibration loss (default: 0.20).
            lambda_sparse: Sparsity regularization coefficient (default: 0.01).
            kappa: Bayesian smoothing constant (default: 10.0).
            logger: Optional logger instance.
        """
        if device is None:
            if "PJRT_DEVICE" in os.environ and os.environ.get("PJRT_DEVICE") == "TPU" or "TPU_NAME" in os.environ:
                try:
                    import torch_xla.core.xla_model as xm
                    self.device = xm.xla_device()
                except Exception:
                    self.device = torch.device("xla")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        elif isinstance(device, str):
            if "xla" in device.lower() or "tpu" in device.lower():
                try:
                    import torch_xla.core.xla_model as xm
                    self.device = xm.xla_device()
                except Exception:
                    self.device = torch.device("xla")
            else:
                self.device = torch.device(device)
        else:
            self.device = device

        self.is_xla = getattr(self.device, "type", "") == "xla"
        self.precision = precision
        self.use_amp = (self.precision in ("amp_fp16", "amp_bf16")) and (self.device.type == "cuda")
        self.use_xla_amp = self.is_xla and (self.precision in ("amp_bf16", "amp_fp16"))
        self.amp_dtype = torch.bfloat16 if self.precision == "amp_bf16" else torch.float16
        self.use_scaler = (self.precision == "amp_fp16") and (self.device.type == "cuda")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.val_fold = val_fold

        # Load fold statistics tensors onto device
        if n_q is not None:
            self.n_q = n_q.to(self.device)
        else:
            self.n_q = torch.zeros(self.model.num_questions, dtype=torch.int64, device=self.device)

        if d_q is not None:
            self.d_q = d_q.to(self.device)
        else:
            self.d_q = torch.full((self.model.num_questions,), 0.20, dtype=torch.float32, device=self.device)

        if omega_q is not None:
            self.omega_q = omega_q.to(self.device)
        else:
            self.omega_q = torch.zeros(self.model.num_questions, dtype=torch.float32, device=self.device)

        if mu_concept_prior is not None:
            self.mu_concept_prior = mu_concept_prior.to(self.device)
        else:
            self.mu_concept_prior = torch.full((self.model.num_questions,), -1.37, dtype=torch.float32, device=self.device)

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.grad_accum_steps = grad_accum_steps
        self.save_dir = save_dir
        self.model_cfg = model_cfg or {}
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        self.beta_d = beta_d
        self.lambda_sparse = lambda_sparse
        self.kappa = kappa
        self.logger = logger or get_logger("acd_v3_trainer")

        os.makedirs(self.save_dir, exist_ok=True)

        # Multi-task loss
        self.criterion = ACDMMKTv3Loss(
            beta_d=self.beta_d,
            lambda_sparse=self.lambda_sparse,
            kappa=self.kappa,
        )

        # Scaler for AMP fp16 on CUDA (not needed for bf16)
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)

        # Optimizer setup with decoupled weight decay parameter groups
        self.optimizer = self._build_optimizer()
        self.scheduler = None

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Build AdamW optimizer with zero weight decay on 1D parameters, biases, norms, and embeddings.
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Exclude biases, 1D parameters, LayerNorms, response vectors, and decay parameters
            if (
                param.ndim <= 1
                or "norm" in name.lower()
                or "bias" in name.lower()
                or "b_" in name.lower()
                or "q_pool" in name.lower()
                or "v_correct" in name.lower()
                or "v_wrong" in name.lower()
                or "f_correct" in name.lower()
                or "f_wrong" in name.lower()
                or "h_init" in name.lower()
                or "omega_" in name.lower()
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.logger.info(
            f"AdamW Parameter Groups: {len(decay_params)} tensor matrices (weight_decay={self.weight_decay}), "
            f"{len(no_decay_params)} bias/norm/embedding vectors (weight_decay=0.0)."
        )

        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=self.lr)

    def _build_cosine_scheduler(self, num_epochs: int) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Build linear warmup followed by Cosine Annealing learning rate scheduler.
        """
        warmup_steps = self.warmup_epochs

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_steps:
                return float(epoch + 1) / float(max(1, warmup_steps))
            progress = float(epoch - warmup_steps) / float(max(1, num_epochs - warmup_steps))
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return float(self.eta_min / self.lr + (1.0 - self.eta_min / self.lr) * cosine_decay)

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Execute one training epoch.
        """
        self.model.train()
        self.model.clear_item_bank_cache()
        if self.is_xla:
            import torch_xla.core.xla_model as xm
            xm.mark_step()

        total_loss = 0.0
        total_bce = 0.0
        total_diff = 0.0
        total_sparse = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device, non_blocking=True)

            targets = batch["r_seq"]
            loss_mask = batch["loss_mask"]

            if self.use_xla_amp:
                with torch.autocast(device_type="xla", dtype=torch.bfloat16):
                    outputs = self.model(batch, mu_concept_prior=self.mu_concept_prior)
                    loss, metrics = self.criterion(
                        model_outputs=outputs,
                        targets=targets,
                        loss_mask=loss_mask,
                        n_q_all=self.n_q,
                        d_q_all=self.d_q,
                        unanchored_concept=self.model.unanchored_concept,
                    )
                    loss = loss / self.grad_accum_steps
                loss.backward()
                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    import torch_xla.core.xla_model as xm
                    xm.optimizer_step(self.optimizer)
                    self.optimizer.zero_grad()
            elif self.use_amp:
                autocast_ctx = (
                    torch.amp.autocast("cuda", dtype=self.amp_dtype)
                    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
                    else torch.cuda.amp.autocast(dtype=self.amp_dtype)
                )
                with autocast_ctx:
                    outputs = self.model(batch, mu_concept_prior=self.mu_concept_prior)
                    loss, metrics = self.criterion(
                        model_outputs=outputs,
                        targets=targets,
                        loss_mask=loss_mask,
                        n_q_all=self.n_q,
                        d_q_all=self.d_q,
                        unanchored_concept=self.model.unanchored_concept,
                    )
                    loss = loss / self.grad_accum_steps

                if self.use_scaler:
                    self.scaler.scale(loss).backward()
                    if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                        self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    loss.backward()
                    if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        if self.is_xla:
                            import torch_xla.core.xla_model as xm
                            xm.optimizer_step(self.optimizer)
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad()
            else:
                outputs = self.model(batch, mu_concept_prior=self.mu_concept_prior)
                loss, metrics = self.criterion(
                    model_outputs=outputs,
                    targets=targets,
                    loss_mask=loss_mask,
                    n_q_all=self.n_q,
                    d_q_all=self.d_q,
                    unanchored_concept=self.model.unanchored_concept,
                )
                loss = loss / self.grad_accum_steps
                loss.backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if self.is_xla:
                        import torch_xla.core.xla_model as xm
                        xm.optimizer_step(self.optimizer)
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += metrics["loss_total"]
            total_bce += metrics["loss_bce"]
            total_diff += metrics["loss_diff"]
            total_sparse += metrics["loss_sparse"]
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_bce = total_bce / max(1, num_batches)
        avg_diff = total_diff / max(1, num_batches)
        avg_sparse = total_sparse / max(1, num_batches)

        if self.is_xla:
            import torch_xla.core.xla_model as xm
            xm.mark_step()

        return {
            "loss": avg_loss,
            "bce": avg_bce,
            "diff": avg_diff,
            "sparse": avg_sparse,
        }

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, desc: str = "Val") -> Dict[str, float]:
        """
        Evaluate model using precomputed offline item bank cache for high throughput.
        """
        self.model.eval()
        self.model.precompute_item_bank_cache(self.mu_concept_prior)

        all_preds = []
        all_targets = []
        all_masks = []

        total_loss = 0.0
        num_batches = 0

        for batch in data_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device, non_blocking=True)

            targets = batch["r_seq"]
            loss_mask = batch["loss_mask"]
            eval_mask = batch.get("eval_mask", loss_mask)

            outputs = self.model(batch, mu_concept_prior=self.mu_concept_prior)
            loss, _ = self.criterion(
                model_outputs=outputs,
                targets=targets,
                loss_mask=eval_mask,
                n_q_all=self.n_q,
                d_q_all=self.d_q,
                unanchored_concept=self.model.unanchored_concept,
            )

            total_loss += loss.item()
            num_batches += 1

            probs = outputs["probs"]
            mask_b = eval_mask.bool()
            if mask_b.any():
                all_preds.append(probs[mask_b].detach().cpu().numpy())
                all_targets.append(targets[mask_b].detach().cpu().numpy())

        self.model.clear_item_bank_cache()
        if self.is_xla:
            import torch_xla.core.xla_model as xm
            xm.mark_step()

        if len(all_preds) > 0:
            flat_preds = np.concatenate(all_preds)
            flat_targets = np.concatenate(all_targets)
            metrics = calculate_metrics(flat_preds, flat_targets)
        else:
            metrics = {"auc": 0.0, "acc": 0.0}
        metrics["loss"] = total_loss / max(1, num_batches)

        return metrics

    def train(self, num_epochs: int = 50) -> Dict[str, Any]:
        """
        Execute full training loop with early stopping on validation AUC.
        """
        self.logger.info("=" * 60)
        self.logger.info("=== Starting ACD-MMKTv3 End-to-End Training ===")
        self.logger.info(
            f"Device: {self.device}, Precision: {self.precision}, Batch Size: {self.train_loader.batch_size}, "
            f"Grad Accum: {self.grad_accum_steps}, Learning Rate: {self.lr}, Epochs: {num_epochs}."
        )

        self.scheduler = self._build_cosine_scheduler(num_epochs)

        best_val_auc = 0.0
        best_epoch = 0
        patience_counter = 0
        best_model_path = os.path.join(self.save_dir, "best_model.pt")

        start_time = time.time()

        for epoch in range(num_epochs):
            t_epoch_start = time.time()
            current_lr = self.optimizer.param_groups[0]["lr"]

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, desc="Val")

            if self.scheduler is not None:
                self.scheduler.step()
                if self.is_xla:
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()

            epoch_dur = time.time() - t_epoch_start
            val_auc = val_metrics.get("auc", 0.0)
            val_acc = val_metrics.get("acc", 0.0)

            improved = val_auc > best_val_auc
            if improved:
                best_val_auc = val_auc
                best_epoch = epoch
                patience_counter = 0

                tmp_path = best_model_path + ".tmp"
                torch.save(
                    {
                        "epoch": epoch,
                        "val_fold": self.val_fold,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_val_auc": best_val_auc,
                        "val_metrics": val_metrics,
                        "model_cfg": self.model_cfg,
                    },
                    tmp_path,
                )
                os.replace(tmp_path, best_model_path)
            else:
                patience_counter += 1

            self.logger.info(
                f"Epoch [{epoch+1:02d}/{num_epochs:02d}] ({epoch_dur:.1f}s, lr={current_lr:.6f}) - "
                f"Train Loss: {train_metrics['loss']:.4f} (BCE: {train_metrics['bce']:.4f}, "
                f"Diff: {train_metrics['diff']:.4f}, Sparse: {train_metrics['sparse']:.4f}) | "
                f"Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f} "
                f"{'[BEST]' if improved else f'(Patience: {patience_counter}/{self.patience})'}"
            )

            if patience_counter >= self.patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs (Best Val AUC: {best_val_auc:.4f} at epoch {best_epoch+1}).")
                break

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/60:.2f} mins. Best Val AUC: {best_val_auc:.4f} (Epoch {best_epoch+1}).")

        # Load best model for test evaluation if test_loader provided
        test_metrics = {}
        if self.test_loader is not None and os.path.exists(best_model_path):
            self.logger.info("Loading best checkpoint for final Test set evaluation...")
            ckpt = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            test_metrics = self.evaluate(self.test_loader, desc="Test")
            self.logger.info(f"Final Test Evaluation: AUC = {test_metrics.get('auc', 0.0):.4f}, Acc = {test_metrics.get('acc', 0.0):.4f}")

        return {
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
            "test_metrics": test_metrics,
            "total_time_mins": total_time / 60,
        }

