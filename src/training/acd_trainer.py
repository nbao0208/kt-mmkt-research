"""
Two-Stage Curriculum Trainer for ACD-MMKT (Stage 1 Item-Bank Pre-training & Stage 2 KT Fine-tuning).

Implements Section 4 of docs/models/ACD-MMKT.md and Step 6 & 7 of
plans/acd_mmkt_implementation_and_training_plan.md:
- Stage 1: Item-Bank Multimodal Pre-training on 7,652 questions for rapid convergence of
  fine-grained cross-modal attention, 2-layer HGNN, solution distillation, and CTT calibration.
- Stage 2: Sequential KT Fine-tuning with Linear Warmup + Cosine Annealing, Gradient Accumulation,
  Selective AMP on CUDA, and Unified Memory garbage collection on Apple Silicon Metal (MPS).
- Separated AdamW parameter groups preventing weight decay on canonical null tokens and BOS vectors.
- Early Stopping on validation Global Micro ROC-AUC and comprehensive checkpoint serialization.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.training.losses import ACDMMKTLoss
from src.evaluation.metrics import calculate_metrics
from src.utils.logger import get_logger


class ACDMMKTTrainer:
    """
    Two-Stage Curriculum Trainer for ACD-MMKT.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        d_ctt: Optional[torch.Tensor] = None,
        ctt_valid_mask: Optional[torch.Tensor] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        patience: int = 10,
        device: Optional[Union[str, torch.device]] = None,
        precision: str = "fp32",
        grad_accum_steps: int = 1,
        save_dir: str = "outputs/checkpoints/acd_mmkt_fold0",
        model_cfg: Optional[Dict[str, Any]] = None,
        warmup_epochs: int = 2,
        eta_min: float = 1e-5,
        beta_distill: float = 0.10,
        beta_ctt: float = 0.20,
        beta_disen: float = 0.05,
        lambda_res: float = 1e-3,
        tau_temp: float = 0.10,
        logger: Optional[Any] = None,
    ):
        """
        Initialize ACDMMKTTrainer.

        Args:
            model: ACDMMKT model instance.
            train_loader: DataLoader for training sequences (4 merged folds).
            val_loader: DataLoader for validation sequences (1 designated fold).
            test_loader: Optional DataLoader for test set.
            d_ctt: Fold-specific CTT error rate labels [7652].
            ctt_valid_mask: Fold-specific CTT reliability mask [7652].
            lr: Learning rate for AdamW optimizer (default: 1e-3).
            weight_decay: L2 regularization coefficient (default: 1e-4).
            max_grad_norm: Gradient clipping threshold (default: 1.0).
            patience: Early stopping patience epochs (default: 10).
            device: Compute device ('cuda', 'cuda:0', 'mps', or 'cpu').
            precision: Computation precision ('fp32', 'amp_fp16', or 'amp_bf16').
            grad_accum_steps: Gradient accumulation steps K_acc (default: 1).
            save_dir: Checkpoint save directory.
            model_cfg: Architecture configuration dictionary to embed in checkpoints.
            warmup_epochs: Linear learning rate warmup epochs (default: 2).
            eta_min: Minimum learning rate for cosine scheduler (default: 1e-5).
            beta_distill: Distillation loss weight (default: 0.10).
            beta_ctt: CTT difficulty loss weight (default: 0.20).
            beta_disen: Rasch orthogonality loss weight (default: 0.05).
            lambda_res: DeepIRT residual regularization weight (default: 1e-3).
            tau_temp: InfoNCE temperature (default: 0.10).
            logger: Logger instance.
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.precision = precision.lower()
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.save_dir = save_dir
        self.model_cfg = model_cfg or getattr(model, "config", None)
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        self.logger = logger or get_logger("ACDMMKTTrainer")

        os.makedirs(self.save_dir, exist_ok=True)

        # Cache fold-specific CTT labels
        if d_ctt is not None:
            self.d_ctt = d_ctt.to(self.device)
        elif hasattr(train_loader.dataset, "ctt_difficulty"):
            self.d_ctt = train_loader.dataset.ctt_difficulty.to(self.device)
        else:
            self.d_ctt = torch.zeros(self.model.num_questions, device=self.device)

        if ctt_valid_mask is not None:
            self.ctt_valid_mask = ctt_valid_mask.to(self.device)
        elif hasattr(train_loader.dataset, "ctt_valid_mask"):
            self.ctt_valid_mask = train_loader.dataset.ctt_valid_mask.to(self.device)
        else:
            self.ctt_valid_mask = torch.zeros(self.model.num_questions, dtype=torch.bool, device=self.device)

        # Multi-task loss objective with ablation awareness (Issue M6)
        eff_beta_distill = beta_distill if getattr(self.model, "use_distillation", True) else 0.0
        eff_beta_ctt = beta_ctt if getattr(self.model, "use_hypergraph_rasch", True) else 0.0
        eff_beta_disen = beta_disen if getattr(self.model, "use_hypergraph_rasch", True) else 0.0
        eff_lambda_res = lambda_res if getattr(self.model, "use_deep_irt", True) else 0.0

        self.loss_fn = ACDMMKTLoss(
            beta_distill=eff_beta_distill,
            beta_ctt=eff_beta_ctt,
            beta_disen=eff_beta_disen,
            lambda_res=eff_lambda_res,
            tau_temp=tau_temp,
        )

        # Separate AdamW parameter groups:
        # 1. No decay: null tokens, BOS tokens, decay rates, LayerNorms, and biases (weight_decay = 0.0)
        # 2. Residual branch: 10x weight decay for res_mlp and fc_res to preserve IRT monotonicity (Issue M4, Plan Step 5.3)
        # 3. Standard decay: all remaining matrix weights (weight_decay = self.weight_decay)
        decay_params = []
        residual_params = []
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
                or "bos" in name
                or "raw_theta" in name
                or "raw_eta" in name
            ):
                no_decay_params.append(param)
            elif "res_mlp" in name or "fc_res" in name:
                residual_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": residual_params, "weight_decay": self.weight_decay * 10.0},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)


        # Mixed Precision Scaler for CUDA
        self.use_amp = (
            self.device.type == "cuda"
            and self.precision in ["amp_fp16", "amp_bf16"]
        )
        self.amp_dtype = (
            torch.bfloat16 if self.precision == "amp_bf16" else torch.float16
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.precision == "amp_fp16"))

        # Setup scheduler placeholder
        self._setup_scheduler(num_epochs=50)

        # State tracking
        self.best_val_auc = -1.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.history: List[Dict[str, Any]] = []

    def _setup_scheduler(self, num_epochs: int) -> None:
        """
        Configure Linear Warmup + Cosine Annealing learning rate schedule.
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

    def train_stage1_item_bank(
        self,
        num_epochs: int = 15,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> List[Dict[str, float]]:
        """
        Stage 1: Pre-train multimodal question bank representations and student distillation network.

        Args:
            num_epochs: Number of Stage 1 pre-training epochs (default: 15).
            batch_size: Item batch size (default: 256).
            lr: Stage 1 learning rate (default: 1e-3).

        Returns:
            List of epoch training metrics.
        """
        self.logger.info(
            f"=== Starting Stage 1: Item-Bank Multimodal Pre-training ({num_epochs} epochs, batch_size={batch_size}) ==="
        )
        self.model.train()
        # Stage 1 Optimizer: decouple weight decay and optimize only Subsystem 1 & 2 (Issue H4)
        s1_decay_params = []
        s1_no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if not (name.startswith("subsystem1.") or name.startswith("subsystem2.")):
                continue
            if (
                param.ndim <= 1
                or "bias" in name
                or "ln" in name
                or "LayerNorm" in name
                or "e_null" in name
            ):
                s1_no_decay_params.append(param)
            else:
                s1_decay_params.append(param)

        stage1_optimizer = torch.optim.AdamW(
            [
                {"params": s1_decay_params, "weight_decay": self.weight_decay},
                {"params": s1_no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
        )


        num_items = self.model.num_questions
        indices = torch.arange(num_items, device=self.device)
        stage1_history = []
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            perm = torch.randperm(num_items, device=self.device)
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, num_items, batch_size):
                batch_indices = perm[i : i + batch_size]
                stage1_optimizer.zero_grad()

                outputs = self.model.forward_item_bank(item_indices=batch_indices)
                loss, metrics = self.loss_fn.forward_stage1(
                    item_outputs=outputs,
                    d_ctt=self.d_ctt,
                    ctt_valid_mask=self.ctt_valid_mask,
                )

                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                stage1_optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            record = {"stage1_epoch": epoch, "loss": avg_loss}
            stage1_history.append(record)

            if epoch % 5 == 0 or epoch == num_epochs:
                self.logger.info(
                    f"Stage 1 [Epoch {epoch:02d}/{num_epochs:02d}] Loss: {avg_loss:.4f} | Distill: {metrics['loss_distill']:.4f} | CTT: {metrics['loss_ctt']:.4f}"
                )

        elapsed = time.time() - start_time
        self.logger.info(f"Stage 1 complete in {elapsed:.1f}s.")
        return stage1_history

    def train_epoch_stage2(self, epoch: int) -> Dict[str, float]:
        """
        Execute one training epoch of Stage 2 sequential interaction KT.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            targets = batch["r_seq"]
            loss_mask = batch["loss_mask"]
            d_ctt = batch["ctt_difficulty"]
            ctt_valid = batch["ctt_valid_mask"]

            # Forward pass with selective AMP
            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    outputs = self.model(batch)
                    loss, metrics = self.loss_fn.forward_stage2(
                        seq_outputs=outputs,
                        targets=targets,
                        loss_mask=loss_mask,
                        d_ctt=d_ctt,
                        ctt_valid_mask=ctt_valid,
                    )
            else:
                outputs = self.model(batch)
                loss, metrics = self.loss_fn.forward_stage2(
                    seq_outputs=outputs,
                    targets=targets,
                    loss_mask=loss_mask,
                    d_ctt=d_ctt,
                    ctt_valid_mask=ctt_valid,
                )

            # Gradient accumulation normalization
            scaled_loss = loss / self.grad_accum_steps

            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Optimizer step every grad_accum_steps or at epoch end
            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                    if self.max_grad_norm > 0:
                        clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.max_grad_norm > 0:
                        clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        elapsed = time.time() - start_time

        self.logger.info(
            f"Epoch {epoch:03d} [Train] Loss: {avg_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e} | Time: {elapsed:.1f}s"
        )
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        split_name: str = "Val",
    ) -> Dict[str, float]:
        """
        Evaluate model predictions using pyKT global micro ROC-AUC and Accuracy.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    outputs = self.model(batch)
            else:
                outputs = self.model(batch)

            preds = outputs["preds"]        # [B, L]
            targets = batch["r_seq"]        # [B, L]
            eval_mask = batch["eval_mask"]  # [B, L]

            active_preds = preds[eval_mask].detach().cpu().numpy()
            active_targets = targets[eval_mask].detach().cpu().numpy()

            all_preds.extend(active_preds.tolist())
            all_targets.extend(active_targets.tolist())

        metrics = calculate_metrics(all_preds, all_targets)

        # Release MPS / Metal cache after evaluation pass on Apple Silicon
        if self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        self.logger.info(
            f"[{split_name}] Global AUC: {metrics['auc']:.4f} | ACC: {metrics['acc']:.4f} | Evaluated Interactions: {metrics['num_samples']:,}"
        )
        return metrics

    def fit(
        self,
        num_epochs: int = 50,
        run_stage1: bool = True,
        stage1_epochs: int = 15,
        stage1_batch_size: int = 256,
        stage1_lr: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        Execute full Two-Stage Curriculum Training:
        - Phase 1: Stage 1 Item-Bank Multimodal Pre-training (optional if pre-trained).
        - Phase 2: Stage 2 Sequential KT Fine-tuning with Early Stopping.

        Args:
            num_epochs: Maximum epochs for Stage 2.
            run_stage1: Whether to execute Stage 1 pre-training.
            stage1_epochs: Number of Stage 1 epochs.
            stage1_batch_size: Stage 1 item batch size.
            stage1_lr: Stage 1 learning rate.

        Returns:
            Dictionary containing best validation metrics, history, and checkpoint path.
        """
        self.logger.info(
            f"Starting ACD-MMKT Two-Stage Training on {self.device} (Precision: {self.precision}, Grad Accum: {self.grad_accum_steps})..."
        )

        # Execute Stage 1 if requested
        if run_stage1:
            self.train_stage1_item_bank(
                num_epochs=stage1_epochs,
                batch_size=stage1_batch_size,
                lr=stage1_lr,
            )

        # Configure Stage 2 learning rate schedule
        self._setup_scheduler(num_epochs=num_epochs)
        model_config = getattr(self.model, "config", None) or self.model_cfg

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch_stage2(epoch)
            val_metrics = self.evaluate(self.val_loader, split_name=f"Val-Epoch-{epoch:02d}")

            self.scheduler.step()

            record = {
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "val_auc": val_metrics["auc"],
                "val_acc": val_metrics["acc"],
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(record)

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
                        "model_class": "ACDMMKT",
                    },
                    best_ckpt_path,
                )
                self.logger.info(
                    f"--> Saved new best checkpoint at epoch {epoch} (Val AUC: {val_auc:.4f}) to {best_ckpt_path}"
                )
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    self.logger.info(
                        f"Early stopping triggered after {epoch} epochs (no improvement for {self.patience} epochs)."
                    )
                    break

        # Test set evaluation if provided
        test_metrics = None
        best_ckpt_path = os.path.join(self.save_dir, "best_model.pt")
        if self.test_loader is not None and os.path.exists(best_ckpt_path):
            self.logger.info(f"Loading best checkpoint for final evaluation on Test split...")
            ckpt = torch.load(best_ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            test_metrics = self.evaluate(self.test_loader, split_name="Test-BestModel")

        return {
            "best_epoch": self.best_epoch,
            "best_val_auc": self.best_val_auc,
            "test_metrics": test_metrics,
            "history": self.history,
            "best_checkpoint": best_ckpt_path,
        }
