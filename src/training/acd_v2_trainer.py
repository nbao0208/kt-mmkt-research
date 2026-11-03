"""
Two-Stage Curriculum Trainer for ACD-MMKTv2.
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v2.

Key features:
1. Two-Stage Curriculum Optimization:
   - Stage 1: Item-Bank Multimodal Pre-training on 7,652 questions for 15 epochs with L_D^weighted.
     Warm-starts Subsystem 1 (Alignment) and Subsystem 2 (HGNN, Disentangled Difficulty, Gram-Schmidt).
   - Stage 2: Sequential Knowledge Tracing Fine-tuning across all 4 subsystems with L_ACD-MMKTv2.
     Clean optimizer & cosine scheduler re-initialization at Stage 2 transition.
2. Decoupled Parameter Groups in AdamW:
   - Zero weight decay on biases (b_c, b_theta, b_a), LayerNorms, pooling queries (q_pool),
     response embeddings (v_correct, v_wrong, f_0, f_1), initial history (h_init), and decay (omega_decay).
   - Standard weight decay (1e-4) on projection and HGNN weight matrices.
3. Hardware-Adaptive Execution:
   - Automated device selection (MPS on Apple Silicon, CUDA with selective AMP, CPU fallback).
   - Offline item-bank precomputation cache during validation and testing for ultra-fast evaluation.
4. Early Stopping on Validation Global Micro ROC-AUC and atomic checkpoint saving.
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

from src.training.losses import ACDMMKTv2Loss
from src.evaluation.metrics import calculate_metrics
from src.utils.logger import get_logger


class ACDMMKTv2Trainer:
    """
    Two-Stage Curriculum Trainer for ACD-MMKTv2.
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
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        patience: int = 10,
        device: Optional[Union[str, torch.device]] = None,
        precision: str = "fp32",
        grad_accum_steps: int = 1,
        save_dir: str = "outputs/checkpoints/acd_mmkt_v2_fold0",
        model_cfg: Optional[Dict[str, Any]] = None,
        warmup_epochs: int = 2,
        eta_min: float = 1e-5,
        stage1_epochs: int = 15,
        stage1_batch_size: int = 256,
        stage1_lr: float = 1e-3,
        beta_d: float = 0.20,
        kappa: float = 10.0,
        use_stage1_pretrain: bool = True,
        logger: Optional[Any] = None,
    ):
        """
        Initialize ACDMMKTv2Trainer.

        Args:
            model: ACDMMKTv2 model instance.
            train_loader: DataLoader for train split (4 merged folds).
            val_loader: DataLoader for validation split (1 designated fold).
            test_loader: Optional DataLoader for test split.
            n_q: Fold-specific interaction counts array [7652].
            d_q: Fold-specific empirical error rate array [7652].
            omega_q: Fold-specific Bayesian confidence weights [7652].
            lr: Peak learning rate for Stage 2 (default: 1e-3).
            weight_decay: L2 penalty for weight matrices (default: 1e-4).
            max_grad_norm: Gradient clipping norm threshold (default: 1.0).
            patience: Early stopping patience epochs (default: 10).
            device: Computing device ('cuda:0', 'mps', or 'cpu').
            precision: Precision mode ('fp32', 'amp_fp16').
            grad_accum_steps: Gradient accumulation steps (default: 1).
            save_dir: Destination path for model checkpoints.
            model_cfg: Architecture configuration dictionary.
            warmup_epochs: Linear warmup epochs (default: 2).
            eta_min: Minimum learning rate for cosine annealing (default: 1e-5).
            stage1_epochs: Epochs for Stage 1 Item-Bank Pre-training (default: 15).
            stage1_batch_size: Batch size for Stage 1 pretraining (default: 256).
            stage1_lr: Learning rate for Stage 1 pretraining (default: 1e-3).
            beta_d: Weight for difficulty calibration loss (default: 0.20).
            kappa: Bayesian smoothing constant (default: 10.0).
            use_stage1_pretrain: Whether to execute Stage 1 pretraining (Ablation 6 flag).
            logger: Optional logger instance.
        """
        if device is None:
            if "PJRT_DEVICE" in os.environ and os.environ.get("PJRT_DEVICE") == "TPU" or "TPU_NAME" in os.environ:
                try:
                    import torch_xla.core.xla_model as xm
                    self.device = xm.xla_device()
                except ImportError:
                    self.device = torch.device("xla")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        elif isinstance(device, str):
            if "xla" in device:
                try:
                    import torch_xla.core.xla_model as xm
                    self.device = xm.xla_device()
                except ImportError:
                    self.device = torch.device(device)
            else:
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
        self.model_cfg = model_cfg or {}
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        self.stage1_epochs = stage1_epochs
        self.stage1_batch_size = stage1_batch_size
        self.stage1_lr = stage1_lr
        self.beta_d = beta_d
        self.kappa = kappa
        self.use_stage1_pretrain = use_stage1_pretrain
        self.logger = logger or get_logger("ACDMMKTv2Trainer")

        os.makedirs(self.save_dir, exist_ok=True)

        # Setup N_q and D_q for fold calibration
        if n_q is not None and d_q is not None:
            self.n_q = n_q.to(self.device)
            self.d_q = d_q.to(self.device)
        elif hasattr(train_loader.dataset, "N_q") and hasattr(train_loader.dataset, "D_q"):
            self.n_q = train_loader.dataset.N_q.to(self.device)
            self.d_q = train_loader.dataset.D_q.to(self.device)
        else:
            self.n_q = torch.zeros(self.model.num_questions, dtype=torch.int64, device=self.device)
            self.d_q = torch.zeros(self.model.num_questions, dtype=torch.float32, device=self.device)

        # Initialize multi-task loss
        self.loss_fn = ACDMMKTv2Loss(beta_d=self.beta_d, kappa=self.kappa)

        # Initialize AMP scaler if CUDA and amp_fp16 requested
        self.scaler = None
        self.is_xla = (self.device.type == "xla")
        if "cuda" in str(self.device) and "fp16" in self.precision:
            self.scaler = torch.cuda.amp.GradScaler()

    def _build_optimizer_and_scheduler(
        self,
        lr: float,
        total_epochs: int,
        named_params: Optional[List[Tuple[str, nn.Parameter]]] = None,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Build decoupled AdamW optimizer and linear-warmup cosine annealing scheduler.
        Strictly decouples weight decay (0.0 for 1D tensors, biases, norms, and specific scalar biases).
        """
        no_decay_keys = (
            "bias", "norm", "layernorm", "q_pool", "b_c", "b_z", "b_theta", "b_a",
            "v_correct", "v_wrong", "f_correct", "f_wrong", "h_init", "omega_decay"
        )

        target_named_params = named_params if named_params is not None else list(self.model.named_parameters())
        decay_params = []
        no_decay_params = []

        for name, param in target_named_params:
            if not param.requires_grad:
                continue
            name_lower = name.lower()
            if param.ndim <= 1 or any(k in name_lower for k in no_decay_keys):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=lr)

        # Learning rate schedule: Linear Warmup for warmup_epochs, then Cosine Annealing to eta_min
        warmup_ep = min(self.warmup_epochs, total_epochs // 2)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_ep:
                return float(epoch + 1) / float(max(1, warmup_ep))
            progress = float(epoch - warmup_ep) / float(max(1, total_epochs - warmup_ep))
            cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
            return max(self.eta_min / lr, cosine_factor)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return optimizer, scheduler

    def train_stage1(self) -> None:
        """
        Stage 1: Item-Bank Multimodal Pre-training on 7,652 questions.
        Pre-trains Subsystem 1 and Subsystem 2 with L_D^weighted.
        """
        self.logger.info(
            f"=== Starting Stage 1: Item-Bank Multimodal Pre-training ({self.stage1_epochs} epochs) ==="
        )
        # Select parameters belonging to Subsystem 1 and Subsystem 2 with strict decoupling
        stage1_named_params = [
            (f"sub1.{k}", v) for k, v in self.model.sub1.named_parameters()
        ] + [
            (f"sub2.{k}", v) for k, v in self.model.sub2.named_parameters()
        ]
        stage1_params = [p for _, p in stage1_named_params if p.requires_grad]

        optimizer, scheduler = self._build_optimizer_and_scheduler(
            lr=self.stage1_lr,
            total_epochs=self.stage1_epochs,
            named_params=stage1_named_params,
        )

        num_questions = self.model.num_questions
        q_indices = torch.arange(num_questions, device=self.device)

        start_time = time.time()
        for epoch in range(1, self.stage1_epochs + 1):
            self.model.train()
            perm = torch.randperm(num_questions, device=self.device)
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, num_questions, self.stage1_batch_size):
                batch_q = q_indices[perm[i : i + self.stage1_batch_size]]
                optimizer.zero_grad()

                out = self.model.forward_stage1(q_ids=batch_q)
                tilde_mu = out["tilde_mu"].squeeze(-1)
                d_target = self.d_q[batch_q]
                n_target = self.n_q[batch_q]

                loss, metrics = self.loss_fn.forward_stage1(
                    tilde_mu=tilde_mu,
                    d_err=d_target,
                    n_obs=n_target,
                )

                loss.backward()
                clip_grad_norm_(stage1_params, self.max_grad_norm)
                if self.is_xla:
                    import torch_xla.core.xla_model as xm
                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()
            if self.is_xla:
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            avg_loss = epoch_loss / max(1, num_batches)
            current_lr = scheduler.get_last_lr()[0]
            self.logger.info(
                f"[Stage 1 Epoch {epoch:02d}/{self.stage1_epochs:02d}] "
                f"L_D^weighted: {avg_loss:.5f} | LR: {current_lr:.6f}"
            )

        elapsed = time.time() - start_time
        self.logger.info(f"=== Stage 1 Completed in {elapsed:.2f}s ===")

    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Execute one Stage 2 sequential knowledge tracing training epoch.
        """
        self.model.train()
        self.model.clear_item_cache()
        if self.is_xla:
            import torch_xla.core.xla_model as xm
            xm.mark_step()

        total_loss_accum = 0.0
        total_bce_accum = 0.0
        total_diff_accum = 0.0
        num_steps = 0

        optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            # Transfer batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device, non_blocking=True)

            targets = batch["r_seq"]
            loss_mask = batch["loss_mask"]

            # Forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch, use_cache=False)
                    loss, metrics = self.loss_fn(
                        model_outputs=outputs,
                        targets=targets,
                        loss_mask=loss_mask,
                        n_q_all=self.n_q,
                        d_q_all=self.d_q,
                    )
                    loss = loss / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            elif self.is_xla and ("bf16" in self.precision or "fp16" in self.precision):
                with torch.autocast(device_type="xla", dtype=torch.bfloat16):
                    outputs = self.model(batch, use_cache=False)
                    loss, metrics = self.loss_fn(
                        model_outputs=outputs,
                        targets=targets,
                        loss_mask=loss_mask,
                        n_q_all=self.n_q,
                        d_q_all=self.d_q,
                    )
                    loss = loss / self.grad_accum_steps
                loss.backward()
            else:
                outputs = self.model(batch, use_cache=False)
                loss, metrics = self.loss_fn(
                    model_outputs=outputs,
                    targets=targets,
                    loss_mask=loss_mask,
                    n_q_all=self.n_q,
                    d_q_all=self.d_q,
                )
                loss = loss / self.grad_accum_steps
                loss.backward()

            total_loss_accum += metrics["loss_total"]
            total_bce_accum += metrics["loss_bce"]
            total_diff_accum += metrics["loss_diff"]
            num_steps += 1

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                if self.scaler is not None:
                    self.scaler.unscale_(optimizer)
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if self.is_xla:
                        import torch_xla.core.xla_model as xm
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()
                optimizer.zero_grad()

            if self.device.type == "mps" and (step + 1) % 25 == 0:
                torch.mps.empty_cache()

        if self.device.type == "mps":
            torch.mps.empty_cache()

        avg_loss = total_loss_accum / max(1, num_steps)
        avg_bce = total_bce_accum / max(1, num_steps)
        avg_diff = total_diff_accum / max(1, num_steps)

        return {
            "loss_total": avg_loss,
            "loss_bce": avg_bce,
            "loss_diff": avg_diff,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, split_name: str = "val") -> Dict[str, float]:
        """
        Evaluate model on validation or test split using cached item bank.
        """
        self.model.eval()
        # Precompute static cache once for fast evaluation
        self.model.precompute_item_cache()

        all_preds = []
        all_targets = []

        for batch in dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device, non_blocking=True)

            eval_mask = batch["eval_mask"]  # [B, L]
            targets = batch["r_seq"]        # [B, L]

            outputs = self.model(batch, use_cache=True)
            probs = outputs["probs"]        # [B, L]

            valid_mask = eval_mask.bool()
            if valid_mask.any():
                valid_probs = probs[valid_mask].cpu().numpy()
                valid_targets = targets[valid_mask].cpu().numpy()
                all_preds.extend(valid_probs)
                all_targets.extend(valid_targets)

        self.model.clear_item_cache()
        if self.is_xla:
            import torch_xla.core.xla_model as xm
            xm.mark_step()

        metrics = calculate_metrics(all_preds, all_targets)
        return metrics

    def fit(self, num_epochs: int = 50) -> Dict[str, Any]:
        """
        Full two-stage training loop with Early Stopping on Validation Global Micro ROC-AUC.
        """
        # Stage 1: Optional Item-Bank Pre-training
        if self.use_stage1_pretrain and self.stage1_epochs > 0:
            self.train_stage1()

        # Stage 2: Sequential KT Fine-tuning
        self.logger.info(
            f"=== Starting Stage 2: Sequential KT Fine-tuning ({num_epochs} epochs) ==="
        )
        optimizer, scheduler = self._build_optimizer_and_scheduler(
            lr=self.lr,
            total_epochs=num_epochs,
        )

        best_val_auc = 0.0
        best_epoch = 0
        patience_counter = 0
        best_checkpoint_path = os.path.join(self.save_dir, "best_model.pt")

        history = {
            "train_loss": [],
            "val_auc": [],
            "val_acc": [],
        }

        total_start = time.time()

        for epoch in range(1, num_epochs + 1):
            ep_start = time.time()
            train_metrics = self.train_epoch(optimizer=optimizer, epoch=epoch)
            val_metrics = self.evaluate(self.val_loader, split_name="val")
            scheduler.step()

            ep_time = time.time() - ep_start
            val_auc = val_metrics["auc"]
            val_acc = val_metrics["acc"]
            current_lr = scheduler.get_last_lr()[0]

            history["train_loss"].append(train_metrics["loss_total"])
            history["val_auc"].append(val_auc)
            history["val_acc"].append(val_acc)

            self.logger.info(
                f"[Stage 2 Epoch {epoch:02d}/{num_epochs:02d}] "
                f"Train Loss: {train_metrics['loss_total']:.4f} (BCE: {train_metrics['loss_bce']:.4f}, Diff: {train_metrics['loss_diff']:.4f}) | "
                f"Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | Time: {ep_time:.1f}s"
            )

            # Check early stopping criterion
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                patience_counter = 0

                # Serialize atomic checkpoint with configuration
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "model_cfg": self.model_cfg,
                    "training_cfg": {
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                        "beta_d": self.beta_d,
                        "kappa": self.kappa,
                    },
                }
                torch.save(checkpoint, best_checkpoint_path)
                self.logger.info(f"  --> Saved new best checkpoint to {best_checkpoint_path} (AUC: {val_auc:.5f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info(
                        f"Early stopping triggered after {epoch} epochs (Best Val AUC: {best_val_auc:.5f} at epoch {best_epoch})."
                    )
                    break

        total_elapsed = time.time() - total_start
        self.logger.info(
            f"=== Training Completed in {total_elapsed / 60.0:.2f} mins. Best Val AUC: {best_val_auc:.5f} at epoch {best_epoch} ==="
        )

        # Final evaluation on Test split if provided
        test_metrics = {}
        if self.test_loader is not None and os.path.exists(best_checkpoint_path):
            self.logger.info("Loading best checkpoint for final pyKT Test evaluation...")
            best_ckpt = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(best_ckpt["model_state_dict"])
            test_metrics = self.evaluate(self.test_loader, split_name="test")
            self.logger.info(
                f"=== Final Test Results -> Micro ROC-AUC: {test_metrics['auc']:.5f} | "
                f"Accuracy: {test_metrics['acc']:.5f} (Samples: {test_metrics['num_samples']}) ==="
            )

        return {
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
            "test_metrics": test_metrics,
            "history": history,
            "checkpoint_path": best_checkpoint_path,
        }
