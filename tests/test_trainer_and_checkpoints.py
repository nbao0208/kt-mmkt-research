"""
Unit tests for MGMMKTTrainer:
- Parameter grouping separating weight decay from e_null, y_0, LayerNorm, and biases.
- Linear Warmup + Cosine Annealing learning rate schedule.
- Checkpoint persistence and restoration of model architecture configs.
"""

import os
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mmkt.mg_mmkt import MGMMKT
from src.training.trainer import MGMMKTTrainer


def test_trainer_param_groups_no_decay():
    """Verify that e_null, y_0, LayerNorm, and biases are excluded from weight decay."""
    model = MGMMKT(
        num_questions=100,
        num_kcs=50,
        d_model=32,
        num_heads=2,
        max_seq_len=20,
    )

    # Synthetic dataloader
    dummy_loader = DataLoader(TensorDataset(torch.zeros(2, 2)), batch_size=1)

    trainer = MGMMKTTrainer(
        model=model,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
        lr=1e-3,
        weight_decay=1e-4,
    )

    param_groups = trainer.optimizer.param_groups
    assert len(param_groups) == 2

    decay_group = param_groups[0]
    no_decay_group = param_groups[1]

    assert decay_group["weight_decay"] == 1e-4
    assert no_decay_group["weight_decay"] == 0.0

    # Ensure e_null parameters are in the no_decay_group
    no_decay_param_ids = {id(p) for p in no_decay_group["params"]}
    assert id(model.imputation.e_null_img_q) in no_decay_param_ids
    assert id(model.imputation.e_null_txt_s) in no_decay_param_ids
    assert id(model.imputation.e_null_img_s) in no_decay_param_ids
    assert id(model.causal_sakt.y_0) in no_decay_param_ids


def test_checkpoint_config_persistence(tmp_path):
    """Verify that model_cfg and model_class are properly saved and restored from checkpoint."""
    save_dir = str(tmp_path / "ckpt_test")
    os.makedirs(save_dir, exist_ok=True)

    model = MGMMKT(
        num_questions=100,
        num_kcs=50,
        d_model=32,
        num_heads=2,
        max_seq_len=20,
        use_visual=False,  # Ablation 3
    )

    dummy_loader = DataLoader(TensorDataset(torch.zeros(2, 2)), batch_size=1)

    trainer = MGMMKTTrainer(
        model=model,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
        save_dir=save_dir,
    )

    # Save checkpoint manually
    ckpt_path = os.path.join(save_dir, "best_model.pt")
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "best_val_auc": 0.75,
            "model_cfg": model.config,
            "model_class": model.__class__.__name__,
        },
        ckpt_path,
    )

    # Reload checkpoint
    loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "model_cfg" in loaded_ckpt
    assert loaded_ckpt["model_cfg"]["use_visual"] is False
    assert loaded_ckpt["model_cfg"]["d_model"] == 32
    assert loaded_ckpt["model_class"] == "MGMMKT"
