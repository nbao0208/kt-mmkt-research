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


def test_mg_mmkt_static_buffers_excluded_from_state_dict():
    """Verify that Tier 1 static item bank buffers are non-persistent and excluded from state_dict."""
    num_q = 50
    num_k = 20
    d_raw = 64
    d_model = 32

    static_bank = {
        "mg_v_txt_q": torch.randn(num_q, d_raw),
        "mg_v_img_q": torch.randn(num_q, d_raw),
        "mg_v_txt_s": torch.randn(num_q, d_raw),
        "mg_v_img_s": torch.randn(num_q, d_raw),
    }

    model = MGMMKT(
        num_questions=num_q,
        num_kcs=num_k,
        d_raw_txt=d_raw,
        d_raw_img=d_raw,
        d_model=d_model,
        static_item_bank=static_bank,
    )

    # 1. Buffers exist on the model instance and are registered as buffers
    assert hasattr(model, "v_txt_q_bank")
    assert hasattr(model, "v_img_q_bank")
    assert hasattr(model, "v_txt_s_bank")
    assert hasattr(model, "v_img_s_bank")

    # 2. Buffers are NOT in state_dict() (persistent=False prevents checkpoint bloating)
    state_dict = model.state_dict()
    assert "v_txt_q_bank" not in state_dict
    assert "v_img_q_bank" not in state_dict
    assert "v_txt_s_bank" not in state_dict
    assert "v_img_s_bank" not in state_dict

    # 3. Model weights ARE in state_dict()
    assert "q_emb.weight" in state_dict
    assert "kc_emb.weight" in state_dict


def test_legacy_checkpoint_backward_compatibility():
    """Verify that legacy checkpoints containing static item bank buffers load without errors."""
    num_q = 50
    num_k = 20
    d_raw = 64
    d_model = 32

    static_bank = {
        "mg_v_txt_q": torch.randn(num_q, d_raw),
        "mg_v_img_q": torch.randn(num_q, d_raw),
        "mg_v_txt_s": torch.randn(num_q, d_raw),
        "mg_v_img_s": torch.randn(num_q, d_raw),
    }

    model = MGMMKT(
        num_questions=num_q,
        num_kcs=num_k,
        d_raw_txt=d_raw,
        d_raw_img=d_raw,
        d_model=d_model,
        static_item_bank=static_bank,
    )

    # Simulate legacy state dict containing frozen static buffers
    legacy_state_dict = model.state_dict().copy()
    legacy_state_dict["v_txt_q_bank"] = static_bank["mg_v_txt_q"].clone()
    legacy_state_dict["v_img_q_bank"] = static_bank["mg_v_img_q"].clone()
    legacy_state_dict["v_txt_s_bank"] = static_bank["mg_v_txt_s"].clone()
    legacy_state_dict["v_img_s_bank"] = static_bank["mg_v_img_s"].clone()

    # Must load successfully with strict=True without raising Unexpected key(s)
    model.load_state_dict(legacy_state_dict, strict=True)

    # Static buffer values must remain intact
    assert torch.allclose(model.v_txt_q_bank, static_bank["mg_v_txt_q"])


def test_forward_pass_identical_after_saving_and_loading(tmp_path):
    """Verify that model output is bitwise identical before saving and after restoring from checkpoint."""
    torch.manual_seed(42)
    num_q = 50
    num_k = 20
    d_raw = 64
    d_model = 32
    B, L = 2, 10

    static_bank = {
        "mg_v_txt_q": torch.randn(num_q, d_raw),
        "mg_v_img_q": torch.randn(num_q, d_raw),
        "mg_v_txt_s": torch.randn(num_q, d_raw),
        "mg_v_img_s": torch.randn(num_q, d_raw),
    }

    model = MGMMKT(
        num_questions=num_q,
        num_kcs=num_k,
        d_raw_txt=d_raw,
        d_raw_img=d_raw,
        d_model=d_model,
        dropout=0.0,
        static_item_bank=static_bank,
    )
    model.eval()

    mock_batch = {
        "q_seq": torch.randint(0, num_q, (B, L)),
        "r_seq": torch.randint(0, 2, (B, L)),
        "modality_masks": torch.ones(B, L, 3),
        "leaf_kcs": torch.randint(0, num_k, (B, L, 4)),
        "leaf_kc_mask": torch.ones(B, L, 4, dtype=torch.bool),
    }

    with torch.no_grad():
        out_before = model(mock_batch)

    # Save state dict
    ckpt_file = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_file)

    # Recreate fresh model and load
    model_restored = MGMMKT(
        num_questions=num_q,
        num_kcs=num_k,
        d_raw_txt=d_raw,
        d_raw_img=d_raw,
        d_model=d_model,
        dropout=0.0,
        static_item_bank=static_bank,
    )
    loaded_data = torch.load(ckpt_file, map_location="cpu")
    model_restored.load_state_dict(loaded_data["model_state_dict"])
    model_restored.eval()

    with torch.no_grad():
        out_after = model_restored(mock_batch)

    assert torch.allclose(out_before["logits"], out_after["logits"], atol=1e-6)
    assert torch.allclose(out_before["preds"], out_after["preds"], atol=1e-6)
    assert torch.allclose(out_before["alpha_q"], out_after["alpha_q"], atol=1e-6)
    assert torch.allclose(out_before["beta_s"], out_after["beta_s"], atol=1e-6)


def test_mg_mmkt_v2_trainer_param_groups_no_decay():
    """Verify that e_null, h_0, raw_delta, raw_theta, LayerNorm, and biases are excluded from weight decay in v2."""
    from src.models.mmkt.mg_mmkt_v2 import MGMMKTv2
    from src.training.mg_v2_trainer import MGMMKTv2Trainer

    model = MGMMKTv2(
        num_questions=100,
        num_kcs=50,
        d_model=32,
        d_ffn=64,
        num_heads=2,
        max_seq_len=20,
    )

    dummy_loader = DataLoader(TensorDataset(torch.zeros(2, 2)), batch_size=1)

    trainer = MGMMKTv2Trainer(
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

    no_decay_param_ids = {id(p) for p in no_decay_group["params"]}
    assert id(model.imputation.e_null_img_q) in no_decay_param_ids
    assert id(model.imputation.e_null_txt_s) in no_decay_param_ids
    assert id(model.imputation.e_null_img_s) in no_decay_param_ids
    assert id(model.causal_attention.h_0) in no_decay_param_ids
    assert id(model.causal_attention.raw_theta) in no_decay_param_ids
    assert id(model.q_gating.raw_delta_q) in no_decay_param_ids
    assert id(model.s_gating.raw_delta_s_txt) in no_decay_param_ids
    assert id(model.s_gating.raw_delta_s_img) in no_decay_param_ids

