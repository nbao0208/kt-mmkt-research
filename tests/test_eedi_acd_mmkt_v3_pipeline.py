"""
End-to-End Training, Evaluation, Explainability & Ablation Pipeline Tests for ACD-MMKTv3 on Eedi.

Verifies:
1. Dataset router isolation between XES3G5M and Eedi (checkpoints, reports, logs).
2. EediTestContext alignment across unified (124,960), public (74,276), and private (50,684) splits.
3. Task 3 expert pairwise agreement evaluator and Eedi-aware ACDv3ExplainabilityEngine.
4. Full ACD-MMKTv3 & Ablation 1-3 training/evaluation compatibility on real Eedi processed data,
   including CLI end-to-end checkpoint saving and evaluation via scripts/evaluate_acd_mmkt_v3.py.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset_router import (
    get_dataloaders_for_dataset,
    load_acd_v3_fold_metadata,
    resolve_dataset_key,
    resolve_dataset_paths,
    resolve_split_tag,
)
from src.data.eedi_dataset import collate_eedi
from src.evaluation.acd_v3_explainability import ACDv3ExplainabilityEngine
from src.evaluation.eedi_test_context import EediTestContext, evaluate_task3_expert_pairs
from src.models.mmkt.acd_mmkt_v3 import ACDMMKTv3
from src.training.acd_v3_trainer import ACDMMKTv3Trainer
from src.training.losses import ACDMMKTv3Loss
from src.utils.config import load_yaml_config


EEDI_PROC = Path("data/processed/Eedi")
requires_eedi = pytest.mark.skipif(
    not (EEDI_PROC / "static_item_bank.pt").exists(),
    reason="Processed Eedi artifacts not present in data/processed/Eedi",
)


def test_dataset_router_isolation():
    """Ensure XES3G5M paths stay untouched while Eedi outputs isolate under */eedi/."""
    xes_cfg = load_yaml_config("configs/dataset/xes3g5m.yaml")
    eedi_cfg = load_yaml_config("configs/dataset/eedi_task34.yaml")
    eedi_train_cfg = load_yaml_config("configs/train_acd_mmkt_v3_eedi.yaml")

    assert resolve_dataset_key(dataset_cfg=xes_cfg) == "xes3g5m"
    assert resolve_dataset_key(dataset_cfg=eedi_cfg) == "eedi"
    assert resolve_dataset_key(dataset_arg="eedi") == "eedi"

    xes_paths = resolve_dataset_paths("xes3g5m", dataset_cfg=xes_cfg)
    assert xes_paths["data_dir"] == "data/processed/XES3G5M"
    assert xes_paths["checkpoint_dir"] == "outputs/checkpoints"
    assert xes_paths["output_report_dir"] == "outputs/reports"
    assert xes_paths["log_dir"] == "outputs/logs"

    eedi_paths = resolve_dataset_paths(
        "eedi",
        dataset_cfg=eedi_cfg,
        train_cfg=eedi_train_cfg,
        checkpoint_dir="outputs/checkpoints",
        output_report_dir="outputs/reports",
    )
    assert eedi_paths["data_dir"] == "data/processed/Eedi"
    assert eedi_paths["checkpoint_dir"] == "outputs/checkpoints/eedi"
    assert eedi_paths["output_report_dir"] == "outputs/reports/eedi"
    assert eedi_paths["log_dir"] == "outputs/logs/eedi"

    assert resolve_split_tag("xes3g5m", "unified") == ""
    assert resolve_split_tag("xes3g5m", "public") == ""
    assert resolve_split_tag("eedi", "unified") == ""
    assert resolve_split_tag("eedi", "public") == "_public"
    assert resolve_split_tag("eedi", "private") == "_private"


@requires_eedi
def test_eedi_test_context_and_task3_expert_pairs():
    """Verify EediTestContext counts, strata masks, and Task 3 expert validation."""
    for split, expected_n in [("unified", 124_960), ("public", 74_276), ("private", 50_684)]:
        ctx = EediTestContext(processed_dir=str(EEDI_PROC), test_split=split)
        assert len(ctx.targets) == expected_n
        assert len(ctx.q_ids) == expected_n
        assert len(ctx.student_index) == expected_n
        strata = ctx.strata_masks()
        assert int(strata["overall"].sum()) == expected_n
        assert int((strata["q_stem_image"] | strata["q_text_only"]).sum()) == expected_n
        assert int((strata["after_incorrect"] | strata["after_correct"]).sum()) == expected_n
        assert "after_dom_misconception" in strata
        assert "task4_any_target_split" in strata

    aux = torch.load(EEDI_PROC / "acd_v3_aux_metadata.pt", map_location="cpu", weights_only=False)
    rng = np.random.default_rng(42)
    a_q = rng.uniform(0.5, 2.0, size=948)
    tilde_mu = rng.normal(0.0, 1.0, size=948)
    t3 = evaluate_task3_expert_pairs(a_q, tilde_mu, aux["task3_expert_pairs"])
    assert t3["num_pairs"] == 50
    assert 0.0 <= t3["a_q_agreement"] <= 1.0


@requires_eedi
def test_eedi_explainability_default_aux_loading():
    """Verify ACDv3ExplainabilityEngine auto-selects Eedi English tokens for 948-item models."""
    _, _, test_loader, bank = get_dataloaders_for_dataset(
        dataset_key="eedi",
        data_dir=str(EEDI_PROC),
        val_fold=0,
        batch_size=2,
        test_batch_size=2,
        num_workers=0,
    )
    model = ACDMMKTv3(static_item_bank=bank, d_model=32, num_heads=2, max_seq_len=200).eval()
    engine = ACDv3ExplainabilityEngine(model=model, aux_metadata_path=None, device=torch.device("cpu"))
    assert engine.aux_metadata is not None
    assert len(engine.aux_metadata["token_to_word_q"]) == 948

    batch = next(iter(test_loader))
    valid_len = int(batch["seq_mask"][0].sum().item())
    exp = engine.explain_interaction(batch=batch, sample_idx=0, target_step=min(5, valid_len - 1))
    assert exp["tier1_psychometrics"]["prediction_faithful"] is True


@requires_eedi
def test_eedi_trainer_and_evaluate_cli_smoke(tmp_path: Path):
    """
    Smoke-test ACDMMKTv3Trainer on real Eedi tensors (Full + diagnostic post_diag_mode),
    save a checkpoint, and execute scripts/evaluate_acd_mmkt_v3.py end-to-end.
    """
    train_loader, val_loader, test_loader, bank = get_dataloaders_for_dataset(
        dataset_key="eedi",
        data_dir=str(EEDI_PROC),
        val_fold=0,
        model_type="acd_mmkt_v3",
        batch_size=4,
        test_batch_size=8,
        p_drop=0.15,
        num_workers=0,
        test_split="private",
    )
    fold_meta = load_acd_v3_fold_metadata(str(EEDI_PROC), val_fold=0, num_questions=948)

    model_cfg = {
        "dataset_name": "Eedi NeurIPS 2020 (Task 3 & 4)",
        "num_questions": 948,
        "num_kcs": 85,
        "d_raw_txt": 768,
        "d_raw_img": 768,
        "d_model": 32,
        "num_heads": 2,
        "max_seq_len": 200,
        "dropout": 0.10,
        "hgnn_layers": 2,
        "gamma_vis": 1.5,
        "delta_a": 0.1,
        "use_multimodal": True,
        "use_solution": True,
        "use_visual": True,
        "post_diag_mode": "diagnostic",
    }

    model = ACDMMKTv3(
        static_item_bank=bank,
        d_model=32,
        num_heads=2,
        max_seq_len=200,
        dropout=0.10,
        p_drop=0.15,
        post_diag_mode="diagnostic",
    )
    sub_train_loader = DataLoader(Subset(train_loader.dataset, range(8)), batch_size=4, collate_fn=collate_eedi)
    sub_val_loader = DataLoader(Subset(val_loader.dataset, range(8)), batch_size=4, collate_fn=collate_eedi)

    ckpt_dir = tmp_path / "ckpt"
    trainer = ACDMMKTv3Trainer(
        model=model,
        train_loader=sub_train_loader,
        val_loader=sub_val_loader,
        n_q=fold_meta["n_q"],
        d_q=fold_meta["d_q"],
        omega_q=fold_meta["omega_q"],
        mu_concept_prior=fold_meta["mu_concept_prior"],
        device=torch.device("cpu"),
        precision="fp32",
        save_dir=str(ckpt_dir),
        val_fold=0,
        model_cfg=model_cfg,
    )
    summary = trainer.train(num_epochs=1)
    assert summary["best_val_auc"] >= 0.0
    ckpt_path = ckpt_dir / "best_model.pt"
    assert ckpt_path.exists()

    # Run scripts/evaluate_acd_mmkt_v3.py CLI on private Eedi test split
    rep_json = tmp_path / "eval_eedi.json"
    exp_md = tmp_path / "explain_eedi.md"
    res = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_acd_mmkt_v3.py",
            "--checkpoint",
            str(ckpt_path),
            "--dataset",
            "eedi",
            "--test-split",
            "private",
            "--device",
            "cpu",
            "--batch-size",
            "64",
            "--output-report",
            str(rep_json),
            "--explain-output-md",
            str(exp_md),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"evaluate_acd_mmkt_v3.py failed:\n{res.stdout}\n{res.stderr}"
    assert rep_json.exists() and exp_md.exists()

    report = json.loads(rep_json.read_text(encoding="utf-8"))
    assert report["total_evaluated_interactions"] == 50_684
    assert report["global_metrics"]["decision_faithfulness_pct"] == pytest.approx(100.0)
    assert "Number" in report["domain_gate_intensity"]
    assert "after_dom_misconception" in report["eedi_stratification"]
    assert report["task3_expert_validation"]["num_pairs"] == 50

    # Verify evaluate_acd_mmkt_v3.py auto-detects Eedi from checkpoint metadata even without --dataset
    rep_auto = tmp_path / "eval_eedi_auto.json"
    res_auto = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_acd_mmkt_v3.py",
            "--checkpoint",
            str(ckpt_path),
            "--test-split",
            "private",
            "--device",
            "cpu",
            "--batch-size",
            "64",
            "--output-report",
            str(rep_auto),
        ],
        capture_output=True,
        text=True,
    )
    assert res_auto.returncode == 0, f"Auto-detect evaluate_acd_mmkt_v3.py failed:\n{res_auto.stderr}"
    auto_report = json.loads(rep_auto.read_text(encoding="utf-8"))
    assert auto_report["total_evaluated_interactions"] == 50_684


def test_cli_help_entrypoints():
    """Verify all updated ACD-MMKTv3 CLI scripts parse arguments cleanly."""
    for script in [
        "scripts/train_acd_mmkt_v3.py",
        "scripts/evaluate_acd_mmkt_v3.py",
        "scripts/run_5fold_acd_mmkt_v3.py",
        "scripts/run_acd_v3_modality_ablation_study.py",
    ]:
        res = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True)
        assert res.returncode == 0, f"{script} --help failed: {res.stderr}"
        assert "--dataset" in res.stdout


@requires_eedi
def test_eedi_modality_ablation_pipeline_init_and_eval(tmp_path: Path):
    """Verify ModalityAblationPipeline initializes cleanly on Eedi and evaluates strata & psychometrics."""
    from scripts.run_acd_v3_modality_ablation_study import ModalityAblationPipeline

    pipeline = ModalityAblationPipeline(
        dataset_arg="eedi",
        test_split="private",
        checkpoints_base=str(tmp_path / "ckpts"),
        reports_base=str(tmp_path / "reports"),
        device_str="cpu",
    )
    assert pipeline.dataset_key == "eedi"
    assert pipeline.split_tag == "_private"
    assert pipeline.num_questions == 948
    assert pipeline.num_test_interactions == 50_684

    rng = np.random.default_rng(0)
    fake_preds = rng.uniform(0.1, 0.9, size=50_684)
    strat = pipeline.compute_stratified_evaluation(fake_preds)
    assert strat["overall"]["num_samples"] == 50_684
    assert "layout_horizontal" in strat
    assert "after_dom_misconception" in strat

    fake_props = [
        {
            "tilde_mu": rng.normal(0.0, 1.0, size=948),
            "a_q": rng.uniform(0.5, 2.0, size=948),
        }
    ]
    psy = pipeline.compute_item_difficulty_correlations(fake_props)
    assert "pearson_r" in psy
    assert psy["task3_expert_validation"]["num_pairs"] == 50


@requires_eedi
def test_eedi_ablation_variants_with_diagnostic_post_diag():
    """
    Verify Full and Ablations 1-3 instantiate, forward, and backpropagate cleanly
    with Eedi's post_diag_mode='diagnostic' on real static_item_bank.pt.
    """
    train_loader, _, _, bank = get_dataloaders_for_dataset(
        dataset_key="eedi",
        data_dir=str(EEDI_PROC),
        val_fold=0,
        batch_size=4,
        test_batch_size=4,
        num_workers=0,
    )
    fold_meta = load_acd_v3_fold_metadata(str(EEDI_PROC), val_fold=0, num_questions=948)
    batch = next(iter(train_loader))

    variant_flags = {
        "full": dict(use_multimodal=True, use_solution=True, use_visual=True, beta_d=0.20, lambda_sparse=0.01),
        "ab3": dict(use_multimodal=True, use_solution=True, use_visual=False, beta_d=0.20, lambda_sparse=0.01),
        "ab2": dict(use_multimodal=True, use_solution=False, use_visual=True, beta_d=0.20, lambda_sparse=0.01),
        "ab1": dict(use_multimodal=False, use_solution=False, use_visual=False, beta_d=0.0, lambda_sparse=0.0),
    }
    param_counts = {}

    for name, flags in variant_flags.items():
        model = ACDMMKTv3(
            static_item_bank=bank,
            d_model=32,
            num_heads=2,
            max_seq_len=200,
            dropout=0.10,
            use_multimodal=flags["use_multimodal"],
            use_solution=flags["use_solution"],
            use_visual=flags["use_visual"],
            post_diag_mode="diagnostic",
        )
        param_counts[name] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        loss_fn = ACDMMKTv3Loss(
            beta_d=flags["beta_d"],
            lambda_sparse=flags["lambda_sparse"],
        )
        out = model(batch, mu_concept_prior=fold_meta["mu_concept_prior"])
        loss_total, loss_metrics = loss_fn(
            model_outputs=out,
            targets=batch["r_seq"],
            loss_mask=batch["loss_mask"],
            n_q_all=fold_meta["n_q"],
            d_q_all=fold_meta["d_q"],
        )
        assert "loss_total" in loss_metrics
        assert torch.isfinite(loss_total), f"Non-finite loss for {name}"
        loss_total.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)

    assert param_counts["full"] > param_counts["ab3"] > param_counts["ab1"]
    assert param_counts["full"] > param_counts["ab2"] > param_counts["ab1"]


def test_xes3g5m_defaults_preserved(tmp_path: Path):
    """Ensure XES3G5M configs, paths, post_diag_mode, and ablation pipeline defaults remain unchanged."""
    xes_ds_cfg = load_yaml_config("configs/dataset/xes3g5m.yaml")
    xes_train_cfg = load_yaml_config("configs/train_acd_mmkt_v3.yaml")
    xes_model_cfg = load_yaml_config("configs/model/acd_mmkt_v3.yaml")

    assert resolve_dataset_key(dataset_cfg=xes_ds_cfg, dataset_arg=None) == "xes3g5m"
    assert resolve_dataset_key(dataset_cfg=xes_ds_cfg, dataset_arg="xes3g5m") == "xes3g5m"
    assert xes_model_cfg.get("post_diag_mode", "none") == "none"
    assert "post_diag_mode" not in xes_train_cfg or xes_train_cfg.get("post_diag_mode") in (None, "none")

    xes_paths = resolve_dataset_paths(
        "xes3g5m",
        dataset_cfg=xes_ds_cfg,
        train_cfg=xes_train_cfg,
        checkpoint_dir="outputs/checkpoints",
        output_report_dir="outputs/reports",
    )
    assert xes_paths["data_dir"] == "data/processed/XES3G5M"
    assert xes_paths["checkpoint_dir"] == "outputs/checkpoints"
    assert xes_paths["output_report_dir"] == "outputs/reports"
    assert xes_paths["log_dir"] == "outputs/logs"
    assert resolve_split_tag("xes3g5m", "unified") == ""

    if Path("data/processed/XES3G5M/acd_v3_aux_metadata.pt").exists() and Path("data/processed/XES3G5M/test_context_cache.pt").exists():
        from scripts.run_acd_v3_modality_ablation_study import ModalityAblationPipeline

        pipeline = ModalityAblationPipeline(
            dataset_arg=None,
            checkpoints_base=str(tmp_path / "xes_ckpts"),
            reports_base=str(tmp_path / "xes_reports"),
            device_str="cpu",
        )
        assert pipeline.dataset_key == "xes3g5m"
        assert pipeline.split_tag == ""
        assert pipeline.num_questions == 7652
        assert pipeline.num_test_interactions == 1_099_197
        assert pipeline.model_cfg_base.get("post_diag_mode", "none") == "none"
