#!/usr/bin/env bash
# ==============================================================================
# SCRIPT HUẤN LUYỆN QUA ĐÊM (OVERNIGHT RUNNER): ACD-MMKT & ABLATION STUDIES (5-FOLD)
# Tự động thực thi tuần tự:
# 1. Huấn luyện ACD-MMKT Full Model trên Fold 0 (Stage 1 Item-Bank + Stage 2 Fine-Tuning).
# 2. Đánh giá chi tiết trên tập Test độc lập (1.099.197 tương tác) và xuất báo cáo giải thích 3 cấp độ.
# 3. Huấn luyện 5 Folds chuẩn pyKT (tận dụng lại checkpoint Fold 0 từ Bước 1, tránh huấn luyện dư thừa).
# 4. Huấn luyện 4 biến thể Nghiên cứu Cắt bỏ (Ablation 2, 3, 4, 5) trên Fold 0.
# 5. Đánh giá kiểm thử độc lập cho cả 4 biến thể Ablation và tổng hợp báo cáo.
# ==============================================================================

set -e  # Dừng ngay nếu có lệnh bị lỗi bất thường

START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "=========================================================================="
echo ">>> BẮT ĐẦU CHUỖI HUẤN LUYỆN QUA ĐÊM ACD-MMKT: $START_TIME"
echo "=========================================================================="

mkdir -p outputs/reports/acd_mmkt_full
mkdir -p outputs/reports/acd_ablation2
mkdir -p outputs/reports/acd_ablation3
mkdir -p outputs/reports/acd_ablation4
mkdir -p outputs/reports/acd_ablation5
mkdir -p outputs/checkpoints

# 0. Xác định Python runtime chuẩn theo quy định Conda kt-research-env
if [ -f "/opt/miniconda3/envs/kt-research-env/bin/python" ]; then
    PYTHON_EXEC="/opt/miniconda3/envs/kt-research-env/bin/python"
elif command -v conda &> /dev/null; then
    PYTHON_EXEC="conda run -n kt-research-env python"
else
    PYTHON_EXEC="python"
fi

echo ">>> Sử dụng Python interpreter: $PYTHON_EXEC"

# Tự động phát hiện thiết bị tính toán tối ưu
if $PYTHON_EXEC -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    DEVICE="cuda"
    PRECISION="amp_fp16"
elif $PYTHON_EXEC -c "import torch; exit(0 if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 1)"; then
    DEVICE="mps"
    PRECISION="fp32"
else
    DEVICE="cpu"
    PRECISION="fp32"
fi

echo ">>> Thiết bị tính toán được chọn: $DEVICE (Độ chính xác: $PRECISION)"

# ------------------------------------------------------------------------------
# BƯỚC 1: Huấn luyện ACD-MMKT Full trên Fold 0 (Two-Stage Curriculum)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [1/5] Huấn luyện ACD-MMKT Full trên Fold 0 (Stage 1 + Stage 2)..."
$PYTHON_EXEC scripts/train_acd_mmkt.py \
    --val-fold 0 \
    --device "$DEVICE" \
    --precision "$PRECISION" \
    --num-epochs 30 \
    --save-dir outputs/checkpoints/acd_mmkt_fold0

# ------------------------------------------------------------------------------
# BƯỚC 2: Đánh giá chi tiết tập Test và xuất báo cáo giải thích 3 cấp độ
# ------------------------------------------------------------------------------
echo ""
echo ">>> [2/5] Đánh giá trên tập Test độc lập và xuất báo cáo giải thích 3 cấp độ..."
$PYTHON_EXEC scripts/evaluate_acd_mmkt.py \
    --checkpoint outputs/checkpoints/acd_mmkt_fold0/best_model.pt \
    --val-fold 0 \
    --device "$DEVICE" \
    --output-report outputs/reports/acd_mmkt_full/eval_fold0_detailed.json \
    --explain-output-md outputs/reports/acd_mmkt_full/sample_explanation.md

# ------------------------------------------------------------------------------
# BƯỚC 3: Chạy 5-Fold Cross Validation chuẩn pyKT cho ACD-MMKT Full
# (Huấn luyện các Fold 1, 2, 3, 4; tái sử dụng checkpoint Fold 0 từ Bước 1)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [3/5] Thực thi 5-Fold Cross Validation chuẩn pyKT cho ACD-MMKT Full..."
# Huấn luyện các fold còn lại (1, 2, 3, 4)
$PYTHON_EXEC scripts/run_5fold_acd_mmkt.py \
    --folds 1 2 3 4 \
    --device "$DEVICE" \
    --num-epochs 30 \
    --checkpoint-dir outputs/checkpoints \
    --output-report-dir outputs/reports/acd_mmkt_full

# Đánh giá tổng hợp toàn diện trên cả 5 Folds [0, 1, 2, 3, 4]
echo ">>> Tổng hợp bảng kết quả 5-Fold CV (Mean ± Std)..."
$PYTHON_EXEC scripts/run_5fold_acd_mmkt.py \
    --eval-only \
    --folds 0 1 2 3 4 \
    --checkpoint-dir outputs/checkpoints \
    --output-report-dir outputs/reports/acd_mmkt_full

# ------------------------------------------------------------------------------
# BƯỚC 4: Huấn luyện 4 biến thể Nghiên cứu Cắt bỏ (Ablation 2, 3, 4, 5) trên Fold 0
# ------------------------------------------------------------------------------
echo ""
echo ">>> [4/5] Huấn luyện 4 mô hình Nghiên cứu Cắt bỏ (Ablation Studies) trên Fold 0..."

echo "--- 4.1 Huấn luyện Ablation 2: w/o Solution Distillation ---"
$PYTHON_EXEC scripts/train_acd_mmkt.py \
    --val-fold 0 \
    --ablation 2 \
    --device "$DEVICE" \
    --precision "$PRECISION" \
    --num-epochs 30 \
    --save-dir outputs/checkpoints/acd_ablation2_fold0

echo "--- 4.2 Huấn luyện Ablation 3: w/o Fine-Grained Alignment (Global Pooling) ---"
$PYTHON_EXEC scripts/train_acd_mmkt.py \
    --val-fold 0 \
    --ablation 3 \
    --device "$DEVICE" \
    --precision "$PRECISION" \
    --num-epochs 30 \
    --save-dir outputs/checkpoints/acd_ablation3_fold0

echo "--- 4.3 Huấn luyện Ablation 4: w/o Hypergraph & Rasch Decomposition ---"
$PYTHON_EXEC scripts/train_acd_mmkt.py \
    --val-fold 0 \
    --ablation 4 \
    --device "$DEVICE" \
    --precision "$PRECISION" \
    --num-epochs 30 \
    --save-dir outputs/checkpoints/acd_ablation4_fold0

echo "--- 4.4 Huấn luyện Ablation 5: w/o DeepIRT Readout (Black-box MLP) ---"
$PYTHON_EXEC scripts/train_acd_mmkt.py \
    --val-fold 0 \
    --ablation 5 \
    --device "$DEVICE" \
    --precision "$PRECISION" \
    --num-epochs 30 \
    --save-dir outputs/checkpoints/acd_ablation5_fold0

# ------------------------------------------------------------------------------
# BƯỚC 5: Đánh giá độc lập 4 biến thể Ablation và xuất báo cáo đối sánh
# ------------------------------------------------------------------------------
echo ""
echo ">>> [5/5] Đánh giá độc lập 4 biến thể Ablation trên tập Test..."

$PYTHON_EXEC scripts/evaluate_acd_mmkt.py \
    --checkpoint outputs/checkpoints/acd_ablation2_fold0/best_model.pt \
    --val-fold 0 \
    --device "$DEVICE" \
    --output-report outputs/reports/acd_ablation2/eval_fold0.json

$PYTHON_EXEC scripts/evaluate_acd_mmkt.py \
    --checkpoint outputs/checkpoints/acd_ablation3_fold0/best_model.pt \
    --val-fold 0 \
    --device "$DEVICE" \
    --output-report outputs/reports/acd_ablation3/eval_fold0.json

$PYTHON_EXEC scripts/evaluate_acd_mmkt.py \
    --checkpoint outputs/checkpoints/acd_ablation4_fold0/best_model.pt \
    --val-fold 0 \
    --device "$DEVICE" \
    --output-report outputs/reports/acd_ablation4/eval_fold0.json

$PYTHON_EXEC scripts/evaluate_acd_mmkt.py \
    --checkpoint outputs/checkpoints/acd_ablation5_fold0/best_model.pt \
    --val-fold 0 \
    --device "$DEVICE" \
    --output-report outputs/reports/acd_ablation5/eval_fold0.json

echo ""
echo "=========================================================================="
echo ">>> HOÀN THÀNH TOÀN DIỆN CHUỖI HUẤN LUYỆN ACD-MMKT QUA ĐÊM: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================================="
