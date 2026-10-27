#!/usr/bin/env bash
# ==============================================================================
# SCRIPT CHẠY QUA ĐÊM (OVERNIGHT RUNNER): MG-MMKT & ABLATION STUDIES (5-FOLD)
# Tự động thực thi tuần tự:
# 1. Huấn luyện nốt các Fold 2, 3, 4 cho mô hình gốc MG-MMKT Full.
# 2. Tổng hợp báo cáo 5-Fold hoàn chỉnh (Folds 0..4) cho MG-MMKT Full.
# 3. Huấn luyện Ablation 2 (w/o Solution) trên cả 5 folds.
# 4. Huấn luyện Ablation 3 (w/o Visual) trên cả 5 folds.
# 5. Huấn luyện Ablation 4 (Naive Concat) trên cả 5 folds.
# ==============================================================================

set -e  # Dừng ngay nếu có lệnh bị lỗi bất thường

START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "=========================================================================="
echo ">>> BẮT ĐẦU CHUỖI HUẤN LUYỆN QUA ĐÊM: $START_TIME"
echo "=========================================================================="

mkdir -p outputs/reports/full_model
mkdir -p outputs/reports/ablation2
mkdir -p outputs/reports/ablation3
mkdir -p outputs/reports/ablation4

# ------------------------------------------------------------------------------
# BƯỚC 1: Huấn luyện nốt Fold 2, 3, 4 cho Full MG-MMKT
# (Giữ nguyên cấu hình trong configs/train.yaml, checkpoints lưu outputs/checkpoints/)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [1/5] Huấn luyện Full MG-MMKT trên các Fold còn lại: 2, 3, 4..."
python scripts/run_5fold_mg_mmkt.py \
    --folds 2 3 4 \
    --device mps \
    --save-base-dir outputs/checkpoints \
    --output-report-dir outputs/reports/full_model

# ------------------------------------------------------------------------------
# BƯỚC 2: Tổng hợp báo cáo Test 5-Fold toàn diện (Folds 0..4) cho Full MG-MMKT
# ------------------------------------------------------------------------------
echo ""
echo ">>> [2/5] Tổng hợp báo cáo 5-Fold hoàn chỉnh cho Full MG-MMKT (Folds 0..4)..."
python scripts/run_5fold_mg_mmkt.py \
    --eval-only \
    --folds 0 1 2 3 4 \
    --device mps \
    --save-base-dir outputs/checkpoints \
    --output-report-dir outputs/reports/full_model

# ------------------------------------------------------------------------------
# BƯỚC 3: Nghiên cứu Loại suy 2 (Ablation 2: w/o Solution) trên toàn bộ 5 Folds
# ------------------------------------------------------------------------------
echo ""
echo ">>> [3/5] Huấn luyện Ablation 2 (w/o Solution) trên 5 Folds..."
python scripts/run_5fold_mg_mmkt.py \
    --ablation 2 \
    --folds 0 1 2 3 4 \
    --device mps \
    --save-base-dir outputs/checkpoints/ablation2_wo_solution \
    --output-report-dir outputs/reports/ablation2

# ------------------------------------------------------------------------------
# BƯỚC 4: Nghiên cứu Loại suy 3 (Ablation 3: w/o Visual) trên toàn bộ 5 Folds
# ------------------------------------------------------------------------------
echo ""
echo ">>> [4/5] Huấn luyện Ablation 3 (w/o Visual) trên 5 Folds..."
python scripts/run_5fold_mg_mmkt.py \
    --ablation 3 \
    --folds 0 1 2 3 4 \
    --device mps \
    --save-base-dir outputs/checkpoints/ablation3_wo_visual \
    --output-report-dir outputs/reports/ablation3

# ------------------------------------------------------------------------------
# BƯỚC 5: Nghiên cứu Loại suy 4 (Ablation 4: Naive Concat) trên toàn bộ 5 Folds
# ------------------------------------------------------------------------------
echo ""
echo ">>> [5/5] Huấn luyện Ablation 4 (Naive Concat) trên 5 Folds..."
python scripts/run_5fold_mg_mmkt.py \
    --ablation 4 \
    --folds 0 1 2 3 4 \
    --device mps \
    --save-base-dir outputs/checkpoints/ablation4_naive_concat \
    --output-report-dir outputs/reports/ablation4

END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo ""
echo "=========================================================================="
echo ">>> TOÀN BỘ TIẾN TRÌNH ĐÃ HOÀN THÀNH XUẤT SẮC: $END_TIME"
echo ">>> Kiểm tra các báo cáo tổng hợp tại: outputs/reports/"
echo "=========================================================================="
