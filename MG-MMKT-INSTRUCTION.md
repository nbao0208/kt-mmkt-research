# HƯỚNG DẪN CHI TIẾT HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH MG-MMKT
*(Multimodal Knowledge Tracing with Masked Gated Fusion, Learnable Imputation & First-Order Explainability)*

Tài liệu này cung cấp toàn bộ hướng dẫn kỹ thuật chi tiết để tái lập, kiểm thử, huấn luyện (đơn fold, 5-fold cross-validation, ablation studies), và thực hiện đánh giá phân tầng kèm trích xuất giải thích (explainability) cho mô hình **MG-MMKT** trên tập dữ liệu benchmark **XES3G5M**.

---

## MỤC LỤC
1. [Tổng Quan Kiến Trúc & Luồng Dữ Liệu](#1-tổng-quan-kiến-trúc--luồng-dữ-liệu)
2. [Môi Trường Thực Thi & Quy Định](#2-môi-trường-thực-thi--quy-định)
3. [Giai Đoạn 0: Tiền Xử Lý Dữ Liệu XES3G5M](#3-giai-đoạn-0-tiền-xử-lý-dữ-liệu-xes3g5m)
4. [Giai Đoạn 1: Kiểm Thử Tự Động (Unit & Integration Tests)](#4-giai-đoạn-1-kiểm-thử-tự-động-unit--integration-tests)
5. [Giai Đoạn 2: Huấn Luyện Mô Hình MG-MMKT](#5-giai-đoạn-2-huấn-luyện-mô-hình-mg-mmkt)
   - [5.1 Huấn luyện Đơn Fold (Single Fold)](#51-huấn-luyện-đơn-fold-single-fold)
   - [5.2 Huấn luyện Toàn Diện 5-Fold Cross-Validation](#52-huấn-luyện-toàn-diện-5-fold-cross-validation)
   - [5.3 Thực hiện Nghiên cứu Loại suy (Ablation Studies)](#53-thực-hiện-nghiên-cứu-loại-suy-ablation-studies)
   - [5.4 Huấn luyện Baseline ID-Only SAKT](#54-huấn-luyện-baseline-id-only-sakt)
6. [Giai Đoạn 3: Đánh Giá Độc Lập & Phân Tầng Đa Phương Thức](#6-giai-đoạn-3-đánh-giá-độc-lập--phân-tầng-đa-phương-thức)
7. [Giai Đoạn 4: Trích Xuất Khả Năng Giải Thích (Explainability)](#7-giai-đoạn-4-trích-xuất-khả-năng-giải-thích-explainability)
8. [Cấu Trúc Tệp Cấu Hình & Artifacts](#8-cấu-trúc-tệp-cấu-hình--artifacts)
9. [Xử Lý Sự Cố & Tối Ưu Bộ Nhớ (Troubleshooting)](#9-xử-lý-sự-cố--tối-ưu-bộ-nhớ-troubleshooting)

---

## 1. Tổng Quan Kiến Trúc & Luồng Dữ Liệu

Mô hình **MGMMKT** kế thừa nền tảng Transformer causal attention của SAKT và giải quyết các bài toán cốt lõi của Knowledge Tracing đa phương thức:
- **Xử lý khuyết thiếu tự nhiên & Modality Dropout:** Thay vì zero-padding làm méo dạng phân phối, mô hình sử dụng các vector học được $e_{\text{null}}^{\text{img}, q}, e_{\text{null}}^{\text{txt}, s}, e_{\text{null}}^{\text{img}, s}$ trong `ModalityImputationLayer` (`src/models/modules/modality_imputation.py`).
- **Masked Gated Fusion:** Sử dụng Softmax có mặt nạ trong `QuestionMaskedGatedFusion` và `SolutionMaskedGatedFusion` (`src/models/modules/gated_fusion.py`) để tự động cân bằng đóng góp giữa ID, Text và Image.
- **Tính nhân quả nghiêm ngặt & BOS Attention Sink:** Lời giải chỉ đi vào luồng lịch sử $y_\tau$ ($\tau < t$). Cột đầu tiên trong ma trận Attention được gán cố định cho token $y_0 = e_{\text{BOS}}$ trong `CausalSAKTLayer` (`src/models/modules/causal_attention.py`) để đóng vai trò Prior Sink, loại bỏ hoàn toàn hiện tượng Attention Leakage và Mask Collapse tại $t=0$.
- **Tuân thủ chuẩn pyKT:** Phân chia tập dữ liệu theo 5-fold student partition isolation (0% rò rỉ học sinh) và đánh giá trên tập Test bằng cửa sổ trượt bảo toàn ngữ cảnh (Context-Preserving Strided Windows, $S=100, L=200$).

```
[Student Interaction at step t]
  Target Question q_t:
    ID Emb (E_q) + Text (v_txt_q) + Image (v_img_q or e_null_img_q)
    ──> QuestionMaskedGatedFusion (alpha_t^q) ──> Query x_t [d_model]

  Historical Sequence (tau < t):
    Interaction y_tau = (q_tau, r_tau, delta_t, KC, CTT)
                        + Solution s_tau (v_txt_s, v_img_s, beta_tau^s)
    Prefix with y_0 (e_BOS) ──> Key/Value y_tau [d_model]

  Causal Attention:
    Attention(Query: x_t, Key: y_0..y_{t-1}, Value: y_0..y_{t-1})
    ──> Context Representation h_t ──> Prediction Head ──> P(r_t = 1)
```

---

## 2. Môi Trường Thực Thi & Quy Định

### 2.1 Môi Trường Conda Bắt Buộc
Mọi lệnh Python, pytest, hoặc cài đặt thư viện **bắt buộc** phải sử dụng môi trường Conda `kt-research-env`.

- **Kích hoạt môi trường trong Terminal:**
  ```bash
  conda activate kt-research-env
  ```
- **Hoặc thực thi qua `conda run`:**
  ```bash
  conda run -n kt-research-env python <đường_dẫn_script>
  conda run -n kt-research-env pytest
  ```
- **Đường dẫn Python binary trực tiếp:**
  ```bash
  /opt/miniconda3/envs/kt-research-env/bin/python
  ```

### 2.2 Kiểm Tra Thiết Bị Tính Toán
Mô hình tự động hỗ trợ Apple Silicon (`mps`), NVIDIA CUDA (`cuda`), hoặc fallback về `cpu`:
```bash
conda run -n kt-research-env python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| MPS:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
```

---

## 3. Giai Đoạn 0: Tiền Xử Lý Dữ Liệu XES3G5M

> [!NOTE]
> Nếu trong thư mục `data/processed/XES3G5M/` đã có sẵn các file `mg_static_item_bank.pt`, `sequences_fold_0.pt` .. `4.pt`, và `sequences_test.pt`, bạn có thể **bỏ qua bước này** và chuyển thẳng sang [Giai Đoạn 1](#4-giai-đoạn-1-kiểm-thử-tự-động-unit--integration-tests).

Nếu cần tiền xử lý từ dữ liệu thô XES3G5M:

### 3.1 Cấu Trúc Thư Mục Dữ Liệu Thô Cần Thiết
```
data/raw/XES3G5M/
├── metadata/
│   ├── questions.json
│   └── kc_routes.json
├── features/
│   ├── question_text_embeddings.pt   # [N_q, 768] trích xuất bởi Chinese-RoBERTa
│   ├── question_image_embeddings.pt  # [N_img, 768] trích xuất bởi CLIP-ViT
│   ├── solution_text_embeddings.pt   # [N_s, 768]
│   └── solution_image_embeddings.pt  # [N_s_img, 768]
└── sequences/
    ├── train_val_sequences.csv       # Gồm 5 fold pyKT
    └── test_sequences.csv
```

### 3.2 Lệnh Thực Thi Tiền Xử Lý
Chạy script tiền xử lý 9 bước qua `scripts/preprocess_xes3g5m.py`:
```bash
conda run -n kt-research-env python scripts/preprocess_xes3g5m.py \
    --config configs/dataset/xes3g5m.yaml \
    --raw_dir data/raw/XES3G5M \
    --processed_dir data/processed/XES3G5M
```

### 3.3 Đầu Ra Được Sinh Ra
```
data/processed/XES3G5M/
├── mg_static_item_bank.pt    # [Khuyên dùng] Tier 1 item bank nén siêu nhanh (~48 MB)
├── static_item_bank.pt       # Tier 1 item bank đầy đủ kèm embedding gốc (~4.6 GB)
├── sequences_fold_0.pt       # Tier 2 dynamic sequences Fold 0 (~41 MB)
├── sequences_fold_1.pt       # Tier 2 dynamic sequences Fold 1 (~41 MB)
├── sequences_fold_2.pt       # Tier 2 dynamic sequences Fold 2 (~41 MB)
├── sequences_fold_3.pt       # Tier 2 dynamic sequences Fold 3 (~41 MB)
├── sequences_fold_4.pt       # Tier 2 dynamic sequences Fold 4 (~41 MB)
├── sequences_test.pt         # Tier 2 dynamic sequences Test Set (~51 MB)
└── metadata_summary.json     # Thống kê tổng số lượng questions, KCs, interactions
```

---

## 4. Giai Đoạn 1: Kiểm Thử Tự Động (Unit & Integration Tests)

Trước khi tiến hành huấn luyện, toàn bộ hệ thống phải vượt qua bộ kiểm thử tự động gồm 27 test cases bao quát:
- Tính bất biến nhân quả (Backward Perturbation Causality Invariance: max diff < 1e-7).
- Dòng gradient đến các token khuyết thiếu học được ($e_{\text{null}}^{\text{img}, q}, e_{\text{null}}^{\text{txt}, s}, e_{\text{null}}^{\text{img}, s}$).
- Kiểm tra tính cô lập tuyệt đối 5 fold học sinh (0% leakage).
- Kiểm tra số lượng đánh giá cửa sổ trượt test (chính xác 1,099,197 tương tác).
- Trích xuất Top-$K$ bằng chứng lịch sử và thống kê phân tầng.

### Lệnh Chạy Toàn Bộ Test Suite
```bash
conda run -n kt-research-env pytest -v
```

Kết quả mong đợi:
```text
tests/test_explainability.py ....                                        [ 14%]
tests/test_mg_mmkt_model.py ..........                                   [ 51%]
tests/test_pykt_protocol.py ....                                         [ 66%]
tests/test_trainer_and_checkpoints.py ..                                 [ 74%]
tests/test_xes3g5m_preprocessing.py .......                              [100%]

============================= 27 passed in ~18s ==============================
```

---

## 5. Giai Đoạn 2: Huấn Luyện Mô Hình MG-MMKT

### 5.1 Huấn luyện Đơn Fold (Single Fold)
Sử dụng script `scripts/train_mg_mmkt.py` khi muốn thử nghiệm nhanh hoặc tinh chỉnh trên một fold cụ thể (ví dụ Fold 0):

```bash
conda run -n kt-research-env python scripts/train_mg_mmkt.py \
    --val-fold 0 \
    --num-epochs 30 \
    --batch-size 64 \
    --lr 0.001 \
    --device auto \
    --save-dir outputs/checkpoints/mg_mmkt_fold0
```

#### Bảng Tham Số của `train_mg_mmkt.py`:
| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `--val-fold` | `0` | Chỉ số Fold làm tập Validation (0..4). 4 fold còn lại tạo thành tập Train. |
| `--num-epochs` | `30` | Số lượng epoch tối đa (có Early Stopping theo dõi Val AUC). |
| `--batch-size` | `64` | Kích thước mini-batch huấn luyện. |
| `--lr` | `1e-3` | Tốc độ học cơ sở của AdamW. |
| `--warmup-epochs`| `2` | Số epoch khởi động tuyến tính trước khi áp dụng Cosine Annealing. |
| `--eta-min` | `1e-5` | Tốc độ học tối thiểu ở chu kỳ cuối. |
| `--device` | `"auto"` | Thiết bị tính toán (`cuda`, `mps`, `cpu`, hoặc `auto`). |
| `--save-dir` | `outputs/checkpoints/mg_mmkt_fold0` | Thư mục lưu checkpoint `best_model.pt` và log `train.log`. |

---

### 5.2 Huấn luyện Toàn Diện 5-Fold Cross-Validation
Để thu thập kết quả chuẩn cho bài báo nghiên cứu, chạy quy trình 5-fold tuần tự qua `scripts/run_5fold_mg_mmkt.py`:

```bash
conda run -n kt-research-env python scripts/run_5fold_mg_mmkt.py \
    --folds 0 1 2 3 4 \
    --num-epochs 30 \
    --batch-size 64 \
    --lr 0.001 \
    --device auto \
    --save-base-dir outputs/checkpoints \
    --output-report-dir outputs/reports
```

#### Quy Trình Tự Động Được Kích Hoạt:
1. Huấn luyện lần lượt từng fold từ 0 đến 4 với seed ngẫu nhiên độc lập (`base_seed + fold`).
2. Ghi nhật ký đào tạo riêng biệt cho từng fold vào `outputs/checkpoints/mg_mmkt_fold{fold}/train_fold{fold}.log`.
3. Lưu checkpoint tốt nhất theo Val AUC tại `outputs/checkpoints/mg_mmkt_fold{fold}/best_model.pt`.
4. Đánh giá trực tiếp trên tập Test bằng cơ chế Context-Preserving Strided Windows.
5. Tổng hợp bảng kết quả trung bình cộng và độ lệch chuẩn (Mean ± Std) xuất ra file JSON và Markdown:
   - `outputs/reports/mg_mmkt_5fold_summary.json`
   - `outputs/reports/mg_mmkt_5fold_summary.md`

#### Chế Độ Đánh Giá Chỉ Đọc (Eval-Only Mode):
Nếu các checkpoint 5 fold đã được huấn luyện xong và chỉ cần đánh giá lại tổng thể mà không tốn công huấn luyện lại:
```bash
conda run -n kt-research-env python scripts/run_5fold_mg_mmkt.py \
    --eval-only \
    --save-base-dir outputs/checkpoints \
    --output-report-dir outputs/reports
```

---

### 5.3 Thực hiện Nghiên cứu Loại suy (Ablation Studies)

Mô hình hỗ trợ 4 cấu hình loại suy chính xác theo thiết kế nghiên cứu:

| Cấu hình | Mô tả | Cờ lệnh tương ứng |
| :--- | :--- | :--- |
| **Config 1 (Full)** | Đầy đủ thành phần MG-MMKT (Gated Fusion, Learnable Imputation, Solution, Visual). | `--ablation 1` (Mặc định) |
| **Config 2 (w/o Sol)** | Loại bỏ toàn bộ biểu diễn lời giải $s_\tau$ khỏi tương tác lịch sử. | `--ablation 2` hoặc `--wo-solution` |
| **Config 3 (w/o Visual)**| Loại bỏ hoàn toàn modality hình ảnh ở cả câu hỏi và lời giải. | `--ablation 3` hoặc `--wo-visual` |
| **Config 4 (Naive Concat)**| Thay Gated Fusion bằng phép ghép nối tuyến tính thô (Linear Projection) và dùng Zero-Padding thay Learnable Imputation. | `--ablation 4` hoặc `--naive-concat` |

#### Ví dụ Lệnh Chạy Loại Suy Đơn Fold:
```bash
# Nghiên cứu tác động của Lời giải (Ablation 2)
conda run -n kt-research-env python scripts/train_mg_mmkt.py \
    --val-fold 0 \
    --ablation 2 \
    --save-dir outputs/checkpoints/ablation_wo_solution_fold0

# Nghiên cứu tác động của Hình ảnh (Ablation 3)
conda run -n kt-research-env python scripts/train_mg_mmkt.py \
    --val-fold 0 \
    --ablation 3 \
    --save-dir outputs/checkpoints/ablation_wo_visual_fold0

# Nghiên cứu tác động của Gated Fusion & Learnable Imputation (Ablation 4)
conda run -n kt-research-env python scripts/train_mg_mmkt.py \
    --val-fold 0 \
    --ablation 4 \
    --save-dir outputs/checkpoints/ablation_naive_concat_fold0
```

#### Ví dụ Lệnh Chạy Loại Suy Toàn Diện 5-Fold:
```bash
conda run -n kt-research-env python scripts/run_5fold_mg_mmkt.py \
    --ablation 2 \
    --save-base-dir outputs/checkpoints/ablation2_wo_solution \
    --output-report-dir outputs/reports/ablation2
```

---

### 5.4 Huấn luyện Baseline ID-Only SAKT
Để làm thước đo đối chứng định lượng hiệu quả vượt trội của đa phương thức so với mô hình thuần ID:
```bash
conda run -n kt-research-env python scripts/run_5fold_mg_mmkt.py \
    --model-type sakt \
    --save-base-dir outputs/checkpoints/sakt_baselines \
    --output-report-dir outputs/reports/sakt
```

---

## 6. Giai Đoạn 3: Đánh Giá Độc Lập & Phân Tầng Đa Phương Thức

Script `scripts/evaluate_mg_mmkt.py` là công cụ chuyên sâu để bóc tách hiệu năng mô hình trên tập Test chính thức:
- **Xử lý cửa sổ trượt:** Tự động cắt chuỗi học sinh dài (> 200 bước) theo bước nhảy $S=100$.
- **Offset toàn cục `context_start`:** Bù trừ chỉ số vị trí để ánh xạ chính xác về mốc thời gian thực của học sinh.
- **Tính toán phân tầng (Multimodal Stratification):** Đánh giá AUC và ACC trên từng nhóm câu hỏi chuyên biệt.

### 6.1 Lệnh Chạy Đánh Giá
```bash
conda run -n kt-research-env python scripts/evaluate_mg_mmkt.py \
    --checkpoint outputs/checkpoints/mg_mmkt_fold0/best_model.pt \
    --batch-size 128 \
    --test-stride 100 \
    --device auto \
    --output-report outputs/reports/eval_fold0_detailed.json
```

### 6.2 Đầu Ra Báo Cáo Trên Terminal
Script sẽ in bảng phân tích toàn diện:
```text
==========================================================================
                     MG-MMKT EVALUATION REPORT                            
==========================================================================
Overall Test AUC: 0.8124 | ACC: 0.7456 (N = 1,099,197)
--------------------------------------------------------------------------
Questions With Image: AUC: 0.8245 | ACC: 0.7580 (N = 345,120)
Questions Text-Only:  AUC: 0.8062 | ACC: 0.7398 (N = 754,077)
--------------------------------------------------------------------------
Prior Sol Had Image:  AUC: 0.8198 | ACC: 0.7512 (N = 152,300)
Prior Sol Text-Only:  AUC: 0.8111 | ACC: 0.7447 (N = 946,897)
--------------------------------------------------------------------------
Rare Questions:       AUC: 0.7845 | ACC: 0.7130 (N = 84,200)
Frequent Questions:   AUC: 0.8148 | ACC: 0.7482 (N = 1,014,997)
==========================================================================
                 FIRST-ORDER MODALITY ATTRIBUTION                         
==========================================================================
Question With Image: Alpha_ID: 0.412 | Alpha_TXT: 0.318 | Alpha_IMG: 0.270
Question Text-Only:  Alpha_ID: 0.548 | Alpha_TXT: 0.452 | Alpha_IMG: 0.000
Statistical Test (Mann-Whitney U Test): U=1.45e+09, p-value=1.20e-45 (Significant: True)
--------------------------------------------------------------------------
Prior Solution With Image (tau=t-1): Beta_TXT: 0.582 | Beta_IMG: 0.418
Prior Solution Text-Only  (tau=t-1): Beta_TXT: 1.000 | Beta_IMG: 0.000
==========================================================================
```

---

## 7. Giai Đoạn 4: Trích Xuất Khả Năng Giải Thích (Explainability)

> [!IMPORTANT]
> **Giai đoạn 4 được kích hoạt TỰ ĐỘNG** ngay trong quá trình thực thi lệnh đánh giá của **Giai đoạn 3** (`scripts/evaluate_mg_mmkt.py`). Bạn **không cần** phải chạy thêm một command riêng biệt.
>
> Khi bạn chạy lệnh ở Giai đoạn 3:
> ```bash
> conda run -n kt-research-env python scripts/evaluate_mg_mmkt.py \
>     --checkpoint outputs/checkpoints/mg_mmkt_fold0/best_model.pt \
>     --output-report outputs/reports/eval_fold0_detailed.json
> ```
> Module `FirstOrderExplainer` (`src/evaluation/explainability.py`) sẽ được tự động gọi trong pipeline:
> 1. Tính toán toàn bộ phân phối $\alpha_t^q$ và kiểm định ý nghĩa thống kê Mann-Whitney U.
> 2. Tính toán phân phối $\beta_\tau^s$ cho các tương tác lịch sử có lời giải ($\tau = t - 1$).
> 3. Trích xuất Top-$K$ bằng chứng lịch sử (`sample_evidence`) cho các chuỗi tương tác mẫu.
> 4. Xuất trực tiếp ra Terminal (xem bảng tại Mục 6.2) và ghi đầy đủ vào tệp JSON báo cáo (`outputs/reports/eval_fold0_detailed.json`).

### 7.1 Phân Bổ Trọng Số Cổng (Modality Attribution)
- **Question Gate $\alpha_t^q = (\alpha_{\text{id}}, \alpha_{\text{txt}}, \alpha_{\text{img}})$:** Đo lường mức độ mô hình dựa vào hình ảnh minh họa so với nội dung văn bản và định danh câu hỏi.
- **Kiểm định Mann-Whitney U:** Tự động kiểm tra xem trọng số $\alpha_{\text{img}}$ tại các câu hỏi có ảnh có cao hơn đáng kể (với $p < 0.05$) so với trường hợp khuyết ảnh hay không.
- **Solution Gate $\beta_{\tau}^s = (\beta_{\text{txt}}, \beta_{\text{img}})$ tại $\tau = t - 1$:** Đo lường mức độ ảnh hưởng của hình minh họa trong lời giải của tương tác liền trước đối với trạng thái học tập của học sinh.

### 7.2 Trích Xuất Top-$K$ Bằng Chứng Lịch Sử (Top-K Historical Evidence)
Hàm `extract_top_k_historical_evidence` trong `src/evaluation/explainability.py` trích xuất $K$ tương tác trong quá khứ có trọng số Attention cao nhất ảnh hưởng đến quyết định dự báo tại bước $t$:
```json
{
  "rank": 1,
  "history_step": 14,
  "actual_history_step": 114,
  "target_step": 28,
  "actual_target_step": 128,
  "attention_weight": 0.4125,
  "prior_sink_weight": 0.0842,
  "question_id": 4120,
  "skill_ids": [105, 342],
  "response": 0,
  "has_stem_image": true,
  "has_sol_text": true,
  "has_sol_image": false
}
```
*Lưu ý:* `prior_sink_weight` ghi nhận lượng chú ý đổ về token gốc $y_0 = e_{\text{BOS}}$, phản ánh mức độ tự tin tiên nghiệm của mô hình khi chưa có đủ bằng chứng lịch sử liên quan.

### 7.3 Sử Dụng Độc Lập Qua Python API / Jupyter Notebook
Nếu bạn muốn phân tích sâu case study của một học sinh cụ thể hoặc vẽ biểu đồ trực quan hóa ma trận Attention cho bài báo, bạn có thể gọi trực tiếp `FirstOrderExplainer` như sau:

```python
import torch
from src.evaluation.explainability import FirstOrderExplainer
from src.models.mmkt.mg_mmkt import MGMMKT

# 1. Tải checkpoint mô hình đã huấn luyện
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("outputs/checkpoints/mg_mmkt_fold0/best_model.pt", map_location=device)
model_cfg = checkpoint["model_cfg"]

model = MGMMKT(**model_cfg).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 2. Chạy inference cho 1 batch học sinh
explainer = FirstOrderExplainer()
with torch.no_grad():
    outputs = model(batch)
    attn_weights = outputs["attn_weights"][0].cpu().numpy()  # [L, L] ma trận attention học sinh 0
    alpha_q = outputs["alpha_q"][0].cpu().numpy()            # [L, 3]
    beta_s = outputs["beta_s"][0].cpu().numpy()              # [L, 2]

# 3. Trích xuất Top-3 bằng chứng lịch sử tại bước dự báo t = 20
top_evidence = explainer.extract_top_k_historical_evidence(
    attn_weights=attn_weights,
    step_t=20,
    q_seq=batch["q_seq"][0].cpu().numpy(),
    r_seq=batch["r_seq"][0].cpu().numpy(),
    modality_masks=batch["modality_masks"][0].cpu().numpy(),
    leaf_kcs=batch["leaf_kcs"][0].cpu().numpy(),
    leaf_kc_mask=batch["leaf_kc_mask"][0].cpu().numpy(),
    context_start=int(batch["context_start"][0].item()),
    k=3,
)

print("Top Historical Evidence:", top_evidence)
```

---

## 8. Cấu Trúc Tệp Cấu Hình & Artifacts

### 8.1 Thư Mục Cấu Hình (`configs/`)
- `configs/model/mg_mmkt.yaml`: Định nghĩa kích thước embedding (`d_model: 128`), số head attention (`num_heads: 4`), độ dài chuỗi tối đa (`max_seq_len: 200`), dropout (`dropout: 0.20`), và các cờ loại suy.
- `configs/train.yaml`: Siêu tham số tối ưu hóa (`lr: 1e-3`, `batch_size: 64`, `warmup_epochs: 2`, `patience: 10`, `weight_decay: 1e-4`).
- `configs/dataset/xes3g5m.yaml`: Đường dẫn dữ liệu (`data_dir`), tỉ lệ dropout phương thái (`p_drop: 0.20`), bước nhảy đánh giá (`test_stride: 100`).

### 8.2 Thư Mục Checkpoints & Báo Cáo
- `outputs/checkpoints/mg_mmkt_fold{0..4}/best_model.pt`: Checkpoint trọng số mô hình tốt nhất theo Validation AUC của từng fold.
- `outputs/checkpoints/mg_mmkt_fold{0..4}/train_fold{0..4}.log`: File nhật ký chi tiết quá trình huấn luyện từng epoch của từng fold.
- `outputs/reports/mg_mmkt_5fold_summary.json`: Báo cáo số liệu tổng hợp 5-fold (Mean ± Std của Val AUC, Val ACC, Test AUC, Test ACC).
- `outputs/reports/mg_mmkt_5fold_summary.md`: Bảng Markdown tổng hợp kết quả 5 fold sẵn sàng chèn vào bài báo.

---

## 9. Xử Lý Sự Cố & Tối Ưu Bộ Nhớ (Troubleshooting)

### 9.1 Lỗi Tràn RAM Khi Khởi Tạo Item Bank
- **Nguyên nhân:** Tệp `static_item_bank.pt` đầy đủ có kích thước ~4.6 GB. Khi chạy 5 fold hoặc mở nhiều worker DataLoader, bộ nhớ RAM có thể bị quá tải.
- **Giải pháp:** Hệ thống đã tự động kích hoạt `load_static_item_bank_optimized` (`src/data/xes3g5m_dataset.py`). Hàm này ưu tiên nạp tệp `mg_static_item_bank.pt` (~48 MB - chỉ chứa các trường cần thiết cho MG-MMKT là `v_txt_q`, `v_img_q`, `v_txt_s`, `v_img_s`, `modality_masks`, `question_leaf_kcs`, `question_kc_mask`, `ctt_difficulty`). Giảm 99% dung lượng RAM và tăng tốc độ khởi tạo DataLoader gấp 10 lần.

### 9.2 Lỗi Kích Thước Batch Khi Huấn Luyện Trên GPU Nhỏ (VRAM < 8 GB)
- Nếu gặp lỗi CUDA Out-Of-Memory, giảm mini-batch size xuống 32:
  ```bash
  conda run -n kt-research-env python scripts/train_mg_mmkt.py --batch-size 32
  ```

### 9.3 Đảm Bảo Tính Nhất Quán Về Seed
Mọi script đều gọi hàm `set_seed` (`src/utils/seed.py`) trước khi khởi tạo mô hình và DataLoader:
- Huấn luyện đơn fold: `seed = 42`.
- Huấn luyện 5-fold: `seed = 42 + fold` (đảm bảo mỗi fold có shuffle mẫu ngẫu nhiên độc lập nhưng hoàn toàn tái lập được 100%).

---
*Tài liệu được cập nhật tự động và đồng bộ với phiên bản mã nguồn mới nhất của dự án KT-MMKT Research.*
