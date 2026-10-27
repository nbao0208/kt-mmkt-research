# HƯỚNG DẪN VẬN HÀNH, HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH ACD-MMKT

Tài liệu này hướng dẫn chi tiết cách thiết lập môi trường, cấu hình tham số, thực thi các kịch bản huấn luyện (đơn fold, 5-fold cross-validation, ablation studies) và đánh giá kiểm thử mô hình **ACD-MMKT** trên tập dữ liệu **XES3G5M**.

---

## MỤC LỤC
1. [Môi Trường Thực Thi Bắt Buộc](#1-môi-trường-thực-thi-bắt-buộc)
2. [Cấu Trúc Thư Mục Dữ Liệu & Kết Quả](#2-cấu-trúc-thư-mục-dữ-liệu--kết-quả)
3. [Kiểm Tra Nhanh Tính Toàn Vẹn (Sanity Check)](#3-kiểm-tra-nhanh-tính-toàn-vẹn-sanity-check)
4. [Bảng Tra Cứu & Mô Tả Chi Tiết Tất Cả Tham Số Huấn Luyện](#4-bảng-tra-cứu--mô-tả-chi-tiết-tất-cả-tham-số-huấn-luyện)
   - [4.1 Tham số Dòng lệnh CLI (`scripts/train_acd_mmkt.py`)](#41-tham-số-dòng-lệnh-cli-scriptstrain_acd_mmktpy)
   - [4.2 Tham số Cấu hình Huấn luyện (`configs/train_acd_mmkt.yaml`)](#42-tham-số-cấu-hình-huấn-luyện-configstrain_acd_mmktyaml)
   - [4.3 Tham số Cấu trúc Mô hình (`configs/model/acd_mmkt.yaml`)](#43-tham-số-cấu-trúc-mô-hình-configsmodelacd_mmktyaml)
   - [4.4 Các Cấu hình Phần cứng Định nghĩa sẵn (Hardware Profiles)](#44-các-cấu-hình-phần-cứng-định-nghĩa-sẵn-hardware-profiles)
5. [Các Kịch Bản Huấn Luyện Thực Tế](#5-các-kịch-bản-huấn-luyện-thực-tế)
   - [5.1 Huấn luyện Đơn Fold (Single Fold Training)](#51-huấn-luyện-đơn-fold-single-fold-training)
   - [5.2 Huấn luyện Toàn diện 5-Fold Cross-Validation](#52-huấn-luyện-toàn-diện-5-fold-cross-validation)
   - [5.3 Huấn luyện 4 Cấu hình Cắt bỏ (Ablation Studies)](#53-huấn-luyện-4-cấu-hình-cắt-bỏ-ablation-studies)
   - [5.4 Kịch bản Chạy Tự Động Qua Đêm (`train_acd_overnight.sh`)](#54-kịch-bản-chạy-tự-động-qua-đêm-train_acd_overnightsh)
6. [Đánh Giá Mô Hình & Xuất Báo Cáo Giải Thích](#6-đánh-giá-mô-hình--xuất-báo-cáo-giải-thích)
   - [6.1 Tham số Dòng lệnh Đánh giá (`scripts/evaluate_acd_mmkt.py`)](#61-tham-số-dòng-lệnh-đánh-giá-scriptsevaluate_acd_mmktpy)
   - [6.2 Lệnh Đánh giá & Xuất Báo cáo](#62-lệnh-đánh-giá--xuất-báo-cáo)
   - [6.3 Các Kết quả & Báo cáo Thu được](#63-các-kết-quả--báo-cáo-thu-được)
7. [Xử Lý Sự Cố & Tối Ưu Tốc Độ / Bộ Nhớ](#7-xử-lý-sự-cố--tối-ưu-tốc-độ--bộ-nhớ)
8. [Tóm Tắt Các Lệnh Phổ Biến (Quick Cheatsheet)](#8-tóm-tắt-các-lệnh-phổ-biến-quick-cheatsheet)

---

## 1. Môi Trường Thực Thi Bắt Buộc

Theo quy chuẩn của dự án, mọi thao tác chạy mã nguồn **bắt buộc** phải sử dụng môi trường Conda `kt-research-env`. Tuyệt đối không dùng Python mặc định của hệ điều hành.

- **Kích hoạt môi trường trong Terminal:**
  ```bash
  conda activate kt-research-env
  ```
- **Chạy lệnh qua `conda run`:**
  ```bash
  conda run -n kt-research-env python <đường_dẫn_script> [tham_số]
  ```
- **Hoặc dùng đường dẫn binary trực tiếp:**
  ```bash
  /opt/miniconda3/envs/kt-research-env/bin/python <đường_dẫn_script> [tham_số]
  ```

**Kiểm tra phần cứng nhận diện:**
```bash
conda run -n kt-research-env python -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available: ', torch.cuda.is_available(), '| Count:', torch.cuda.device_count())
print('MPS Available:  ', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
"
```

---

## 2. Cấu Trúc Thư Mục Dữ Liệu & Kết Quả

Hệ thống yêu cầu các tệp dữ liệu đã xử lý nằm sẵn trong `data/processed/XES3G5M/`:

```
kt-mmkt-research/
├── data/
│   └── processed/
│       └── XES3G5M/
│           ├── static_item_bank.pt       # Ngân hàng câu hỏi tĩnh (văn bản, ảnh, KCs, CTT)
│           ├── sequences_fold_0.pt       # Chuỗi tương tác học sinh Fold 0
│           ├── sequences_fold_1.pt       # Chuỗi tương tác học sinh Fold 1
│           ├── sequences_fold_2.pt       # Chuỗi tương tác học sinh Fold 2
│           ├── sequences_fold_3.pt       # Chuỗi tương tác học sinh Fold 3
│           ├── sequences_fold_4.pt       # Chuỗi tương tác học sinh Fold 4
│           └── sequences_test.pt         # Tập Test độc lập (1.099.197 tương tác)
├── outputs/
│   ├── checkpoints/                      # Nơi lưu file trọng số mô hình tốt nhất (.pt)
│   ├── logs/                             # Nhật ký huấn luyện dạng text / console log
│   └── reports/                          # Báo cáo kết quả JSON và Markdown giải thích
```

---

## 3. Kiểm Tra Nhanh Tính Toàn Vẹn (Sanity Check)

Trước khi chạy huấn luyện dài hạn, chạy bộ unit test tự động (9 bài test) để đảm bảo mô hình hoạt động ổn định:

```bash
/opt/miniconda3/envs/kt-research-env/bin/python -m pytest tests/test_acd_mmkt_model.py -v
```

Nếu toàn bộ 9 bài test đều báo `PASSED` trong khoảng 3–5 giây, hệ thống đã sẵn sàng cho quá trình huấn luyện.

---

## 4. Bảng Tra Cứu & Mô Tả Chi Tiết Tất Cả Tham Số Huấn Luyện

Mô hình ACD-MMKT huấn luyện theo **chiến lược 2 giai đoạn (Two-Stage Curriculum Learning)**:
- **Giai đoạn 1 (Stage 1):** Tiền huấn luyện ngân hàng câu hỏi trên toàn bộ 7.652 bài tập (không phụ thuộc chuỗi thời gian).
- **Giai đoạn 2 (Stage 2):** Huấn luyện chuỗi tương tác tuần tự của học sinh với cơ chế nhớ/quên và dự đoán xác suất làm đúng.

### 4.1 Tham số Dòng lệnh CLI (`scripts/train_acd_mmkt.py`)

| Tên tham số CLI | Kiểu dữ liệu | Giá trị mặc định | Diễn giải chi tiết chức năng & tác động |
| :--- | :---: | :---: | :--- |
| `--val-fold` | `int` | `0` | Chỉ định Fold làm tập kiểm định ($0, 1, 2, 3$ hoặc $4$). 4 folds còn lại sẽ tự động gộp thành tập huấn luyện. |
| `--num-epochs` | `int` | `50` | Tổng số epoch huấn luyện cho Giai đoạn 2. Thường chạy 30–50 epoch là hội tụ. |
| `--batch-size` | `int` | `64` | Kích thước mini-batch cho Giai đoạn 2 (số chuỗi học sinh trong một batch). |
| `--lr` | `float` | `0.001` | Tốc độ học (Learning Rate) cho bộ tối ưu AdamW ở Giai đoạn 2. |
| `--seed` | `int` | `42` | Hạt giống ngẫu nhiên để cố định kết quả tái lập (mỗi fold sẽ dùng `seed + fold`). |
| `--device` | `str` | `auto` | Thiết bị chạy: `'mps'` (Mac Silicon), `'cuda'` / `'cuda:0'` / `'cuda:1'` (NVIDIA GPU), hoặc `'cpu'`. |
| `--profile` | `str` | `None` | Nạp sẵn cấu hình tối ưu phần cứng (`'m1_pro'`, `'dual_gpu_5060ti'`, `'dual_gpu_3060'`). Khi dùng profile, các giá trị batch size/precision trong profile sẽ tự động được áp dụng. |
| `--precision` | `str` | `fp32` | Độ chính xác dấu phẩy động: `'fp32'` (khuyên dùng cho Mac MPS), `'amp_fp16'` (bật mixed precision trên NVIDIA GPU), `'amp_bf16'`. |
| `--grad-accum-steps` | `int` | `1` | Số bước tích lũy đạo hàm trước khi cập nhật trọng số. Kích thước batch hiệu dụng = `batch_size` $\times$ `grad_accum_steps`. |
| `--num-workers` | `int` | `0` | Số tiến trình nạp dữ liệu song song của DataLoader (trên Mac để `0`, trên Linux/GPU có thể để `4`). |
| `--save-dir` | `str` | `outputs/checkpoints/acd_mmkt_fold0` | Thư mục lưu tệp checkpoint tốt nhất `best_model.pt`. |
| `--skip-stage1` | `flag` | `False` | Bỏ qua Giai đoạn 1 nếu đã có checkpoint ngân hàng đề hoặc chỉ muốn fine-tune trực tiếp. |
| `--stage1-epochs` | `int` | `15` | Số epoch huấn luyện Giai đoạn 1 (thường chỉ cần 10–15 epoch, chạy rất nhanh khoảng 1 phút). |
| `--stage1-lr` | `float` | `0.001` | Tốc độ học cho bộ tối ưu ở Giai đoạn 1. |
| `--ablation` | `int` | `None` | Chọn nhanh cấu hình nghiên cứu loại suy ($1$: Full, $2$: w/o Distillation, $3$: w/o Fine-Grained, $4$: w/o HGNN/Rasch, $5$: w/o DeepIRT). |
| `--wo-distillation` | `flag` | `False` | Cờ tắt nhánh chưng cất độ phức tạp lời giải (Ablation 2). |
| `--wo-fine-grained` | `flag` | `False` | Cờ thay thế attention từ khóa/vùng ảnh bằng vector gộp trung bình toàn cục (Ablation 3). |
| `--wo-hypergraph-rasch` | `flag` | `False` | Cờ tắt mạng siêu đồ thị và phân tách độ khó Rasch (Ablation 4). |
| `--wo-deep-irt` | `flag` | `False` | Cờ thay tầng đọc ra IRT bằng mạng MLP hộp đen thông thường (Ablation 5). |
| `--config-model` | `str` | `configs/model/acd_mmkt.yaml` | Đường dẫn tệp YAML cấu hình cấu trúc mô hình. |
| `--config-train` | `str` | `configs/train_acd_mmkt.yaml` | Đường dẫn tệp YAML cấu hình tối ưu hóa và huấn luyện. |
| `--config-dataset` | `str` | `configs/dataset/xes3g5m.yaml`| Đường dẫn tệp YAML cấu hình thông số tập dữ liệu. |

---

### 4.2 Tham số Cấu hình Huấn luyện (`configs/train_acd_mmkt.yaml`)

Tệp này quản lý các thiết lập mặc định cho quá trình huấn luyện:

```yaml
val_fold: 0                             # Fold xác thực mặc định
seed: 42                                # Seed ngẫu nhiên
data_dir: "data/processed/XES3G5M"      # Thư mục chứa dữ liệu
save_dir: "outputs/checkpoints/acd_mmkt_fold0" # Nơi lưu checkpoint

# Cấu hình Giai đoạn 1 (Item-Bank Pre-training)
stage1_epochs: 15                       # Số epoch pre-train ngân hàng câu hỏi
stage1_batch_size: 256                  # Kích thước lô câu hỏi cho Stage 1
stage1_lr: 0.001                        # Tốc độ học Stage 1

# Cấu hình Giai đoạn 2 (Sequential Interaction Fine-tuning)
num_epochs: 50                          # Tổng số epoch cho Stage 2
lr: 0.001                               # Tốc độ học cơ sở
weight_decay: 0.0001                    # Hệ số suy giảm trọng số L2
max_grad_norm: 1.0                      # Cắt tỉa đạo hàm chống bùng nổ gradient
patience: 10                            # Số epoch chờ nếu Val AUC không tăng trước khi Early Stop
warmup_epochs: 2                        # Số epoch khởi động tốc độ học từ 0.1x đến 1.0x
eta_min: 0.00001                        # Tốc độ học tối thiểu khi dùng Cosine Annealing decay
p_drop: 0.20                            # Tỷ lệ ngẫu nhiên rơi rớt lời giải (Modality Dropout)
test_stride: 100                        # Bước nhảy cửa sổ trượt khi đánh giá tập test
```

---

### 4.3 Tham số Cấu trúc Mô hình (`configs/model/acd_mmkt.yaml`)

Tệp này định nghĩa kiến trúc mạng và các hệ số hàm mất mát đa nhiệm:

```yaml
model_name: "acd_mmkt"
num_questions: 7652                     # Tổng số câu hỏi trong ngân hàng đề
num_kcs: 1175                           # Tổng số kỹ năng/khái niệm kiến thức (KCs)
d_raw_txt: 768                          # Chiều embedding văn bản thô (Chinese-RoBERTa)
d_raw_img: 768                          # Chiều embedding ảnh thô (CLIP-ViT)
d_ctx: 21                               # Chiều vector ngữ cảnh (15 tĩnh + 4 động + 2 cờ lời giải)
d_model: 128                            # Chiều không gian biểu diễn ẩn chung của mô hình
num_heads: 4                            # Số đầu chú ý (Attention Heads, mỗi đầu 32 chiều)
max_seq_len: 200                        # Chiều dài chuỗi tối đa trong một cửa sổ trượt
dropout: 0.20                           # Tỷ lệ dropout ngẫu nhiên trong mạng
hgnn_layers: 2                          # Số lớp tích chập siêu đồ thị giữa các kỹ năng
tau_temp: 0.10                          # Nhiệt độ chuẩn hóa cho mạng chưng cất lời giải
beta_distill: 0.10                      # Trọng số hàm mất mát chưng cất độ phức tạp lời giải
beta_ctt: 0.20                          # Trọng số hàm mất mát hiệu chuẩn độ khó thực nghiệm CTT
beta_disen: 0.05                        # Trọng số hàm mất mát trực giao hóa vector khái niệm & biến thiên
lambda_res: 0.001                       # Trọng số điều chuẩn nhánh phần dư DeepIRT
irt_scale: 4.0                          # Hệ số tỷ lệ logit tâm trắc học (kappa)

# Các cờ bật/tắt thành phần mô hình (True = đầy đủ mô hình)
use_solution: true                      # Có tích hợp lời giải vào luồng tiếp thu tri thức
use_visual: true                        # Có sử dụng thông tin hình ảnh
use_distillation: true                  # (Ablation 2: đổi thành false để tắt chưng cất lời giải)
use_fine_grained: true                  # (Ablation 3: đổi thành false để dùng vector gộp toàn cục)
use_hypergraph_rasch: true              # (Ablation 4: đổi thành false để tắt siêu đồ thị và Rasch)
use_deep_irt: true                      # (Ablation 5: đổi thành false để thay DeepIRT bằng MLP hộp đen)
```

---

### 4.4 Các Cấu hình Phần cứng Định nghĩa sẵn (Hardware Profiles)

Bạn có thể truyền trực tiếp cờ `--profile <tên_profile>` vào lệnh chạy để nạp nhanh cấu hình phần cứng tối ưu:

| Tên Profile | Thiết bị (`device`) | Độ chính xác (`precision`) | Batch Size | Grad Accum | Workers | Máy phù hợp |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `m1_pro` | `mps` | `fp32` | `32` | `2` | `0` | MacBook Pro M1/M2/M3 (16GB RAM) |
| `dual_gpu_5060ti` | `cuda:0` | `amp_fp16` | `128` | `1` | `4` | NVIDIA GPU $\ge 16\text{ GB}$ (RTX 5060 Ti, RTX 4080, A100) |
| `dual_gpu_3060` | `cuda:1` | `amp_fp16` | `64` | `2` | `4` | NVIDIA GPU $12\text{ GB}$ (RTX 3060, RTX 4070) |
| `kaggle_t4` | `cuda:0` | `amp_fp16` | `64` | `2` | `2` | Kaggle 2x NVIDIA Tesla T4 (16GB VRAM) |

---

## 5. Các Kịch Bản Huấn Luyện Thực Tế

### 5.1 Huấn luyện Đơn Fold (Single Fold Training)

#### Kịch bản A: Chạy trên MacBook M1 Pro (Profile `m1_pro`)
Lệnh chạy huấn luyện Fold 0 (15 epoch Giai đoạn 1 + 30 epoch Giai đoạn 2):
```bash
conda run -n kt-research-env python scripts/train_acd_mmkt.py \
    --val-fold 0 \
    --profile m1_pro \
    --stage1-epochs 15 \
    --num-epochs 30 \
    --save-dir outputs/checkpoints/acd_mmkt_fold0
```

#### Kịch bản B: Chạy trên Card NVIDIA GPU (Profile `dual_gpu_5060ti`)
```bash
conda run -n kt-research-env python scripts/train_acd_mmkt.py \
    --val-fold 0 \
    --profile dual_gpu_5060ti \
    --stage1-epochs 15 \
    --num-epochs 30 \
    --save-dir outputs/checkpoints/acd_mmkt_fold0
```

#### Kịch bản C: Tùy biến tham số trực tiếp (không dùng profile)
Nếu muốn tự chỉnh batch size, learning rate hoặc epoch theo ý muốn:
```bash
conda run -n kt-research-env python scripts/train_acd_mmkt.py \
    --val-fold 1 \
    --device cuda:0 \
    --precision amp_fp16 \
    --batch-size 64 \
    --grad-accum-steps 2 \
    --lr 0.0005 \
    --num-epochs 40 \
    --stage1-epochs 10 \
    --save-dir outputs/checkpoints/acd_mmkt_fold1
```

---

### 5.2 Huấn luyện Toàn diện 5-Fold Cross-Validation

Chạy script `scripts/run_5fold_acd_mmkt.py` để thực hiện giao thức kiểm định chéo 5 fold:

#### Cách 1: Chạy tuần tự trên 1 máy / 1 GPU
```bash
conda run -n kt-research-env python scripts/run_5fold_acd_mmkt.py \
    --folds 0 1 2 3 4 \
    --device mps \
    --batch-size 32 \
    --grad-accum-steps 2 \
    --num-epochs 30 \
    --checkpoint-dir outputs/checkpoints \
    --output-report-dir outputs/reports/acd_mmkt_full
```

#### Cách 2: Chạy song song bất đồng bộ trên 2 GPU NVIDIA
Tự động giao Fold 0, 1, 2 cho GPU 0 (`batch_size=128`) và giao Fold 3, 4 cho GPU 1 (`batch_size=64, grad_accum=2`), không lo nghẽn DDP:
```bash
conda run -n kt-research-env python scripts/run_5fold_acd_mmkt.py \
    --dual-gpu \
    --num-epochs 30 \
    --checkpoint-dir outputs/checkpoints \
    --output-report-dir outputs/reports/acd_mmkt_full
```

#### Cách 3: Chỉ tổng hợp kết quả khi đã có sẵn 5 checkpoint (`--eval-only`)
Nếu bạn đã huấn luyện xong các fold và chỉ muốn tổng hợp bảng kết quả tính Mean ± Std:
```bash
conda run -n kt-research-env python scripts/run_5fold_acd_mmkt.py \
    --eval-only \
    --folds 0 1 2 3 4 \
    --checkpoint-dir outputs/checkpoints \
    --output-report-dir outputs/reports/acd_mmkt_full
```

Tệp báo cáo tổng hợp sẽ được lưu vào:
- `outputs/reports/acd_mmkt_full/5fold_acd_mmkt_summary.md` (dạng bảng Markdown)
- `outputs/reports/acd_mmkt_full/5fold_acd_mmkt_summary.json` (dạng dữ liệu JSON)

---

### 5.3 Huấn luyện 4 Cấu hình Cắt bỏ (Ablation Studies)

Để so sánh và đánh giá tầm quan trọng của từng bộ phận kiến trúc, chạy 4 cấu hình cắt bỏ trên cùng Fold 0 bằng tham số `--ablation <số>`:

```bash
# Cấu hình 2: w/o Solution Distillation (Tắt chưng cất độ phức tạp lời giải)
conda run -n kt-research-env python scripts/train_acd_mmkt.py \
    --val-fold 0 --ablation 2 --num-epochs 30 \
    --save-dir outputs/checkpoints/acd_ablation2_fold0

# Cấu hình 3: w/o Fine-Grained Alignment (Dùng vector gộp toàn cục thay vì Attention hạt mịn)
conda run -n kt-research-env python scripts/train_acd_mmkt.py \
    --val-fold 0 --ablation 3 --num-epochs 30 \
    --save-dir outputs/checkpoints/acd_ablation3_fold0

# Cấu hình 4: w/o Hypergraph & Rasch (Tắt mạng siêu đồ thị HGNN và phân tách độ khó Rasch)
conda run -n kt-research-env python scripts/train_acd_mmkt.py \
    --val-fold 0 --ablation 4 --num-epochs 30 \
    --save-dir outputs/checkpoints/acd_ablation4_fold0

# Cấu hình 5: w/o DeepIRT Readout (Dùng mạng MLP hộp đen thay vì công thức tâm trắc học IRT)
conda run -n kt-research-env python scripts/train_acd_mmkt.py \
    --val-fold 0 --ablation 5 --num-epochs 30 \
    --save-dir outputs/checkpoints/acd_ablation5_fold0
```

---

### 5.4 Kịch bản Chạy Tự Động Qua Đêm (`train_acd_overnight.sh`)

Script shell `scripts/train_acd_overnight.sh` tự động hoá hoàn toàn 5 bước thực nghiệm lớn:
1. Huấn luyện mô hình Full ACD-MMKT trên Fold 0 (Stage 1 + Stage 2).
2. Đánh giá chi tiết tập Test và xuất báo cáo giải thích 3 cấp độ.
3. Huấn luyện các Fold 1, 2, 3, 4 còn lại (tự động nhận lại checkpoint Fold 0 từ Bước 1, không chạy lại gây phí thời gian) và tổng hợp kết quả 5-Fold CV.
4. Huấn luyện tuần tự cả 4 biến thể Ablation 2, 3, 4, 5 trên Fold 0.
5. Đánh giá kiểm thử độc lập cả 4 biến thể Ablation và xuất báo cáo đối sánh.

**Lệnh khởi chạy chạy ngầm không bị ngắt khi đóng terminal:**
```bash
nohup bash scripts/train_acd_overnight.sh > overnight_acd.log 2>&1 &
```

**Xem tiến trình chạy trực tiếp:**
```bash
tail -f overnight_acd.log
```

---

## 6. Đánh Giá Mô Hình & Xuất Báo Cáo Giải Thích

### 6.1 Tham số Dòng lệnh Đánh giá (`scripts/evaluate_acd_mmkt.py`)

| Tham số CLI | Kiểu dữ liệu | Mặc định | Chức năng chi tiết |
| :--- | :---: | :---: | :--- |
| `--checkpoint` | `str` | **Bắt buộc** | Đường dẫn tới tệp checkpoint mô hình cần đánh giá (`best_model.pt`). |
| `--val-fold` | `int` | `0` | Chỉ số Fold tương ứng để nạp đúng nhãn độ khó CTT không bị rò rỉ. |
| `--batch-size` | `int` | `128` | Kích thước batch khi duyệt qua tập Test. |
| `--test-stride` | `int` | `100` | Bước nhảy cửa sổ trượt bảo toàn ngữ cảnh ($S = 100$). |
| `--device` | `str` | `auto` | Thiết bị tính toán (`'cuda'`, `'mps'`, `'cpu'`). |
| `--output-report` | `str` | `None` | Tùy chọn đường dẫn tệp JSON để lưu kết quả đánh giá số liệu. |
| `--explain-output-md`| `str` | `None` | Tùy chọn đường dẫn tệp Markdown để xuất báo cáo giải thích sư phạm mẫu. |

---

### 6.2 Lệnh Đánh giá & Xuất Báo cáo

Đánh giá checkpoint Fold 0 trên toàn bộ **1.099.197 tương tác hợp lệ** của tập Test độc lập:

```bash
conda run -n kt-research-env python scripts/evaluate_acd_mmkt.py \
    --checkpoint outputs/checkpoints/acd_mmkt_fold0/best_model.pt \
    --val-fold 0 \
    --batch-size 128 \
    --output-report outputs/reports/acd_mmkt_full/eval_fold0_detailed.json \
    --explain-output-md outputs/reports/acd_mmkt_full/sample_explanation.md
```

---

### 6.3 Các Kết quả & Báo cáo Thu được

Sau khi chạy xong, terminal sẽ in bảng tổng kết chi tiết gồm:
1. **Hiệu năng tổng thể trên tập Test (Global Test Metrics):**
   - Micro AUC và Accuracy (ACC).
2. **Đánh giá phân tầng (Stratified Metrics):**
   - **Ảnh vs Chữ:** Tách riêng nhóm câu hỏi có hình ảnh (`visual_questions`) so với câu hỏi chỉ có văn bản (`text_only_questions`).
   - **Ảnh Lời giải:** Tách riêng nhóm học sinh vừa xem ảnh lời giải ở bước trước ($t-1$) so với chỉ xem chữ lời giải.
   - **Khởi đầu lạnh:** Tách riêng nhóm câu hỏi hiếm gặp ($<10$ lần xuất hiện trong train) so với câu hỏi thường gặp.
3. **Độ tương quan tâm trắc học Rasch:**
   - Hệ số tương quan Pearson $r$ và Spearman $\rho$ giữa độ khó mô hình ước lượng và tỷ lệ làm sai thực tế của học sinh.
4. **Báo cáo giải thích 3 cấp độ (`sample_explanation.md`):**
   - **Cấp độ 1:** Chỉ số năng lực học sinh, độ khó đề bài, độ phân biệt câu hỏi và khoảng cách $\Delta_{\text{psy}}$.
   - **Cấp độ 2:** Bản đồ chú ý chéo từ khóa — vùng ảnh patch $7 \times 7$ (đã lọc sạch token padding `[PAD]`), trọng số chú ý lịch sử làm bài.
   - **Cấp độ 3:** Lời phê sư phạm bằng ngôn ngữ tự nhiên: đánh giá mức độ thành thạo, phân tích nguyên nhân và khuyến nghị bài tập tiếp theo.

---

## 7. Xử Lý Sự Cố & Tối Ưu Tốc Độ / Bộ Nhớ

### 7.1 Lỗi Tràn Bộ Nhớ GPU (`CUDA out of memory` hoặc `MPS out of memory`)
- **Nguyên nhân:** Kích thước mini-batch quá lớn hoặc chuỗi học sinh dài kèm nhiều hình ảnh.
- **Cách khắc phục:**
  - Giảm `--batch-size` xuống một nửa (ví dụ từ `128` xuống `64`, hoặc từ `64` xuống `32`).
  - Tăng `--grad-accum-steps` lên gấp đôi (ví dụ từ `1` lên `2`, hoặc từ `2` lên `4`) để giữ nguyên kích thước batch hiệu dụng.

### 7.2 Lỗi NaN hoặc Mất Ổn Định Số Học Trên Apple Silicon (`mps`)
- **Nguyên nhân:** Backend `mps` của macOS đôi khi gặp hiện tượng underflow khi tính toán hàm softmax hoặc LayerNorm ở chế độ nửa độ chính xác `fp16`.
- **Cách khắc phục:** Luôn giữ `--precision fp32` khi chạy trên thiết bị Mac (`device=mps`). Tốc độ tính toán trên M1/M2/M3 với `fp32` vẫn rất nhanh và đảm bảo độ chính xác tuyệt đối.

### 7.3 Thiết Lập Số Lượng DataLoader Workers (`--num-workers`)
- **Trên macOS:** Khuyến nghị đặt `--num-workers 0` để tránh xung đột tài nguyên chia sẻ giữa các tiến trình con.
- **Trên Linux / GPU Server:** Nên đặt `--num-workers 4` để nạp dữ liệu song song từ ổ đĩa SSD lên GPU nhanh hơn, tránh để GPU bị rảnh chờ dữ liệu.

### 7.4 Tách Nhóm Trọng Số và Tránh Co Rút Vector Khuyết Thiếu
Bộ tối ưu hóa của ACD-MMKT đã được tự động chia thành 3 nhóm tham số:
- **Nhóm 1 (Trọng số ma trận thông thường):** Áp dụng `weight_decay = 0.0001`.
- **Nhóm 2 (Nhánh phần dư DeepIRT):** Áp dụng `weight_decay = 0.001` (lớn gấp 10 lần) để bảo toàn tính chi phối của công thức tâm trắc học.
- **Nhóm 3 (Vector bù khuyết $E_{\text{null}}$, token BOS, độ lệch bias, LayerNorm):** Áp dụng `weight_decay = 0.0` để các vector này không bị co rút về 0 khi huấn luyện lâu dài.

---

## 8. Tóm Tắt Các Lệnh Phổ Biến (Quick Cheatsheet)

```bash
# 1. Chạy bộ kiểm thử tự động 9 tests (Mất khoảng 4 giây)
/opt/miniconda3/envs/kt-research-env/bin/python -m pytest tests/test_acd_mmkt_model.py -v

# 2. Huấn luyện nhanh 1 fold trên Mac M1/M2/M3
conda run -n kt-research-env python scripts/train_acd_mmkt.py \
    --val-fold 0 --profile m1_pro --num-epochs 30

# 3. Huấn luyện 1 fold trên GPU NVIDIA (RTX / A100)
conda run -n kt-research-env python scripts/train_acd_mmkt.py \
    --val-fold 0 --profile dual_gpu_5060ti --num-epochs 30

# 4. Đánh giá checkpoint tập test và xuất báo cáo giải thích Markdown
conda run -n kt-research-env python scripts/evaluate_acd_mmkt.py \
    --checkpoint outputs/checkpoints/acd_mmkt_fold0/best_model.pt \
    --val-fold 0 \
    --output-report outputs/reports/eval_fold0.json \
    --explain-output-md outputs/reports/sample_explanation.md

# 5. Chạy 5-Fold Cross Validation song song trên 2 GPU
conda run -n kt-research-env python scripts/run_5fold_acd_mmkt.py \
    --dual-gpu --num-epochs 30

# 6. Khởi chạy toàn bộ chuỗi thực nghiệm tự động qua đêm (Full 5 Folds + 4 Ablations)
nohup bash scripts/train_acd_overnight.sh > overnight_acd.log 2>&1 &
```
