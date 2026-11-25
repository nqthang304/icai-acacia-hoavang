# Tái hiện ICAI 2026 – Lập bản đồ rừng keo và sàng lọc khu vực khảo sát, xã Hòa Vang

Source code dựng lại các thực nghiệm của bài báo *"Mapping Acacia Plantation Land-Cover States and Prioritizing
Field-Assessment Sites for Large-Timber Development Using Sentinel-1/2 Time Series in Hoa Vang Commune, Da Nang"*
(ICAI 2026), giữ nguyên đầu vào/đầu ra và giao thức đánh giá. Code bổ sung các đầu ra phân tích để trả lời phản biện:
ma trận nhầm lẫn, bản đồ dự đoán, phân tách theo nguồn nhãn và pixel lấp, đánh giá trên tập mảnh chung, mIoU riêng trên
các quý bị xóa quang học, bảng ngân sách huấn luyện.

Mọi chỗ tài liệu không ghi rõ được liệt kê trong **[ASSUMPTIONS.md](ASSUMPTIONS.md)**.

## Cấu trúc

```
configs/
  base.yaml                 tham số chung (gắn nguồn [paper]/[thesis]/[data]/[ASSUMPTION])
  split_blocks.json         gán khối 3x3 -> train/val/test (khớp mọi mốc của bài báo)
  models/*.yaml             unet, utae, tsvit, quast, plain_tatt
  experiments/*.yaml        main_test, context_sweep, capacity, short_window, modality,
                            quast_ablation, buffer, optical_dropout, smoke
src/acacia/
  data/       cache.py (GeoTIFF -> npy), split.py, normalize.py, datasets.py, context.py
  models/     unet.py, temporal.py (PlainTAtt, U-TAE, TSViT), quast.py, blocks.py
  train/      losses.py (WCE + Dice, monotonicity), trainer.py
  eval/       predict.py, evaluator.py, metrics.py, optical_dropout.py, stats.py
  polygon/    classify.py (RF, SVM-RBF, XGBoost, QuaST encoder + MLP)
  screening/  screen.py (chỉ báo G/T/D/V, điểm, nhóm, độ nhạy trọng số)
  report/     build.py (bảng + hình)
scripts/      00_package_kaggle … 08_report
tests/        test_anchors.py (mốc kiểm chứng)
notebooks/    kaggle_run.ipynb
```

## Quy trình

| Bước | Lệnh | Kết quả |
|---|---|---|
| 1 | `python scripts/01_build_cache.py` | `work/cache/` (X.npy 2,1 GB, Y.npy, dem.npy) |
| 2 | `python scripts/02_prepare.py` | `work/prepared/` chuẩn hóa, mốc kiểm chứng, bảng lớp theo tập |
| – | `pytest -q tests` | kiểm tra mốc |
| 3 | `python scripts/03_run.py <thực nghiệm>` | `work/runs/<thực nghiệm>/<cấu hình>/s<seed>/` |
| 4 | `python scripts/04_optical_dropout.py [--extra]` | Bảng IV |
| 5 | `python scripts/05_stats.py` | kiểm định cặp, tập mảnh chung |
| 6 | `python scripts/06_polygon.py` | Bảng V |
| 7 | `python scripts/07_screening.py` | Bảng VI, Hình 8–9 |
| 8 | `python scripts/08_report.py` | `work/results/report/REPORT.md` + CSV + PNG |

Thứ tự thực nghiệm ở bước 3: `smoke` (chạy thử) → `main_test` → `context_sweep` → `modality` → `quast_ablation` →
`short_window` → `buffer` → `capacity`. Bước 4 cần `main_test` (seed 101) và `modality/s1s2_q`.

Ghi đè config từ dòng lệnh: `--set train.max_epochs=50 loss.dice_mask_ignore=true`.
Lần chạy đã xong (có `summary.json`) được bỏ qua, nên chạy lại cùng lệnh sẽ tiếp tục từ chỗ dừng.

## Chạy trên Kaggle

1. Trên máy: `python scripts/01_build_cache.py` rồi `python scripts/00_package_kaggle.py`, tạo ra
   `kaggle_upload/acacia-hoavang-code.zip` và `kaggle_upload/acacia-hoavang-data.zip` (khoảng 2,4 GB).
2. Upload mỗi file thành một Kaggle Dataset, đặt tên `acacia-hoavang-code` và `acacia-hoavang-data`.
3. Import `notebooks/kaggle_run.ipynb`, gắn hai dataset, chọn GPU T4, bật Internet, rồi chạy từng cell.

Biến môi trường: `ACACIA_DATA` (thư mục data), `ACACIA_CACHE` (thư mục cache), `ACACIA_WORK` (thư mục ghi kết quả).

## Bản đối chiếu sửa lỗi Dice

Mặc định tái hiện đúng hành vi gốc: pixel không nhãn vẫn nằm trong mẫu số Dice. Muốn chạy bản đã sửa (phản biện #25)
mà không đè lên kết quả gốc, đặt thư mục làm việc khác:
`ACACIA_WORK=work_dicefix python scripts/03_run.py main_test --set loss.dice_mask_ignore=true`.
