# Những điểm đã kiểm chứng và những giả định

Bài báo ICAI 2026 và báo cáo tốt nghiệp không công bố source code, file chia tập hay đầy đủ cấu hình.
Tài liệu này ghi lại (1) những gì code khớp với số liệu đã công bố, và (2) mọi chỗ phải tự chọn.
Trong `configs/*.yaml`, mỗi tham số được gắn nhãn `[paper]`, `[thesis]`, `[data]` hoặc `[ASSUMPTION]`.

## 1. Mốc đã kiểm chứng (tự động trong `tests/test_anchors.py`)

| Mốc | Bài báo | Code | Ghi chú |
|---|---|---|---|
| Pixel-quý có nhãn hợp lệ | 11.891.184 / 38.928.384 (30,55%) | khớp chính xác | dữ liệu đúng phiên bản bài báo dùng |
| Số polygon | 521 | 521 | 186 gốc + 335 vẽ thêm |
| Mảnh có nhãn thủ công | 201 / 297 | 201 | |
| Số mảnh train / val / test | 186 / 54 / 57 | khớp | |
| Mảnh "đủ chuỗi 32 quý" train / val / test | 170 / 51 / 43 | khớp | tiêu chí = mảnh có ≥ 1 quý có nhãn (phản biện #21) |
| Polygon vắt ranh giới các tập | 60 | 64 | mốc duy nhất không khớp tuyệt đối |
| Cặp mảnh kề nằm ở hai tập | 205 | khớp | kề 8 hướng |
| Train đủ điều kiện sau buffer 640 m | 143 | 143 | buffer quanh tập val |
| Mảnh test chung cho kiểm định cặp | 34 | 34 | = mảnh test còn pixel hợp lệ sau khi che polygon vắt ranh |
| Buffer 1.280 m: không còn Water trong train | có | có | |
| KS trung bình 2018-2019 (21 kênh quang học) | 0,410 | 0,404 | khác nhỏ do lấy mẫu pixel |
| Số tham số U-Net base-16 / 12 / 9 | 1,95 / 1,10 / 0,62 M | 1,946 / 1,096 / 0,618 M | U-Net chuẩn 5 tầng, conv không bias + BN |
| QuaST / TSViT / U-TAE / PlainTAtt | 1,16 / 1,03 / 0,63 / 0,63 M | 1,160 / 1,026 / 0,623 / 0,641 M | độ rộng chọn để khớp số tham số |
| Polygon từng là Keo cho sàng lọc | 105 | 105 | với ngưỡng ≥ 25 pixel (giả định A15) |

## 2. Giả định

### Dữ liệu và chia tập
- **A1. Cách gán khối 3×3 vào train/val/test** (`configs/split_blocks.json`). Tài liệu không công bố. File được tìm
  bằng simulated annealing (`acacia.data.split.search_split`, `scripts/02_prepare.py --search-split`) để khớp
  **đồng thời** mọi mốc ở mục 1, cộng ba ràng buộc suy ra từ bài báo: tập mảnh chung có 34 mảnh, mọi lớp đều có mặt ở tập test (Bảng II có IoU
  lớp Water trên test) và buffer 1.280 m làm mất hết pixel Water trong train (mục V.A). Có thể có nhiều cách chia cùng
  khớp các mốc này; không đảm bảo trùng với cách chia gốc.
- **A2. Buffer** chỉ loại các mảnh train nằm trong bán kính Chebyshev 1 mảnh (640 m) / 2 mảnh (1.280 m) quanh **tập val**
  (thí nghiệm buffer là chẩn đoán trên val).
- **A3. Chuẩn hóa**: phân vị và median/IQR ước lượng trên 2.000.000 pixel lấy mẫu từ mọi pixel của mảnh train (kể cả pixel
  không nhãn), tách TOA (2018) / SR (2019+). Kênh có IQR = 0 (`co_lap`, gần như nhị phân) chia cho 1.
- **A4. Kênh 27 `co_lap`** trong dữ liệu là cờ lấp (thực tế chỉ có giá trị 0/1 vì S1 không quý nào phải lấp), được dùng
  như kênh "sensor overlap" của bài báo. Can thiệp optical dropout gán giá trị 1 (= "S2 phải lấp") đúng như bài báo.

### Mô hình
- **A5. Đầu vào U-Net**: cả 27 kênh của quý mục tiêu. DEM không đưa vào phân đoạn [paper].
- **A6. PlainTAtt / WindowNet-TAtt** (phản biện #6): encoder CNN dùng chung mọi quý [16,32,64,112] → 2 khối Transformer
  hai chiều theo thời gian ở tỉ lệ 1/8 → decoder U-Net với skip của chính quý đó; các kênh ghép thẳng, không hợp nhất.
  WindowNet-TAtt = cùng kiến trúc với cửa sổ T quý kết thúc ở quý mục tiêu (chỉ quý mục tiêu có nhãn); quý trước 2018Q1
  được đệm và che khỏi attention.
- **A7. U-TAE xuất theo quý**: L-TAE gốc dùng master query gộp chuỗi thành 1 bản đồ. Để đánh giá theo quý, mỗi quý là một
  query; ma trận chú ý [head, t, s] ở tỉ lệ thấp nhất được nội suy lên các tầng skip để tổng hợp skip theo nhóm kênh.
  Mã hóa vị trí dùng chỉ số quý (0-31) thay cho ngày trong năm.
- **A8. TSViT xuất theo quý**: bỏ token lớp; patch 4×4, encoder thời gian 4 tầng → encoder không gian 3 tầng, mỗi token
  giải mã thành patch 4×4×5.
- **A9. QuaST**: hai encoder [16,32,64,128]; hợp nhất ở 16×16 và 8×8; MLP chất lượng 3→64→C; `h_s` = conv1×1(3C→C)–ReLU–
  conv1×1(C→C), cổng theo kênh và pixel; `B_s` = Linear(C→C) trên Q (hằng theo không gian); 2 khối Transformer hai chiều
  ở 8×8; decoder lấy skip quang học ở 64×64, 32×32 và đặc trưng đã hợp nhất ở 16×16 (Hình 3).
- **A10. Monotonicity penalty**: chất lượng của một phần tử (mẫu, quý) = S2 count chuẩn hóa (trung bình không gian);
  độ tin cậy quang học = trung bình cổng g; phạt `relu(g_i − g_j)` trên mọi cặp trong batch có q_i < q_j, trung bình
  qua hai tỉ lệ.

### Huấn luyện và đánh giá
- **A11. Huấn luyện**: tối đa 100 epoch, dừng sớm 15 epoch; một epoch U-Net = mọi (mảnh, quý) train có nhãn; một epoch
  mô hình chuỗi = mọi mảnh train đủ điều kiện (hoặc mọi (mảnh, quý mục tiêu) có nhãn với cửa sổ). Oversample theo mẫu: mẫu
  chứa Water ×5, chứa Residential ×1,5 (lấy hệ số lớn nhất). Trọng số lớp tính trên nhãn train. Dice smooth = 1.
  Mixed precision trên GPU. Mã hóa lỗi Dice của bản gốc được giữ (`loss.dice_mask_ignore: false`).
- **A12. Đánh giá**: pixel thuộc 60 polygon vắt ranh giới bị loại khỏi đánh giá val/test, nhưng vẫn dùng khi huấn luyện.
  Tập mảnh chung cho kiểm định cặp = mảnh test có pixel hợp lệ (sau khi loại polygon vắt ranh) ở mọi mô hình; với cách
  chia đã chọn, đúng 34 mảnh như bài báo.
- **A13. Optical dropout**: mỗi mảnh một hoán vị quý theo mask seed, D quý đầu bị xóa (tập lồng nhau theo D).

### Polygon và sàng lọc
- **A14. Đặc trưng polygon-quý**: mean, std, p10, p50, p90 của 27 kênh trên pixel mẫu trong `features_quy` + số pixel;
  không dùng DEM (tùy chọn `--include-dem`). Lưới siêu tham số trong `acacia/polygon/classify.py`, chọn theo Macro-F1 val.
  QuaST encoder + MLP: đặc trưng 16×16 sau tầng giải mã đầu, gộp theo diện tích polygon, encoder đông cứng.
- **A15. Polygon sàng lọc**: polygon từng mang nhãn Keo và có ≥ 25 pixel ở quý lớn nhất (cho đúng 105).
- **A16. Sự kiện trồng lại**: lần chuyển nhãn Đất trống → Keo cuối cùng; nếu không có thì đáy NDVI cuối cùng sau khi giảm
  ≥ 0,15 so với đỉnh trước đó. Growth = độ dốc hồi quy NDVI theo quý từ sự kiện đó (cần ≥ 4 quý).
- **A17. Disturbance quanh 2020Q4** = NDVI trung bình (2020Q2, 2020Q3) − NDVI thấp nhất (2020Q4, 2021Q1). Lý do chọn mốc
  (phản biện #30): chuỗi bão cuối 2020 ở miền Trung (bão Molave đổ bộ Quảng Nam - Đà Nẵng ngày 28/10/2020).
- **A18. Bảy bộ trọng số thay thế**: xem `ALT_WEIGHTS` trong `acacia/screening/screen.py`.
- **A19. Nhãn gần nhất** của polygon = lớp đa số của polygon tại quý 2025Q4; không có → "unlabeled".
