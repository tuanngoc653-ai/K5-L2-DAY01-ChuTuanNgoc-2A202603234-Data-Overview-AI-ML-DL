# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.13.18 / 2.11.0+cpu / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

## 1. Phân loại ảnh

Ảnh `traffic`, dòng `rank = 1` trong `classification_predictions.json`.

1. `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về cả bức ảnh. Trong record không có box hay polygon nào, nên không biết "cab" nằm ở đâu trong ảnh. Mà ảnh thì có đủ xe buýt, ô tô, người đi bộ chứ không chỉ một chiếc xe.

3. Class list đi kèm taxonomy/checkpoint. Model này chỉ chọn trong 1000 lớp của ImageNet, nên ảnh không khớp lớp nào thì nó vẫn phải trả về một lớp trong danh sách đó.

4. Chỉ ghi tên lớp thì dễ nhầm khi có nhiều bộ dữ liệu. Ví dụ `class_id=0` trong COCO-80 là `person`, sang ImageNet-1K lại là lớp khác. Có đủ cả mã, tên và taxonomy thì mới tra lại được đúng danh sách gốc.

5. Guideline phải nói chọn lớp theo cái gì: chủ thể chính, vật chiếm nhiều diện tích nhất, hay nhãn cho cả cảnh. Cũng cần nói nếu không có lớp nào hợp thì làm sao, ví dụ hỏi Lab Coach thay vì tự chọn đại. Với ảnh `traffic`, model chọn `cab` (0.510915), sau đó là `minibus` (0.164284), nhưng nếu không có quy tắc thì hai người gán có thể chọn hai nhãn khác nhau.

6. Score chỉ là mức model tin vào dự đoán của nó. Model tin cao vẫn có thể sai. Nếu lấy luôn dự đoán đó làm ground truth thì giống như để model tự chấm bài của mình.

## 2. Phát hiện vật thể

Ảnh `kitchen` (640 × 427), file `detection_predictions.json`.

1. `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Đây là hộp bao quanh người đứng quay lưng ở phía sau. Góc trên-trái ở (385.33, 69.24), góc dưới-phải ở (498.92, 348.92), tính từ góc trên-trái của ảnh. Hộp rộng khoảng 114 pixel, cao khoảng 280 pixel.

3. Mình chọn hai ngưỡng 0.35 và 0.60.
   - 0.35: 11 prediction (2 `person`, 5 `bowl`, 2 `oven`, 2 `cup`)
   - 0.60: 6 prediction (2 `person`, 2 `bowl`, 2 `oven`)

4. Hạ ngưỡng thì có nhiều hộp hơn nên reviewer phải xem nhiều hơn: bỏ hộp sai lớp, hộp trùng, hộp lệch, và xem những hộp score thấp có đúng là vật thật không. Ngưỡng chỉ dùng để lọc prediction. Người gán nhãn không được bỏ vật thể chỉ vì model cho score thấp.

5. Quy tắc box: mỗi vật một hộp, vẽ sát phần nhìn thấy được, không chừa nền và không đoán phần bị che hoặc nằm ngoài ảnh.

6. Mình chọn `person` 0.610931 tại [0.12, 263.18, 61.55, 310.78]. Đây chỉ là một cánh tay ở sát mép trái ảnh, phần còn lại của người bị cắt mất. Trường hợp này mình đưa vào guideline, vì vật bị cắt mép gặp ở rất nhiều ảnh: vẫn gán `person` và vẽ theo phần thấy được. Nếu chỉ thấy một mẩu quá nhỏ, không chắc là gì, thì mới hỏi Lab Coach.

## 3. Phân đoạn theo từng đối tượng

Ảnh `kitchen`, file `segmentation_predictions.json`.

1. `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; `polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]

2. Box chỉ là hình chữ nhật bao ngoài nên vẫn dính nền, ví dụ phần tường cạnh người. Polygon đi theo đường viền nên thấy được dáng đầu, tay, chân và biết pixel nào thuộc về người.

3. `instance_id` để phân biệt từng đối tượng trong output. Ví dụ `kitchen-001` và `kitchen-009` là hai đối tượng khác nhau nhưng đều có `class_id`=0, nên nó không phải class ID. Nó cũng không phải tracking ID vì chỉ đánh số trong output của một ảnh, không theo một vật qua nhiều khung hình.

4. Quy tắc vẽ mask: vẽ sát biên phần nhìn thấy, mỗi instance một mask riêng, không tính vật khác đang đặt lên hoặc che phía trước, không đoán phần bị che.

5. Mask `dining table` (`kitchen-005`, 0.578416) phủ gần hết mặt bàn, tính luôn cả bột và dụng cụ đặt trên đó, lại lan xuống cả dưới gầm bàn. Chỗ vật đặt trên bàn thì dùng guideline được, vì gặp thường xuyên: vật trên bàn không tính vào mask của bàn. Còn bàn bếp này có được gọi là `dining table` không thì mình sẽ hỏi Lab Coach, vì đó là chuyện định nghĩa lớp.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho cả ảnh | `traffic`: top-1 là `cab` (0.510915) trong khi ảnh có nhiều xe buýt, ô tô và người, khó nói chủ thể chính là gì | Chọn một lớp theo guideline, không có lớp hợp thì hỏi | Lớp có đúng taxonomy và đúng tiêu chí chọn không |
| Phát hiện vật thể | Một lớp và một hộp cho mỗi vật | `kitchen`: xoong chảo treo không có box vì COCO-80 không có lớp này. Cái bowl ở khoảng [155, 169, 182, 184] được 0.701283 ở detection nhưng chỉ 0.372406 ở segmentation | Vẽ hộp sát từng vật, gán lớp theo taxonomy | Hộp có sát không, có sai lớp, trùng hay bỏ sót không |
| Instance segmentation | Một lớp và một polygon cho mỗi instance | `kitchen-004`: bó lá treo bị gán `potted plant` (0.632235) dù không thấy chậu. `kitchen-005`: mask bàn phủ cả vật trên bàn | Vẽ biên sát từng instance, tách các vật chạm nhau | Biên có sát không, có bị gộp hay lẫn vật khác không, lớp có đúng không |

## 5. An toàn dữ liệu

1. Mình chỉ dùng ba ảnh COCO công khai có sẵn trong notebook, không đưa ảnh cá nhân hay dữ liệu nội bộ lên Colab/GitHub.

2. Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi thì mình dừng lại và báo GV/Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
