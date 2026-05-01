# Phân Tích Điểm Số Sinh Viên

## Mô tả đề tài

Đề tài phân tích bộ dữ liệu điểm thi của 1000 sinh viên, gồm 3 môn Math, Reading và Writing. Ngoài điểm số, dữ liệu còn có thêm thông tin về giới tính, nhóm dân tộc, trình độ học vấn phụ huynh, chế độ bữa trưa và việc có tham gia khóa ôn thi hay không.

Mục tiêu chính:
- Làm sạch dữ liệu điểm, xử lý lỗi và tạo biến phụ trợ
- Phân tích điểm trung bình theo lớp (nhóm) và theo từng môn
- Xác định yếu tố nào ảnh hưởng đến kết quả học tập
- Trực quan hóa phân bố điểm bằng nhiều dạng biểu đồ
- Dự đoán điểm cuối kỳ bằng mô hình Regression

## Cấu trúc thư mục

```
StudentPerformance/
├── StudentsPerformance.csv     # Dữ liệu gốc (1000 bản ghi)
├── main.py                    # Điểm vào – điều phối toàn bộ pipeline
├── utils.py                   # Hằng số, hàm tiện ích dùng chung
├── data_cleaning.py           # Phần 1 – Làm sạch dữ liệu
├── eda.py                     # Phần 2 – Phân tích khám phá (EDA)
├── visualization.py           # Phần 3 – Trực quan hóa (18 biểu đồ)
├── modeling.py                # Phần 4 – Xây dựng mô hình
├── insight.py                 # Phần 5 – Insight & Kết luận
├── README.md
└── output/                    # Kết quả (tạo tự động khi chạy)
    ├── cleaned_data.csv
    ├── insights_summary.txt
    └── *.png                  # 26 biểu đồ
```

## Cài đặt

Python 3.8+. Cài thư viện:

```
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Chạy chương trình

```
cd StudentPerformance
python main.py
```

Kết quả in ra console, biểu đồ và file csv lưu trong `output/`.

## Nội dung chi tiết

### Phần 1 - Làm sạch dữ liệu

Đọc file CSV, kiểm tra kiểu dữ liệu, tìm missing values, loại bản ghi trùng lặp. Chuẩn hóa tên cột cho gọn (ví dụ `race/ethnicity` thành `ethnicity`). Kiểm tra điểm có nằm trong khoảng 0-100 không, nếu lệch thì sửa. Phát hiện outlier bằng IQR nhưng giữ nguyên vì điểm thi thực tế có thể rất thấp hoặc rất cao.

Sau đó tạo thêm một số cột:
- `total_score`: tổng 3 môn
- `average_score`: điểm trung bình
- `xep_loai`: xếp loại học lực (Xuất sắc / Giỏi / Khá / Trung bình khá / Trung bình / Yếu)
- `pass_fail`: Đạt (>= 50) hay Không đạt

Dữ liệu sạch lưu ra `output/cleaned_data.csv`.

### Phần 2 - EDA (Phân tích khám phá)

- Thống kê mô tả: mean, median, std, skewness, kurtosis cho từng môn
- Phân tích điểm trung bình theo từng môn (Math thường thấp hơn Reading/Writing)
- Phân tích theo nhóm (group A-E): so sánh điểm giữa các nhóm, dùng ANOVA test xem có khác biệt có ý nghĩa không
- Phân bố xếp loại, tỉ lệ Đạt/Không đạt
- So sánh nam vs nữ: t-test hai mẫu độc lập
- Điểm theo trình độ phụ huynh (6 mức từ some high school đến master's degree)
- Ảnh hưởng khóa ôn thi: t-test so sánh completed vs none
- Ảnh hưởng bữa trưa: t-test standard vs free/reduced
- Ma trận tương quan Pearson giữa 3 môn
- Phân tích chéo: giới tính × khóa ôn thi, giới tính × bữa trưa
- Kiểm định Chi-square: giới tính vs xếp loại, khóa ôn thi vs đạt/không đạt

### Phần 3 - Trực quan hóa

Tổng cộng 26 biểu đồ, gồm:

| # | Biểu đồ | Mô tả |
|---|---------|-------|
| 1 | Histogram + KDE | Phân bố điểm từng môn |
| 2 | Histogram | Phân bố điểm trung bình |
| 3 | Boxplot | So sánh 3 môn |
| 4 | Boxplot | Điểm theo giới tính |
| 5 | Bar chart | Điểm TB theo nhóm |
| 6 | Grouped bar | Điểm từng môn theo nhóm |
| 7 | Bar chart | Điểm theo trình độ phụ huynh |
| 8 | Boxplot | Khóa ôn thi |
| 9 | Violin plot | Bữa trưa |
| 10 | Heatmap | Tương quan 3 môn |
| 11 | Heatmap | Tương quan mở rộng (có biến mã hóa) |
| 12 | Scatter | Math vs Reading (màu = Writing) |
| 13 | Pair plot | 3 môn theo giới tính |
| 14 | Pie chart | Xếp loại học lực |
| 15 | Count plot | SV theo nhóm và giới tính |
| 16 | KDE overlay | So sánh mật độ 3 môn |
| 17 | Stacked bar | Xếp loại theo giới tính |
| 18 | Grouped bar | Khóa ôn thi × bữa trưa |
| 19-20 | Scatter + Residual | Regression actual vs predicted (6 mô hình) |
| 21 | Bar chart | Feature importance (Regression) |
| 22-23 | Scatter | Dự đoán điểm TB từ nhân khẩu học |
| 24 | Confusion Matrix | 3 mô hình phân loại |
| 25 | Bar chart | So sánh accuracy phân loại |
| 26 | Bar chart | So sánh R2 hồi quy |

### Phần 4 - Modeling

**4.1 Dự đoán điểm cuối kỳ (Writing) bằng Regression**

Dùng các đặc điểm nhân khẩu học + điểm Math, Reading để dự đoán điểm Writing (coi như điểm cuối kỳ). Thử 6 mô hình:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Đánh giá bằng MAE, RMSE, R² và Cross-validation 5-fold. Kèm biểu đồ Actual vs Predicted, phân tích phần dư (Residual), và feature importance.

**4.2 Dự đoán điểm trung bình chỉ từ nhân khẩu học**

Thử dùng 5 đặc điểm (gender, ethnicity, parent_education, lunch, test_prep) để đoán điểm trung bình. R² thấp hơn nhiều so với 4.1, cho thấy điểm phụ thuộc vào nỗ lực học tập chứ không chỉ hoàn cảnh.

**4.3 Phân loại Đạt / Không đạt**

Với ngưỡng 50 điểm trung bình, dùng 3 mô hình: Logistic Regression, Decision Tree, Random Forest. Đánh giá bằng Accuracy, Precision, Recall, F1, Confusion Matrix.

### Phần 5 - Insight & Kết luận

Tổng hợp các phát hiện chính từ toàn bộ phân tích: mối liên hệ giữa các yếu tố nhân khẩu học với điểm số, tương quan giữa các môn, hiệu quả dự đoán, và đề xuất cải thiện.

## Dữ liệu

Bộ dữ liệu "Students Performance in Exams" từ Kaggle:
https://www.kaggle.com/datasets/spscientist/students-performance-in-exams

## Thành viên

| STT | Họ và tên        | MSSV     | Phụ trách |
|-----|------------------|----------|-----------|
| 1   | Bùi Minh hiếu    | 20222227 | Làm sạch dữ liệu (Phần 1) + Insight & Kết luận (Phần 5) |
| 2   | Nguyễn Quốc Bảo  | 20221333 | Phân tích khám phá EDA (Phần 2) |
| 3   | Nguyễn Đức Dũng  | 20222220 | Trực quan hóa dữ liệu (Phần 3) |
| 4   | Nguyễn Mạnh Hùng | 20222173 | Xây dựng mô hình & Đánh giá (Phần 4) + Viết báo cáo tổng hợp |

## Ghi chú

- Code chia thành nhiều module cho dễ đọc và bảo trì. Chạy `main.py` để thực thi toàn bộ pipeline.
- Thư mục `output/` tự động tạo khi chạy, không cần tạo trước.
