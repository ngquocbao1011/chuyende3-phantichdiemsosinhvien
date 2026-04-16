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
├── StudentsPerformance.csv
├── main.py
├── utils.py
├── data_cleaning.py
├── eda.py
├── visualization.py
├── modeling.py
├── insight.py
├── README.md
└── output/
```

## Cài đặt

```
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Chạy chương trình

```
python main.py
```

## Thành viên

| STT | Họ và tên        | MSSV     | Phụ trách |
|-----|------------------|----------|-----------|
| 1   | Bùi Minh hiếu    | 20222227 | Làm sạch dữ liệu (Phần 1) + Insight & Kết luận (Phần 5) |
| 2   | Nguyễn Quốc Bảo  | 20221333 | Phân tích khám phá EDA (Phần 2) |
| 3   | Nguyễn Đức Dũng  | 20222220 | Trực quan hóa dữ liệu (Phần 3) |
| 4   | Nguyễn Mạnh Hùng | 20222173 | Xây dựng mô hình & Đánh giá (Phần 4) + Viết báo cáo tổng hợp |

## Phân công công việc chi tiết

### Thành viên 1 – Làm sạch dữ liệu + Insight
- Làm sạch dữ liệu
- Xử lý lỗi
- Tạo biến mới
- Viết insight và kết luận

### Thành viên 2 – EDA
- Thống kê mô tả
- Phân tích dữ liệu
- Kiểm định thống kê

### Thành viên 3 – Visualization
- Vẽ biểu đồ
- Trực quan hóa dữ liệu

### Thành viên 4 – Modeling + Báo cáo
- Xây dựng mô hình
- Đánh giá mô hình
- Viết báo cáo tổng hợp

## Ghi chú
- Chạy file main.py để thực thi toàn bộ chương trình
- Kết quả lưu trong thư mục output/

.