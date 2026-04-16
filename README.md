1 Giới thiệu

Module này thực hiện Exploratory Data Analysis (EDA) trên bộ dữ liệu điểm số sinh viên.
Mục tiêu là:

Hiểu phân bố dữ liệu
Phát hiện mối quan hệ giữa các biến
Xác định các yếu tố ảnh hưởng đến kết quả học tập

2 Công nghệ sử dụng

Python
Pandas
NumPy
SciPy (thống kê)

3 Nội dung phân tích

1. Thống kê mô tả
Mean, min, max, std
Skewness (độ lệch)
Kurtosis (độ nhọn)
2. Phân tích theo môn
So sánh điểm trung bình:
Toán (Math)
Đọc (Reading)
Viết (Writing)
3. Phân tích theo nhóm (ethnicity)
Điểm trung bình theo nhóm
Kiểm định ANOVA
Kiểm tra sự khác biệt giữa các nhóm
4. Phân bố học lực
Số lượng từng loại xếp loại
Tỷ lệ Đạt / Không đạt
5. Phân tích theo giới tính
So sánh điểm trung bình Nam vs Nữ
Kiểm định T-test
6. Trình độ phụ huynh
Ảnh hưởng đến điểm trung bình
7. Khóa ôn thi (test preparation)
So sánh:
Có học (completed)
Không học (none)
Kiểm định T-test
Tính chênh lệch điểm trung bình (diff_prep)
8. Bữa trưa
Ảnh hưởng của:
standard
free/reduced
Kiểm định thống kê
9. Ma trận tương quan
Tương quan giữa:
Math, Reading, Writing
Total, Average
10. Phân tích chéo
Giới tính × Khóa ôn thi
Giới tính × Bữa trưa
11. Kiểm định Chi-square
Giới tính vs Xếp loại
Khóa ôn thi vs Đạt/Không đạt
12. Tương quan Pearson
Math vs Reading
Math vs Writing
Reading vs Writing

4 Ý nghĩa

Xác định yếu tố ảnh hưởng đến kết quả học tập
Hỗ trợ xây dựng mô hình Machine Learning
Làm cơ sở cho phần Insight & Visualization