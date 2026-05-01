# -*- coding: utf-8 -*-
"""
Phần 5: Insight & Kết luận
"""

import os
from utils import OUTPUT_DIR, get_fig_count


def run_insight(df, corr_mat, diff_prep, results_reg, results_clf):
    """
    Tổng hợp insight và lưu ra file.
    """
    print("\n\n" + "=" * 65)
    print("PHẦN 5: INSIGHT & KẾT LUẬN")
    print("=" * 65)

    best_reg_name = max(results_reg, key=lambda k: results_reg[k]['R2'])
    best_reg_r2 = results_reg[best_reg_name]['R2']
    best_clf_name = max(results_clf, key=lambda k: results_clf[k]['accuracy'])
    best_clf_acc = results_clf[best_clf_name]['accuracy']

    insights_text = f"""
KẾT QUẢ CHÍNH TỪ PHÂN TÍCH:

1. TỔNG QUAN BỘ DỮ LIỆU
   - {len(df)} sinh viên, 3 môn thi: Math, Reading, Writing.
   - Điểm TB: Math = {df['math_score'].mean():.1f}, Reading = {df['reading_score'].mean():.1f}, Writing = {df['writing_score'].mean():.1f}.
   - Dữ liệu sạch, không có missing values.

2. PHÂN BỐ ĐIỂM
   - Điểm 3 môn phân bố gần dạng chuẩn, hơi lệch trái.
   - Đa số SV đạt mức Khá và Trung bình khá.
   - Tỷ lệ Yếu: {(df['xep_loai']=='Yếu').mean()*100:.1f}%, Xuất sắc: {(df['xep_loai']=='Xuất sắc').mean()*100:.1f}%.

3. ĐIỂM THEO MÔN VÀ NHÓM (LỚP)
   - Reading và Writing có điểm TB cao hơn Math.
   - Nhóm E có điểm TB cao nhất, nhóm A thấp nhất.
   - ANOVA: sự khác biệt giữa các nhóm có ý nghĩa thống kê (p < 0.05).

4. YẾU TỐ ẢNH HƯỞNG ĐẾN KẾT QUẢ HỌC TẬP

   a) Giới tính:
      - Nam: Math cao hơn ({df[df['gender']=='male']['math_score'].mean():.1f} vs {df[df['gender']=='female']['math_score'].mean():.1f}).
      - Nữ: Reading và Writing cao hơn nam.

   b) Trình độ phụ huynh:
      - Trình độ càng cao, điểm con cái càng tốt.
      - Master's degree: TB {df[df['parent_education']=="master's degree"]['average_score'].mean():.1f}.

   c) Khóa ôn thi:
      - SV hoàn thành khóa ôn thi có điểm cao hơn {diff_prep:.1f} điểm.
      - T-test: p < 0.001 (có ý nghĩa thống kê).

   d) Bữa trưa:
      - Standard lunch: điểm cao hơn free/reduced.
      - Phản ánh ảnh hưởng của điều kiện kinh tế.

5. TƯƠNG QUAN GIỮA CÁC MÔN
   - Reading-Writing: r = {corr_mat.loc['reading_score','writing_score']:.3f} (rất mạnh).
   - Math-Reading: r = {corr_mat.loc['math_score','reading_score']:.3f}.
   - Math-Writing: r = {corr_mat.loc['math_score','writing_score']:.3f}.

6. MÔ HÌNH DỰ ĐOÁN
   - Dự đoán điểm cuối kỳ (Writing): {best_reg_name} tốt nhất, R2 = {best_reg_r2:.4f}.
   - Dự đoán chỉ từ nhân khẩu học: R2 thấp (< 0.2), điểm phụ thuộc nhiều vào nỗ lực.
   - Phân loại Đạt/Không đạt: {best_clf_name}, Accuracy = {best_clf_acc:.2%}.

7. ĐỀ XUẤT
   - Mở rộng chương trình ôn thi cho SV.
   - Hỗ trợ tài chính/dinh dưỡng cho SV khó khăn.
   - Xây dựng phương pháp học theo giới tính và thế mạnh.
   - Nghiên cứu thêm các yếu tố khác (thời gian học, động lực, ...).
"""

    print(insights_text)

    with open(os.path.join(OUTPUT_DIR, "insights_summary.txt"), 'w',
              encoding='utf-8') as f:
        f.write(insights_text)
    print("=> Đã lưu Insights vào output/insights_summary.txt")

    print("\n" + "=" * 65)
    print(f"  HOÀN THÀNH! Tổng cộng {get_fig_count()} biểu đồ.")
    print(f"  Kết quả lưu tại: ./{OUTPUT_DIR}/")
    print("=" * 65)
