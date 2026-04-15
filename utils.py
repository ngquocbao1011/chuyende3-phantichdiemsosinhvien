# -*- coding: utf-8 -*-
"""
Module tiện ích dùng chung cho toàn bộ dự án.
Chứa hằng số, hàm save_fig, hàm xếp loại, cấu hình biểu đồ.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Cấu hình biểu đồ
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams['figure.dpi'] = 130
plt.rcParams['savefig.bbox'] = 'tight'

# Thư mục lưu kết quả
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Danh sách cột điểm
SCORE_COLS = ['math_score', 'reading_score', 'writing_score']

# Thứ tự trình độ phụ huynh
EDU_ORDER = [
    'some high school', 'high school', 'some college',
    "associate's degree", "bachelor's degree", "master's degree"
]

# Thứ tự xếp loại
XL_ORDER = ['Xuất sắc', 'Giỏi', 'Khá', 'Trung bình khá', 'Trung bình', 'Yếu']

# Bộ đếm biểu đồ
_fig_counter = 0


def save_fig(name):
    """Lưu biểu đồ vào thư mục output và đóng lại."""
    global _fig_counter
    _fig_counter += 1
    fname = f"{_fig_counter:02d}_{name}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close('all')
    print(f"  [{_fig_counter}] Đã lưu: {fname}")
    return fname


def get_fig_count():
    """Trả về số biểu đồ đã lưu."""
    return _fig_counter


def xep_loai(d):
    """Xếp loại học lực theo điểm trung bình."""
    if d >= 90: return 'Xuất sắc'
    if d >= 80: return 'Giỏi'
    if d >= 70: return 'Khá'
    if d >= 60: return 'Trung bình khá'
    if d >= 50: return 'Trung bình'
    return 'Yếu'
