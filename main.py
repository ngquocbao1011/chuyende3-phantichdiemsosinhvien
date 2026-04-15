# -*- coding: utf-8 -*-
"""
Main – Điều phối toàn bộ pipeline phân tích kết quả học tập sinh viên.

Chạy:  python main.py
"""

import os
import matplotlib
matplotlib.use('Agg')

from utils import OUTPUT_DIR
from data_cleaning import run_data_cleaning
from eda import run_eda
from visualization import run_visualization
from modeling import run_modeling
from insight import run_insight


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = 'StudentsPerformance.csv'

    # Phần 1: Làm sạch dữ liệu
    df = run_data_cleaning(csv_path)

    # Phần 2: Phân tích khám phá (EDA)
    corr_mat, diff_prep = run_eda(df)

    # Phần 3: Trực quan hóa
    run_visualization(df)

    # Phần 4: Modeling
    results_reg, results_clf = run_modeling(df)

    # Phần 5: Insight & Kết luận
    run_insight(df, corr_mat, diff_prep, results_reg, results_clf)


if __name__ == '__main__':
    main()
