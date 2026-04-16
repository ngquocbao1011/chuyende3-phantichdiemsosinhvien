# -*- coding: utf-8 -*-
"""
Phần 1: Đọc và làm sạch dữ liệu (Data Cleaning)
"""

import pandas as pd
import numpy as np
import os
from utils import OUTPUT_DIR, SCORE_COLS, xep_loai


def run_data_cleaning(filepath):
    """
    Đọc file CSV, làm sạch và tạo biến phụ trợ.
    Trả về DataFrame đã xử lý.
    """
    print("=" * 65)
    print("  PHÂN TÍCH ĐIỂM SỐ SINH VIÊN  ")
    print("=" * 65)

    print("\n" + "=" * 65)
    print("PHẦN 1: DATA CLEANING")
    print("=" * 65)

    # --- 1.1 Đọc dữ liệu ---
    df_raw = pd.read_csv(filepath)
    df = df_raw.copy()

    print(f"\n1.1 Đọc dữ liệu thành công")
    print(f"    Kích thước: {df.shape[0]} bản ghi, {df.shape[1]} trường")
    print(f"    Các trường: {list(df.columns)}")
    print(f"\n    5 bản ghi đầu:")
    print(df.head().to_string(index=False))

    # --- 1.2 Kiểu dữ liệu ---
    print(f"\n1.2 Kiểu dữ liệu:")
    for col in df.columns:
        nunique = df[col].nunique()
        dtype = df[col].dtype
        print(f"    {col:40s} | {str(dtype):8s} | {nunique} giá trị duy nhất")

    # --- 1.3 Kiểm tra missing values ---
    print(f"\n1.3 Kiểm tra giá trị thiếu (Missing Values):")
    miss = df.isnull().sum()
    miss_pct = (df.isnull().sum() / len(df) * 100).round(2)
    miss_df = pd.DataFrame({'Số lượng': miss, 'Tỷ lệ (%)': miss_pct})
    print(miss_df.to_string())
    total_miss = miss.sum()
    if total_miss == 0:
        print("    => Không có giá trị thiếu nào. Dữ liệu đầy đủ.")
    else:
        print(f"    => Tổng {total_miss} giá trị thiếu.")
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                med = df[col].median()
                df[col].fillna(med, inplace=True)
                print(f"    -> Điền {col} = trung vị ({med})")
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isnull().sum() > 0:
                mod = df[col].mode()[0]
                df[col].fillna(mod, inplace=True)
                print(f"    -> Điền {col} = mode ({mod})")

    # --- 1.4 Kiểm tra trùng lặp ---
    print(f"\n1.4 Kiểm tra bản ghi trùng lặp:")
    n_dup = df.duplicated().sum()
    print(f"    Số bản ghi trùng lặp: {n_dup}")
    if n_dup > 0:
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"    => Đã loại bỏ. Còn lại {len(df)} bản ghi.")
    else:
        print(f"    => Không có bản ghi trùng lặp.")

    # --- 1.5 Chuẩn hóa tên cột ---
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    df.rename(columns={
        'race/ethnicity': 'ethnicity',
        'parental_level_of_education': 'parent_education',
        'test_preparation_course': 'test_prep'
    }, inplace=True)
    print(f"\n1.5 Chuẩn hóa tên cột:")
    print(f"    {list(df.columns)}")

    # --- 1.6 Kiểm tra giá trị điểm hợp lệ ---
    print(f"\n1.6 Kiểm tra khoảng giá trị điểm (kỳ vọng 0-100):")
    for col in SCORE_COLS:
        vmin, vmax = df[col].min(), df[col].max()
        status = "OK" if (vmin >= 0 and vmax <= 100) else "CẢNH BÁO"
        print(f"    {col:20s}: [{vmin}, {vmax}]  [{status}]")
        if vmin < 0:
            cnt = (df[col] < 0).sum()
            df.loc[df[col] < 0, col] = 0
            print(f"      -> Đã thay {cnt} giá trị âm thành 0")
        if vmax > 100:
            cnt = (df[col] > 100).sum()
            df.loc[df[col] > 100, col] = 100
            print(f"      -> Đã thay {cnt} giá trị >100 thành 100")

    # --- 1.7 Phát hiện outliers (IQR) ---
    print(f"\n1.7 Phát hiện outliers (phương pháp IQR):")
    for col in SCORE_COLS:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        print(f"    {col:20s}: IQR={IQR:.1f}, [{lo:.1f}, {hi:.1f}], "
              f"outliers={n_out}")
    print("    => Giữ nguyên outliers (điểm thi thực tế có thể rất thấp/cao).")

    # --- 1.8 Tạo các biến phụ trợ ---
    print(f"\n1.8 Tạo biến phụ trợ:")
    df['total_score'] = df[SCORE_COLS].sum(axis=1)
    df['average_score'] = (df['total_score'] / 3).round(2)
    df['xep_loai'] = df['average_score'].apply(xep_loai)
    df['pass_fail'] = np.where(df['average_score'] >= 50, 'Đạt', 'Không đạt')
    df['math_read_diff'] = df['math_score'] - df['reading_score']
    df['read_write_diff'] = df['reading_score'] - df['writing_score']

    print(f"    Thêm cột: total_score, average_score, xep_loai, pass_fail")
    print(f"    Thêm cột: math_read_diff, read_write_diff")
    print(f"\n    Mẫu dữ liệu sau xử lý:")
    print(df[['math_score', 'reading_score', 'writing_score',
              'total_score', 'average_score', 'xep_loai']].head(8).to_string())

    df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), index=False)
    print(f"\n    => Lưu dữ liệu sạch: output/cleaned_data.csv")

    return df
    #.