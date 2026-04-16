# -*- coding: utf-8 -*-
"""
Phần 2: EDA - Phân tích khám phá dữ liệu
"""

import pandas as pd
import numpy as np
from scipy import stats
from utils import SCORE_COLS, EDU_ORDER, XL_ORDER


def run_eda(df):
    """
    Thực hiện phân tích khám phá dữ liệu.
    Trả về (corr_mat, diff_prep) để dùng ở phần Insight.
    """
    print("\n\n" + "=" * 65)
    print("PHẦN 2: EDA - PHÂN TÍCH KHÁM PHÁ DỮ LIỆU")
    print("=" * 65)

    # --- 2.1 Thống kê mô tả ---
    print("\n2.1 Thống kê mô tả điểm số:")
    desc_cols = SCORE_COLS + ['total_score', 'average_score']
    desc = df[desc_cols].describe().round(2)
    print(desc.to_string())

    print("\n    Độ lệch (Skewness) và độ nhọn (Kurtosis):")
    for col in SCORE_COLS:
        sk = df[col].skew()
        ku = df[col].kurtosis()
        print(f"    {col:20s}: skewness={sk:+.3f}, kurtosis={ku:+.3f}")

    # --- 2.2 Phân tích điểm trung bình theo môn ---
    print("\n2.2 Phân tích điểm trung bình theo từng môn:")
    subject_stats = pd.DataFrame({
        'Môn': ['Math', 'Reading', 'Writing'],
        'Trung bình': [df['math_score'].mean(), df['reading_score'].mean(),
                       df['writing_score'].mean()],
        'Trung vị': [df['math_score'].median(), df['reading_score'].median(),
                     df['writing_score'].median()],
        'Độ lệch chuẩn': [df['math_score'].std(), df['reading_score'].std(),
                          df['writing_score'].std()],
        'Min': [df['math_score'].min(), df['reading_score'].min(),
                df['writing_score'].min()],
        'Max': [df['math_score'].max(), df['reading_score'].max(),
                df['writing_score'].max()],
    })
    print(subject_stats.round(2).to_string(index=False))

    # --- 2.3 Phân tích điểm trung bình theo lớp (nhóm) ---
    print("\n2.3 Điểm trung bình theo nhóm (lớp):")
    group_stats = df.groupby('ethnicity').agg(
        sĩ_số=('ethnicity', 'count'),
        math_tb=('math_score', 'mean'),
        reading_tb=('reading_score', 'mean'),
        writing_tb=('writing_score', 'mean'),
        average_tb=('average_score', 'mean'),
        average_std=('average_score', 'std')
    ).round(2)
    print(group_stats.to_string())

    # ANOVA test giữa các nhóm
    groups_anova = [g['average_score'].values for _, g in df.groupby('ethnicity')]
    f_stat, p_anova = stats.f_oneway(*groups_anova)
    print(f"\n    ANOVA test giữa các nhóm: F={f_stat:.3f}, p={p_anova:.6f}")
    if p_anova < 0.05:
        print("    => Khác biệt có ý nghĩa thống kê giữa các nhóm (p < 0.05)")
    else:
        print("    => Không có sự khác biệt có ý nghĩa thống kê")

    # --- 2.4 Phân bố xếp loại ---
    print("\n2.4 Phân bố xếp loại học lực:")
    xl = df['xep_loai'].value_counts()
    xl_pct = (xl / len(df) * 100).round(1)
    for grade in XL_ORDER:
        if grade in xl.index:
            print(f"    {grade:18s}: {xl[grade]:4d} SV ({xl_pct[grade]:5.1f}%)")

    print(f"\n    Đạt/Không đạt:")
    pf = df['pass_fail'].value_counts()
    for k in ['Đạt', 'Không đạt']:
        if k in pf.index:
            print(f"    {k:18s}: {pf[k]:4d} SV ({pf[k]/len(df)*100:.1f}%)")

    # --- 2.5 Phân tích theo giới tính ---
    print("\n2.5 Phân tích theo giới tính:")
    gen = df.groupby('gender')[SCORE_COLS + ['average_score']].agg(
        ['mean', 'std']).round(2)
    print(gen.to_string())

    male = df[df['gender'] == 'male']['average_score']
    female = df[df['gender'] == 'female']['average_score']
    t_gen, p_gen = stats.ttest_ind(male, female)
    print(f"\n    T-test (nam vs nữ): t={t_gen:.3f}, p={p_gen:.6f}")

    # --- 2.6 Trình độ phụ huynh ---
    print("\n2.6 Điểm trung bình theo trình độ phụ huynh:")
    edu_stats = df.groupby('parent_education').agg(
        số_lượng=('average_score', 'count'),
        điểm_tb=('average_score', 'mean'),
        điểm_std=('average_score', 'std')
    ).reindex(EDU_ORDER).round(2)
    print(edu_stats.to_string())

    # --- 2.7 Khóa ôn thi ---
    print("\n2.7 Ảnh hưởng của khóa ôn thi:")
    prep = df.groupby('test_prep')[SCORE_COLS + ['average_score']].mean().round(2)
    print(prep.to_string())

    prep_yes = df[df['test_prep'] == 'completed']['average_score']
    prep_no = df[df['test_prep'] == 'none']['average_score']
    t_prep, p_prep = stats.ttest_ind(prep_yes, prep_no)
    diff_prep = prep_yes.mean() - prep_no.mean()
    print(f"    T-test (completed vs none): t={t_prep:.3f}, p={p_prep:.6f}")
    print(f"    Chênh lệch trung bình: {diff_prep:+.2f} điểm")

    # --- 2.8 Bữa trưa ---
    print("\n2.8 Ảnh hưởng của chế độ bữa trưa:")
    lunch = df.groupby('lunch')[SCORE_COLS + ['average_score']].mean().round(2)
    print(lunch.to_string())

    t_lunch, p_lunch = stats.ttest_ind(
        df[df['lunch'] == 'standard']['average_score'],
        df[df['lunch'] == 'free/reduced']['average_score']
    )
    print(f"    T-test (standard vs free/reduced): t={t_lunch:.3f}, p={p_lunch:.6f}")

    # --- 2.9 Ma trận tương quan ---
    print("\n2.9 Ma trận tương quan:")
    corr_full = df[SCORE_COLS + ['total_score', 'average_score']].corr().round(3)
    print(corr_full.to_string())

    # --- 2.10 Phân tích chéo ---
    print("\n2.10 Phân tích chéo - Giới tính x Khóa ôn thi:")
    cross1 = df.groupby(['gender', 'test_prep'])['average_score'].agg(
        ['mean', 'count']).round(2)
    print(cross1.to_string())

    print("\n2.11 Phân tích chéo - Giới tính x Bữa trưa:")
    cross2 = df.groupby(['gender', 'lunch'])['average_score'].agg(
        ['mean', 'count']).round(2)
    print(cross2.to_string())

    # --- 2.12 Kiểm định Chi-square ---
    print("\n2.12 Kiểm định Chi-square:")
    ct_gender_grade = pd.crosstab(df['gender'], df['xep_loai'])
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct_gender_grade)
    print(f"    Giới tính vs Xếp loại: chi2={chi2:.3f}, p={p_chi:.6f}, df={dof}")

    ct_prep_pass = pd.crosstab(df['test_prep'], df['pass_fail'])
    chi2_2, p_chi_2, dof_2, _ = stats.chi2_contingency(ct_prep_pass)
    print(f"    Khóa ôn thi vs Đạt/Không đạt: chi2={chi2_2:.3f}, "
          f"p={p_chi_2:.6f}, df={dof_2}")

    # --- 2.13 Hệ số tương quan Pearson ---
    print("\n2.13 Hệ số tương quan Pearson (các cặp điểm):")
    pairs = [('math_score', 'reading_score'),
             ('math_score', 'writing_score'),
             ('reading_score', 'writing_score')]
    for c1, c2 in pairs:
        r, p = stats.pearsonr(df[c1], df[c2])
        print(f"    {c1} vs {c2}: r={r:.4f}, p={p:.2e}")

    # Ma trận tương quan để dùng ở phần Insight
    corr_mat = df[SCORE_COLS].corr()

    return corr_mat, diff_prep
