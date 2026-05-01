# -*- coding: utf-8 -*-
"""
Phần 3: Trực quan hóa dữ liệu (Visualization)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from utils import SCORE_COLS, EDU_ORDER, XL_ORDER, save_fig


def run_visualization(df):
    """Tạo toàn bộ biểu đồ trực quan hóa."""

    print("\n\n" + "=" * 65)
    print("PHẦN 3: TRỰC QUAN HÓA")
    print("=" * 65)

    colors = ['#2980b9', '#27ae60', '#c0392b']

    # --- 3.1 Phân bố điểm từng môn (histogram + KDE) ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for i, (col, c) in enumerate(zip(SCORE_COLS, colors)):
        sns.histplot(df[col], bins=25, kde=True, color=c, ax=axes[i],
                     edgecolor='white', alpha=0.8)
        mu = df[col].mean()
        axes[i].axvline(mu, color='black', ls='--', lw=1.5,
                        label=f'TB = {mu:.1f}')
        axes[i].axvline(df[col].median(), color='orange', ls=':', lw=1.5,
                        label=f'Trung vị = {df[col].median():.0f}')
        axes[i].set_title(col.replace('_', ' ').title(), fontweight='bold')
        axes[i].set_xlabel('Điểm')
        axes[i].set_ylabel('Tần suất')
        axes[i].legend(fontsize=9)
    plt.suptitle('Phân bố điểm theo từng môn', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig("phan_bo_diem_tung_mon")

    # --- 3.2 Phân bố điểm trung bình ---
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['average_score'], bins=30, kde=True, color='#8e44ad',
                 edgecolor='white', ax=ax)
    ax.axvline(df['average_score'].mean(), color='red', ls='--', lw=2,
               label=f"TB = {df['average_score'].mean():.1f}")
    ax.axvline(50, color='black', ls='-', lw=1.5, label='Ngưỡng đạt (50)')
    ax.set_title('Phân bố điểm trung bình', fontweight='bold', fontsize=13)
    ax.set_xlabel('Điểm trung bình')
    ax.set_ylabel('Tần suất')
    ax.legend()
    plt.tight_layout()
    save_fig("phan_bo_diem_trung_binh")

    # --- 3.3 Boxplot so sánh 3 môn ---
    fig, ax = plt.subplots(figsize=(8, 6))
    melted = df[SCORE_COLS].melt(var_name='Môn', value_name='Điểm')
    melted['Môn'] = melted['Môn'].map({
        'math_score': 'Math', 'reading_score': 'Reading',
        'writing_score': 'Writing'
    })
    sns.boxplot(data=melted, x='Môn', y='Điểm', palette=colors, ax=ax, width=0.5)
    ax.set_title('So sánh phân bố điểm giữa 3 môn', fontweight='bold')
    ax.set_ylabel('Điểm')
    plt.tight_layout()
    save_fig("boxplot_so_sanh_3_mon")

    # --- 3.4 Boxplot điểm theo giới tính ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, (col, lbl) in enumerate(zip(SCORE_COLS, ['Math', 'Reading', 'Writing'])):
        sns.boxplot(data=df, x='gender', y=col, ax=axes[i],
                    palette=['#FF69B4', '#4169E1'], width=0.5)
        axes[i].set_title(f'{lbl} theo giới tính', fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Điểm')
    plt.suptitle('Điểm theo giới tính', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig("boxplot_diem_theo_gioi_tinh")

    # --- 3.5 Điểm theo nhóm (lớp) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    g_avg = df.groupby('ethnicity')['average_score'].mean().sort_values()
    bars = ax.barh(g_avg.index, g_avg.values,
                   color=sns.color_palette('viridis', len(g_avg)),
                   edgecolor='white', height=0.6)
    for bar, v in zip(bars, g_avg.values):
        ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{v:.1f}', va='center', fontweight='bold', fontsize=10)
    ax.set_xlabel('Điểm trung bình')
    ax.set_title('Điểm trung bình theo nhóm (lớp)', fontweight='bold')
    plt.tight_layout()
    save_fig("diem_theo_nhom")

    # --- 3.6 Điểm từng môn theo nhóm ---
    fig, ax = plt.subplots(figsize=(12, 6))
    gm = df.groupby('ethnicity')[SCORE_COLS].mean().reindex(
        ['group A', 'group B', 'group C', 'group D', 'group E'])
    gm.columns = ['Math', 'Reading', 'Writing']
    gm.plot(kind='bar', ax=ax, width=0.75, edgecolor='white',
            color=['#2980b9', '#27ae60', '#c0392b'])
    ax.set_title('Điểm trung bình từng môn theo nhóm', fontweight='bold')
    ax.set_xlabel('Nhóm')
    ax.set_ylabel('Điểm')
    ax.legend(title='Môn')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    save_fig("diem_tung_mon_theo_nhom")

    # --- 3.7 Biểu đồ trình độ phụ huynh ---
    fig, ax = plt.subplots(figsize=(12, 6))
    edu_avg = df.groupby('parent_education')[SCORE_COLS].mean().reindex(EDU_ORDER)
    edu_avg.columns = ['Math', 'Reading', 'Writing']
    edu_avg.plot(kind='bar', ax=ax, width=0.75, edgecolor='white',
                 color=['#2980b9', '#27ae60', '#c0392b'])
    ax.set_title('Điểm theo trình độ học vấn phụ huynh', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Điểm trung bình')
    ax.legend(title='Môn')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    plt.tight_layout()
    save_fig("diem_theo_trinh_do_phu_huynh")

    # --- 3.8 Ảnh hưởng khóa ôn thi ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, (col, lbl) in enumerate(zip(SCORE_COLS, ['Math', 'Reading', 'Writing'])):
        sns.boxplot(data=df, x='test_prep', y=col, ax=axes[i],
                    palette=['#2ecc71', '#e74c3c'], width=0.5)
        axes[i].set_title(f'{lbl}', fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Điểm')
    plt.suptitle('Ảnh hưởng của khóa ôn thi đến điểm số',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig("anh_huong_khoa_on_thi")

    # --- 3.9 Violin plot bữa trưa ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, (col, lbl) in enumerate(zip(SCORE_COLS, ['Math', 'Reading', 'Writing'])):
        sns.violinplot(data=df, x='lunch', y=col, ax=axes[i],
                       palette=['#27ae60', '#c0392b'], inner='quartile')
        axes[i].set_title(f'{lbl}', fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Điểm')
    plt.suptitle('Điểm theo chế độ bữa trưa', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig("violin_bua_trua")

    # --- 3.10 Heatmap tương quan ---
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_mat = df[SCORE_COLS].corr()
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    sns.heatmap(corr_mat, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, linewidths=1.5,
                square=True, ax=ax, mask=mask,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Ma trận tương quan giữa các điểm số', fontweight='bold')
    plt.tight_layout()
    save_fig("heatmap_tuong_quan")

    # --- 3.11 Heatmap tương quan mở rộng ---
    fig, ax = plt.subplots(figsize=(10, 8))
    df_corr = df[SCORE_COLS + ['average_score']].copy()
    df_corr['gender_enc'] = LabelEncoder().fit_transform(df['gender'])
    df_corr['lunch_enc'] = LabelEncoder().fit_transform(df['lunch'])
    df_corr['test_prep_enc'] = LabelEncoder().fit_transform(df['test_prep'])
    corr_ext = df_corr.corr()
    sns.heatmap(corr_ext, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, linewidths=0.8, ax=ax, square=True)
    ax.set_title('Ma trận tương quan mở rộng (gồm biến mã hóa)',
                 fontweight='bold')
    plt.tight_layout()
    save_fig("heatmap_mo_rong")

    # --- 3.12 Scatter Math vs Reading ---
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(df['math_score'], df['reading_score'],
                    c=df['writing_score'], cmap='RdYlGn', alpha=0.55,
                    edgecolors='gray', linewidth=0.3, s=35)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label('Writing Score')
    z = np.polyfit(df['math_score'], df['reading_score'], 1)
    p_line = np.poly1d(z)
    xr = np.linspace(df['math_score'].min(), df['math_score'].max(), 100)
    ax.plot(xr, p_line(xr), 'r--', lw=2,
            label=f'y = {z[0]:.2f}x + {z[1]:.1f}')
    ax.set_xlabel('Math Score')
    ax.set_ylabel('Reading Score')
    ax.set_title('Tương quan Math - Reading (màu = Writing)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_fig("scatter_math_vs_reading")

    # --- 3.13 Pair plot ---
    pair_df = df[SCORE_COLS + ['gender']].copy()
    g = sns.pairplot(pair_df, hue='gender', palette=['#FF69B4', '#4169E1'],
                     diag_kind='kde', plot_kws={'alpha': 0.45, 's': 25})
    g.figure.suptitle('Pair Plot các điểm số theo giới tính',
                      y=1.02, fontweight='bold', fontsize=13)
    save_fig("pair_plot")

    # --- 3.14 Pie chart xếp loại ---
    fig, ax = plt.subplots(figsize=(8, 8))
    xl_data = df['xep_loai'].value_counts().reindex(
        [g for g in XL_ORDER if g in df['xep_loai'].unique()])
    pie_colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c', '#95a5a6']
    wedges, texts, autotexts = ax.pie(
        xl_data.values, labels=xl_data.index, autopct='%1.1f%%',
        colors=pie_colors[:len(xl_data)],
        explode=[0.04] * len(xl_data), startangle=140,
        pctdistance=0.82, textprops={'fontsize': 11})
    for t in autotexts:
        t.set_fontweight('bold')
    ax.set_title('Phân bố xếp loại học lực', fontweight='bold', fontsize=13)
    plt.tight_layout()
    save_fig("pie_xep_loai")

    # --- 3.15 Count plot: SV theo nhóm và giới tính ---
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x='ethnicity', hue='gender',
                  palette=['#FF69B4', '#4169E1'], ax=ax, edgecolor='white')
    ax.set_title('Số lượng sinh viên theo nhóm và giới tính', fontweight='bold')
    ax.set_xlabel('Nhóm')
    ax.set_ylabel('Số lượng')
    ax.legend(title='Giới tính')
    plt.tight_layout()
    save_fig("countplot_nhom_gioi_tinh")

    # --- 3.16 KDE overlay 3 môn ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, lbl, c in zip(SCORE_COLS, ['Math', 'Reading', 'Writing'], colors):
        sns.kdeplot(df[col], ax=ax, color=c, lw=2.5, label=lbl,
                    fill=True, alpha=0.15)
    ax.set_title('So sánh phân bố mật độ 3 môn', fontweight='bold')
    ax.set_xlabel('Điểm')
    ax.set_ylabel('Mật độ')
    ax.legend()
    plt.tight_layout()
    save_fig("kde_3_mon")

    # --- 3.17 Stacked bar: xếp loại theo giới tính ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ct = pd.crosstab(df['xep_loai'], df['gender'])
    ct = ct.reindex([g for g in XL_ORDER if g in ct.index])
    ct.plot(kind='barh', stacked=True, ax=ax,
            color=['#FF69B4', '#4169E1'], edgecolor='white')
    ax.set_title('Xếp loại học lực theo giới tính', fontweight='bold')
    ax.set_xlabel('Số lượng sinh viên')
    ax.set_ylabel('')
    ax.legend(title='Giới tính')
    plt.tight_layout()
    save_fig("stacked_xeploai_gioitinh")

    # --- 3.18 Grouped: test_prep + lunch ---
    fig, ax = plt.subplots(figsize=(9, 6))
    cross_avg = df.groupby(['test_prep', 'lunch'])['average_score'].mean().unstack()
    cross_avg.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'],
                   edgecolor='white', width=0.6)
    ax.set_title('Điểm TB theo khóa ôn thi và bữa trưa', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Điểm trung bình')
    ax.set_xticklabels(['Đã hoàn thành', 'Không tham gia'], rotation=0)
    ax.legend(title='Bữa trưa')
    plt.tight_layout()
    save_fig("grouped_testprep_lunch")
