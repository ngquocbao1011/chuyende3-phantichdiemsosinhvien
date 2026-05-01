# -*- coding: utf-8 -*-
"""
Phần 4: Xây dựng mô hình (Modeling)
  4.1 Dự đoán điểm cuối kỳ bằng Regression
  4.2 Dự đoán điểm trung bình từ nhân khẩu học
  4.3 Phân loại Đạt / Không đạt
  4.4 Tổng hợp kết quả
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from utils import SCORE_COLS, save_fig


def run_modeling(df):
    """
    Huấn luyện và đánh giá các mô hình.
    Trả về (results_reg, results_clf).
    """
    print("\n\n" + "=" * 65)
    print("PHẦN 4: MODELING")
    print("=" * 65)

    # Chuẩn bị dữ liệu
    df_ml = df.copy()
    cat_cols = ['gender', 'ethnicity', 'parent_education', 'lunch', 'test_prep']
    for col in cat_cols:
        le = LabelEncoder()
        df_ml[col + '_enc'] = le.fit_transform(df_ml[col])

    features = [c + '_enc' for c in cat_cols]

    # ============================================================
    # 4.1 DỰ ĐOÁN ĐIỂM CUỐI KỲ BẰNG REGRESSION
    # ============================================================
    print("\n--- 4.1 DỰ ĐOÁN ĐIỂM CUỐI KỲ (WRITING SCORE) BẰNG REGRESSION ---")
    print("    Bài toán: dùng thông tin SV + điểm Math, Reading để dự đoán Writing")
    print("    (Writing xem như điểm thi cuối kỳ)")

    X_reg = df_ml[features + ['math_score', 'reading_score']]
    y_reg = df_ml['writing_score']
    feat_names = features + ['math_score', 'reading_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    results_reg = {}
    all_preds = {}

    # a) Linear Regression
    print("\n  a) Linear Regression:")
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)
    results_reg['Linear Regression'] = _eval_reg(y_test, y_pred_lr, lr,
                                                  scaler.transform(X_reg), y_reg)
    all_preds['Linear Reg.'] = y_pred_lr
    _print_reg_metrics(results_reg['Linear Regression'])
    print("     Hệ số hồi quy:")
    for fn, coef in zip(feat_names, lr.coef_):
        print(f"       {fn:25s}: {coef:+.4f}")
    print(f"       {'Intercept':25s}: {lr.intercept_:+.4f}")

    # b) Ridge Regression
    print("\n  b) Ridge Regression (alpha=1.0):")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_sc, y_train)
    y_pred_ridge = ridge.predict(X_test_sc)
    results_reg['Ridge Regression'] = _eval_reg(y_test, y_pred_ridge, ridge,
                                                 scaler.transform(X_reg), y_reg)
    all_preds['Ridge'] = y_pred_ridge
    _print_reg_metrics(results_reg['Ridge Regression'])

    # c) Lasso Regression
    print("\n  c) Lasso Regression (alpha=0.1):")
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_sc, y_train)
    y_pred_lasso = lasso.predict(X_test_sc)
    results_reg['Lasso Regression'] = _eval_reg(y_test, y_pred_lasso, lasso,
                                                 scaler.transform(X_reg), y_reg)
    all_preds['Lasso'] = y_pred_lasso
    _print_reg_metrics(results_reg['Lasso Regression'])

    # d) Decision Tree Regressor
    print("\n  d) Decision Tree Regressor:")
    dtr = DecisionTreeRegressor(max_depth=8, random_state=42)
    dtr.fit(X_train, y_train)
    y_pred_dtr = dtr.predict(X_test)
    results_reg['Decision Tree'] = _eval_reg(y_test, y_pred_dtr, dtr,
                                              X_reg, y_reg)
    all_preds['Decision Tree'] = y_pred_dtr
    _print_reg_metrics(results_reg['Decision Tree'])

    # e) Random Forest Regressor
    print("\n  e) Random Forest Regressor:")
    rfr = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rfr.fit(X_train, y_train)
    y_pred_rfr = rfr.predict(X_test)
    results_reg['Random Forest'] = _eval_reg(y_test, y_pred_rfr, rfr,
                                              X_reg, y_reg)
    all_preds['Random Forest'] = y_pred_rfr
    _print_reg_metrics(results_reg['Random Forest'])

    # f) Gradient Boosting Regressor
    print("\n  f) Gradient Boosting Regressor:")
    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                     learning_rate=0.08, random_state=42)
    gbr.fit(X_train, y_train)
    y_pred_gbr = gbr.predict(X_test)
    results_reg['Gradient Boosting'] = _eval_reg(y_test, y_pred_gbr, gbr,
                                                  X_reg, y_reg)
    all_preds['Gradient Boost'] = y_pred_gbr
    _print_reg_metrics(results_reg['Gradient Boosting'])

    # Bảng tổng hợp
    print("\n  --- TỔNG HỢP MÔ HÌNH HỒI QUY ---")
    print(f"  {'Mô hình':25s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s} {'CV_R2':>8s}")
    print(f"  {'-'*57}")
    for name, m in results_reg.items():
        print(f"  {name:25s} {m['MAE']:8.4f} {m['RMSE']:8.4f} "
              f"{m['R2']:8.4f} {m['CV_R2']:8.4f}")

    best_reg = max(results_reg, key=lambda k: results_reg[k]['R2'])
    print(f"\n  => Mô hình tốt nhất: {best_reg} "
          f"(R2 = {results_reg[best_reg]['R2']:.4f})")

    # Biểu đồ Actual vs Predicted
    _plot_actual_vs_predicted(y_test, all_preds)

    # Biểu đồ Residuals
    _plot_residuals(y_test, all_preds)

    # Feature Importance (Gradient Boosting)
    fig, ax = plt.subplots(figsize=(9, 5))
    imp = pd.Series(gbr.feature_importances_, index=feat_names).sort_values()
    imp.plot(kind='barh', color='#2980b9', edgecolor='white', ax=ax)
    ax.set_title('Độ quan trọng đặc trưng (Gradient Boosting Regression)',
                 fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    save_fig("feature_importance_regression")

    # ============================================================
    # 4.2 DỰ ĐOÁN ĐIỂM TRUNG BÌNH TỪ NHÂN KHẨU HỌC
    # ============================================================
    print("\n\n--- 4.2 DỰ ĐOÁN ĐIỂM TRUNG BÌNH TỪ ĐẶC ĐIỂM NHÂN KHẨU HỌC ---")
    print("    Chỉ dùng: gender, ethnicity, parent_education, lunch, test_prep")

    X_avg = df_ml[features]
    y_avg = df_ml['average_score']
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X_avg, y_avg, test_size=0.2, random_state=42)

    avg_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8,
                                                random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                        learning_rate=0.1,
                                                        random_state=42)
    }
    avg_preds = {}
    print(f"  {'Mô hình':25s} {'RMSE':>8s} {'R2':>8s}")
    print(f"  {'-'*45}")
    for name, model in avg_models.items():
        model.fit(X_tr2, y_tr2)
        yp = model.predict(X_te2)
        r2 = r2_score(y_te2, yp)
        rmse = np.sqrt(mean_squared_error(y_te2, yp))
        print(f"  {name:25s} {rmse:8.4f} {r2:8.4f}")
        avg_preds[name] = yp

    print("  => R2 thấp -> các yếu tố nhân khẩu học chỉ giải thích")
    print("     một phần nhỏ kết quả học tập.")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (name, yp) in zip(axes, avg_preds.items()):
        ax.scatter(y_te2, yp, alpha=0.4, s=25, edgecolors='gray', linewidth=0.3)
        ax.plot([y_te2.min(), y_te2.max()],
                [y_te2.min(), y_te2.max()], 'r--', lw=1.5)
        ax.set_title(f'{name}\nR2={r2_score(y_te2, yp):.4f}', fontweight='bold')
        ax.set_xlabel('Thực tế')
        ax.set_ylabel('Dự đoán')
    plt.suptitle('Dự đoán điểm TB chỉ từ đặc điểm nhân khẩu học',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig("regression_average_nhan_khau")

    # ============================================================
    # 4.3 PHÂN LOẠI ĐẠT / KHÔNG ĐẠT
    # ============================================================
    print("\n\n--- 4.3 PHÂN LOẠI ĐẠT / KHÔNG ĐẠT ---")
    print("    Ngưỡng: average_score >= 50 => Đạt")

    X_clf = df_ml[features]
    y_clf = (df_ml['average_score'] >= 50).astype(int)
    print(f"    Phân bố: Đạt={y_clf.sum()}, Không đạt={len(y_clf)-y_clf.sum()}")

    X_tc, X_ec, y_tc, y_ec = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    models_clf = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=6,
                                                 random_state=42)
    }

    results_clf = {}
    predictions_clf = {}
    for name, model in models_clf.items():
        model.fit(X_tc, y_tc)
        yp = model.predict(X_ec)
        acc = accuracy_score(y_ec, yp)
        cv_acc = cross_val_score(model, X_clf, y_clf, cv=5, scoring='accuracy')
        results_clf[name] = {'accuracy': acc, 'cv_mean': cv_acc.mean(),
                              'cv_std': cv_acc.std()}
        predictions_clf[name] = yp
        print(f"\n  {name}:")
        print(f"    Accuracy = {acc:.4f}")
        print(f"    CV Accuracy (5-fold) = {cv_acc.mean():.4f} "
              f"+/- {cv_acc.std():.4f}")
        print(classification_report(y_ec, yp,
              target_names=['Không đạt', 'Đạt'], zero_division=0))

    # Confusion Matrix
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for i, (name, yp) in enumerate(predictions_clf.items()):
        cm = confusion_matrix(y_ec, yp)
        ConfusionMatrixDisplay(cm, display_labels=['Không đạt', 'Đạt']).plot(
            ax=axes[i], cmap='Blues', colorbar=False)
        axes[i].set_title(name, fontweight='bold')
    plt.suptitle('Confusion Matrix - Phân loại Đạt/Không đạt',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig("confusion_matrix")

    # So sánh accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    names_c = list(results_clf.keys())
    accs_c = [results_clf[n]['accuracy'] for n in names_c]
    cv_accs = [results_clf[n]['cv_mean'] for n in names_c]
    x_pos = np.arange(len(names_c))
    w = 0.35
    b1 = ax.bar(x_pos - w / 2, accs_c, w, label='Test Accuracy',
                color='#3498db', edgecolor='white')
    b2 = ax.bar(x_pos + w / 2, cv_accs, w, label='CV Accuracy',
                color='#2ecc71', edgecolor='white')
    for b in b1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f'{b.get_height():.3f}', ha='center', fontsize=9,
                fontweight='bold')
    for b in b2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f'{b.get_height():.3f}', ha='center', fontsize=9,
                fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names_c, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Accuracy')
    ax.set_title('So sánh mô hình phân loại', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_fig("so_sanh_classification")

    # Feature importance (RF)
    fig, ax = plt.subplots(figsize=(8, 5))
    imp_clf = pd.Series(
        models_clf['Random Forest'].feature_importances_, index=features
    ).sort_values()
    imp_clf.plot(kind='barh', color='#e74c3c', edgecolor='white', ax=ax)
    ax.set_title('Độ quan trọng đặc trưng (RF Classification)', fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    save_fig("feature_importance_classification")

    # ============================================================
    # 4.4 TỔNG HỢP SO SÁNH
    # ============================================================
    print("\n\n--- 4.4 TỔNG HỢP KẾT QUẢ MODELING ---")
    print("\n  MÔ HÌNH HỒI QUY (dự đoán điểm Writing):")
    print(f"  {'Mô hình':25s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s} {'CV_R2':>8s}")
    print(f"  {'='*57}")
    for name, m in results_reg.items():
        print(f"  {name:25s} {m['MAE']:8.4f} {m['RMSE']:8.4f} "
              f"{m['R2']:8.4f} {m['CV_R2']:8.4f}")

    print(f"\n  MÔ HÌNH PHÂN LOẠI (Đạt/Không đạt):")
    print(f"  {'Mô hình':25s} {'Accuracy':>10s} {'CV Accuracy':>12s}")
    print(f"  {'='*50}")
    for name, m in results_clf.items():
        print(f"  {name:25s} {m['accuracy']:10.4f} {m['cv_mean']:10.4f} "
              f"+/- {m['cv_std']:.4f}")

    # So sánh R2
    fig, ax = plt.subplots(figsize=(10, 5))
    reg_names = list(results_reg.keys())
    r2_vals = [results_reg[n]['R2'] for n in reg_names]
    cv_vals = [results_reg[n]['CV_R2'] for n in reg_names]
    x_p = np.arange(len(reg_names))
    w = 0.35
    ax.bar(x_p - w / 2, r2_vals, w, label='Test R2',
           color='#3498db', edgecolor='white')
    ax.bar(x_p + w / 2, cv_vals, w, label='CV R2',
           color='#f39c12', edgecolor='white')
    ax.set_xticks(x_p)
    ax.set_xticklabels(reg_names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('R2 Score')
    ax.set_title('So sánh R2 các mô hình hồi quy', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_fig("so_sanh_r2_regression")

    return results_reg, results_clf


# ----------------------------------------------------------------
# Hàm hỗ trợ nội bộ
# ----------------------------------------------------------------

def _eval_reg(y_true, y_pred, model, X_cv, y_cv):
    """Tính các metric hồi quy."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    cv = cross_val_score(model, X_cv, y_cv, cv=5, scoring='r2')
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'CV_R2': cv.mean(),
            'CV_std': cv.std()}


def _print_reg_metrics(m):
    """In metric hồi quy."""
    print(f"     MAE  = {m['MAE']:.4f}")
    print(f"     RMSE = {m['RMSE']:.4f}")
    print(f"     R2   = {m['R2']:.4f}")
    print(f"     CV R2 (5-fold) = {m['CV_R2']:.4f} +/- {m['CV_std']:.4f}")


def _plot_actual_vs_predicted(y_test, preds_dict):
    """Biểu đồ Actual vs Predicted cho tất cả mô hình."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors_reg = ['#3498db', '#9b59b6', '#1abc9c', '#e67e22', '#27ae60', '#e74c3c']
    for idx, (name, yp) in enumerate(preds_dict.items()):
        ax = axes[idx // 3][idx % 3]
        ax.scatter(y_test, yp, alpha=0.45, color=colors_reg[idx],
                   edgecolors='gray', s=25, linewidth=0.3)
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'r--', lw=1.5)
        r2_val = r2_score(y_test, yp)
        ax.set_title(f'{name}\nR2={r2_val:.4f}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Thực tế')
        ax.set_ylabel('Dự đoán')
    plt.suptitle('Actual vs Predicted - Dự đoán điểm Writing (cuối kỳ)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig("regression_actual_vs_predicted")


def _plot_residuals(y_test, preds_dict):
    """Biểu đồ phân dư (Residual Analysis)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors_reg = ['#3498db', '#9b59b6', '#1abc9c', '#e67e22', '#27ae60', '#e74c3c']
    for idx, (name, yp) in enumerate(preds_dict.items()):
        ax = axes[idx // 3][idx % 3]
        residuals = y_test.values - yp
        ax.scatter(yp, residuals, alpha=0.4, color=colors_reg[idx],
                   edgecolors='gray', s=25, linewidth=0.3)
        ax.axhline(0, color='red', ls='--', lw=1.5)
        ax.set_title(f'{name}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Giá trị dự đoán')
        ax.set_ylabel('Residual')
    plt.suptitle('Phân tích phần dư (Residual Analysis)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig("regression_residuals")
