import os
import numpy as np
import pandas as pd
import xgboost as xgb

from datetime import datetime
from packaging import version
import sklearn

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve
)

import torch  # 用 torch 保存为 .pt

# ✅ 新增：用于保存PDF图
import matplotlib.pyplot as plt

# =====================
# 配置区
# =====================
INPUT_XLSX = "./trait_train.xlsx"
LABEL_COL = "Label"

PAIR_COL = "CDR3a_CDR3b"
RAW_COLS_FOR_PAIR = ["CDR3a", "CDR3b"]
FEATURE_COLS = [PAIR_COL]

N_SPLITS = 5
RANDOM_STATE = 42

MIN_FREQUENCY = 5
MAX_CATEGORIES = None

XGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
)

CURVE_POINTS = 1001


def make_ohe():
    skl_ver = version.parse(sklearn.__version__)
    if skl_ver >= version.parse("1.1"):
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=MIN_FREQUENCY,
            max_categories=MAX_CATEGORIES,
            sparse_output=True
        )
    else:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True
        )


def mean_std(arr):
    return float(np.mean(arr)), float(np.std(arr))


# =====================
# ✅ 新增：画图并保存为 PDF
# =====================
def save_roc_pdf(fpr, tpr, auc_value, pdf_path, title_prefix="ROC"):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf")  # 保存PDF :contentReference[oaicite:2]{index=2}
    plt.close()


def save_pr_pdf(recall, precision, aupr_value, pdf_path, title_prefix="Precision-Recall"):
    plt.figure()
    plt.plot(recall, precision, label=f"AUPR={aupr_value:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf")  # 保存PDF :contentReference[oaicite:3]{index=3}
    plt.close()


def main():
    # =====================
    # 1) 读取数据
    # =====================
    df = pd.read_excel(INPUT_XLSX)

    required_cols = RAW_COLS_FOR_PAIR + [LABEL_COL]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少列：{col}")

    df[PAIR_COL] = df["CDR3a"].astype(str) + "_" + df["CDR3b"].astype(str)

    X = df[FEATURE_COLS].astype(str)
    y = df[LABEL_COL].astype(int).values

    # =====================
    # 2) 输出目录（时间戳命名）
    # =====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"xgb_models_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # =====================
    # 3) 五折交叉验证
    # =====================
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    auc_scores, aupr_scores = [], []
    precisions, recalls, f1s = [], [], []

    fpr_grid = np.linspace(0, 1, CURVE_POINTS)
    recall_grid = np.linspace(0, 1, CURVE_POINTS)
    tpr_folds = []
    pr_precision_folds = []

    # 用于总体PDF（基于插值后的平均曲线）
    # 注意：PR曲线原始 recall 是递减/递增的序列，画总体平均用插值网格更稳。
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")

        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        ohe = make_ohe()
        preprocessor = ColumnTransformer(
            transformers=[("cat", ohe, FEATURE_COLS)],
            remainder="drop",
            sparse_threshold=1.0
        )

        clf = xgb.XGBClassifier(**XGB_PARAMS)

        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", clf)
        ])

        pipe.fit(X_train, y_train)

        # 保存模型 pt
        model_path = os.path.join(fold_dir, f"xgb_pipeline_fold{fold}_{timestamp}.pt")
        payload = {
            "pipeline": pipe,
            "fold": fold,
            "timestamp": timestamp,
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__,
        }
        torch.save(payload, model_path)

        # 预测 & 指标
        y_prob = pipe.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_val, y_prob)
        aupr = average_precision_score(y_val, y_prob)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        print(f"AUC={auc:.4f}, AUPR={aupr:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        auc_scores.append(auc)
        aupr_scores.append(aupr)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        # =====================
        # 4) 计算ROC/PR曲线 + 保存npy + 保存PDF
        # =====================
        # ROC（原始点）
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        # PR（原始点）
        pr_prec, pr_rec, _ = precision_recall_curve(y_val, y_prob)

        # ——保存每折 PDF（用原始曲线点更直观）
        save_roc_pdf(
            fpr, tpr, auc,
            pdf_path=os.path.join(fold_dir, f"ROC_fold{fold}_{timestamp}.pdf"),
            title_prefix=f"ROC Curve (Fold {fold})"
        )
        save_pr_pdf(
            pr_rec, pr_prec, aupr,
            pdf_path=os.path.join(fold_dir, f"PR_fold{fold}_{timestamp}.pdf"),
            title_prefix=f"PR Curve (Fold {fold})"
        )

        # ——插值到固定网格：用于总体平均曲线
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_folds.append(tpr_interp)

        pr_prec_interp = np.interp(recall_grid, pr_rec[::-1], pr_prec[::-1])
        pr_precision_folds.append(pr_prec_interp)

        # ——保存每折 npy（你原本就需要）
        roc_payload_fold = {"fpr_grid": fpr_grid, "tpr_interp": tpr_interp, "auc": auc}
        pr_payload_fold = {"recall_grid": recall_grid, "precision_interp": pr_prec_interp, "aupr": aupr}

        np.save(os.path.join(fold_dir, f"roc_curve_payload_fold{fold}_{timestamp}.npy"),
                roc_payload_fold, allow_pickle=True)
        np.save(os.path.join(fold_dir, f"pr_curve_payload_fold{fold}_{timestamp}.npy"),
                pr_payload_fold, allow_pickle=True)

        metrics_fold = {"auc": auc, "aupr": aupr, "precision": precision, "recall": recall, "f1": f1}
        np.save(os.path.join(fold_dir, f"metrics_fold{fold}_{timestamp}.npy"),
                metrics_fold, allow_pickle=True)

    # =====================
    # 5) 保存五折平均到 .txt（总目录）
    # =====================
    auc_m, auc_s = mean_std(auc_scores)
    aupr_m, aupr_s = mean_std(aupr_scores)
    p_m, p_s = mean_std(precisions)
    r_m, r_s = mean_std(recalls)
    f1_m, f1_s = mean_std(f1s)

    txt_path = os.path.join(out_dir, f"cv_summary_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("===== XGBoost 5-Fold CV Results (CDR3a+CDR3b OneHot + infrequent merge) =====\n")
        f.write(f"timestamp: {timestamp}\n")
        f.write(f"sklearn_version: {sklearn.__version__}\n")
        f.write(f"xgboost_version: {xgb.__version__}\n")
        f.write(f"min_frequency: {MIN_FREQUENCY}\n")
        f.write(f"max_categories: {MAX_CATEGORIES}\n\n")
        f.write(f"AUC  (mean ± std): {auc_m:.6f} ± {auc_s:.6f}\n")
        f.write(f"AUPR (mean ± std): {aupr_m:.6f} ± {aupr_s:.6f}\n")
        f.write(f"Precision (mean ± std): {p_m:.6f} ± {p_s:.6f}\n")
        f.write(f"Recall    (mean ± std): {r_m:.6f} ± {r_s:.6f}\n")
        f.write(f"F1-score  (mean ± std): {f1_m:.6f} ± {f1_s:.6f}\n")

    # =====================
    # 6) 保存整体平均 ROC/PR：npy + PDF（总目录）
    # =====================
    tpr_folds = np.vstack(tpr_folds)
    pr_precision_folds = np.vstack(pr_precision_folds)

    roc_payload_all = {
        "fpr_grid": fpr_grid,
        "tpr_folds": tpr_folds,
        "tpr_mean": tpr_folds.mean(axis=0),
        "tpr_std": tpr_folds.std(axis=0),
        "auc_scores": np.array(auc_scores),
        "auc_mean": float(np.mean(auc_scores)),
    }
    pr_payload_all = {
        "recall_grid": recall_grid,
        "precision_folds": pr_precision_folds,
        "precision_mean": pr_precision_folds.mean(axis=0),
        "precision_std": pr_precision_folds.std(axis=0),
        "aupr_scores": np.array(aupr_scores),
        "aupr_mean": float(np.mean(aupr_scores)),
    }

    np.save(os.path.join(out_dir, f"roc_curve_payload_all_{timestamp}.npy"), roc_payload_all, allow_pickle=True)
    np.save(os.path.join(out_dir, f"pr_curve_payload_all_{timestamp}.npy"), pr_payload_all, allow_pickle=True)

    # ——总体PDF：用平均曲线画
    save_roc_pdf(
        roc_payload_all["fpr_grid"],
        roc_payload_all["tpr_mean"],
        roc_payload_all["auc_mean"],
        pdf_path=os.path.join(out_dir, f"ROC_mean_{timestamp}.pdf"),
        title_prefix="ROC Curve (Mean over 5 folds)"
    )
    save_pr_pdf(
        pr_payload_all["recall_grid"],
        pr_payload_all["precision_mean"],
        pr_payload_all["aupr_mean"],
        pdf_path=os.path.join(out_dir, f"PR_mean_{timestamp}.pdf"),
        title_prefix="PR Curve (Mean over 5 folds)"
    )

    print("完成！输出目录：", out_dir)
    print("每个 fold 目录下新增：ROC_foldX_*.pdf 和 PR_foldX_*.pdf")
    print("总目录下新增：ROC_mean_*.pdf 和 PR_mean_*.pdf")


if __name__ == "__main__":
    main()
