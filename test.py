import os
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.ndimage import gaussian_filter1d  # [新增] 用于曲线平滑
from scipy.interpolate import make_interp_spline # [新增] 另一种平滑方式

# ==========================================
# [全局设置] 绘图风格
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_theme(style="white", font="Arial")
except ImportError:
    HAS_SEABORN = False
    print("提示: 未检测到 seaborn 库，将使用 matplotlib 默认配色。")

from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
)

# =====================
# 配置区
# =====================
MODEL_ROOT = "/mnt/data/caobf/XGboost/model_comparison_20260112_132837"
TEST_XLSX  = "./health.xlsx"
LABEL_COL = "Label"

GENE_COLS = ["TRAV", "TRAJ", "TRBV", "TRBJ"]
CDR3_COLS = ["CDR3a", "CDR3b"]
FEATURE_COLS = GENE_COLS + CDR3_COLS 
MAX_CDR3_LEN = 25 

CURVE_POINTS = 1001
THRESHOLD = 0.5

SMOOTH_SIGMA = 2.0 

# =====================
# 辅助类
# =====================
class SequenceSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, cols, max_len=20, padding_char='X'):
        self.cols = cols
        self.max_len = max_len
        self.padding_char = padding_char

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        dfs = []
        for col in self.cols:
            if col not in X.columns:
                 continue
            seqs = X[col].fillna('').astype(str)
            split_data = []
            for seq in seqs:
                if len(seq) > self.max_len:
                    processed_seq = seq[:self.max_len]
                else:
                    processed_seq = seq.ljust(self.max_len, self.padding_char)
                split_data.append(list(processed_seq))
            col_names = [f"{col}_Pos{i+1}" for i in range(self.max_len)]
            dfs.append(pd.DataFrame(split_data, columns=col_names, index=X.index))
        return pd.concat(dfs, axis=1)

def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(width=1.5, labelsize=12)
    ax.grid(False)

def main():
    # 1) 读取测试集
    if not os.path.exists(TEST_XLSX):
        raise FileNotFoundError(f"找不到测试集文件: {TEST_XLSX}")
        
    df = pd.read_excel(TEST_XLSX)
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"测试集缺少列：{missing_cols}")

    X_raw = df[FEATURE_COLS]

    has_label = (LABEL_COL in df.columns)
    if has_label:
        y_test = df[LABEL_COL].astype(int).values
    else:
        y_test = None
        print("⚠️ 测试集没有 Label 列：将只输出预测概率，不计算指标。")

    print("正在对测试集进行序列分割处理...")
    splitter = SequenceSplitter(cols=CDR3_COLS, max_len=MAX_CDR3_LEN)
    X_cdr3_split = splitter.transform(X_raw[CDR3_COLS])
    X_test_final = pd.concat([X_raw[GENE_COLS], X_cdr3_split], axis=1)

    # 2) 搜索 .pt 文件
    pt_files = sorted(glob.glob(os.path.join(MODEL_ROOT, "xgboost_fold_*.pt")))
    
    if not pt_files:
        raise FileNotFoundError(f"在 {MODEL_ROOT} 下没找到 'xgboost_fold_*.pt' 文件")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    probs_all_folds = []
    metrics_rows = []
    fold_prob_cols = {}

    print(f"找到 {len(pt_files)} 个模型文件，开始预测...")

    for pt_path in pt_files:
        filename = os.path.basename(pt_path)
        try:
            fold_id = filename.split('_')[-1].replace('.pt', '')
        except:
            fold_id = filename

        print(f">>> 正在处理模型: {filename} ...")

        payload = safe_torch_load(pt_path)
        model = payload["model"]
        ohe = payload["preprocessor"]

        try:
            X_test_encoded = ohe.transform(X_test_final)
        except Exception as e:
            print(f"编码出错 (Fold {fold_id}): {e}")
            raise e

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_encoded)[:, 1]
        else:
            y_prob = model.decision_function(X_test_encoded)
        
        probs_all_folds.append(y_prob)
        fold_prob_cols[f"prob_fold{fold_id}"] = y_prob
        
        if has_label:
            y_pred = (y_prob >= THRESHOLD).astype(int)
            auc = roc_auc_score(y_test, y_prob)
            aupr = average_precision_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            metrics = {"fold": fold_id, "auc": auc, "aupr": aupr, "f1": f1}
            metrics_rows.append(metrics)

    # 3) TREX (Average)
    probs_all_folds = np.vstack(probs_all_folds)
    y_prob_trex = probs_all_folds.mean(axis=0)
    y_pred_trex = (y_prob_trex >= THRESHOLD).astype(int)

    np.save(os.path.join(MODEL_ROOT, f"test_probs_TREX_{timestamp}.npy"), y_prob_trex)

    df_out = df.copy()
    for col, arr in fold_prob_cols.items():
        df_out[col] = arr

    df_out["prob_TREX"] = y_prob_trex
    df_out["pred_TREX"] = y_pred_trex

    if has_label:
        df_out["correct_TREX"] = (df_out[LABEL_COL].astype(int).values == y_pred_trex).astype(int)

    pred_xlsx_path = os.path.join(MODEL_ROOT, f"test_predictions_{timestamp}.xlsx")
    df_out.to_excel(pred_xlsx_path, index=False)
    print(f"\n✅ 预测结果已保存: {pred_xlsx_path}")

    # =========================
    # 绘图与最终评估 (TREX)
    # =========================
    if has_label:
        print("\n正在绘制测试集结果图表...")
        
        # --- [修改开始] 计算并打印详细指标 ---
        trex_auc = roc_auc_score(y_test, y_prob_trex)
        trex_aupr = average_precision_score(y_test, y_prob_trex)
        
        # 新增 Precision 和 Recall 计算
        trex_precision = precision_score(y_test, y_pred_trex, zero_division=0)
        trex_recall = recall_score(y_test, y_pred_trex, zero_division=0)
        trex_f1 = f1_score(y_test, y_pred_trex, zero_division=0)
        
        print("-" * 60)
        print(f" >>> TREX 最终效果评估 (Threshold={THRESHOLD}):")
        print(f"     AUC       : {trex_auc:.4f}")
        print(f"     AUPR      : {trex_aupr:.4f}")
        print(f"     Precision : {trex_precision:.4f}")
        print(f"     Recall    : {trex_recall:.4f}")
        print(f"     F1-score  : {trex_f1:.4f}")
        print("-" * 60)

        if len(metrics_rows) > 0:
            metrics_df = pd.DataFrame(metrics_rows)
            with pd.ExcelWriter(pred_xlsx_path, mode="a", engine="openpyxl") as writer:
                metrics_df.to_excel(writer, sheet_name="fold_metrics", index=False)

        # =======================================================
        # [修改] 使用高斯滤波 (Gaussian Filter) 让曲线变圆滑
        # =======================================================

        # --- Plot 1: ROC Curve (TREX) ---
        fpr, tpr, _ = roc_curve(y_test, y_prob_trex)
        
        # 1. 简单插值（如果点太少）
        if len(fpr) < 50: 
            # 如果点很少，先进行线性插值增加点数
            new_fpr = np.linspace(0, 1, 200)
            new_tpr = np.interp(new_fpr, fpr, tpr)
            fpr, tpr = new_fpr, new_tpr
            
        # 2. 高斯平滑
        tpr_smooth = gaussian_filter1d(tpr, sigma=SMOOTH_SIGMA)
        # 修正：保证平滑后不超过0和1，且单调递增
        tpr_smooth = np.clip(tpr_smooth, 0, 1)
        tpr_smooth = np.maximum.accumulate(tpr_smooth) # 保证单调递增（符合ROC特性）

        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.5)
        color_roc = sns.color_palette("bright")[0] if HAS_SEABORN else '#1f77b4'
        
        # 绘制平滑后的曲线
        plt.plot(fpr, tpr_smooth, color=color_roc, lw=3, label=f'TREX (AUC={trex_auc:.3f})')
        
        style_axis(plt.gca())
        plt.xlabel('False Positive Rate', fontweight='bold', fontsize=14)
        plt.ylabel('True Positive Rate', fontweight='bold', fontsize=14)
        plt.title('Test ROC Curve (TREX)', fontweight='bold', fontsize=16)
        plt.legend(loc="lower right", frameon=False, fontsize=12)
        plt.savefig(os.path.join(MODEL_ROOT, f"Test_ROC_TREX_{timestamp}.pdf"), bbox_inches='tight')
        plt.close()

        # --- Plot 2: PR Curve (TREX) ---
        precision, recall, _ = precision_recall_curve(y_test, y_prob_trex)
        
        # PR曲线的顺序通常是 Recall 递减，Precision 递增/震荡
        # 为了平滑，我们最好对 Recall 进行排序
        sorted_indices = np.argsort(recall)
        recall_sorted = recall[sorted_indices]
        precision_sorted = precision[sorted_indices]
        
        # 1. 简单插值
        if len(recall_sorted) < 50:
            new_recall = np.linspace(0, 1, 200)
            new_precision = np.interp(new_recall, recall_sorted, precision_sorted)
            recall_sorted, precision_sorted = new_recall, new_precision

        # 2. 高斯平滑
        precision_smooth = gaussian_filter1d(precision_sorted, sigma=SMOOTH_SIGMA)
        precision_smooth = np.clip(precision_smooth, 0, 1)

        plt.figure(figsize=(8, 8))
        color_pr = sns.color_palette("bright")[1] if HAS_SEABORN else '#ff7f0e'
        
        # 绘制平滑后的曲线
        plt.plot(recall_sorted, precision_smooth, color=color_pr, lw=3, label=f'TREX (AUPR={trex_aupr:.3f})')
        
        style_axis(plt.gca())
        plt.xlabel('Recall', fontweight='bold', fontsize=14)
        plt.ylabel('Precision', fontweight='bold', fontsize=14)
        plt.title('Test PR Curve (TREX)', fontweight='bold', fontsize=16)
        plt.legend(loc="lower left", frameon=False, fontsize=12)
        plt.savefig(os.path.join(MODEL_ROOT, f"Test_PR_TREX_{timestamp}.pdf"), bbox_inches='tight')
        plt.close()

        # --- Plot 3: Confusion Matrix ---
        cm = confusion_matrix(y_test, y_pred_trex)
        
        print("\n" + "="*40)
        print(">>> 混淆矩阵 (TREX) 文字版:")
        print(f"[[TN, FP],\n [FN, TP]]")
        print("-" * 20)
        print(cm)
        print("-" * 20)
        print("TN (True Negative):", cm[0, 0])
        print("FP (False Positive):", cm[0, 1])
        print("FN (False Negative):", cm[1, 0])
        print("TP (True Positive): ", cm[1, 1])
        print("="*40 + "\n")

        plt.figure(figsize=(7, 6))
        if HAS_SEABORN:
            ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                             annot_kws={"size": 16, "weight": "bold"},
                             cbar=False)
            ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=14)
            ax.set_ylabel('True Label', fontweight='bold', fontsize=14)
            ax.set_title('Confusion Matrix (TREX)', fontweight='bold', fontsize=16)
        else:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues', values_format='d', ax=plt.gca(), colorbar=False)
            plt.title('Confusion Matrix (TREX)', fontweight='bold', fontsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_ROOT, f"Test_ConfusionMatrix_TREX_{timestamp}.pdf"), bbox_inches='tight')
        plt.close()

        print(f"\n全部图表已保存至: {MODEL_ROOT}")

if __name__ == "__main__":
    main()