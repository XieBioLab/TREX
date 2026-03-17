import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# ==========================================
# [绘图设置: Arial + 无网格]
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.grid'] = False # 强制关闭网格

try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_theme(style="white", font="Arial")
except ImportError:
    HAS_SEABORN = False

from datetime import datetime
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    roc_curve, roc_auc_score, 
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score
)

# 防止无图形界面报错
plt.switch_backend('Agg')

# =====================
# 配置区
# =====================
INPUT_XLSX = "./trait_train_cleaned.xlsx"
LABEL_COL = "Label"

MAX_CDR3_LEN = 25 
N_SPLITS = 5
RANDOM_STATE = 42
CURVE_POINTS = 1001

# 定义对比实验组 (字典顺序即图例顺序)
# 这里的 List 代表特征的拼接顺序
EXPERIMENTS = {
    "CDR3b Only": ["CDR3b"],
    "Dual CDR3 (a+b)": ["CDR3a", "CDR3b"],
    "Full Chain": ["TRAV", "CDR3a", "TRAJ", "TRBV", "CDR3b", "TRBJ"]
}

# 区分序列列和类别列
SEQ_COLS_SET = {"CDR3a", "CDR3b"}

# 颜色映射 (绿 -> 蓝 -> 红)
COLORS = ['#2ca02c', '#1f77b4', '#d62728']

# XGBoost 参数
XGB_PARAMS = dict(
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=6, 
    subsample=0.8, 
    colsample_bytree=0.6, 
    reg_lambda=1.0, 
    eval_metric="logloss", 
    random_state=RANDOM_STATE, 
    n_jobs=-1
)

# =====================
# 辅助类
# =====================
class SequenceSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, col, max_len=20, padding_char='X'):
        self.col = col
        self.max_len = max_len
        self.padding_char = padding_char

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        seqs = X[self.col].fillna('').astype(str)
        split_data = []
        for seq in seqs:
            if len(seq) > self.max_len:
                processed_seq = seq[:self.max_len]
            else:
                processed_seq = seq.ljust(self.max_len, self.padding_char)
            split_data.append(list(processed_seq))
        
        col_names = [f"{self.col}_Pos{i+1}" for i in range(self.max_len)]
        return pd.DataFrame(split_data, columns=col_names, index=X.index)

def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(width=1.5, labelsize=12)
    ax.grid(False)

def main():
    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f"找不到文件: {INPUT_XLSX}")
    df = pd.read_excel(INPUT_XLSX)
    y = df[LABEL_COL].astype(int).values
    
    # 准备输出
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"ablation_onehot_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    results = {}
    bar_metrics = []
    
    fpr_grid = np.linspace(0, 1, CURVE_POINTS)
    recall_grid = np.linspace(0, 1, CURVE_POINTS)

    # ==========================================
    # 循环实验组
    # ==========================================
    for exp_name, cols_to_use in EXPERIMENTS.items():
        print(f"\n>>> Running Experiment: {exp_name} ...")
        print(f"    Features: {cols_to_use}")
        
        # 1. 动态构建特征矩阵
        feature_blocks = []
        for col in cols_to_use:
            # 检查列是否存在
            if col not in df.columns:
                print(f"    Warning: Column {col} not found, skipping.")
                continue
                
            if col in SEQ_COLS_SET:
                # 序列 -> 拆解
                splitter = SequenceSplitter(col=col, max_len=MAX_CDR3_LEN)
                block = splitter.transform(df)
            else:
                # 基因 -> 保持原样 (转字符串)
                block = df[[col]].fillna('Unknown').astype(str)
            
            feature_blocks.append(block)
        
        if not feature_blocks:
            print("    No features available for this set.")
            continue
            
        # 按顺序拼接 DataFrame
        X_df_combined = pd.concat(feature_blocks, axis=1)
        
        # 2. One-Hot 编码 (Train/Val Split 会在 CV 内部做吗？为了方便，这里全量 fit_transform)
        # 注意：为了防止 Data Leakage (验证集出现新类别)，这里应该在 CV 循环内 fit
        # 但 OneHotEncoder 的 handle_unknown='ignore' 可以处理这个问题
        # 为了代码简洁且特征对齐，我们在循环外先 transform 为 dense/sparse 矩阵
        # (注：如果内存不够，可换成 sparse=True，XGBoost 支持)
        
        print("    Encoding features...")
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_encoded = ohe.fit_transform(X_df_combined)
        print(f"    Feature Matrix Shape: {X_encoded.shape}")

        # 3. 交叉验证
        tpr_folds = []
        precision_interp_folds = []
        aucs, auprs = [], []
        precs, recs, f1s = [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_encoded, y), 1):
            X_train, X_val = X_encoded[train_idx], X_encoded[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            clf = xgb.XGBClassifier(**XGB_PARAMS)
            clf.fit(X_train, y_train)
            
            # 预测
            y_prob = clf.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            
            # Metrics
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            tpr_interp = np.interp(fpr_grid, fpr, tpr); tpr_interp[0] = 0.0
            tpr_folds.append(tpr_interp)
            aucs.append(roc_auc_score(y_val, y_prob))
            
            p_curve, r_curve, _ = precision_recall_curve(y_val, y_prob)
            p_interp = np.interp(recall_grid, r_curve[::-1], p_curve[::-1])
            precision_interp_folds.append(p_interp)
            auprs.append(average_precision_score(y_val, y_prob))
            
            precs.append(precision_score(y_val, y_pred, zero_division=0))
            recs.append(recall_score(y_val, y_pred, zero_division=0))
            f1s.append(f1_score(y_val, y_pred, zero_division=0))

        # 汇总结果
        m_auc, s_auc = np.mean(aucs), np.std(aucs)
        m_aupr, s_aupr = np.mean(auprs), np.std(auprs)
        
        print(f"    Result: AUC={m_auc:.3f}, AUPR={m_aupr:.3f}")
        
        results[exp_name] = {
            "tpr": np.mean(tpr_folds, axis=0), 
            "auc": m_auc, "auc_std": s_auc,
            "prec_curve": np.mean(precision_interp_folds, axis=0), 
            "aupr": m_aupr
        }
        
        bar_metrics.append({"Feature Set": exp_name, "Metric": "AUPR", "Value": m_aupr})
        bar_metrics.append({"Feature Set": exp_name, "Metric": "Precision", "Value": np.mean(precs)})
        bar_metrics.append({"Feature Set": exp_name, "Metric": "Recall", "Value": np.mean(recs)})
        bar_metrics.append({"Feature Set": exp_name, "Metric": "F1-Score", "Value": np.mean(f1s)})

    # ==========================================
    # 4. 绘图
    # ==========================================
    print("\nDrawing Plots...")
    
    # Plot 1: ROC
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.5)
    for i, (name, res) in enumerate(results.items()):
        plt.plot(fpr_grid, res["tpr"], color=COLORS[i], lw=2.5,
                 label=f'{name} (AUC={res["auc"]:.3f})')
    style_axis(plt.gca())
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=14)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=14)
    plt.title('Ablation Study: ROC Curve', fontweight='bold', fontsize=16)
    plt.legend(loc="lower right", frameon=False, fontsize=12)
    plt.savefig(os.path.join(out_dir, "Ablation_ROC.pdf"), bbox_inches='tight')
    plt.close()

    # Plot 2: PR
    plt.figure(figsize=(8, 8))
    for i, (name, res) in enumerate(results.items()):
        plt.plot(recall_grid, res["prec_curve"], color=COLORS[i], lw=2.5,
                 label=f'{name} (AUPR={res["aupr"]:.3f})')
    style_axis(plt.gca())
    plt.xlabel('Recall', fontweight='bold', fontsize=14)
    plt.ylabel('Precision', fontweight='bold', fontsize=14)
    plt.title('Ablation Study: PR Curve', fontweight='bold', fontsize=16)
    plt.legend(loc="lower left", frameon=False, fontsize=12)
    plt.savefig(os.path.join(out_dir, "Ablation_PR.pdf"), bbox_inches='tight')
    plt.close()

    # Plot 3: Metrics Bar
    df_bar = pd.DataFrame(bar_metrics)
    plt.figure(figsize=(10, 6))
    if HAS_SEABORN:
        ax = sns.barplot(data=df_bar, x="Metric", y="Value", hue="Feature Set", 
                         palette=COLORS, alpha=0.9, errorbar=None)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10, fontfamily='Arial')
    style_axis(plt.gca())
    plt.ylim([0, 1.15])
    plt.ylabel('Score', fontweight='bold', fontsize=14)
    plt.title('Feature Contribution Analysis', fontweight='bold', fontsize=16)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., frameon=False, fontsize=12)
    plt.savefig(os.path.join(out_dir, "Ablation_Metrics_Bar.pdf"), bbox_inches='tight')
    plt.close()

    # 保存数据
    df_bar.to_excel(os.path.join(out_dir, "ablation_summary.xlsx"), index=False)
    print(f"\nDone! Results saved to: {out_dir}")

if __name__ == "__main__":
    main()