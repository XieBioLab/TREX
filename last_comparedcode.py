import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib  # [新增] 用于保存 sklearn 的预处理对象

# ==========================================
# [全局设置] 绘图字体与风格
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
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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

GENE_COLS = ["TRAV", "TRAJ", "TRBV", "TRBJ"]
CDR3_COLS = ["CDR3a", "CDR3b"]
MAX_CDR3_LEN = 25 

N_SPLITS = 5
RANDOM_STATE = 42
CURVE_POINTS = 1001

# 模型定义
MODELS = {
    "XGBoost": xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6, 
        subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, 
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_split=5, 
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "SVM (RBF)": SVC(
        kernel='rbf', C=1.0, probability=True, random_state=RANDOM_STATE
    ),
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000, solver='lbfgs', random_state=RANDOM_STATE
    )
}

if HAS_SEABORN:
    COLORS = sns.color_palette("bright", len(MODELS))
else:
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

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

def main():
    # 1. 准备数据
    if not os.path.exists(INPUT_XLSX):
        # 如果找不到文件，为了演示代码运行，生成一些假数据
        print(f"警告: 找不到文件 {INPUT_XLSX}，将生成随机测试数据用于演示。")
        data = {
            "TRAV": np.random.choice(["TRAV1", "TRAV2"], 100),
            "TRAJ": np.random.choice(["TRAJ1", "TRAJ2"], 100),
            "TRBV": np.random.choice(["TRBV1", "TRBV2"], 100),
            "TRBJ": np.random.choice(["TRBJ1", "TRBJ2"], 100),
            "CDR3a": ["CASSSQGTGVYEQYF"] * 100,
            "CDR3b": ["CASSLAGGGEQYF"] * 100,
            "Label": np.random.randint(0, 2, 100)
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_excel(INPUT_XLSX)
    
    X_raw = df[GENE_COLS + CDR3_COLS]
    y = df[LABEL_COL].astype(int).values
    
    print("正在处理特征...")
    splitter = SequenceSplitter(cols=CDR3_COLS, max_len=MAX_CDR3_LEN)
    X_cdr3_split = splitter.transform(X_raw[CDR3_COLS])
    X_final_df = pd.concat([X_raw[GENE_COLS], X_cdr3_split], axis=1)
    
    # 初始化 OneHotEncoder
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 
    X_encoded = ohe.fit_transform(X_final_df)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"model_comparison_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    
    # [新增] 保存 OneHotEncoder，以便将来预测新数据时使用
    joblib.dump(ohe, os.path.join(out_dir, "preprocessor_ohe.joblib"))
    print(f"预处理器已保存至: {os.path.join(out_dir, 'preprocessor_ohe.joblib')}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    # 结果容器
    model_results = {}
    bar_metrics = [] 

    # =====================
    # 2. 训练与评估循环
    # =====================
    fpr_grid = np.linspace(0, 1, CURVE_POINTS)
    recall_grid = np.linspace(0, 1, CURVE_POINTS)

    for model_name, model_instance in MODELS.items():
        print(f"\n>>> 正在评估模型 (5-Fold CV): {model_name} ...")
        
        tpr_folds = []
        precision_interp_folds = []
        
        aucs, auprs = [], []
        precs, recs, f1s = [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_encoded, y), 1):
            X_train, X_val = X_encoded[train_idx], X_encoded[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            clf = sklearn.base.clone(model_instance)
            clf.fit(X_train, y_train)
            
            # [修改点] 如果是 XGBoost，保存每一折的模型
            if model_name == "XGBoost":
                fold_model_path = os.path.join(out_dir, f"xgboost_fold_{fold}.json")
                # XGBClassifier 自带 save_model 方法
                clf.save_model(fold_model_path)
            
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_val)[:, 1]
            else:
                y_prob = clf.decision_function(X_val)
            
            y_pred = (y_prob >= 0.5).astype(int)

            # --- ROC ---
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            tpr_interp = np.interp(fpr_grid, fpr, tpr)
            tpr_interp[0] = 0.0
            tpr_folds.append(tpr_interp)
            aucs.append(roc_auc_score(y_val, y_prob))
            
            # --- PR ---
            p_curve, r_curve, _ = precision_recall_curve(y_val, y_prob)
            p_interp = np.interp(recall_grid, r_curve[::-1], p_curve[::-1])
            precision_interp_folds.append(p_interp)
            auprs.append(average_precision_score(y_val, y_prob))
            
            # --- Scalar Metrics ---
            precs.append(precision_score(y_val, y_pred, zero_division=0))
            recs.append(recall_score(y_val, y_pred, zero_division=0))
            f1s.append(f1_score(y_val, y_pred, zero_division=0))

        # 计算平均曲线
        mean_tpr = np.mean(tpr_folds, axis=0); mean_tpr[-1] = 1.0
        mean_precision = np.mean(precision_interp_folds, axis=0)
        
        # 计算平均指标
        m_auc, s_auc = np.mean(aucs), np.std(aucs)
        m_aupr, s_aupr = np.mean(auprs), np.std(auprs)
        m_prec, s_prec = np.mean(precs), np.std(precs)
        m_rec, s_rec = np.mean(recs), np.std(recs)
        m_f1, s_f1 = np.mean(f1s), np.std(f1s)
        
        print(f"   {model_name} [平均结果]: AUC={m_auc:.3f}, F1={m_f1:.3f}")
        
        model_results[model_name] = {
            "tpr": mean_tpr, "auc": m_auc, "auc_std": s_auc,
            "prec_curve": mean_precision, "aupr": m_aupr, "aupr_std": s_aupr
        }
        
        bar_metrics.append({"Model": model_name, "Metric": "AUPR", "Value": m_aupr})
        bar_metrics.append({"Model": model_name, "Metric": "Precision", "Value": m_prec})
        bar_metrics.append({"Model": model_name, "Metric": "Recall", "Value": m_rec})
        bar_metrics.append({"Model": model_name, "Metric": "F1-Score", "Value": m_f1})

    # =====================
    # [新增] 最终全量训练 XGBoost
    # =====================
    # 交叉验证仅用于评估性能。为了后续使用，通常在所有数据上重新训练一个最终模型。
    print("\n>>> 正在使用全部数据重新训练最终 XGBoost 模型并保存...")
    final_xgb = sklearn.base.clone(MODELS["XGBoost"])
    final_xgb.fit(X_encoded, y)
    final_model_path = os.path.join(out_dir, "xgboost_final_full.json")
    final_xgb.save_model(final_model_path)
    print(f"最终完整模型已保存: {final_model_path}")

    # =====================
    # 3. 绘图函数 (统一风格)
    # =====================
    def style_axis(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(width=1.5, labelsize=12)
        ax.grid(False)

    print("\n正在绘制图表 (Arial Font)...")

    # --- Plot 1: ROC Curve ---
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.5)
    
    for i, (name, res) in enumerate(model_results.items()):
        plt.plot(fpr_grid, res["tpr"], color=COLORS[i % len(COLORS)], lw=2.5,
                 label=f'{name} (AUC={res["auc"]:.3f})')
    
    ax = plt.gca()
    style_axis(ax)
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=14)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=14)
    plt.title('ROC Curve Comparison (5-Fold Avg)', fontweight='bold', fontsize=16)
    plt.legend(loc="lower right", frameon=False, fontsize=12)
    plt.savefig(os.path.join(out_dir, "Compare_ROC_Arial.pdf"), bbox_inches='tight')
    plt.close()

    # --- Plot 2: PR Curve ---
    plt.figure(figsize=(8, 8))
    for i, (name, res) in enumerate(model_results.items()):
        plt.plot(recall_grid, res["prec_curve"], color=COLORS[i % len(COLORS)], lw=2.5,
                 label=f'{name} (AUPR={res["aupr"]:.3f})')
    
    ax = plt.gca()
    style_axis(ax)
    plt.xlabel('Recall', fontweight='bold', fontsize=14)
    plt.ylabel('Precision', fontweight='bold', fontsize=14)
    plt.title('PR Curve Comparison (5-Fold Avg)', fontweight='bold', fontsize=16)
    plt.legend(loc="lower left", frameon=False, fontsize=12)
    plt.savefig(os.path.join(out_dir, "Compare_PR_Arial.pdf"), bbox_inches='tight')
    plt.close()

    # --- Plot 3: Metrics Bar Chart ---
    df_bar = pd.DataFrame(bar_metrics)
    
    plt.figure(figsize=(10, 6))
    
    if HAS_SEABORN:
        ax = sns.barplot(data=df_bar, x="Metric", y="Value", hue="Model", 
                         palette=COLORS, alpha=0.9, errorbar=None)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10, fontfamily='Arial')
    else:
        print("未检测到 seaborn，跳过高级柱状图绘制")
        pass

    ax = plt.gca()
    style_axis(ax)
    plt.ylim([0, 1.15]) 
    plt.ylabel('Score', fontweight='bold', fontsize=14)
    plt.xlabel('')
    plt.title('Model Performance Metrics (5-Fold Avg)', fontweight='bold', fontsize=16)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., frameon=False, fontsize=12)
    
    plt.savefig(os.path.join(out_dir, "Compare_Metrics_Bar_Arial.pdf"), bbox_inches='tight')
    plt.close()

    # 保存数据
    df_bar.to_excel(os.path.join(out_dir, "all_metrics_summary.xlsx"), index=False)
    print(f"\n全部完成！结果保存在: {out_dir}")

if __name__ == "__main__":
    main()