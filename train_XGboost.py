import os 
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import torch

# ==========================================
# 绘图字体与风格
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

# -------------- Bootstrap 配置 --------------
N_BOOTSTRAPS = 1000
CI_LEVEL = 0.95
BOOTSTRAP_SEED = 42
# --------------------------------------------------

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

# =====================
# Bootstrap CI 函数（基于 OOF）
# =====================
def bootstrap_ci_auc_ap(y_true, y_score, n_bootstraps=1000, ci_level=0.95, seed=42, stratified=True):
    """
    返回：
      - auc_point, ap_point
      - auc_ci_low, auc_ci_high
      - ap_ci_low, ap_ci_high
    说明：
      - PR-AUC 这里用 average_precision_score (AP) 作为 AUPR 的常见实现。
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)

    auc_point = roc_auc_score(y_true, y_score)
    ap_point = average_precision_score(y_true, y_score)

    rng = np.random.RandomState(seed)
    aucs = []
    aps = []

    if stratified:
        idx_pos = np.where(y_true == 1)[0]
        idx_neg = np.where(y_true == 0)[0]
        n_pos = len(idx_pos)
        n_neg = len(idx_neg)

        for _ in range(n_bootstraps):
            # 分层 bootstrap：分别在正/负类内有放回采样，保持类比例更稳定
            samp_pos = rng.choice(idx_pos, size=n_pos, replace=True)
            samp_neg = rng.choice(idx_neg, size=n_neg, replace=True)
            samp = np.concatenate([samp_pos, samp_neg])

            # 计算
            aucs.append(roc_auc_score(y_true[samp], y_score[samp]))
            aps.append(average_precision_score(y_true[samp], y_score[samp]))
    else:
        for _ in range(n_bootstraps):
            samp = rng.randint(0, n, n)
            # 避免某次抽样全是一个类别导致 AUC 报错
            if len(np.unique(y_true[samp])) < 2:
                continue
            aucs.append(roc_auc_score(y_true[samp], y_score[samp]))
            aps.append(average_precision_score(y_true[samp], y_score[samp]))

    alpha = (1.0 - ci_level) / 2.0
    low_q = 100 * alpha
    high_q = 100 * (1 - alpha)

    auc_ci_low, auc_ci_high = np.percentile(aucs, [low_q, high_q])
    ap_ci_low, ap_ci_high = np.percentile(aps, [low_q, high_q])

    return auc_point, ap_point, auc_ci_low, auc_ci_high, ap_ci_low, ap_ci_high


def main():
    # 1. 准备数据
    if not os.path.exists(INPUT_XLSX):
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
    
    # 保存 OneHotEncoder
    joblib.dump(ohe, os.path.join(out_dir, "preprocessor_ohe.joblib"))
    print(f"预处理器已保存至: {os.path.join(out_dir, 'preprocessor_ohe.joblib')}")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    # 结果容器
    model_results = {}
    bar_metrics = []
    detailed_fold_metrics = []

    # -------- 收集所有 OOF 预测 --------
    all_oof_predictions = []
    # -------------------------------------

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
            
            # 为 XGBoost 保存每一折的模型
            if model_name == "XGBoost":
                fold_json_path = os.path.join(out_dir, f"xgboost_fold_{fold}.json")
                clf.save_model(fold_json_path)

                fold_pt_path = os.path.join(out_dir, f"xgboost_fold_{fold}.pt")
                payload = {
                    "model": clf,
                    "preprocessor": ohe,
                    "fold": fold,
                    "params": clf.get_params()
                }
                torch.save(payload, fold_pt_path)
            
            # 获取预测概率
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_val)[:, 1]
            else:
                y_prob = clf.decision_function(X_val)
            
            y_pred = (y_prob >= 0.5).astype(int)

            # -------- 记录 OOF（每折验证集预测）--------
            all_oof_predictions.append(pd.DataFrame({
                "Model": model_name,
                "Fold": fold,
                "y_true": y_val,
                "y_score": y_prob
            }))
            # ------------------------------------------------

            # --- 计算指标 ---
            # ROC / AUC
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            tpr_interp = np.interp(fpr_grid, fpr, tpr)
            tpr_interp[0] = 0.0
            tpr_folds.append(tpr_interp)
            
            val_auc = roc_auc_score(y_val, y_prob)
            aucs.append(val_auc)
            
            # PR / AUPR (AP)
            p_curve, r_curve, _ = precision_recall_curve(y_val, y_prob)
            p_interp = np.interp(recall_grid, r_curve[::-1], p_curve[::-1])
            precision_interp_folds.append(p_interp)
            
            val_aupr = average_precision_score(y_val, y_prob)
            auprs.append(val_aupr)
            
            # Scalar Metrics
            val_prec = precision_score(y_val, y_pred, zero_division=0)
            val_rec = recall_score(y_val, y_pred, zero_division=0)
            val_f1 = f1_score(y_val, y_pred, zero_division=0)
            
            precs.append(val_prec)
            recs.append(val_rec)
            f1s.append(val_f1)

            detailed_fold_metrics.append({
                "Model": model_name,
                "Fold": fold,
                "AUC": val_auc,
                "AUPR": val_aupr,
                "F1": val_f1,
                "Precision": val_prec,
                "Recall": val_rec,
                "Parameters": str(clf.get_params())
            })

        # 计算平均曲线
        mean_tpr = np.mean(tpr_folds, axis=0); mean_tpr[-1] = 1.0
        mean_precision = np.mean(precision_interp_folds, axis=0)
        
        # 计算平均指标
        m_auc, s_auc = np.mean(aucs), np.std(aucs)
        m_aupr, s_aupr = np.mean(auprs), np.std(auprs)
        m_prec, s_prec = np.mean(precs), np.std(precs)
        m_rec, s_rec = np.mean(recs), np.std(recs)
        m_f1, s_f1 = np.mean(f1s), np.std(f1s)

        print(f"   {model_name} [平均结果]: AUC={m_auc:.3f}, AUPR={m_aupr:.3f}, F1={m_f1:.3f}")
        
        model_results[model_name] = {
            "tpr": mean_tpr, "auc": m_auc, "auc_std": s_auc,
            "prec_curve": mean_precision, "aupr": m_aupr, "aupr_std": s_aupr
        }
        
        bar_metrics.append({"Model": model_name, "Metric": "AUPR", "Value": m_aupr})
        bar_metrics.append({"Model": model_name, "Metric": "Precision", "Value": m_prec})
        bar_metrics.append({"Model": model_name, "Metric": "Recall", "Value": m_rec})
        bar_metrics.append({"Model": model_name, "Metric": "F1-Score", "Value": m_f1})

    # =====================
    # 保存详细的每折数据 (Excel)
    # =====================
    df_detailed = pd.DataFrame(detailed_fold_metrics)
    cols_order = ["Model", "Fold", "AUC", "AUPR", "F1", "Precision", "Recall", "Parameters"]
    df_detailed = df_detailed[cols_order]
    detailed_excel_path = os.path.join(out_dir, "detailed_fold_metrics.xlsx")
    df_detailed.to_excel(detailed_excel_path, index=False)
    print(f"\n每一折的详细指标与参数已保存至: {detailed_excel_path}")

    # =====================
    # 保存所有 out-of-fold 预测 (Excel)
    # =====================
    df_oof_all = pd.concat(all_oof_predictions, ignore_index=True)
    oof_excel_path = os.path.join(out_dir, "all_out_of_fold_predictions.xlsx")
    df_oof_all.to_excel(oof_excel_path, index=False)
    print(f"所有模型所有折的 out-of-fold 预测已保存至: {oof_excel_path}")

    # =====================
    # 基于 OOF 做 1000 次 bootstrap，输出 95% CI
    # =====================
    print(f"\n>>> 正在基于 OOF 预测进行 bootstrap（n={N_BOOTSTRAPS}）估计 {int(CI_LEVEL*100)}% CI ...")

    bootstrap_rows = []
    for model_name in MODELS.keys():
        sub = df_oof_all[df_oof_all["Model"] == model_name]
        y_true_sub = sub["y_true"].values
        y_score_sub = sub["y_score"].values

        auc_point, ap_point, auc_lo, auc_hi, ap_lo, ap_hi = bootstrap_ci_auc_ap(
            y_true_sub, y_score_sub,
            n_bootstraps=N_BOOTSTRAPS,
            ci_level=CI_LEVEL,
            seed=BOOTSTRAP_SEED,
            stratified=True
        )

        bootstrap_rows.append({
            "Model": model_name,
            "OOF_ROC_AUC": auc_point,
            f"ROC_AUC_CI_low_{int(CI_LEVEL*100)}": auc_lo,
            f"ROC_AUC_CI_high_{int(CI_LEVEL*100)}": auc_hi,
            "OOF_PR_AUC(AP)": ap_point,
            f"PR_AUC_CI_low_{int(CI_LEVEL*100)}": ap_lo,
            f"PR_AUC_CI_high_{int(CI_LEVEL*100)}": ap_hi,
            "Bootstraps": N_BOOTSTRAPS,
            "Stratified": True
        })

        print(f"  [{model_name}] ROC-AUC={auc_point:.3f} ({auc_lo:.3f}-{auc_hi:.3f}), "
              f"PR-AUC={ap_point:.3f} ({ap_lo:.3f}-{ap_hi:.3f})")

    df_boot = pd.DataFrame(bootstrap_rows)
    ci_excel_path = os.path.join(out_dir, "bootstrap_ci_summary.xlsx")
    df_boot.to_excel(ci_excel_path, index=False)
    print(f"\nBootstrap CI 汇总已保存至: {ci_excel_path}")

    # =====================
    # 最终全量训练 XGBoost
    # =====================
    print("\n>>> 正在使用全部数据重新训练最终 XGBoost 模型并保存...")
    final_xgb = sklearn.base.clone(MODELS["XGBoost"])
    final_xgb.fit(X_encoded, y)
    
    final_model_path_json = os.path.join(out_dir, "xgboost_final_full.json")
    final_xgb.save_model(final_model_path_json)
    
    final_model_path_pt = os.path.join(out_dir, "xgboost_final_full.pt")
    torch.save({
        "model": final_xgb,
        "preprocessor": ohe,
        "type": "final_full_model"
    }, final_model_path_pt)
    
    print(f"最终完整模型已保存: {final_model_path_pt}")

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
        # 若无 seaborn，可按需自行实现 matplotlib 的柱状图，这里保持与你原逻辑一致不展开
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

    # 保存平均数据
    df_bar.to_excel(os.path.join(out_dir, "avg_metrics_summary.xlsx"), index=False)

    print(f"\n全部完成！结果保存在: {out_dir}")


if __name__ == "__main__":
    main()
