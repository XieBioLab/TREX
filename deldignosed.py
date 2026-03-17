import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# =====================
# 配置区 (请根据实际修改)
# =====================
INPUT_FILE = "trait_train.xlsx"           # 输入文件
OUTPUT_FILE = "trait_train_cleaned.xlsx"  # 输出文件
LABEL_COL = "Label"

# 参与特征计算的列
SEQ_COLS = ["CDR3a", "CDR3b"] 

# ESM2 模型路径
ESM2_MODEL_PATH = "/mnt/data/caobf/XGboost/models"

# 【关键参数】剔除比例
# 0.1 表示剔除掉 10% “长得最像阳性”的阴性样本
# 如果你想删得更狠一点，确保阴阳界限分明，可以设为 0.15 或 0.2
DROP_SIMILAR_RATIO = 0.1 

BATCH_SIZE = 32

# =====================
# 功能函数
# =====================
def get_esm2_embeddings(text_list, model_path, batch_size=32):
    """提取 Embedding 用于计算相似度"""
    model_path = os.path.abspath(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"正在加载 ESM2: {model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
        model.eval()
    except Exception as e:
        raise ValueError(f"模型加载失败: {e}")

    embeddings = []
    # 简单清洗
    text_list = [t if len(str(t)) > 0 else "<unk>" for t in text_list]
    
    for i in tqdm(range(0, len(text_list), batch_size), desc="提取特征"):
        batch_texts = text_list[i : i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            # Mean Pooling
            mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            embeddings.append(mean_embeddings.cpu().numpy())
            
    return np.vstack(embeddings)

def main():
    # 1. 读取数据
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"找不到文件: {INPUT_FILE}")
    
    df = pd.read_excel(INPUT_FILE)
    print(f"原始数据总数: {len(df)}")
    
    # 2. 分离阳性和阴性
    df_pos = df[df[LABEL_COL] == 1].copy()
    df_neg = df[df[LABEL_COL] == 0].copy()
    
    print(f"阳性样本 (Label=1): {len(df_pos)}")
    print(f"阴性样本 (Label=0): {len(df_neg)}")
    
    # 3. 准备序列字符串 (CDR3a + " " + CDR3b)
    print("正在拼接序列...")
    pos_seqs = (df_pos[SEQ_COLS[0]].fillna('').astype(str) + " " + df_pos[SEQ_COLS[1]].fillna('').astype(str)).tolist()
    neg_seqs = (df_neg[SEQ_COLS[0]].fillna('').astype(str) + " " + df_neg[SEQ_COLS[1]].fillna('').astype(str)).tolist()
    
    # 4. 提取特征
    print("\n--- 步骤 A: 提取阳性样本特征 ---")
    pos_embeddings = get_esm2_embeddings(pos_seqs, ESM2_MODEL_PATH, BATCH_SIZE)
    
    print("\n--- 步骤 B: 提取阴性样本特征 ---")
    neg_embeddings = get_esm2_embeddings(neg_seqs, ESM2_MODEL_PATH, BATCH_SIZE)
    
    # 5. 计算相似度矩阵
    # 矩阵形状: [阴性数量, 阳性数量]
    print("\n正在计算交叉相似度矩阵 (Neg vs Pos)...")
    sim_matrix = cosine_similarity(neg_embeddings, pos_embeddings)
    
    # 6. 找出每个阴性样本与阳性群体的“最大相似度”
    # 意思：这个阴性样本，最像哪一个阳性样本？像到了什么程度？
    max_sim_scores = np.max(sim_matrix, axis=1)
    df_neg['max_sim_to_pos'] = max_sim_scores
    
    # 7. 确定剔除阈值
    # 我们要剔除分数最高的那些 (Top X%)
    threshold = np.quantile(max_sim_scores, 1.0 - DROP_SIMILAR_RATIO)
    print(f"\n相似度阈值 (剔除前 {DROP_SIMILAR_RATIO*100}%): {threshold:.4f}")
    
    # 保留分数 < 阈值的 (即和阳性不像的)
    df_neg_clean = df_neg[df_neg['max_sim_to_pos'] < threshold].copy()
    df_removed = df_neg[df_neg['max_sim_to_pos'] >= threshold].copy()
    
    # 8. 输出报告
    print("-" * 30)
    print("清洗结果:")
    print(f"保留阴性: {len(df_neg_clean)} (与阳性差异较大)")
    print(f"剔除阴性: {len(df_removed)} (与阳性过于相似)")
    print("-" * 30)
    
    # 打印一些被剔除的样本看看
    if len(df_removed) > 0:
        print("被剔除的样本示例 (可能包含假阴性):")
        print(df_removed[SEQ_COLS + ['max_sim_to_pos']].head(3))
        
    # 9. 合并保存
    # 删掉辅助列
    df_neg_clean = df_neg_clean.drop(columns=['max_sim_to_pos'])
    
    df_final = pd.concat([df_pos, df_neg_clean], ignore_index=True)
    
    # 打乱顺序
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_final.to_excel(OUTPUT_FILE, index=False)
    print(f"\n清洗完成！新文件已保存至: {OUTPUT_FILE} (总数: {len(df_final)})")

if __name__ == "__main__":
    main()