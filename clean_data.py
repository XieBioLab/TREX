import pandas as pd
import os

def check_overlap(file_a_path, file_b_path):
    # 定义需要比较的六列
    cols_to_compare = ['TRAV', 'TRAJ', 'CDR3a', 'TRBV', 'TRBJ', 'CDR3b']

    print(f"正在读取文件 A: {file_a_path}")
    if not os.path.exists(file_a_path):
        print(f"错误: 找不到文件 {file_a_path}")
        return
    df_a = pd.read_excel(file_a_path)

    print(f"正在读取文件 B: {file_b_path}")
    if not os.path.exists(file_b_path):
        print(f"错误: 找不到文件 {file_b_path}")
        return
    df_b = pd.read_excel(file_b_path)

    # 检查列是否存在
    for col in cols_to_compare:
        if col not in df_a.columns:
            print(f"错误: 文件 A 中缺少列 {col}")
            return
        if col not in df_b.columns:
            print(f"错误: 文件 B 中缺少列 {col}")
            return

    # 数据预处理：
    # 1. 提取这六列
    # 2. 转换为字符串类型 (防止有的识别为数字有的识别为文本导致不匹配)
    # 3. 去除首尾空格 (防止肉眼看不见的空格导致不匹配)
    # 4. 填充空值为 ""
    
    print("正在进行数据标准化处理...")
    sub_a = df_a[cols_to_compare].fillna("").astype(str).apply(lambda x: x.str.strip())
    sub_b = df_b[cols_to_compare].fillna("").astype(str).apply(lambda x: x.str.strip())

    # 为了避免文件内部自身的重复影响统计（例如 A 里有 2 条一样的，B 里有 1 条一样的），
    # 通常我们比较的是“唯一序列”的重合情况。
    # 如果你想看 A 里有多少行出现在了 B 里（不去重），请注释掉下面这两行 drop_duplicates
    sub_a_unique = sub_a.drop_duplicates()
    sub_b_unique = sub_b.drop_duplicates()

    print(f"文件 A 去重后有 {len(sub_a_unique)} 条唯一序列")
    print(f"文件 B 去重后有 {len(sub_b_unique)} 条唯一序列")

    # 方法：使用 merge 求交集 (Inner Join)
    overlap = pd.merge(sub_a_unique, sub_b_unique, on=cols_to_compare, how='inner')
    
    num_overlap = len(overlap)
    
    print("=" * 30)
    print(f"结果：发现 {num_overlap} 条完全相同的序列")
    print("=" * 30)
    
    # 如果需要保存重复的数据
    if num_overlap > 0:
        save_path = "overlap_results.xlsx"
        overlap.to_excel(save_path, index=False)
        print(f"重复的具体数据已保存至: {save_path}")

    return overlap

# ==========================================
# 配置路径并运行
# ==========================================
path_a = "/mnt/data/caobf/XGboost/McPAS.xlsx"
path_b = "/mnt/data/caobf/XGboost/trait_train_cleaned.xlsx"

if __name__ == "__main__":
    check_overlap(path_a, path_b)