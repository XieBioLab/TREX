import pandas as pd

# 读取两个 Excel 文件
df1 = pd.read_excel("./McPAS.xlsx")
df2 = pd.read_excel("./trait_train.xlsx")

# Label 列名
label_col = "Label"

# 用于判断是否重合的 6 列（除 Label 外）
key_cols = [c for c in df1.columns if c != label_col]

# 只保留用于比较的列
df1_key = df1[key_cols].drop_duplicates()
df2_key = df2[key_cols].drop_duplicates()

# 通过 merge 找到重合数据
overlap = df1_key.merge(df2_key, on=key_cols, how="inner")

# 统计数量
overlap_count = len(overlap)

print(f"两个数据集中【除 Label 外 6 列完全相同】的重合数据数量为: {overlap_count}")

# 如需保存重合数据
# overlap.to_excel("./overlap_result.xlsx", index=False)

