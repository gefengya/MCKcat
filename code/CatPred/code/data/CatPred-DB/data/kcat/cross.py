# import pandas as pd
# from sklearn.model_selection import KFold
#
# # 读取 CSV 文件
# file_path = '/mnt/usb3/code/gfy/code/catpred_pipeline/data/CatPred-DB/data/kcat/kcat_train_data_4.csv'
# data = pd.read_csv(file_path)
#
# # 获取数据集的索引
# indices = data.index
#
# # 创建 KFold 对象，进行五折交叉验证
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
#
# # 进行五折交叉验证划分
# fold = 1
# for train_index, test_index in kf.split(indices):
#     # 获取训练集和测试集的索引
#     train_data = data.iloc[train_index]
#     test_data = data.iloc[test_index]
#
#     # 打印每一折的信息
#     print(f"Fold {fold}:")
#     print(f"Training set size: {len(train_data)}")
#     print(f"Test set size: {len(test_data)}")
#
#     # 可以将每一折的训练集和测试集保存为 CSV 文件
#     train_data.to_csv(f'kcat_train_fold_{fold}.csv', index=False)
#     test_data.to_csv(f'kcat_test_fold_{fold}.csv', index=False)
#
#     fold += 1
import pandas as pd
from sklearn.model_selection import KFold
import random

# 读取训练集数据
train_file_path = 'kcat_train_reaction.csv'
train_data = pd.read_csv(train_file_path)

# 按 Sequence ID 分组，保证相同氨基酸序列的酶在一起
train_grouped = train_data.groupby('Reaction ID')

# 获取分组的键列表
train_groups = list(train_grouped.groups.keys())

# 打乱分组（如果需要）
random.seed(42)
random.shuffle(train_groups)

# 创建 KFold 对象进行五折交叉验证划分
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 进行五折交叉验证
fold = 1
for train_index, val_index in kf.split(train_groups):
    # 获取当前折的训练子集和验证子集的分组
    train_sub_groups = [train_groups[i] for i in train_index]
    val_sub_groups = [train_groups[i] for i in val_index]

    # 根据分组获取对应的训练子集和验证子集数据
    train_sub_data = pd.concat([train_grouped.get_group(group) for group in train_sub_groups])
    val_sub_data = pd.concat([train_grouped.get_group(group) for group in val_sub_groups])

    # 保存每一折的训练子集和验证子集
    train_sub_data.to_csv(f'train_fold_{fold}_rea.csv', index=False)
    val_sub_data.to_csv(f'val_fold_{fold}_rea.csv', index=False)

    fold += 1
