import numpy as np
from Bio.Align import substitution_matrices

def extract_features_from_file(input_file, output_file):
    # 加载BLOSUM62矩阵
    blosum62 = substitution_matrices.load("BLOSUM62")
    # 定义20个标准的天然氨基酸
    standard_amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    # 读取FASTA文件中的序列
    with open(input_file, 'r') as f:
        sequences = [line.strip().replace('U','X').replace('Z','X').replace('O','X').replace('-','X') for line in f if not line.startswith('>')]

    # 初始化一个用于存储特征的列表
    features_list = []

    # 对于每个序列，计算特征
    for seq in sequences:
        # 初始化一个用于存储当前序列特征的数组
        features = np.zeros((len(seq), len(standard_amino_acids)))

        # 使用BLOSUM62矩阵提取特征
        for i, aa in enumerate(seq):
            if aa == 'X':
                features[i] = np.zeros(len(standard_amino_acids))
            elif aa in standard_amino_acids:
                features[i] = np.array([blosum62[aa][aa2] for aa2 in standard_amino_acids])

        # 将特征添加到特征列表中
        features_list.append(features)

    # 转换成NumPy数组并计算最小值和最大值
    features = np.array(features_list)
    min_value = np.min(features)
    max_value = np.max(features)
    features = (features - min_value) / (max_value - min_value)

    print(features.shape, np.max(features), np.min(features))
    np.save(output_file, features)

for length in range(7, 70, 2):
    extract_features_from_file(f"../../data_preprocessing/processed_database/train_dataset_{length}.fasta", f"blosum_train_{length}.npy")
    extract_features_from_file(f"../../data_preprocessing/processed_database/test_dataset_{length}.fasta", f"blosum_test_{length}.npy")
