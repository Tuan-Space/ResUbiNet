import h5py
import numpy as np

def extract_data_from_h5(h5_filename, npy_name):
    with h5py.File(h5_filename, 'r') as f:
        # 获取总数据量
        total_data = len(f.keys())
        # 初始化一个数组来存储数据
        data_list = []
        for i in range(1, total_data+1):
            key = f'{str(i).zfill(5)}'  # 使用.zfill(5)使数字总是5位数
            data = f[key][()]
            data_list.append(data)
        data_array = np.array(data_list)
    
    print(data_array.shape)
    # 保存为.npy格式
    np.save(npy_name, data_array)

for length in range(7, 70, 2):
    extract_data_from_h5(f"train_dataset_{length}_per_protein.h5", f"prot_train_{length}_per_protein.npy")
    extract_data_from_h5(f"test_dataset_{length}_per_protein.h5", f"prot_test_{length}_per_protein.npy")
