import h5py
import numpy as np

def extract_data_from_h5(h5_filename, npy_name):
    with h5py.File(h5_filename, 'r') as f:
        total_data = len(f.keys())
        data_list = []
        for i in range(1, total_data+1):
            key = f'{str(i).zfill(5)}'
            data = f[key][()]
            data_list.append(data)
        data_array = np.array(data_list)
    
    print(data_array.shape)
    np.save(npy_name, data_array)

extract_data_from_h5("train_dataset_per_protein.h5", "prot_train_per_protein.npy")
extract_data_from_h5("test_dataset_per_protein.h5", "prot_test_per_protein.npy")
