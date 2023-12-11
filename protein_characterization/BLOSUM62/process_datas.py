import numpy as np
from Bio.Align import substitution_matrices

def extract_features_from_file(input_file, output_file):
    blosum62 = substitution_matrices.load("BLOSUM62")
    standard_amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    with open(input_file, 'r') as f:
        sequences = [line.strip().replace('U','X').replace('Z','X').replace('O','X').replace('-','X') for line in f if not line.startswith('>')]

    features_list = []

    for seq in sequences:
        features = np.zeros((len(seq), len(standard_amino_acids)))

        for i, aa in enumerate(seq):
            if aa == 'X':
                features[i] = np.zeros(len(standard_amino_acids))
            elif aa in standard_amino_acids:
                features[i] = np.array([blosum62[aa][aa2] for aa2 in standard_amino_acids])

        features_list.append(features)

    features = np.array(features_list)
    min_value = np.min(features)
    max_value = np.max(features)
    features = (features - min_value) / (max_value - min_value)

    print(features.shape, np.max(features), np.min(features))
    np.save(output_file, features)

extract_features_from_file("../../data_preprocessing/processed_database/train_dataset.fasta", "blosum_train.npy")
extract_features_from_file("../../data_preprocessing/processed_database/test_dataset.fasta", "blosum_test.npy")
