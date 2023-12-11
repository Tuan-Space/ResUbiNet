import numpy as np
import pandas as pd

def parse_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        sequence = ''
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    sequence = ''
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
    return sequences

fasta_file = "independent_69.fasta"
sequences = parse_fasta(fasta_file)

labels_file = "independent_labels.npy"
labels = np.load(labels_file)

data = pd.DataFrame({'Label': labels, 'Sequence': sequences})

csv_file = "independent_data_69.csv"
data.to_csv(csv_file, index=False, sep=',')

fasta_file = "training_69.fasta"
sequences = parse_fasta(fasta_file)

labels_file = "training_labels.npy"
labels = np.load(labels_file)

data = pd.DataFrame({'Label': labels, 'Sequence': sequences})

csv_file = "training_data_69.csv"
data.to_csv(csv_file, index=False, sep=',')