import numpy as np

def process_file(filename):
    sequences = []
    labels = []

    with open(filename, 'r') as file:
        lines = file.readlines()[4:]
        for line in lines:
            data = line.strip().split('\t')
            subsequence = data[0][3:24]
            sequences.append(subsequence)
            labels.append(int(data[3]))

    return sequences, labels

def sequences_to_fasta(sequences, output_file):
    with open(output_file, 'w') as file:
        for idx, seq in enumerate(sequences, 1):
            file.write(f'>{idx:05}\n')
            file.write(seq + '\n')

training_sequences, training_labels = process_file('DatasetForhCKSAAP_UbSite/TrainingDataset.txt')
independent_sequences, independent_labels = process_file('DatasetForhCKSAAP_UbSite/IndependentDataset.txt')

sequences_to_fasta(training_sequences, 'processed_database/train_dataset.fasta')
sequences_to_fasta(independent_sequences, 'processed_database/test_dataset.fasta')

np.save('processed_database/train_label.npy', training_labels)
np.save('processed_database/test_label.npy', independent_labels)
