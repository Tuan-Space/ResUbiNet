import numpy as np

def process_fasta_and_labels(fasta_file, npy_file, output_file):
    with open(fasta_file, 'r') as f:
        lines = f.readlines()

    sequences = []
    for line in lines:
        if line.startswith(">"):
            continue
        sequences.append(line.strip())

    labels = np.load(npy_file)

    output_sequences = []
    for i in range(len(sequences)):
        sequence = sequences[i]
        label = labels[i]

        if label == 1:
            sequence = sequence[:17] + '#' + sequence[17:]

        output_sequences.append(sequence)

    with open(output_file, "w") as f:
        for i, sequence in enumerate(output_sequences):
            f.write(f">{i + 1}\n{sequence}\n")

fasta_file = "training_33.fasta"
npy_file = "training_labels.npy"
output_file = "modified_training_33.fasta"
process_fasta_and_labels(fasta_file, npy_file, output_file)
