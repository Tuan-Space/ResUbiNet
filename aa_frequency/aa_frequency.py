import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = './Helvetica.ttf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus']=False

def read_fasta(filename):
    with open(filename, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if not line.startswith('>'):
                seq += f'{line[:12]}{line[13:]}'
    return seq

def amino_acid_freq(seq):
    freq = {}
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for aa in amino_acids:
        count = seq.count(aa)
        freq[aa] = round(count / len(seq), 4)
    return freq

def plot_histogram(pos_freq, neg_freq):
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    pos_values = []
    neg_values = []
    for aa in amino_acids:
        pos_values.append(pos_freq[aa])
        neg_values.append(neg_freq[aa])
    x = np.arange(len(amino_acids))
    width = 0.4
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, pos_values, width, color=(56/255,108/255,176/255), label='Ubiquitination')
    ax.bar(x + width/2, neg_values, width, color=(255/255,127/255,0/255), label='Non-ubiquitination')
    ax.set_xticks(x)
    ax.set_xticklabels(amino_acids)
    ax.set_xlabel('Amino Acid')
    ax.set_ylabel('Frequency')
    ax.legend()
    fig.tight_layout()
    plt.savefig(f'./aa_frequency.pdf')
    plt.show()

pos_seq = read_fasta('../diff_length/data_preprocessing/pos_neg/pos.fasta')
neg_seq = read_fasta('../diff_length/data_preprocessing/pos_neg/neg.fasta')
pos_freq = amino_acid_freq(pos_seq)
neg_freq = amino_acid_freq(neg_seq)
plot_histogram(pos_freq, neg_freq)
