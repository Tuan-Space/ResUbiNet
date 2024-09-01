import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path_bold = './Helvetica-Bold.ttf'
font_prop_bold = fm.FontProperties(fname=font_path_bold)
fm.fontManager.addfont(font_path_bold)

font_path = './Helvetica.ttf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus']=False

index_df = pd.read_csv("index.csv")
index_dict = index_df.set_index("index").to_dict(orient="index")

def read_fasta(filename):
    sequences = []
    with open(filename, "r") as f:
        for line in f:
            if not line.startswith(">"):
                sequences.append(line.strip())
    return sequences

def calc_mean_index(sequences, index_type):
    seq_len = len(sequences[0])
    index_sum = np.zeros(seq_len)
    index_count = np.zeros(seq_len)
    for seq in sequences:
        for i in range(seq_len):
            aa = seq[i]
            if aa not in ['-','X']:
                index_count[i] += 1
                index_value = index_dict[index_type][aa]
                index_sum[i] += index_value
    index_mean = index_sum / index_count
    return index_mean

pos_sequences = read_fasta("../diff_length/data_preprocessing/pos_neg/pos.fasta")
neg_sequences = read_fasta("../diff_length/data_preprocessing/pos_neg/neg.fasta")

index_types = ["VHSE1", "VHSE3", "VHSE5"]
index_colors = ["red", "green", "blue"]

plt.figure(figsize=(14, 20))  # 调整整体图的大小
plt.subplots_adjust(hspace=0.3)  # 这里的0.5可以根据需要调整

for i, (index_type, index_color) in enumerate(zip(index_types, index_colors), start=1):
    pos_mean_index = calc_mean_index(pos_sequences, index_type)
    neg_mean_index = calc_mean_index(neg_sequences, index_type)

    plt.subplot(3, 1, i)  # 创建一个 3 行 1 列的子图，i 表示当前子图的位置
    plt.plot(range(-12, 13), pos_mean_index, color=index_color, marker='o', label="Ubiquitination")
    plt.plot(range(-12, 13), neg_mean_index, color=index_color, marker='o', linestyle="--", label="Non-ubiquitination")

    plt.tick_params(axis='both', direction='in', top=True, right=True, width=1.5, which='major')
    plt.xlim(-13, 13)
    plt.xticks(np.arange(-12, 13, 1))
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=1.5)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.xlabel("Position")
    plt.ylabel(f'Mean of {index_type} values')

fig = plt.gcf()
fig.text(0.07, 0.89, 'a', fontproperties=font_prop_bold, fontsize=24)
fig.text(0.07, 0.61, 'b', fontproperties=font_prop_bold, fontsize=24)
fig.text(0.07, 0.33, 'c', fontproperties=font_prop_bold, fontsize=24)

plt.savefig('./index_plot.pdf')
plt.show()


