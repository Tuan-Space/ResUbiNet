import os
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

def read_accuracy(csv_file):
    with open(csv_file, 'r') as f:
        train_acc = 0
        val_acc = 0
        train_auc = 0
        val_auc = 0
        for line in f.readlines():
            if line.startswith('Train Fold'):
                train_acc += float(line.split(',')[1])
                train_auc += float(line.split(',')[6])
            elif line.startswith('Validation Fold'):
                val_acc += float(line.split(',')[1])
                val_auc += float(line.split(',')[6])
            elif line.startswith('Average Test Predictions'):
                test_acc = float(line.split(',')[1])
                test_auc = float(line.split(',')[6])
        return train_acc/5, val_acc/5, test_acc, train_auc/5, val_auc/5, test_auc

lengths = []
train_accs = []
val_accs = []
test_accs = []
train_aucs = []
val_aucs = []
test_aucs = []
for file in os.listdir('./resubinet'):
    if file.endswith('.csv'):
        length = int(file.split('_')[-1].split('.')[0])
        train_acc, val_acc, test_acc, train_auc, val_auc, test_auc= read_accuracy('./resubinet/' + file)
        lengths.append(length)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        test_aucs.append(test_auc)

combined_list = list(zip(lengths, train_accs, val_accs, test_accs, train_aucs, val_aucs, test_aucs))
sorted_list = sorted(combined_list, key=lambda x: x[0])
lengths, train_accs, val_accs, test_accs, train_aucs, val_aucs, test_aucs = zip(*sorted_list)

def draw_pic(lengths, train_index, val_index, test_index, my_index, position):
    plt.subplot(2, 1, position[0])
    plt.plot(lengths, train_index, color='red', marker='o', label='Training')
    plt.plot(lengths, val_index, color='green', marker='o', label='Validation')
    plt.plot(lengths, test_index, color='blue', marker='o', label='Test')
    fig = plt.gcf()
    fig.text(position[1], position[2], position[3], fontproperties=font_prop_bold, fontsize=20)

    plt.tick_params(axis='both', direction='in', top=True, right=True, width=1.5, which='major')
    plt.xlim(lengths[0]-1, lengths[-1]+1)
    # plt.ylim(0, 100)
    plt.xticks(lengths)
    # plt.yticks(np.arange(-10, 11, 1))
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=':', linewidth=1.5)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.xlabel("Length")
    plt.ylabel(my_index)

plt.figure(figsize=(12, 12))  # 调整整体图的大小
plt.subplots_adjust(hspace=0.3)  # 这里的0.5可以根据需要调整

draw_pic(lengths, train_accs, val_accs, test_accs, 'Accuracy', [1, 0.07, 0.89, 'a'])
draw_pic(lengths, train_aucs, val_aucs, test_aucs, 'AUC', [2, 0.07, 0.46, 'b'])

plt.savefig('./diff_length_metrics.pdf')
plt.show()
