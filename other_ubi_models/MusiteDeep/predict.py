import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef

def evaluate_metrics(true_labels, predictions):
    preds_class = (predictions > 0.5).astype(int).reshape(-1)
    acc = accuracy_score(true_labels, preds_class)
    auc = roc_auc_score(true_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(true_labels, preds_class).ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    mcc = matthews_corrcoef(true_labels, preds_class)
    return acc, auc, sn, sp, mcc

with open('./models/my_model/_results.txt', 'r') as file:
    lines = file.readlines()

pred_labels = []

for line in lines:
    parts = line.split('\t')
    if not line.startswith('>') and parts[0] == '17':
        pred_label = float(parts[2].split(":")[1])
        pred_labels.append(pred_label)

pred_labels = np.array(pred_labels)
true_labels = np.load('./independent_labels.npy')
print(true_labels.shape)
print(pred_labels.shape)
results = []
acc, auc, sn, sp, mcc = evaluate_metrics(true_labels, pred_labels)
results.append([acc, auc, sn, sp, mcc])
df = pd.DataFrame(results, columns=["ACC", "AUC", "Sn", "Sp", "MCC"])
df = df.round(4)
print(df)
df.to_csv(f'./models/my_model.csv', index=False)