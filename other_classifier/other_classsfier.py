import os
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef, roc_curve
import matplotlib.font_manager as fm
font_path = './Helvetica.ttf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus']=False


# for i in ['DNN', 'CNN', 'LSTM']:
#     aa_inputs = layers.Input(shape=(25,31))
#     blo_inputs = layers.Input(shape=(25,20))
#     prot_inputs = layers.Input(shape=(1024))
#     x = layers.Concatenate()([aa_inputs, blo_inputs])
#     if i == 'DNN':
#         x = layers.Flatten()(x)
#         x = layers.Concatenate()([x,prot_inputs])
#         x = layers.Dense(128)(x)
#     elif i =='CNN':
#         x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
#         x = layers.Flatten()(x)
#         x = layers.Concatenate()([x,prot_inputs])
#     elif i == 'LSTM':
#         x = layers.LSTM(32, return_sequences = True)(x)
#         x = layers.Flatten()(x)
#         x = layers.Concatenate()([x,prot_inputs])
#     outputs = layers.Dense(1, activation='sigmoid')(x)
#     model = keras.Model([aa_inputs, blo_inputs, prot_inputs], outputs)
#     model.summary()
# exit(0)

def train_data(train_data, train_label, test_data):
    train_scores = []
    test_scores = []
    combined_train = np.concatenate([train_data[0].reshape(train_data[0].shape[0], -1), train_data[1].reshape(train_data[1].shape[0], -1), train_data[2]], axis=-1)
    combined_test = np.concatenate([test_data[0].reshape(test_data[0].shape[0], -1), test_data[1].reshape(test_data[1].shape[0], -1), test_data[2]], axis=-1)

    clf = SVC(probability=True)
    clf.fit(combined_train, train_label)
    svm_train_score = clf.predict_proba(combined_train)
    svm_score = clf.predict_proba(combined_test)
    train_scores.append(svm_train_score)
    test_scores.append(svm_score)
    with open('svm_model.pkl', 'wb') as file:
        pickle.dump(clf, file)
    
    clf = RandomForestClassifier()
    clf.fit(combined_train, train_label)
    rf_train_score = clf.predict_proba(combined_train)
    rf_score = clf.predict_proba(combined_test)
    train_scores.append(rf_train_score)
    test_scores.append(rf_score)
    with open('rf_model.pkl', 'wb') as file:
        pickle.dump(clf, file)

    clf = KNeighborsClassifier()
    clf.fit(combined_train, train_label)
    knn_train_score = clf.predict_proba(combined_train)
    knn_score = clf.predict_proba(combined_test)
    train_scores.append(knn_train_score)
    test_scores.append(knn_score)
    with open('knn_model.pkl', 'wb') as file:
        pickle.dump(clf, file)

    model = xgb.XGBClassifier()
    model.fit(combined_train, train_label)
    xgboost_train_score = model.predict_proba(combined_train)
    xgboost_score = model.predict_proba(combined_test)
    train_scores.append(xgboost_train_score)
    test_scores.append(xgboost_score)
    with open('xgb_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    for i in ['DNN', 'CNN', 'LSTM']:
        kfold = KFold(5, shuffle=True, random_state=42)

        # 初始化存储预测结果的列表
        all_train_scores = []
        all_test_scores = []
        num_fold = 1
        # 开始5折交叉验证
        for train, test in kfold.split(train_data[0]):
            aa_inputs = layers.Input(shape=(25,31))
            blo_inputs = layers.Input(shape=(25,20))
            prot_inputs = layers.Input(shape=(1024))
            x = layers.Concatenate()([aa_inputs, blo_inputs])
            if i == 'DNN':
                x = layers.Flatten()(x)
                x = layers.Concatenate()([x, prot_inputs])
                x = layers.Dense(128)(x)
            elif i == 'CNN':
                x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
                x = layers.Flatten()(x)
                x = layers.Concatenate()([x, prot_inputs])
            elif i == 'LSTM':
                x = layers.LSTM(32, return_sequences=True)(x)
                x = layers.Flatten()(x)
                x = layers.Concatenate()([x, prot_inputs])
            
            outputs = layers.Dense(1, activation='sigmoid')(x)
            model = keras.Model([aa_inputs, blo_inputs, prot_inputs], outputs)
            model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0005), metrics=['accuracy'])
            
            # 保存最佳模型
            checkpoint = ModelCheckpoint(f'{i}/best_model_fold_{num_fold}.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')

            # 训练模型
            model.fit([train_data[0][train], train_data[1][train], train_data[2][train]], 
                    train_label[train], 
                    epochs=50, 
                    validation_data=([train_data[0][test], train_data[1][test], train_data[2][test]],train_label[test]),
                    callbacks=[checkpoint],
                    batch_size=128, 
                    verbose=1,
                    shuffle=True)
            model = keras.models.load_model(f'{i}/best_model_fold_{num_fold}.h5')
            # 预测
            train_score = model.predict([train_data[0], train_data[1], train_data[2]])
            test_score = model.predict([test_data[0], test_data[1], test_data[2]])

            # 存储每一折的预测结果
            all_train_scores.append(train_score)
            all_test_scores.append(test_score)
            num_fold += 1
        train_scores.append(np.mean(all_train_scores, axis=0))
        test_scores.append(np.mean(all_test_scores, axis=0))
    return train_scores, test_scores


train_label = np.load('../../workspace_final/data_preprocessing/extracted_datas/training_labels.npy')
test_label = np.load('../../workspace_final/data_preprocessing/extracted_datas/independent_labels.npy')

aa_train_data = np.load(f'../diff_length/protein_characterization/AAindex/aa_train_25.npy', allow_pickle=True)
aa_test_data = np.load(f'../diff_length/protein_characterization/AAindex/aa_test_25.npy', allow_pickle=True)

blosum_train_data = np.load(f'../diff_length/protein_characterization/BLOSUM62/blosum_train_25.npy', allow_pickle=True)
blosum_test_data = np.load(f'../diff_length/protein_characterization/BLOSUM62/blosum_test_25.npy', allow_pickle=True)
print(blosum_train_data.shape, blosum_test_data.shape)

prot_train_data = np.load(f'../diff_length/protein_characterization/ProtTrans/prot_train_25_per_protein.npy', allow_pickle=True)
prot_test_data = np.load(f'../diff_length/protein_characterization/ProtTrans/prot_test_25_per_protein.npy', allow_pickle=True)

# train_scores, test_scores = train_data([aa_train_data, blosum_train_data, prot_train_data], train_label, [aa_test_data, blosum_test_data, prot_test_data])

# with open('./train_scores.pkl', 'wb') as f:
#     pickle.dump(train_scores, f)

# with open('./test_scores.pkl', 'wb') as f:
#     pickle.dump(test_scores, f)

with open('./train_scores.pkl', 'rb') as f:
    train_scores = pickle.load(f)

with open('./test_scores.pkl', 'rb') as f:
    test_scores = pickle.load(f)

def plot_detail():
    plt.plot([0, 100], [0, 100], linestyle='--', lw=2, color='r', label='Random Guess', alpha=.8)
    plt.tick_params(axis='both', direction='in', top=True, right=True, width=1.5, which='major')
    plt.xlabel('1-Specificity(%)')
    plt.ylabel('Sensitivity(%)')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xticks(np.arange(0, 110, 10))
    plt.yticks(np.arange(0, 110, 10))
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', linewidth=1.5)

    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    plt.savefig(f'./auc.pdf')

def evaluate_metrics(true_labels, predictions):
    preds_class = (predictions > 0.5).astype(int).reshape(-1)
    acc = accuracy_score(true_labels, preds_class)
    auc = roc_auc_score(true_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(true_labels, preds_class).ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    prec = tp / (tp + fp)
    f1 = 2 * tp / (2 * tp + fn + fp)
    mcc = matthews_corrcoef(true_labels, preds_class)
    std = np.std(predictions, ddof=0)
    return acc, sn, sp, prec, f1, auc, mcc, std

results = []
final_auc=[]

for n,i in enumerate(['SVM', 'RF', 'KNN', 'XGBoost', 'DNN', 'CNN', 'LSTM']):
    tprs = []
    fprs = []
    if n >= 4:
        pred_train = train_scores[n][:, 0]
        pred_test = test_scores[n][:, 0]
    else:
        pred_train = train_scores[n][:, 1]
        pred_test = test_scores[n][:, 1]
    acc, sn, sp, prec, f1, auc, mcc, std = evaluate_metrics(train_label, pred_train)
    results.append([f'{i}_train', acc, sn, sp, prec, f1, auc, mcc, std])
    acc, sn, sp, prec, f1, auc, mcc, std = evaluate_metrics(test_label, pred_test)
    fpr, tpr, _ = roc_curve(test_label, pred_test)
    tprs.append(tpr)
    fprs.append(fpr)
    final_auc.append([fpr, tpr, auc])
    results.append([f'{i}_test', acc, sn, sp, prec, f1, auc, mcc, std])

# df = pd.DataFrame(results, columns=["Type", "ACC", "Sn", "Sp", "Precision", "F1-score", "AUC", "MCC", "Std"])
# df = df.round(4)
# print(df)
# df.to_csv(f'./evaluation_results.csv', index=False)

with open(f'../diff_length/ResUbiNet/resubinet/test_roc_data_25.pkl', 'rb') as f:
    roc_data = pickle.load(f)

plt.figure(figsize=(10, 8))
plt.plot(final_auc[2][0]*100, final_auc[2][1]*100, lw=2, alpha=0.8, color=(176/255,74/255,70/255), label=f'KNN: AUC={final_auc[2][2]:.4f}')
plt.plot(final_auc[3][0]*100, final_auc[3][1]*100, lw=2, alpha=0.8, color=(255/255,172/255,55/255), label=f'XGBoost: AUC={final_auc[3][2]:.4f}')
plt.plot(final_auc[1][0]*100, final_auc[1][1]*100, lw=2, alpha=0.8, color=(254/255,55/255,149/255), label=f'RF: AUC={final_auc[1][2]:.4f}')
plt.plot(final_auc[4][0]*100, final_auc[4][1]*100, lw=2, alpha=0.8, color=(0/255,0/255,0/255), label=f'DNN: AUC={final_auc[4][2]:.4f}')
plt.plot(final_auc[0][0]*100, final_auc[0][1]*100, lw=2, alpha=0.8, color=(81/255,161/255,93/255), label=f'SVM: AUC={final_auc[0][2]:.4f}')
plt.plot(final_auc[5][0]*100, final_auc[5][1]*100, lw=2, alpha=0.8, color=(246/255,20/255,48/255), label=f'CNN: AUC={final_auc[5][2]:.4f}')
plt.plot(final_auc[6][0]*100, final_auc[6][1]*100, lw=2, alpha=0.8, color=(198/255,26/255,165/255), label=f'LSTM: AUC={final_auc[6][2]:.4f}')
plt.plot(roc_data[0][5]*100, roc_data[1][5]*100, lw=2, alpha=0.8, color=(18/255,15/255,253/255), label=f'ResUbiNet: AUC={roc_data[2][5]:.4f}')
plot_detail()