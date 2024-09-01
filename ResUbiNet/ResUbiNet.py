import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = './Helvetica.ttf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus']=False
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers,optimizers,regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef, roc_curve
from sklearn.model_selection import StratifiedKFold


def transformer_block(inputs, num_heads=2, dff=32):
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dff)(inputs, inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    x2 = layers.Dense(dff, activation='relu')(x)
    x2 = layers.Dense(K.int_shape(inputs)[-1])(x2)
    x = layers.LayerNormalization(epsilon=1e-6)(x + x2)
    return x

def multi_kernel_conv(inputs, filters):
    x1 = layers.Conv1D(filters, 1, padding='same')(inputs)
    x2 = layers.Conv1D(filters, 3, padding='same')(inputs)
    x3 = layers.Conv1D(filters, 5, padding='same')(inputs)
    x = layers.Concatenate()([x1, x2, x3])
    x = layers.Activation('relu')(x)
    return x

def residual_block(inputs, filters):
    x1 = multi_kernel_conv(inputs, filters)
    x1 = layers.MaxPooling1D()(x1)
    x1 = layers.Conv1D(filters, 3, padding='same')(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv1D(filters, 3, padding='same')(x1)
    x1 = se_block(x1)
    x2 = layers.Conv1D(filters, 1, padding='same')(inputs)
    x2 = layers.MaxPooling1D()(x2)
    x = layers.Add()([x1, x2])
    x = layers.Activation('relu')(x)
    return x

def se_block(input_tensor, ratio=8):
    filters = K.int_shape(input_tensor)[-1]
    se_shape = (1, filters)
    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return layers.multiply([input_tensor, se])

def ResUbiNet(length):
    aa_inputs = layers.Input(shape=(length,31))
    aa_x = transformer_block(aa_inputs, num_heads=2, dff=32)
    aa_x = residual_block(aa_x, 32)

    blo_inputs = layers.Input(shape=(length,20))
    blo_x = transformer_block(blo_inputs, num_heads=2, dff=32)
    blo_x = residual_block(blo_x, 32)

    both_x = layers.Concatenate()([aa_x, blo_x])
    both_x = layers.Flatten()(both_x)
    both_x = layers.Dropout(0.5)(both_x)
    both_x = layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l1(1e-4))(both_x)
    both_x = layers.Dropout(0.5)(both_x)
    both_x = layers.Dense(16, activation='relu')(both_x)

    prot_inputs = layers.Input(shape=(1024,))
    prot_x = layers.Dropout(0.5)(prot_inputs)
    prot_x = layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l1(1e-4))(prot_x)
    prot_x = layers.Dropout(0.5)(prot_x)
    prot_x = layers.Dense(16, activation='relu')(prot_x)

    x = layers.Concatenate()([both_x, prot_x])
    x = layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l1(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model([aa_inputs, blo_inputs, prot_inputs], outputs)
    model.compile(loss='binary_crossentropy', optimizer=optimizers.legacy.Adam(learning_rate=0.0005), metrics=['accuracy',keras.metrics.AUC(name='auc')])
    # model.summary()
    return model

def k_flod_train(train_data, train_label, test_data, test_label, length):
    results = []
    train_fprs = []
    train_tprs = []
    train_aucs = []
    val_fprs = []
    val_tprs = []
    val_aucs = []
    test_fprs = []
    test_tprs = []
    test_aucs = []
    all_test_preds = []
    fold_no = 1
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train, test in kfold.split(train_data[0], train_label):
        print(f'Training for fold {fold_no} ...')
        model_path = f'./resubinet/Ubiquitination_fold_{length}_{fold_no}.h5'
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        model=ResUbiNet(length)
        model.fit([train_data[0][train], train_data[1][train], train_data[2][train]], train_label[train], epochs=50, validation_data=([train_data[0][test], train_data[1][test], train_data[2][test]], train_label[test]),
                            callbacks=[checkpoint],
                            batch_size=128, verbose=1,shuffle=True)
        
        model = keras.models.load_model(model_path)
        train_preds = model.predict([train_data[0][train], train_data[1][train], train_data[2][train]])
        train_acc, train_sn, train_sp, train_prec, train_f1, train_auc, train_mcc = evaluate_metrics(train_label[train], train_preds)
        train_fpr, train_tpr, _ = roc_curve(train_label[train], train_preds)
        train_fprs.append(train_fpr)
        train_tprs.append(train_tpr)
        train_aucs.append(train_auc)
        print(f"Train Fold -> ACC: {train_acc}, Sn: {train_sn}, Sp: {train_sp}, Precision: {train_prec}, F1: {train_f1}, AUC: {train_auc}, MCC: {train_mcc}")
        results.append([f"Train Fold {fold_no}", train_acc, train_sn, train_sp, train_prec, train_f1, train_auc, train_mcc])

        val_preds = model.predict([train_data[0][test], train_data[1][test], train_data[2][test]])
        val_acc, val_sn, val_sp, val_prec, val_f1, val_auc, val_mcc = evaluate_metrics(train_label[test], val_preds)
        val_fpr, val_tpr, _ = roc_curve(train_label[test], val_preds)
        val_fprs.append(val_fpr)
        val_tprs.append(val_tpr)
        val_aucs.append(val_auc)
        print(f"Validation -> ACC: {val_acc}, Sn: {val_sn}, Sp: {val_sp}, Precision: {val_prec}, F1: {val_f1}, AUC: {val_auc}, MCC: {val_mcc}")
        results.append([f"Validation Fold {fold_no}", val_acc, val_sn, val_sp, val_prec, val_f1, val_auc, val_mcc])

        test_preds = model.predict(test_data)
        test_acc, test_sn, test_sp, test_prec, test_f1, test_auc, test_mcc = evaluate_metrics(test_label, test_preds)
        test_fpr, test_tpr, _ = roc_curve(test_label, test_preds)
        test_fprs.append(test_fpr)
        test_tprs.append(test_tpr)
        test_aucs.append(test_auc)
        all_test_preds.append(test_preds)
        print(f"Test Fold -> ACC: {test_acc}, Sn: {test_sn}, Sp: {test_sp}, Precision: {test_prec}, F1: {test_f1}, AUC: {test_auc}, MCC: {test_mcc}")
        results.append([f"Test Fold {fold_no}", test_acc, test_sn, test_sp, test_prec, test_f1, test_auc, test_mcc])
        fold_no += 1
    
    final_test_preds = sum(all_test_preds) / 5
    test_acc, test_sn, test_sp, test_prec, test_f1, test_auc, test_mcc = evaluate_metrics(test_label, final_test_preds)
    print(f"Average Test Predictions ->  ACC: {test_acc}, Sn: {test_sn}, Sp: {test_sp}, Precision: {test_prec}, F1: {test_f1}, AUC: {test_auc}, MCC: {test_mcc}")
    results.append([f"Average Test Predictions", test_acc, test_sn, test_sp, test_prec, test_f1, test_auc, test_mcc])

    test_fpr, test_tpr, _ = roc_curve(test_label, final_test_preds)
    test_fprs.append(test_fpr)
    test_tprs.append(test_tpr)
    test_aucs.append(test_auc)

    with open(f'./resubinet/train_roc_data_{length}.pkl', 'wb') as f:
        pickle.dump([train_fprs, train_tprs, train_aucs], f)

    with open(f'./resubinet/val_roc_data_{length}.pkl', 'wb') as f:
        pickle.dump([val_fprs, val_tprs, val_aucs], f)

    with open(f'./resubinet/test_roc_data_{length}.pkl', 'wb') as f:
        pickle.dump([test_fprs, test_tprs, test_aucs], f)

    df = pd.DataFrame(results, columns=["Type", "ACC", "Sn", "Sp", "Precision", "F1-score", "AUC", "MCC"])
    df = df.round(4)
    print(df)
    df.to_csv(f'./resubinet/evaluation_results_{length}.csv', index=False)

np.random.seed(42)
tf.random.set_seed(42)

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
    return acc, sn, sp, prec, f1, auc, mcc

def plot_detail(type, length):
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
    plt.savefig(f'./resubinet/{type}_auc_{length}.pdf')

for length in range(7, 70, 2):
    # train_label = np.load('../../../workspace_final/data_preprocessing/extracted_datas/training_labels.npy')
    # test_label = np.load('../../../workspace_final/data_preprocessing/extracted_datas/independent_labels.npy')

    # aa_train_data = np.load(f'../protein_characterization/AAindex/aa_train_{length}.npy', allow_pickle=True)
    # aa_test_data = np.load(f'../protein_characterization/AAindex/aa_test_{length}.npy', allow_pickle=True)

    # blosum_train_data = np.load(f'../protein_characterization/BLOSUM62/blosum_train_{length}.npy', allow_pickle=True)
    # blosum_test_data = np.load(f'../protein_characterization/BLOSUM62/blosum_test_{length}.npy', allow_pickle=True)
    # print(blosum_train_data.shape, blosum_test_data.shape)

    # prot_train_data = np.load(f'../protein_characterization/ProtTrans/prot_train_{length}_per_protein.npy', allow_pickle=True)
    # prot_test_data = np.load(f'../protein_characterization/ProtTrans/prot_test_{length}_per_protein.npy', allow_pickle=True)

    # k_flod_train([aa_train_data, blosum_train_data, prot_train_data], train_label, [aa_test_data, blosum_test_data, prot_test_data], test_label, length)

    with open(f'./resubinet/train_roc_data_{length}.pkl', 'rb') as f:
        roc_data = pickle.load(f)
        plt.figure(figsize=(10, 8))
        plt.plot(roc_data[0][0]*100, roc_data[1][0]*100, lw=2, alpha=0.8, color=(81/255,161/255,93/255), label=f'Fold 1: AUC={roc_data[2][0]:.4f}')
        plt.plot(roc_data[0][1]*100, roc_data[1][1]*100, lw=2, alpha=0.8, color=(254/255,55/255,149/255), label=f'Fold 2: AUC={roc_data[2][1]:.4f}')
        plt.plot(roc_data[0][2]*100, roc_data[1][2]*100, lw=2, alpha=0.8, color=(176/255,74/255,70/255), label=f'Fold 3: AUC={roc_data[2][2]:.4f}')
        plt.plot(roc_data[0][3]*100, roc_data[1][3]*100, lw=2, alpha=0.8, color=(255/255,172/255,55/255), label=f'Fold 4: AUC={roc_data[2][3]:.4f}')
        plt.plot(roc_data[0][4]*100, roc_data[1][4]*100, lw=2, alpha=0.8, color=(198/255,26/255,165/255), label=f'Fold 5: AUC={roc_data[2][4]:.4f}')
        plot_detail('train', length)

    with open(f'./resubinet/val_roc_data_{length}.pkl', 'rb') as f:
        roc_data = pickle.load(f)
        plt.figure(figsize=(10, 8))
        plt.plot(roc_data[0][0]*100, roc_data[1][0]*100, lw=2, alpha=0.8, color=(81/255,161/255,93/255), label=f'Fold 1: AUC={roc_data[2][0]:.4f}')
        plt.plot(roc_data[0][1]*100, roc_data[1][1]*100, lw=2, alpha=0.8, color=(254/255,55/255,149/255), label=f'Fold 2: AUC={roc_data[2][1]:.4f}')
        plt.plot(roc_data[0][2]*100, roc_data[1][2]*100, lw=2, alpha=0.8, color=(176/255,74/255,70/255), label=f'Fold 3: AUC={roc_data[2][2]:.4f}')
        plt.plot(roc_data[0][3]*100, roc_data[1][3]*100, lw=2, alpha=0.8, color=(255/255,172/255,55/255), label=f'Fold 4: AUC={roc_data[2][3]:.4f}')
        plt.plot(roc_data[0][4]*100, roc_data[1][4]*100, lw=2, alpha=0.8, color=(198/255,26/255,165/255), label=f'Fold 5: AUC={roc_data[2][4]:.4f}')
        plot_detail('val', length)

    with open(f'./resubinet/test_roc_data_{length}.pkl', 'rb') as f:
        roc_data = pickle.load(f)
        plt.figure(figsize=(10, 8))
        plt.plot(roc_data[0][0]*100, roc_data[1][0]*100, lw=2, alpha=0.8, color=(81/255,161/255,93/255), label=f'Fold 1: AUC={roc_data[2][0]:.4f}')
        plt.plot(roc_data[0][1]*100, roc_data[1][1]*100, lw=2, alpha=0.8, color=(254/255,55/255,149/255), label=f'Fold 2: AUC={roc_data[2][1]:.4f}')
        plt.plot(roc_data[0][2]*100, roc_data[1][2]*100, lw=2, alpha=0.8, color=(176/255,74/255,70/255), label=f'Fold 3: AUC={roc_data[2][2]:.4f}')
        plt.plot(roc_data[0][3]*100, roc_data[1][3]*100, lw=2, alpha=0.8, color=(255/255,172/255,55/255), label=f'Fold 4: AUC={roc_data[2][3]:.4f}')
        plt.plot(roc_data[0][4]*100, roc_data[1][4]*100, lw=2, alpha=0.8, color=(198/255,26/255,165/255), label=f'Fold 5: AUC={roc_data[2][4]:.4f}')
        plt.plot(roc_data[0][5]*100, roc_data[1][5]*100, lw=2, alpha=0.8, color=(18/255,15/255,253/255), label=f'ResUbiNet: AUC={roc_data[2][5]:.4f}')
        plot_detail('test', length)
