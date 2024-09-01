import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = './Helvetica.ttf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus']=False
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers,optimizers,regularizers
from tensorflow.keras import backend as K
from sklearn.metrics import auc, precision_recall_curve


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

test_label = np.load('../../../workspace_final/data_preprocessing/extracted_datas/independent_labels.npy')
aa_test_data = np.load(f'../protein_characterization/AAindex/aa_test_25.npy', allow_pickle=True)
blosum_test_data = np.load(f'../protein_characterization/BLOSUM62/blosum_test_25.npy', allow_pickle=True)
prot_test_data = np.load(f'../protein_characterization/ProtTrans/prot_test_25_per_protein.npy', allow_pickle=True)

all_test_preds = []
plt.figure(figsize=(10, 8))
for fold_no, colo in zip([1, 2, 3, 4, 5], [(81/255,161/255,93/255), (254/255,55/255,149/255), (176/255,74/255,70/255), (255/255,172/255,55/255), (198/255,26/255,165/255)]):
    model = keras.models.load_model(f'./resubinet/Ubiquitination_fold_25_{fold_no}.h5')
    test_preds = model.predict([aa_test_data, blosum_test_data, prot_test_data])
    all_test_preds.append(test_preds)
    precision, recall, thresholds = precision_recall_curve(test_label, test_preds)
    pr_auc = auc(recall, precision)
    plt.plot(recall*100, precision*100, lw=2, alpha=0.8, color=colo, label=f'Fold {fold_no}: area={pr_auc:.4f}')

final_test_preds = sum(all_test_preds) / 5

precision, recall, thresholds = precision_recall_curve(test_label, final_test_preds)
pr_auc = auc(recall, precision)

plt.plot(recall*100, precision*100, lw=2, alpha=0.8, color=(18/255,15/255,253/255), label=f'ResUbiNet area={pr_auc:.4f}')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 110, 10))
plt.tick_params(axis='both', direction='in', top=True, right=True, width=1.5, which='major')
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', linewidth=1.5)
ax = plt.gca()
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig(f'./PR_curve.pdf')
plt.show()