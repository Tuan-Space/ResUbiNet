import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import matplotlib.pyplot as plt
# from getData import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc


def training_vis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax1.plot(loss, label='train_loss')
    ax1.plot(val_loss, label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    #ax2.plot(acc, label='train_acc')
    ax2.plot(acc, label='train_accuracy')
    ax2.plot(val_acc, label='val_accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    # plt.savefig('photo/Loss_Acc.png')
    plt.show()


def calculate_performance(test_num,labels,predict_y):
    tp=0
    fp=0
    tn=0
    fn=0
    for index in range(test_num):
        if(labels[index]==1):
            if(labels[index] == predict_y[index]):
                tp += 1
            else:
                fn += 1
        else:
            if(labels[index] == predict_y[index]):
                tn += 1
            else:
                fp += 1
    acc = float(tp+tn)/test_num
    precision = float(tp)/(tp+fp+ sys.float_info.epsilon)
    sensitivity = float(tp)/(tp+fn+ sys.float_info.epsilon)
    specificity = float(tn)/(tn+fp+ sys.float_info.epsilon)
    f1 = 2 * precision * sensitivity / (precision + sensitivity)
    mcc = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return acc, sensitivity, specificity, mcc, f1


def calculate_performance2(test_num,labels,predict_y):
    tp=0
    fp=0
    tn=0
    fn=0
    for index in range(test_num):
        if(labels[index]==1):
            if(labels[index] == predict_y[index]):
                tp += 1
            else:
                fn += 1
        else:
            if(labels[index] == predict_y[index]):
                tn += 1
            else:
                fp += 1
    acc = float(tp+tn)/test_num
    precision = float(tp)/(tp+fp+ sys.float_info.epsilon)
    sensitivity = float(tp)/(tp+fn+ sys.float_info.epsilon)
    specificity = float(tn)/(tn+fp+ sys.float_info.epsilon)
    f1 = 2 * precision * sensitivity / (precision + sensitivity)
    mcc = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return acc, sensitivity, specificity, mcc, f1, precision


def precision_recall(true_label, pred_proba):
    lw = 2
    precision, recall, _ = precision_recall_curve(true_label, pred_proba)
    average_precision = average_precision_score(true_label, pred_proba)
    for i in range(len(precision)):
        print(precision[i], recall[i])
    plt.clf()
    plt.plot(recall, precision, color='gold', lw=lw,
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve to binary-class')
    plt.legend(loc="lower right")
    plt.show()


def roc(true_label, pred_proba):
    fpr, tpr, thresholds = roc_curve(true_label, pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
