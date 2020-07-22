from tensorflow.keras import backend as K
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def get_roc_curve(targets, predicts):
    fpr_micro, tpr_micro, _ = roc_curve(targets.ravel(), predicts.ravel())
    auc_score = auc(fpr_micro, tpr_micro)
    return (fpr_micro, tpr_micro, auc_score)