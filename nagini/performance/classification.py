import numpy as np


def accuracy_score(y_true, y_pred):
    score_array = y_true == y_pred
    sum_score = sum(score_array)
    count_score = len(score_array)
    score = sum_score / count_score
    return score

def precision_score(y_true, y_pred):
    pos_pred_count = sum(y_pred)
    true_pos_pred_count = sum(np.where(y_pred == 1, y_true * y_pred, 0))
    score = true_pos_pred_count / pos_pred_count
    return score

def recall_score(y_true, y_pred):
    pos_true_count = sum(y_true)
    true_pos_true_count = sum(np.where(y_true == 1, y_true * y_pred, 0))
    score = true_pos_true_count / pos_true_count
    return score
