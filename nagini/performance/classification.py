"functions that calculate metrics to check the performance of classification models"

import numpy as np


def accuracy_score(y_true, y_pred):
    """ Compute Accuracy score classification metric

    Args:
        y_true (np.array): Array of correct target values. 1=True and 0=False
        y_pred (np.array): Array of predicted target values. 1=True and 0=False

    Returns:
        float: Aggregated accuracy score
    """
    score_array = y_true == y_pred
    sum_score = sum(score_array)
    count_score = len(score_array)
    score = sum_score / count_score
    return score


def precision_score(y_true, y_pred):
    """ Compute precision score classification metric

    Args:
        y_true (np.array): Array of correct target values. 1=True and 0=False
        y_pred (np.array): Array of predicted target values. 1=True and 0=False

    Returns:
        float: Aggregated precision score
    """
    pos_pred_count = sum(y_pred)
    true_pos_pred_count = sum(np.where(y_pred == 1, y_true * y_pred, 0))
    score = true_pos_pred_count / pos_pred_count
    return score


def recall_score(y_true, y_pred):
    """ Compute recall score classification metric

    Args:
        y_true (np.array): Array of correct target values. 1=True and 0=False
        y_pred (np.array): Array of predicted target values. 1=True and 0=False

    Returns:
        float: Aggregated recall score
    """
    pos_true_count = sum(y_true)
    true_pos_true_count = sum(np.where(y_true == 1, y_true * y_pred, 0))
    score = true_pos_true_count / pos_true_count
    return score


def f1_score(y_true, y_pred):
    """ Compute f1 score classification metric

    Args:
        y_true (np.array): Array of correct target values. 1=True and 0=False
        y_pred (np.array): Array of predicted target values. 1=True and 0=False

    Returns:
        float: Aggregated f1 score
    """
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = 2 * (recall * precision) / (recall + precision)
    return f1
