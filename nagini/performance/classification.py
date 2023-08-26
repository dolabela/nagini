import numpy as np


def calculate_accuracy_score(y_true, y_pred):
    score_array = y_true == y_pred
    sum_score = sum(score_array)
    count_score = len(score_array)
    score = sum_score/count_score
    return score





