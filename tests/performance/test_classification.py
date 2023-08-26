
from nagini.performance.classification import *
import numpy as np

def test_accuracy_score():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    accuracy_score = calculate_accuracy_score(y_true, y_pred)
    assert accuracy_score == 0.75