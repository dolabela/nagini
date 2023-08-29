
from nagini.performance.classification import *
import numpy as np

def test_accuracy_score():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    actual_score = accuracy_score(y_true, y_pred)
    assert actual_score == 0.75

def test_precision_score():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    actual_score = precision_score(y_true, y_pred)
    assert round(actual_score, 2) == 0.67

def test_recall_score():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    actual_score = recall_score(y_true, y_pred)
    assert round(actual_score, 2) == 0.5

def test_f1_score():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([1, 1, 1, 0])
    actual_score = f1_score(y_true, y_pred)
    assert round(actual_score, 2) == 0.4
    
