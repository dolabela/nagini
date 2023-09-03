
from nagini.performance.classification import ClassificationMetrics
import numpy as np
import pytest 

y_true = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])
y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 1, 1, 0])
actual_metrics = [ClassificationMetrics(y_pred=y_pred,y_true=y_true)]

@pytest.mark.parametrize("actual_metrics", actual_metrics)
def test_accuracy_score(actual_metrics):
    assert round(actual_metrics.accuracy_score,2) == 0.7

@pytest.mark.parametrize("actual_metrics", actual_metrics)
def test_precision_score(actual_metrics):
    assert round(actual_metrics.precision_score,2) == 0.67

@pytest.mark.parametrize("actual_metrics", actual_metrics)
def test_recall_score(actual_metrics):
    assert round(actual_metrics.recall_score,2) == 0.8

@pytest.mark.parametrize("actual_metrics", actual_metrics)
def test_f1_score(actual_metrics):
    assert round(actual_metrics.f1_score,2)== 0.73
    
y_true = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 0])
y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 1])
y_probs= np.array([[0.51, 0.49], [0.26, 0.74], [0.75, 0.25], [0.66, 0.34], [0.62, 0.38], [0.14, 0.86], [0.35, 0.65], [0.86, 0.14], [0.85, 0.15], [0.14, 0.86]])
actual_metrics = [ClassificationMetrics(y_pred=y_pred,y_true=y_true,y_probs=y_probs)]

@pytest.mark.parametrize("actual_metrics", actual_metrics)
def test_auc_roc_score(actual_metrics):
    assert round(actual_metrics.auc_roc,2)== 0.52

@pytest.mark.parametrize("actual_metrics", actual_metrics)
def test_auc_pr_score(actual_metrics):
    assert round(actual_metrics.auc_pr,2)== 0.55