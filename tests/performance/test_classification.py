
from nagini.performance.classification import ClassificationMetrics
import numpy as np
import pytest 

y_true = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])
y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 1, 1, 0])
y_probs = 
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
    
 