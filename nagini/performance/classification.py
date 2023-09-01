"functions that calculate metrics to check the performance of classification models"

import numpy as np



class ClassificationMetrics():
    def __init__(self, y_pred, y_true, positive_value = 1):
        self.y_pred = y_pred
        self.y_true = y_true
        self._positive_value = positive_value
        self._get_metrics()

    def _get_metrics(self):
        self._calculate_tp()
        self._calculate_tn()
        self._calculate_fp()
        self._calculate_fn()
        self._accuracy_score()
        self._recall_score()
        self._precision_score()
        self._f1_score()

    def print_metrics(self):
        print(f"""
            Accuracy: {self.accuracy_score}
            Precision: {self.precision_score}
            Recall: {self.recall_score}
            F1: {self.f1_score}
        """)

    def _calculate_tp(self):
        tp = (self.y_true == self.y_pred) & \
            (self.y_true == self._positive_value) & \
            (self.y_pred == self._positive_value) 
        self.tp = sum(tp)

    def _calculate_tn(self):
        tn = (self.y_true == self.y_pred) & \
            (self.y_true != self._positive_value) & \
            (self.y_pred != self._positive_value) 
        self.tn = sum(tn)

    def _calculate_fp(self):
        fp = (self.y_true != self.y_pred) & \
            (self.y_true != self._positive_value) & \
            (self.y_pred == self._positive_value) 
        self.fp = sum(fp)

    def _calculate_fn(self):
        fn = (self.y_true != self.y_pred) & \
            (self.y_true == self._positive_value) & \
            (self.y_pred != self._positive_value) 
        self.fn = sum(fn)

    def _accuracy_score(self):
        self.accuracy_score = (self.tp + self.tn)/(self.tp + \
                                self.tn + self.fp + self.fn )
                              
    def _precision_score(self):
        self.precision_score = self.tp/(self.tp + self.fp)

    def _recall_score(self):
        self.recall_score = self.tp/(self.tp + self.fn)

    def _f1_score(self):
        self.f1_score = 2 / (1/self.recall_score + 1/self.precision_score)

    def _calculate_false_negative_rate(self):
        self.fnr = self.fn/(self.tp + self.fn)

    def _calculate_true_negative_rate(self):
        self.fnr = self.tn/(self.tn + self.fp)

 
# >>> direct_recall_score(x,y)
# 0.25
# >>> direct_precision_score(x,y)
# 0.5
# >>> direct_accuracy_score(x,y)
# 0.2
