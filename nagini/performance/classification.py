"functions that calculate metrics to check the performance of classification models"

import numpy as np

class ClassificationMetrics():
    def __init__(self, y_pred: np.array, y_true: np.array, y_probs: np.array = None, positive_value = 1):
        self.y_pred = y_pred
        self.y_true = y_true
        self.y_probs = y_probs
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
        self._calculate_false_negative_rate()
        self._calculate_true_negative_rate()
        self._calculate_false_positive_rate()
        self._calculate_true_positive_rate()
        if self.y_probs is not None :
            self._calculate_roc_pr_ranges()
            self._calculate_auc()

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
        self.tp = int(sum(tp))

    def _calculate_tn(self):
        tn = (self.y_true == self.y_pred) & \
            (self.y_true != self._positive_value) & \
            (self.y_pred != self._positive_value) 
        self.tn = int(sum(tn))

    def _calculate_fp(self):
        fp = (self.y_true != self.y_pred) & \
            (self.y_true != self._positive_value) & \
            (self.y_pred == self._positive_value) 
        self.fp = int(sum(fp))

    def _calculate_fn(self):
        fn = (self.y_true != self.y_pred) & \
            (self.y_true == self._positive_value) & \
            (self.y_pred != self._positive_value) 
        self.fn = int(sum(fn))

    def _accuracy_score(self):
        try:
            self.accuracy_score = (self.tp + self.tn)/(self.tp + \
                                self.tn + self.fp + self.fn )
        except ZeroDivisionError:
            self.accuracy_score = 0

    def _precision_score(self):
        try:
            self.precision_score = float(self.tp/(self.tp + self.fp))
        except ZeroDivisionError:
            self.precision_score = 0

    def _recall_score(self):            
        try:
            self.recall_score = float(self.tp/(self.tp + self.fn))
        except ZeroDivisionError:
            self.recall_score = 0

    def _f1_score(self):
        try:
            self.f1_score = 2 / (1/self.recall_score + 1/self.precision_score)
        except ZeroDivisionError:
            self.f1_score = 0

    def _calculate_true_positive_rate(self):
        self.tpr = self.tp/(self.tp + self.fn)

    def _calculate_false_positive_rate(self):
        self.fpr = self.fp/(self.tn + self.fp)

    def _calculate_false_negative_rate(self):
        self.fnr = self.fn/(self.tp + self.fn)

    def _calculate_true_negative_rate(self):
        self.tnr = self.tn/(self.tn + self.fp)

    def _calculate_roc_pr_ranges(self):
        array_threshold = np.arange(0, 1, 1/100)
        y_probs = self.y_probs[:,1]
        self.array_precision = np.zeros(shape=(100))
        self.array_recall = np.zeros(shape=(100))
        self.array_tpr = np.zeros(shape=(100))
        self.array_fpr = np.zeros(shape=(100))
        for idx, threshhold in enumerate(array_threshold):
            th_y_pred = np.where(y_probs > threshhold, 1, 0)
            metrics = ClassificationMetrics(y_true=self.y_true, y_pred = th_y_pred)
            self.array_precision[idx] = metrics.precision_score
            self.array_recall[idx] = metrics.recall_score
            self.array_tpr[idx] = metrics.tpr
            self.array_fpr[idx] = metrics.fpr

    def _calculate_auc(self):
        self.auc_pr = abs(np.trapz(self.array_precision, self.array_recall))
        self.auc_roc = abs(np.trapz(self.array_fpr,self.array_tpr))


