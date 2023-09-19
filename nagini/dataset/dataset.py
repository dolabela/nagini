from nagini.performance.classification import ClassificationMetrics

from nagini.data_integrity import *



class BaseDataset():
    def __init__(self, X, type_model, y = None) -> None:
        self._y = y
        self._X = X 
        if type_model not in ["single_classification"]:
            raise Exception()
        else:
            self.type_model = type_model

    def calculate_features_integrity(self):
        duplication_result = calculate_duplication(df = self._X)
        self.duplication_count, self.duplication_rows = duplication_result
        self.missing_values_result = calculate_duplication(df = self._X)
            


class TestData(BaseDataset):
    def __init__(self, X, y, type_model, y_pred = None, y_probs = None) -> None:
        super().__init__(X = X, y=y, type_model=type_model)
        self.X = self._X
        self.y_true = self._y
        self.y_pred = y_pred
        if type not in ["single_classification"]:
            raise Exception()
        else:
            self.type = type

    def calculate_performance_metrics(self):
        if self.type == "single_classification":
            ClassificationMetrics(y_pred=self.y, y_true=self.y)
