 

class BaseData():
    def __init__(self, y, X, type) -> None:
        self.y = y
        self.X = X 
        if type not in ["single_classification"]:
            raise Exception()
        else:
            self.type = type

class TrainData(BaseData):
    def __init__(self, y, X, type):
        super().__init__(X=X, y=y, type=type)

class TestData(BaseData):
    def __init__(self, y, X, type):
        super().__init__(X=X, y=y, type=type)
        self.calculate_metrics()

    def calculate_metrics(self):
        

