"""Models"""
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


class LinRegression:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, x, y):
        x = x.reshape((x.shape[0], -1))
        self.__init__()
        self.model.fit(x, y)

    def predict(self, x):
        x = x.reshape((x.shape[0], -1))
        return self.model.predict(x)
