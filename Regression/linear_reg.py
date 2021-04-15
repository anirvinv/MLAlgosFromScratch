import numpy as np
from matplotlib import pyplot as plt

class Linear_Model:
    def __init__(self, feature_count):
        self.size = feature_count
        self.coeffs = np.random.random((self.size, 1))
        self.bias = np.random.random((1,1))
    def fit(self, X, y, epochs=10000, lr=0.01):
        assert(len(X[0]) == self.size)
        X = np.array(X)
        y = np.array(y)

        for i in range(epochs):
            y_pred = np.dot(X, self.coeffs) + self.bias

            dJ_W = np.dot(X.T, (y_pred - y))
            dJ_db = np.sum(y_pred-y, keepdims=True) 

            self.coeffs -= dJ_W * lr
            self.bias -= dJ_db * lr
        

    def predict(self, X):
        return np.dot(X, self.coeffs) + self.bias

    def __str__(self):
        return f"Coeffs: \n{self.coeffs} \nBias: \n{self.bias}"

model = Linear_Model(1)

X = [[1],[2],[3],[4]]
y = [[2],[4],[6],[8]]

model.fit(X, y)
print(model.predict(X))




