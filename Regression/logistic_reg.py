import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class Logistic_Model:
    def __init__(self, feature_count):
        self.size = feature_count
        self.coeffs = np.random.random((self.size, 1))
        self.bias = np.random.random((1,1))

    def fit(self, X, y, epochs=10000, lr=0.01):
        X = np.array(X)
        y = np.array(y)

        for i in range(epochs):
            z1 = np.dot(X, self.coeffs) + self.bias
            y_pred = sigmoid(z1)

            dW = -1 * np.dot(X.T, (y - y_pred))
            db = -1 * np.sum((y-y_pred), keepdims=True)

            self.coeffs -= dW * lr
            self.bias -= db * lr

    def predict(self, X):
        return sigmoid(np.dot(X, self.coeffs) + self.bias)
    
    def __str__(self):
        return f"Coeffs: \n{self.coeffs} \nBias: {self.bias}"
    
model = Logistic_Model(1)

X = [[0], [1], [0], [1], [0]]
y = [[0],[1], [0], [1], [0]]

model.fit(X, y, lr=0.01)

print(model.predict([0]))
