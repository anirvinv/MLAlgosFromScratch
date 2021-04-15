import numpy as np
from matplotlib import pyplot as plt

class Network:
    def __init__(self, layers, feature_count, activation=lambda x: 1/(1+np.exp(-x))):
        
        self.activation = activation
        self.layers = [] 
        self.bias = []
        self.layers.append(np.random.random((layers[0], feature_count)))
        self.bias.append(np.random.random((layers[0], 1)))
        for index, layer in enumerate(layers):
            if index == 0:
                continue
            else:
                self.layers.append(np.random.random(size=(layer, layers[index -1])))
                self.bias .append(np.random.random(size=(layer, 1)))

    def forward(self, X):
        X_copy = X.copy().T
        for layer, bias in zip(self.layers, self.bias):
            X_copy = np.dot(layer, X_copy) + bias
            X_copy = self.activation(X_copy)
            # print(X_copy)

        return X_copy

    def fit(self):
        pass

    def __str__(self):
        representation = ""

        for layer, bias in zip(self.layers, self.bias):
            representation += f"Layer : \n{layer} \nBias:\n {bias} \n\n" 
        return representation


X = np.random.random(size=(5, 10))

net = Network((3,2,1), 10)

output = net.forward(X)

print(output)