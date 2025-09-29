# # Implementation of Perceptron Algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=50):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def train(self, X, y):
        # Initialize weights
        n_samples, n_features = X.shape
        # Set initial weights to zero
        self.weights = np.zeros(n_features)
        self.bias = 0
        # Set True Class Labels
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.where(linear_output >= 0, 1, -1)
                # Update rule
                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(('red', 'blue')))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Decision Boundary')
    plt.show()

if __name__ == "__main__":
    # # Manually create a linearly separable dataset (10 examples)
    # X = np.array([[1, 2], [2, 3], [2, 1], [3, 2], [1.5, 1.5], # Class 0
    #               [6, 5], [7, 8], [8, 6], [5, 7], [6, 6]]) # Class 1
    # y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  
    
    # Manually create a non-linearly separable dataset (10 examples)
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [0, 1], # Class 0
                  [1, 0], [2, 3], [3, 2], [1.5, 2.5], [2.5, 1.5] ]) # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    # Initialize and train the Perceptron
    ppn = Perceptron(learning_rate=0.1, n_iter=20)
    ppn.train(X, y)

    # Make predictions
    y_pred = ppn.predict(X)
    y_true = np.where(y == 0, -1, 1)
    accuracy = np.mean(y_pred == y_true)

    print("Predictions:", y_pred)
    print("True Labels:", y_true)
    print(f"Training accuracy: {accuracy:.2f}")

    # Plot the decision boundary
    plot_decision_boundary(X, y, ppn)