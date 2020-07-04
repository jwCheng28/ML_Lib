import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import random

# Import or Generate Training Set
X, y = make_classification(n_samples=100, n_features=2,
                           n_redundant=0, n_informative=1,
                           n_clusters_per_class=1, random_state=random.randint(10,20))

# Initialize Theta Values
theta = np.zeros(3)

# Add column of 1's for X_0
X = np.column_stack([np.ones(y.size), X])

# Function for plotting Result and Cost History
def plotData(X, y, theta, history):
    fig = plt.figure(1)
    pos = y == 1
    neg = y == 0
    # Plot different cluster with different color
    plt.plot(X[neg, 1], X[neg, 2], 'ro', label = "Negative Class")
    plt.plot(X[pos, 1], X[pos, 2], 'bo', label = "Positive Class")

    # y = mx + b => X_2 = -(theta_0 + theta_1 * X_1)/theta_2
    slope = -(theta[1] / theta[2])
    b = -(theta[0] / theta[2])
    ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = b + (slope * x_vals)
    plt.plot(x_vals, y_vals, c="k", label = "Hypothesis Function");
    plt.xlabel("Random Data X_1")
    plt.ylabel("Random Data X_2")
    plt.legend()

    # History of Cost
    fig = plt.figure(2)
    plt.plot([i for i in range(len(history))], history, '-')
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()

# Sigmoid Function
def sigmoid(z):
    z = np.array(z)
    g = 1 / (1 + np.exp(-z))
    return g

# Hypothesis Function
def hypothesis_func(X, theta):
    z = np.dot(X, theta)
    return sigmoid(z)

# Cost Function
def costFunction(X, y, theta):
    h = hypothesis_func(X, theta)
    m = y.size
    cost = (1 / m) * sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
    return cost

# Gradient Descent implementation
def gradientDescent(X, y, theta, alpha, iteration):
    theta = theta.copy()
    history = []
    h = hypothesis_func(X, theta)
    m = y.size
    for i in range(iteration):
        theta = theta - ((alpha / m) * (h - y).dot(X))
        history.append(costFunction(X, y, theta))
    return theta, history


theta, history = gradientDescent(X, y, theta, 0.003, 2800)

print("Cost History: ")
print(history[:5])
print('.....')
print(history[len(history)-5:])
print("Final Theta: ")
print(theta)

plotData(X, y, theta, history)
