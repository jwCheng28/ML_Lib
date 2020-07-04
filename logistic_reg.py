import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5, 6])
y = np.array([0, 0, 0, 1, 1, 1])
theta = np.zeros(2)

X = np.stack([np.ones(y.size), X], axis=1)
print(X)

fig = plt.figure()
plt.plot(X[y==0, 1], y[y==0], 'ro')
plt.plot(X[y==1, 1], y[y==1], 'bo')

def sigmoid(z):
    z = np.array(z)
    g = 1 / (1 + np.exp(-z))
    return g

def hypothesis_func(X, theta):
    z = np.dot(X, theta)
    return sigmoid(z)

def costFunction(X, y, theta):
    h = hypothesis_func(X, theta)
    m = y.size
    cost = (1 / m) * sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
    return cost

def gradientDescent(X, y, theta, alpha, iteration):
    theta = theta.copy()
    history = []
    h = hypothesis_func(X, theta)
    m = y.size
    for i in range(iteration):
        theta = theta - ((alpha / m) * (h - y).dot(X))
        history.append(costFunction(X, y, theta))
    return theta, history

theta, history = gradientDescent(X, y, theta, 0.000006, 2400)
print(hypothesis_func(np.array([1, 5]), theta))
print(history[:2])
print('.....')
print(history[len(history)-2:])
plt.plot(X[:, 1], hypothesis_func(X, theta), '-')
print(hypothesis_func(X, theta))
plt.show()

