import numpy as np
import matplotlib.pyplot as plt

# Import or Data Sets for Testing

# Random Training Set
def randomData(start, end, size):
    X = np.random.randint(start, end, (size,))
    y = np.random.randint(start, end, (size,))
    return X, y

# y = x clustered data
def linearCluster(interval, amount, size):
    X = np.random.randint(0, interval + 20, (size//amount,))
    y = np.random.randint(0, interval + 20, (size//amount,))
    for i in range(2, amount + 1):
        temp1 = np.random.randint(-10 + interval * (i-1), interval*i + 20, (size//amount,))
        temp2 = np.random.randint(-10 + interval * (i-1), interval*i + 20, (size//amount,))
        X = np.append(X, temp1)
        y = np.append(y, temp2)
    return X, y

# Basic Feature Scaling
def featureScaling(X, y):
    rangeX = max(X) - min(X)
    rangey = max(y) - min(y)
    return (X/rangeX), (y/rangey)

# Initial X (input/features), y (output)
X, y = linearCluster(50, 2, 90)

# Initial Theta Val
theta = [0, 0]

# Training set Size
m = y.size

# Feature Scale Training Set
#X, y = featureScaling(X, y)

# Create additional dimension of 1 for X
X = np.stack([np.ones(m), X], axis=1)

# Implementation of Cost Function, single variable
def cost_function(X, y, theta):
    m = y.size
    J = (1 / (2 * m)) * sum(np.square(np.dot(X, theta) - y))
    return J

# Implementation of Gradient Descent, single variable
def gradient_descent(X, y, theta, alpha, iteration):
    m = y.size
    theta = theta.copy()
    history = []
    for i in range(iteration):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        history.append(cost_function(X, y, theta))
    return theta, history

# Implementation of Normal Equation
def normalEquation(X, y):
    X_t = np.transpose(X)
    A = np.dot(X_t, X)
    I = np.linalg.inv(A)
    theta = np.dot(I, X_t).dot(y)
    return theta

# Show and Return Results of Gradient Descent
def results(X, y, theta, alpha, iteration):
    theta, history = gradient_descent(X, y, theta, alpha, iteration)
    print("Gradient Descent Results")
    print("Final Theta: ")
    print(theta)
    print("Cost Function History: ")
    print(history[:10])
    print("......")
    print(history[iteration-10:iteration])
    return theta, history

# Show plots of result line and data, with Cost over Iterations
def show(X, y, theta, history):
    fig = plt.figure(1)
    plt.plot(X[:, 1], y, 'o', color='#c6aadf', ms=10, mec='k', label="Training Set")
    plt.plot(X[:, 1], np.dot(X, theta), '-', color='#bae4ff', label="Gradient Descent Method")
    plt.plot(X[:, 1], np.dot(X, normalEquation(X, y)), '-', color='#f8a6a9', label="Normal Equation Method")
    plt.ylabel('Random Data')
    plt.xlabel('Random Data')
    plt.legend()
    checkProg(history)
    plt.show()

# Show Cost over Iterations
def checkProg(history):
    fig = plt.figure(2)
    time = [i for i in range(len(history))]
    plt.plot(time, history, '-', color='#c6aadf')
    plt.ylabel("Cost")
    plt.xlabel("No. of Iterations")

# Testing
iteration = 12000
alpha = 0.0001
theta, history = results(X, y, theta, alpha, iteration)
print("Compare with Normal Equation Results")
print("Final Theta: ")
print(normalEquation(X, y))
print("Final Cost: ")
print(cost_function(X, y, normalEquation(X, y)))
show(X, y, theta, history)
