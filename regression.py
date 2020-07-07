import numpy as np
import matplotlib.pyplot as plt

# Import or Generate Data Sets for Testing

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
def featureScaling(X):
    X = X.copy()
    u_X = np.mean(X)
    sigma = np.std(X)
    return (X - u_X) / sigma

# Initial X (input/features), y (output)
X, y = linearCluster(50, 2, 90)

# Initial Theta Val
theta = np.zeros(2)

# Training set Size
m = y.size

# Feature Scale Training Set
X_s = featureScaling(X)

# Create additional dimension of 1 for X_0
X = np.stack([np.ones(m), X], axis=1)
X_s = np.stack([np.ones(m), X_s], axis=1)

# Manipulate X for polynomial regression
def polyX(X, deg):
    theta = np.zeros(deg+1)
    M = np.ones(X.shape[0])
    for i in range(1, deg + 1):
        S = (X[:,1]).copy() ** i
        M = np.column_stack([M, S])
    return M, theta

# Implementation of Cost Function
def cost_function(X, y, theta, lambda_):
    m = y.size
    J = (1 / (2 * m)) * (sum(np.square(np.dot(X, theta) - y)) + lambda_ * sum(theta ** 2))
    return J

# Implementation of Gradient Descent
def gradient_descent(X, y, theta, alpha, lambda_, iteration):
    m = y.size
    theta = theta.copy()
    reg = theta.copy()
    reg[0] = 0
    history = []
    for i in range(iteration):
        theta = theta - (alpha / m) * ((np.dot(X, theta) - y).dot(X) + (lambda_ / m) * reg)
        history.append(cost_function(X, y, theta, lambda_))
    return theta, history

# Implementation of Normal Equation
def normalEquation(X, y, lambda_):
    X_t = np.transpose(X)
    reg = np.eye(X.shape[1])
    reg[0][0] = 0
    A = np.dot(X_t, X) + lambda_ * reg
    I = np.linalg.inv(A)
    theta = np.dot(I, X_t).dot(y)
    return theta

# Show and Return Results of Gradient Descent
def results(X, y, theta, alpha, lambda_, iteration):
    theta, history = gradient_descent(X, y, theta, alpha, lambda_, iteration)
    print("Gradient Descent Results")
    print("Final Theta: ")
    print(theta)
    print("Cost Function History: ")
    print(history[:10])
    print("......")
    print(history[iteration-10:iteration])
    return theta, history

# Show plots of result line and data, with Cost over Iterations
def show(X, y, theta, history, lambda_):
    fig = plt.figure(1)
    plt.plot(X[:, 1], y, 'o', color='#c6aadf', ms=10, mec='k', label="Training Set")
    plt.plot(X[:, 1], np.dot(X, theta), '-', color='#bae4ff', label="Gradient Descent Method")
    plt.plot(X[:, 1], np.dot(X, normalEquation(X, y, lambda_)), '-', color='#f8a6a9', label="Normal Equation Method")
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
iteration = 4000
alpha = 0.0001
lambda_ = 3
theta, history = results(X, y, theta, alpha, lambda_, iteration)
print("Compare with Normal Equation Results")
print("Final Theta: ")
print(normalEquation(X, y, lambda_))
print("Final Cost: ")
print(cost_function(X, y, normalEquation(X, y, lambda_), lambda_))
show(X, y, theta, history, lambda_)
