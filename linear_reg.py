import numpy as np
import matplotlib.pyplot as plt

# Import or Create random Data Sets for Testing
def randomData(start, end, size):
	X = np.random.randint(start, end, (size,))
	y = np.random.randint(start, end, (size,))
	return X, y

def linearCluster(interval, amount, size):
	X = np.random.randint(0, interval + 20, (size//amount,))
	y = np.random.randint(0, interval + 20, (size//amount,))
	for i in range(2, amount + 1):
		temp1 = np.random.randint(-10 + interval * (i-1), interval*i + 20, (size//amount,))
		temp2 = np.random.randint(-10 + interval * (i-1), interval*i + 20, (size//amount,))
		X = np.append(X, temp1)
		y = np.append(y, temp2)
	return X, y

X, y = linearCluster(50, 2, 90)

def featureScaling(X, y):
	rangeX = max(X) - min(X)
	rangey = max(y) - min(y)
	return (X/rangeX), (y/rangey)

# Initial Theta Val
theta = [0, 0]

# Training set Size
m = y.size

# Feature Scale Training Set
#X, y = featureScaling(X, y)

# Create additional dimension of 1 for X
X = np.stack([np.ones(m), X], axis=1)

def cost_function(X, y, theta):
	m = y.size
	J = (1 / (2 * m)) * sum(np.square(np.dot(X, theta) - y))
	return J

def gradient_descent(X, y, theta, alpha, iteration):
	m = y.size
	theta = theta.copy()
	history = []
	for i in range(iteration):
		theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
		history.append(cost_function(X, y, theta))	
	return theta, history

def results(X, y, theta, alpha, iteration):
	theta, history = gradient_descent(X, y, theta, alpha, iteration)
	print("Final Theta:")
	print(theta)
	print("Cost Function History")
	print(history[:10])
	print("......")
	print(history[iteration-10:iteration])
	return theta, history

def show(X, y, theta, history):
	fig = plt.figure(1)
	plt.plot(X[:,1], y, 'o', color='#c6aadf', ms=10, mec='k', label = "Training Set")
	plt.plot(X[:,1], np.dot(X, theta), '-',color='#bae4ff', label = "Hypothesis Function")
	plt.ylabel('Random Data')
	plt.xlabel('Random Data')
	plt.legend()
	checkProg(history)
	plt.show()

def checkProg(history):
	fig = plt.figure(2)
	time = [i for i in range(len(history))]
	plt.plot(time, history, '-', color='#c6aadf')
	plt.ylabel("Cost")
	plt.xlabel("No. of Iterations")

iteration = 20
alpha = 0.00005
theta, history = results(X, y, theta, alpha, iteration)
show(X, y, theta, history)


