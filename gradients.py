import numpy as np
import tensorflow as tf
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


X = 2*np.random.rand(100,1)
X_b = np.c_[np.ones((100,1)), X]
y = 4 + 3*X + np.random.rand(100,1)


eta = 0.1
n_iteration = 10
m = 100


theta = np.random.rand(2,1)

for i in range(n_iteration):
    
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta*gradients
    y_reg = theta[0] + theta[1]*np.linspace(0,2)
    if i % 1 == 0:
        print(*gradients)
        plt.plot(np.linspace(0,2), y_reg)
    
    
#print(theta)
plt.plot(X,y, 'ob')