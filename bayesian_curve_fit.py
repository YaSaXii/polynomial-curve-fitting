from math import *
from random import *
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

"""
Bayesian curve fitting of sin(2*pi*x) over [0, 1] using polynomials and
Gaussian distributions for noise and the weights prior.
"""

# Degree of the polynomial
M = 9
# Number of training points
N = 10
# Precision of targets
beta = 11.1
# Precision of prior weights distribution
alpha = 0.00

def f(x):
    return sin(2*pi*x)

def noise():
    return gauss(0, beta**-1)

#training = [random() for i in range(N)]
training = np.arange(0, 1.0*(1 + 1.0 / N), 1.0 / N)
targets = [f(x) + noise() for x in training]

def phi(x):
    return np.array([[x**i] for i in range(M + 1)])

I = np.identity(M + 1)

result = np.sum([phi(x).dot(phi(x).T) for x in training], axis=0)
S = np.linalg.inv(beta*result + alpha*I)

# Mean of Gaussian distribution of t corresponding to x
def m(x):
    result = np.sum([t*phi(a) for a, t in zip(training, targets)], axis=0)
    return beta*(phi(x).T).dot(S.dot(result))[0][0]

# Variance of Gaussian distribution of t corresponding to x
def var(x):
    return (beta**-1 + (phi(x).T).dot(S.dot(phi(x))))[0][0]

def color(a):
    return (1 - (norm.cdf(a) - norm.cdf(a - 0.05)) / (norm.cdf(0.05) - norm.cdf(0)))

points = np.arange(0, 1.005, 0.005)
    
for a in np.arange(0.1, 3.05, 0.05):
    c = matplotlib.colors.rgb2hex((1, color(a), color(a)))
    plt.fill_between(points, [m(x) + (a - 0.1)*var(x) for x in points], [m(x) + a*var(x) for x in points], color=c)
    plt.fill_between(points, [m(x) - a*var(x) for x in points], [m(x) - (a - 0.1)*var(x) for x in points], color=c)

plt.plot(points, [m(x) for x in points], color='0')    

for x, t in zip(training, targets):
    plt.scatter(x, t, color='b')

plt.plot(points, [f(x) for x in points], color='g')


axes = plt.gca()
axes.set_ylim([-1.5, 1.5])
axes.set_xlim([0, 1])

    
plt.show()




