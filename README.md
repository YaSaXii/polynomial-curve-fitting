# polynomial-curve-fitting
Python implementation of polynomial curve fitting using Bayesian methods.

This program produces a visualisation of fitting the function f(x) = sin(2*pi*x), including the precision beta (set to 11.1 by default), using polynomial curves of a given degree M (set to 9 by default). The best fitting curve is then displayed as a black curve. The best fitting curve is the mean of a Gaussian distribution of polynomial curves. The variance of the values these curves take may be calculated for each x, and therefore one may visualise the probability density of a value y for each x given a new data point. The colour red is used, where the intensity correspondeds to the probability density, normalised by the probability density of the interval within 0.05 of the mean. An example output for the default values is provided in example.png.

A detailed explanation and mathematical derivation of the algorithm may be found in 'Pattern Recognition and Machine Learning' by Bishop.

