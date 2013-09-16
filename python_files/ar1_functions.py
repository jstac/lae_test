
"""

John Stachurski, April 2012

Functions for manipulating AR(1) processes

"""

from __future__ import division
import numpy as np
from scipy.stats import norm, t
from numpy.random import uniform
from scipy import randn
from numpy import dot
from numpy.linalg import inv


def cont_time_params(kappa, b, sigma, delta):
    """
    A function to convert the Vasicek model  

       dX = kappa(b - X) dt + sigma dW,   W standard Brownian motion

    into a discrete AR1 model.  Param delta in (0, 1] is a fraction of one
    year, and indicates the time length of the step.  Discretized, the model
    is represented as the AR1 process

        X' = beta + alpha X + s Z

    """
    alpha = np.exp(- kappa * delta)
    beta = b * (1 - np.exp(- kappa * delta))
    s2 = sigma * sigma * (1 - np.exp(- 2 * kappa * delta)) / (2 * kappa)
    s = np.sqrt(s2)
    return beta, alpha, s


def fit(X):
    """
    Fits parameters of AR1 model X' = beta + alpha X + s W to the data in
    vector X by max likelihood.
    """
    n = len(X) - 1
    xx = np.ones((n, 2))
    xx[:,1] = X[:-1]
    xx_trans = xx.transpose()
    yy = X[1:]
    coeffs = dot(dot(inv(dot(xx_trans, xx)), xx_trans), yy)
    betahat, alphahat = coeffs
    uhat = yy - dot(xx, coeffs)
    sigma2hat = dot(uhat, uhat) / n
    return betahat, alphahat, np.sqrt(sigma2hat)


class AR1:
    """
    A class to represent AR1 processes.  The method set_constants needs to be
    called whenever parameter values are changed.
    """

    def __init__(self, beta=0.0, alpha=0.0, s=1.0):
        self.beta, self.alpha, self.s = beta, alpha, s
        self.set_constants()

    def set_constants(self):
        # Additional constants
        self.stat_mean = self.beta / (1 - self.alpha)  # Mean of stationary 
        self.stat_var = self.s**2 / (1 - self.alpha**2)  # Variance of stationary
        self.stat_sd = np.sqrt(self.stat_var)

    def psi(self, x):
        "The stationary distribution for the Vasicek model."
        return norm.pdf(x, loc=self.stat_mean, scale=self.stat_sd)

    def sim_data(self, K, N):
        """
        Draws K stationary time series of length N from the Vasicek model and
        returns them as a K x N array.
        """
        X = np.zeros((K, N))
        X[:,0] = norm.rvs(size=K, loc=self.stat_mean, scale=self.stat_sd)
        for t in range(1, N):
            X[:,t] = self.beta + self.alpha * X[:,t-1] + self.s * norm.rvs(size=K)
        return X

    def p(self, x, y):
        '''
        The transition density: A density in the variable y.  The function is 
        vectorized in both arguments.
        '''
        return norm.pdf(y - self.beta - self.alpha * x, scale=self.s)

    def est_params(self, Xdata):
        self.alpha, self.beta, self.s = fit(Xdata)
        self.set_constants()

