"""

John Stachurski, Feb 2012

A collection of models that serve as the alternative.  All classes provide the
method ts(), where ts(n) simulates and returns a time series of length n.

"""

from __future__ import division
import numpy as np
from scipy.stats import norm
from scipy.stats import t as tdist
from scipy.stats import gamma as G
from numpy.random import uniform
from scipy import randn
import discrete_mc 



class LE:
    """
    Continuous time level effects model

        dX = kappa(b - X) dt + sigma X^gamma dW,   W Brownian motion

    The discretized version of the model has form

        X' = X + kappa (b - X) delta + X^gamma sigma sqrt(delta) Z

    where a prime denotes next period's value, and Z is N(0,1).
    """

    def __init__(self, kappa=0, b=0, sigma=0, gamma=0, delta=0):
        self.kappa = kappa
        self.b = b
        self.sigma = sigma
        self.gamma = gamma
        self.delta = delta

    def ts(self, n): 
        """ 
        Generates a series from the discretized version.  As currently
        written, will fail if state is negative.
        """
        V = self.sigma * np.sqrt(self.delta) * norm.rvs(size=n) 
        X = np.empty(n)
        X[0] = self.b
        for t in range(1, n): 
            x = max(X[t-1], 0)  # Can't raise negative number to gamma
            X[t] = x + self.kappa * (self.b - x) * self.delta \
                    + (x**self.gamma) * V[t] 
        return X


class ARMA:
    """
    ARMA model X' = beta + alpha X + gamma Z + s Z'
    """

    def __init__(self, alpha=None, beta=None, gamma=None, s=None):
        self.alpha, self.beta, self.gamma, self.s = alpha, beta, gamma, s

    def ts(self, n): 
        Z = norm.rvs(size=n) 
        X = np.empty(n)
        X[0] = self.beta / (1 - self.alpha)  # Stationary mean
        for t in range(1, n): 
            X[t] = self.beta + self.alpha * X[t-1] \
                    + self.gamma * Z[t-1] + self.s * Z[t]
        return X


class ART:
    """
    AR1 model X' = beta + alpha X + s Z' where Z is t(1/gamma)
    """

    def __init__(self, alpha=None, beta=None, gamma=None, s=None):
        self.alpha, self.beta, self.gamma, self.s = alpha, beta, gamma, s

    def ts(self, n): 
        Z = tdist.rvs(1/self.gamma, size=n) 
        X = np.empty(n)
        X[0] = self.beta / (1 - self.alpha)  # Stationary mean
        for t in range(1, n): 
            X[t] = self.beta + self.alpha * X[t-1] + self.s * Z[t]
        return X


class ARSKEW:
    """
    AR1 model X' = beta + alpha X + s Z' where Z is some skewed distribution
    """

    def __init__(self, alpha=None, beta=None, gamma=None, s=None):
        self.alpha, self.beta, self.gamma, self.s = alpha, beta, gamma, s

    def ts(self, n): 
        Z = G.rvs(2, scale=2, size=n) 
        X = np.empty(n)
        X[0] = self.beta / (1 - self.alpha)  # Stationary mean
        for t in range(1, n): 
            X[t] = self.beta + self.alpha * X[t-1] + self.s * Z[t]
        return X


class SV:
    """
    Stochastic Volatility model

        X[t+1] = beta + alpha X[t] + s[t] Z[t+1]
        s[t+1] = b * s[t]^rho * exp{ gamma W[t+1] }

    where both shocks are independent N(0,1).  When gamma = 0 we want
    to recover a certain null, where s is specified and constant at s0.
    To do so we can set b = s0^(1-rho).
    """
    def __init__(self, alpha=0, beta=0, s0=0, gamma=0, rho=0.5):
        self.s0 = s0  # Initial condition for s
        self.rho = rho
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.b = self.s0**(1 - self.rho)

    def set_s0(self, x):
        self.s0 = x  # Initial condition for s
        self.b = self.s0**(1 - self.rho)

    def ts(self, n): 
        Z = norm.rvs(size=n)
        W = norm.rvs(size=n)
        X = np.empty(n)
        s = np.empty(n)  # Holds log of s
        s[0] = self.s0
        X[0] = self.beta / (1 - self.alpha)
        for t in range(1, n): 
            s[t] = self.b * (s[t-1]**(1 - self.rho)) * np.exp(self.gamma * W[t])
            X[t] = self.beta + self.alpha * X[t-1] + s[t-1] * Z[t]
        return X


class NLAR:
    """
    Nonlinear autoregression model X' = beta + alpha X + gamma g(X) + s Z'
    where g(x) = |x| / |x + 2|.
    """

    def __init__(self, alpha=None, beta=None, gamma=None, s=None):
        self.alpha, self.beta, self.gamma, self.s = alpha, beta, gamma, s

    def ts(self, n): 
        Z = norm.rvs(size=n) 
        X = np.empty(n)
        X[0] = self.beta / (1 - self.alpha)  # Stationary mean
        for t in range(1, n): 
            X[t] = self.beta + self.alpha * X[t-1] + self.gamma * \
                np.abs(X[t-1]) / np.abs(X[t-1] + 2) + self.s * Z[t]
        return X


class LSTAR:
    """
    Logistic STAR model X' = beta + alpha X + gamma g(X) + s Z'
    where g(x) = x / (1 + exp(-x/2)).
    """

    def __init__(self, alpha=None, beta=None, gamma=None, s=None):
        self.alpha, self.beta, self.gamma, self.s = alpha, beta, gamma, s

    def ts(self, n): 
        Z = norm.rvs(size=n) 
        X = np.empty(n)
        X[0] = self.beta / (1 - self.alpha)  # Stationary mean
        for t in range(1, n): 
            X[t] = self.beta + self.alpha * X[t-1] + self.gamma * \
                X[t-1] / (1 + np.exp(- X[t-1] / 2)) + self.s * Z[t]
        return X


class RW:
    """
    Gaussian random walk.
    """
    def __init__(self, sigma, initial):
        self.sigma, self.initial = sigma, initial

    def ts(self, n):
        V = self.sigma * norm.rvs(size=n) 
        X = np.empty(n)
        X[0] = self.initial
        for t in range(1, n): 
            X[t] = X[t-1] + V[t] 
        return X


class GARCH:
    "GARCH DGP"

    def __init__(self, a0=0.05, a1=0.05, b=0.9):
        self.a0, self.a1, self.b = a0, a1, b

    def ts(self, n):
        R = np.empty(n)
        X = np.empty(n)
        X[0] = 1
        for t in range(n-1):
            R[t] = np.sqrt(X[t]) * randn(1)
            X[t+1] = self.a0 + self.b * X[t] + self.a1 * R[t]**2
        R[n-1] = np.sqrt(X[n-1]) * randn(1)
        return R



class RSW:
    " Regime switching DGP "

    def __init__(self, p, beta, alpha, s):
        self.p = p  # A 
        self.alpha, self.beta, self.s = alpha, beta, s

    def ts(self, n, y0=0):
        """ Generates a time series of length n, and returns it as
        a numpy array.  """
        sobs = discrete_mc.sample_path(self.p, init=0, sample_size=n)
        Z = norm.rvs(size=n)
        y = np.empty(n)
        for i in range(n - 1):
            y[i+1] = self.beta[sobs[i]] + self.alpha * y[i] + self.s * Z[i]
        return y



class Setar:
    " SETAR DGP "

    def __init__(self, params = [-1, .8, 1, .8]):
        self.params = params
        self.sigma = 1.5

    def sigmoid(self, x): return 1.0 / (1.0 + np.exp(-x))
                
    def mu(self, x):
        a = (self.params[0] + self.params[1] * x) * (1 - self.sigmoid(x))
        b = (self.params[2] + self.params[3] * x) * self.sigmoid(x)
        return a + b

    def ts(self, n):
        X = np.empty(n)
        X[0] = 0
        W = randn(n)
        for t in range(n - 1):
            X[t+1] = self.mu(X[t]) + self.sigma * W[t]
        return X


