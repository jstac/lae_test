
""" 

John Stachurski, April 2012

Implements the test with a one-dimensional Gaussian AR1 null X' = beta + alpha
X + s Z.  Here a prime is next period's value and Z is standard normal.  The
three parameters are beta, alpha and s.  Given the Gaussian AR1 assumption,
the three parameters beta, alpha and s define a particular density kernel p.
The test statistic is computed using this p and data that is passed to the
class methods.

"""

from __future__ import division
import numpy as np
from scipy.integrate import quad, simps
import ar1_functions


class Test:
    """
    A Test object stores the test size.
    """

    def __init__(self, test_size=0.05):
        self.test_size = test_size

    def test_stat(self, beta, alpha, s, X):
        """
        Evaluate the test statistic given the model (kernel) defined by null
        X' = beta + alpha X + s Z, and a real-valued time series X.  A
        vectorized routine.
        """
        null = ar1_functions.AR1(beta=beta, alpha=alpha, s=s)
        a1 = null.stat_mean - 4 * null.stat_sd
        b1 = null.stat_mean + 4 * null.stat_sd
        xm = np.mean(X)
        xsd = np.sqrt(np.var(X))
        a2 = xm - 4 * xsd
        b2 = xm + 4 * xsd
        a = min(a1, a2)
        b = max(b1, b2)
        K = 50      # No. of grid points for numerical integration
        Y = np.linspace(a, b, K)  # Grid points
        n = len(X.flatten())  # Number of data points
        CX = np.reshape(X, (n, 1))  # Make X a column vec
        RY = np.reshape(Y, (1, K))  # Make Y a row vec
        V = null.p(CX, RY) 
        vals = (np.mean(V, axis=0) - null.psi(Y))**2
        return n * simps(vals.flatten(), Y.flatten())  # Numerical integral


    def test_stat_nv(self, beta, alpha, s, X):
        """
        Non-vectorized test_stat routine (much slower, for debugging).
        """
        null = ar1_functions.AR1(beta=beta, alpha=alpha, s=s)
        if self.w:
            g = lambda y: self.w(y) * (np.mean(null.p(X, y)) - null.psi(y))**2 
        else:
            g = lambda y: (np.mean(null.p(X, y)) - null.psi(y))**2 
        return len(X) * quad(g, -np.inf, np.inf)[0]


    def compute_critical(self, beta, alpha, s, ep=True, num_reps=500, ts_length=264):
        """ 
        Compute the critical value of the test stat associated with parameters
        (beta, alpha, s) and AR1 null by Monte Carlo.  The flag ep determines
        whether or not the test is with estimated parameters.  (Note that the
        test statistic has a different asymptotic distribution when parameters
        are estimated---see the paper.)
        """
        # Step 1: Compute num_reps observations of the test statistic 
        T = np.empty(num_reps)        
        a1 = ar1_functions.AR1(beta=beta, alpha=alpha, s=s)
        data = a1.sim_data(num_reps, ts_length)  # Simulate under null
        for k in range(num_reps):
            if ep:
                betahat, alphahat, shat = ar1_functions.fit(data[k,:])
                T[k] = self.test_stat(betahat, alphahat, shat, data[k,:])
            else:
                T[k] = self.test_stat(beta, alpha, s, data[k,:])
        # Step 2: Compute and return (1 - size) quantile of test stat observations
        T.sort()
        return T[int((1 - self.test_size) * num_reps)]

