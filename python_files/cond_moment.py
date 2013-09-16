
"""

John Stachurski, April 2012

Implements the conditional moment test used by Pritsker (1998).

"""

from __future__ import division
import numpy as np
import scikits.statsmodels.api as sm
import ar1_functions


def cm_test(X):
    """
    Conditional moment test.  X is a flat numpy array.
    """
    betahat, alphahat, shat = ar1_functions.fit(X)
    n = len(X)
    xL = X[:(n-1)]  #  All but the last one
    xF = X[1:]      #  All but the first one
    Z = (xF - betahat - alphahat * xL)**2 
    XX = sm.add_constant(xL)
    out = sm.OLS(Z, XX).fit()
    return np.abs(out.tvalues[0]) > 1.96
