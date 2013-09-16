
"""

John Stachurski, April 2012

Implements test of the AR1 null against an alternative specified below.  Other
alternatives can be selected from the classes in alternatives.py.

"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as G
import ar1_null 
import ar1_functions
import alternatives
import cond_moment


def compute_power(alt_mod=None, ts_length=500, replications=1000, cv_reps=1000):
    """
    Computes the rejection frequency for a given null and alternative.
    """
    outcomes = []
    for m in range(replications):
        # Generate time series from alternative
        X = alt_mod.ts(ts_length)
        # Estimate parameters under the null
        betahat, alphahat, shat = ar1_functions.fit(X)
        cvs = v_test.compute_critical(betahat, alphahat, shat, 
                num_reps=cv_reps, ts_length=ts_length)
        T = v_test.test_stat(betahat, alphahat, shat, X)
        outcomes.append(T > cvs)
    return sum(outcomes) / replications


v_test = ar1_null.Test()  

alt = alternatives.ART(beta=1.0, alpha=0.9, s=1, gamma=0)
gammas = np.linspace(0.01, 0.3, 2)
for gamma in gammas:
    alt.gamma = gamma
    lae_freq = compute_power(alt_mod=alt, replications=1000, cv_reps=1000)
    print("%f %f\n" % (gamma, lae_freq))



