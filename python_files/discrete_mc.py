""" 

John Stachurski, 2011

Provides a function for generating sample paths of discrete Markov chains 

"""

import numpy as np
from random import uniform


class DiscreteRV:
    """
    Each instance is provided with an array of probabilities q. 
    The draw() method returns x with probability q[x].
    """

    def __init__(self, q):
        self.set_q(q)

    def set_q(self, q):
        self.Q = np.cumsum(q)   # Cumulative sum

    def draw(self):
        """
        Returns n draws from q
        """
        return self.Q.searchsorted(uniform(0, 1)) 


def sample_path(p, init=0, sample_size=1000): 
    """
    A function that generates sample paths of a finite Markov chain with 
    kernel p on state space S = [0,...,N-1], starting from state init.

    Parameters: 

        * p is a 2D NumPy array, nonnegative, rows sum to 1
        * init is an integer in S
        * sample_size is an integer

    Returns: A flat NumPy array containing the sample
    """
    N = len(p)
    # Let P[x] be the distribution corresponding to p[x,:]
    P = [DiscreteRV(p[x,:]) for x in range(N)]
    X = np.empty(sample_size, dtype=int)
    X[0] = init
    for t in range(sample_size - 1):
        X[t+1] = P[X[t]].draw()
    return X
