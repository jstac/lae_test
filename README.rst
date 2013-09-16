.. _lae_test:

************************************************
Goodness of Fit for Markov Models
************************************************


This page collects files and computer code for the paper **Goodness of Fit for
Markov Models: A Density Approach** by Vance Martin, Yoshihiko Nishiyama and
John Stachurski.

Abstract
----------

We propose a density-based goodness of fit test suitable for time series data.
The test compares the data against a parametric class of models specified in
the null hypothesis.   Estimation of smoothing parameters is not required, and
the test has nontrivial power against :math:`1/\sqrt{n}` local alternatives. 

Code
-------

Implementations are provided in both Python and C.  The Python code is more
straightforward and portable, but the C code is significantly faster.

