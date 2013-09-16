
/*
John Stachurski, March 2012
A collection of functions for working with the AR1 model defined by

    X' = beta + alpha x + s Z    where   Z ~ N(0,1)

*/


#include <stdio.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_fit.h>

#include "utilities.h"

int ar1_ts (double * x, double * params, int n, unsigned long int seed)
{
    double beta = params[0];
    double alpha = params[1];
    double s = params[2];
    /* create a generator chosen by the environment variable GSL_RNG_TYPE */
    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, seed);

    int i;
    x[0] = beta / (1 - alpha);  // Start at mean of stationary dist
    for (i = 1; i < n; i++) 
     {
       x[i] = beta + alpha * x[i-1] + gsl_ran_gaussian(r, s);
     }

    gsl_rng_free (r);
    return 0;
}

int cont_time_to_ar1(double kappa, double b, double sigma, double delta, double * beta, double * alpha, double * s)
{
    *beta = b * (1 - exp(- kappa * delta));
    *alpha = exp(- kappa * delta);
    double s2 = sigma * sigma * (1 - exp(- 2 * kappa * delta)) / (2 * kappa);
    *s = sqrt(s2);
    return 0;
}

int fit_params(double *x, int n, double * beta, double * alpha, double * s)
{
    double cov00, cov01, cov11, chisq;
    double * y = x + 1;
    gsl_fit_linear (x, 1, y, 1, n-1, 
                        beta, alpha, &cov00, &cov01, &cov11, &chisq);
    int i;
    double u;
    double sum = 0.0;
    for (i = 0; i < n - 1; i++) 
     {
         u = y[i] - (* beta) - (* alpha) * x[i];
         sum += u * u;
     }
    *s = sqrt(sum / (n - 1));
    return 0;
}
