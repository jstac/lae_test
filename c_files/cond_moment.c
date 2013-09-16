
/*
    John Stachurski, March 2012

    Functions for implementing the conditional moment test in Pritsker

*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_fit.h>

#include "utilities.h"
#include "ar1_functions.h"

     
double cond_m_compute_power(int (*generator)(double *, double *, int, unsigned long int), 
        double * params, 
        int n,
        int power_reps)
{
    /* 
     * Computes power of conditional moment test in Pritsker 1998 RFStud
     *
     * The first two arguments to the function are 'generator',
     * a function pointer that points to a function for generating time series
     * from the alternative, and a double pointer 'params' that points to an
     * array of parameters that will be passed to generator.
     *
     */
    double X[n];
    double R[n-1];
    double T[power_reps];
    int i, t;
    double r, sx, se, ts, sighat, u, cov00, cov01, cov11, chisq;
    double betahat, alphahat, shat;
    srand(time(0));
    int k = rand() % 1000000;
    for (i = 0; i < power_reps; i++) 
    {
        generator(X, params, n, k + i);
        fit_params(X, n, &betahat, &alphahat, &shat);
        // Compute squared residuals
        for (t = 0; t < n - 1; t++) 
         {
             r = X[t+1] - betahat - alphahat * X[t];
             R[t] = r * r;
         }
        // Regress R on X and a constant
        gsl_fit_linear (X, 1, R, 1, n-1, 
                        &betahat, &alphahat, &cov00, &cov01, &cov11, &chisq);
        // Compute sigma hat for this regression
        double sum = 0.0;
        for (t = 0; t < n - 1; t++) 
         {
             u = R[t] - betahat - alphahat * X[t];
             sum += u * u;
         }
        sighat = sqrt(sum / (n - 1));
        // And compute the test statistic for H_0 : alphahat = 0
        sx = sqrt(var(X, n-1));
        se = sighat / (sx * sqrt(n-1));
        ts = alphahat / se;
        //printf("ts = %g\n", ts);
        T[i] =  abs(ts) > 1.96;
    }
    return mean(T, power_reps);
}

