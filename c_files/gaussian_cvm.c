
/*
 
    John Stachurski, Feb 2012

    Functions for implementing the CvM test with null being N(mu, s^2)

*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>

#include "utilities.h"
#include "ar1_functions.h"
#include "alternatives.h"

     
double cvm_test_stat(double beta, double alpha, double s, double * X, int n)
{
    double mu = beta / (1 - alpha);
    double sigma = sqrt( (s * s) / (1 - alpha * alpha) );
    gsl_sort(X, 1, n);
    double sum = (1.0 / (12.0 * n));
    int i;
    double a;
    for (i = 0; i < n; i++)
    {
        a = gsl_cdf_gaussian_P(X[i] - mu, sigma) - (2.0 * i - 1.0) / (2.0 * n);
        sum += a * a;
    }
    return sum;
}


double cvm_cv_fixed_params(double beta, double alpha, double s, int n, int num_reps)
{
    double X[n];
    double T[num_reps];
    int i;
    srand(time(0));
    int k = rand() % 1000000;
    double params[3] = {beta, alpha, s}; 
    for (i = 0; i < num_reps; i++) 
    {
        ar1_ts(X, params, n, k + i);  // IID draws, because alpha=0
        T[i] = cvm_test_stat(beta, alpha, s, X, n);
    }
    gsl_sort(T, 1, num_reps);
    return gsl_stats_quantile_from_sorted_data(T, 1, num_reps, 0.95);
}



double cvm_cv_est_params(double beta, double alpha, double s, int n, int num_reps)
{
    // Critical value with estimated parameters
    double X[n];
    double T[num_reps];
    int i;
    double params[3] = {beta, alpha, s}; 
    double betahat;
    double alphahat;
    double shat;
    srand(time(0));
    int k = rand() % 1000000;
    for (i = 0; i < num_reps; i++) 
    {
        ar1_ts(X, params, n, k + i);
        fit_params(X, n, &betahat, &alphahat, &shat);
        T[i] = cvm_test_stat(betahat, alphahat, shat, X, n);
    }
    gsl_sort(T, 1, num_reps);
    return gsl_stats_quantile_from_sorted_data(T, 1, num_reps, 0.95);
}


double cvm_compute_power(int (*generator)(double *, double *, int, unsigned long int), 
        double * params, 
        int n,
        int power_reps,
        int cv_reps)
{
    /* 
     * Computes power of test with Gaussian null, estimated parameters, and
     * given alternative.  The first two arguments to the function are 'generator',
     * a function pointer that points to a function for generating time series
     * from the alternative, and a double pointer 'params' that points to an
     * array of parameters that will be passed to generator.
     *
     */
    double X[n];
    double T[power_reps];
    int i;
    double betahat, alphahat, shat;
    double cv, ts;
    srand(time(0));
    int k = rand() % 1000000;
    for (i = 0; i < power_reps; i++) 
    {
        //printf("i = %d\n", i);
        generator(X, params, n, k + i);
        fit_params(X, n, &betahat, &alphahat, &shat);
        cv = cvm_cv_est_params(betahat, alphahat, shat, n, cv_reps);
        //printf("cv = %g\n", cv);
        ts = cvm_test_stat(betahat, alphahat, shat, X, n);
        //printf("ts = %g\n", ts);
        T[i] =  ts > cv;
    }
    return mean(T, power_reps);
}

