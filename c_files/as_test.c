
/*
 
    John Stachurski, Feb 2012

    Functions for implementing the Ait-Sahalia test with null being N(mu, s^2)

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


double comp_bandwidth(double * X, int n)
{
    // Computes bandwidth by Silverman's rule
    double shat = sqrt(var(X, n));
    return 1.06 * shat * pow(n, - 1.0 / 5.0);
}

double npkde(double * X, double n, double bw, double y)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < n; i++)
    {
        sum += gsl_ran_gaussian_pdf((X[i] - y) / bw, 1);
    }
    return (1.0 / bw) * (sum / n);
}
     

double as_test_stat(double beta, double alpha, double s, double * X, int n)
{
    double mu = beta / (1 - alpha);
    double sigma = sqrt( (s * s) / (1 - alpha * alpha) );

    double c1 = 1.0 / sqrt(2.0 * 3.141592);
    double c2 = 1.0 / (2.0 * sqrt(3.141592)); 

    //double bw = 0.0124;
    double bw = comp_bandwidth(X, n);

    int i;
    double esum = 0.0;
    double vsum = 0.0;
    double msum = 0.0;
    for (i = 0; i < n; i++)
    {
        double t = npkde(X, n, bw, X[i]);
        double u = gsl_ran_gaussian_pdf(X[i] - mu, sigma);
        esum += t;
        vsum += t * t * t;
        msum += (u - t) * (u - t);
    }
    double E = c2 * esum / n;
    double V = c1 * vsum / n;
    double M = n * bw * msum / n;
    return fabs(pow(bw * V, -0.5) * (M - E));
}


double as_cv_fixed_params(double beta, double alpha, double s, int n, int num_reps)
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
        T[i] = as_test_stat(beta, alpha, s, X, n);
    }
    gsl_sort(T, 1, num_reps);
    return gsl_stats_quantile_from_sorted_data(T, 1, num_reps, 0.95);
}



double as_cv_est_params(double beta, double alpha, double s, int n, int num_reps)
{
    double X[n];
    double T[num_reps];
    int i;
    srand(time(0));
    int k = rand() % 1000000;
    double params[3] = {beta, alpha, s}; 
    double betahat;
    double alphahat;
    double shat;
    for (i = 0; i < num_reps; i++) 
    {
        ar1_ts(X, params, n, k + i);  // IID draws, because alpha=0
        fit_params(X, n, &betahat, &alphahat, &shat);
        T[i] = as_test_stat(betahat, alphahat, shat, X, n);
    }
    gsl_sort(T, 1, num_reps);
    return gsl_stats_quantile_from_sorted_data(T, 1, num_reps, 0.95);
}


double as_compute_power(int (*generator)(double *, double *, int, unsigned long int), 
        double * params, 
        int n,
        int power_reps,
        int cv_reps)
{
    /* 
     * Computes power of AS test with AR(1) null, estimated parameters, and
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
        generator(X, params, n, k + i);
        fit_params(X, n, &betahat, &alphahat, &shat);
        cv = as_cv_est_params(betahat, alphahat, shat, n, cv_reps);
        ts = as_test_stat(betahat, alphahat, shat, X, n);
        T[i] =  ts > cv;
    }
    return mean(T, power_reps);
}


