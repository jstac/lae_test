

/*
 
John Stachurski, Jan 2012
Functions for implementing the LAE test with null being the AR1 model

    X' = beta + alpha x + s Z    where   Z ~ N(0,1)

*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>

#include "utilities.h"
#include "ar1_functions.h"
#include "alternatives.h"

     
double ar1_integrand(double y, void * params) {
    /* 
     * Integrand for the test statistic.  This will form the function part of
     * the gsl_function used for integration.  Hence params is a void pointer.
     * It will be cast into a pointer to a double pointer.  It has the form
     *
     *    params = &beta, &alpha, &s, &X, &n, &stat_mean, &stat_sd
     *
     */
    double ** p = (double **) params;
    double beta = *p[0]; double alpha = *p[1]; double s = *p[2];
    double * X = p[3];
    int n = (int) *p[4];
    double stat_mean = *p[5];
    double stat_sd = *p[6];
    double sum = 0.0;
    int i;
    for (i = 0; i < n; i++) 
     {
         sum += gsl_ran_gaussian_pdf(y - beta - alpha * X[i], s);
     }
    double q = gsl_ran_gaussian_pdf(y - stat_mean, stat_sd);
    return pow(sum / n - q, 2);
}
     

double ar1_test_stat(double beta, double alpha, double s, double * X, int n)
{
    double stat_mean = beta / (1 - alpha); 
    double stat_sd = sqrt((s * s) / (1 - alpha * alpha));

    // Find a suitable interval for integration
    double a1 = stat_mean - 4 * stat_sd;
    double b1 = stat_mean + 4 * stat_sd;
    double xm = mean(X, n);
    double xsd = sqrt(var(X, n));
    double a2 = xm - 4 * xsd;
    double b2 = xm + 4 * xsd;
    double a = (a1 < a2) ? a1 : a2;
    double b = (b1 > b2) ? b1 : b2;

    // Build an array of pointers to doubles in order to convert the 
    // function 'ar1_integrand' into a gsl_function suitable for integration
    double m = (double) n;
    double * params[7];
    params[0] = &beta;
    params[1] = &alpha;
    params[2] = &s;
    params[3] = X;
    params[4] = &m;
    params[5] = &stat_mean;
    params[6] = &stat_sd;
    gsl_function F;
    F.function = &ar1_integrand;
    F.params = params;

    double result; 
    //double error;
    //gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    //gsl_integration_qags(&F, a, b, 10, 10, 1000, w, &result, &error);
    
    //double error;
    //size_t ne;
    //gsl_integration_qng(&F, a, b, 10, 10, &result, &error, &ne); 
    
    gsl_integration_glfixed_table * ft = gsl_integration_glfixed_table_alloc (60);
    result = gsl_integration_glfixed (&F, a, b, ft);
    gsl_integration_glfixed_table_free(ft);

    return n * result;
}


double ar1_cv_fixed_params(double beta, double alpha, double s, int n, int num_reps)
{
    // Critical value when the null is a fixed (i.e., parameterized) AR(1)
    double X[n];
    double T[num_reps];
    int i;
    srand(time(0));
    int k = rand() % 1000000;
    double params[3] = {beta, alpha, s}; 
    for (i = 0; i < num_reps; i++) 
    {
        ar1_ts(X, params, n, k + i);
        T[i] = ar1_test_stat(beta, alpha, s, X, n);
    }
    gsl_sort(T, 1, num_reps);
    return gsl_stats_quantile_from_sorted_data(T, 1, num_reps, 0.95);
}



double ar1_cv_est_params(double beta, double alpha, double s, int n, int num_reps)
{
    // Critical value for the test with estimated parameters, calculated by
    // simulating under null.  The calculation returns critical value
    // c(theta), where theta = (beta, alpha, s).
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
        T[i] = ar1_test_stat(betahat, alphahat, shat, X, n);
    }
    gsl_sort(T, 1, num_reps);
    return gsl_stats_quantile_from_sorted_data(T, 1, num_reps, 0.95);
}



double ar1_size_distortion(int large_n, int small_n, double beta, double alpha, double s)
{
    /* Computes the asymtotic critical value using n = large_n, and then the
     * rejection frequency when n = small_n and the asymtotic critical value
     * is used.
     */
    int num_reps = 5000;
    double params[3] = {beta, alpha, s}; 
    double asymptotic_cv;
    asymptotic_cv = ar1_cv_est_params(beta, alpha, s, large_n, num_reps);
    printf("Asymptotic cv = %g, ", asymptotic_cv);
    double X[small_n];
    double T[num_reps];
    int i;
    double betahat;
    double alphahat;
    double shat;
    srand(time(0));
    int k = rand() % 1000000;
    for (i = 0; i < num_reps; i++) 
    {
        ar1_ts(X, params, small_n, k + i);
        fit_params(X, small_n, &betahat, &alphahat, &shat);
        T[i] = ar1_test_stat(betahat, alphahat, shat, X, small_n) > asymptotic_cv;
    }
    return mean(T, num_reps);
}


double ar1_compute_power(int (*generator)(double *, double *, int, unsigned long int), 
        double * params, 
        int n,
        int power_reps,
        int cv_reps)
{
    /* 
     * Computes power of test with AR(1) null, estimated parameters, and
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
        cv = ar1_cv_est_params(betahat, alphahat, shat, n, cv_reps);
        //printf("cv = %g\n", cv);
        ts = ar1_test_stat(betahat, alphahat, shat, X, n);
        //printf("ts = %g\n", ts);
        T[i] =  ts > cv;
    }
    return mean(T, power_reps);
}


