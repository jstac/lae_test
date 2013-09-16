
/* This file is depreciated and not used currently for any simulations */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "utilities.h"
#include <gsl/gsl_errno.h>
#include "ar1_functions.h"
#include "ar1_null.h"
#include "alternatives.h"
#include "gaussian_cvm.h"
#include "cond_moment.h"
#include "as_test.h"
     

int main(void)
{
    gsl_set_error_handler_off();

    int power_reps = 100;
    int cv_reps = 100;
    int n = 264;
    
    double beta, alpha, s;
    // Specify the null
    double delta = 1 / 12.0;
    double kappa0 = 0.85837;
    //double kappa0 = 1.71674;
    double b0 = 0.089102;
    double sigma0 = sqrt(0.002185);
    //double sigma0 = sqrt(0.004371);
    cont_time_to_ar1(kappa0, b0, sigma0, delta, &beta, &alpha, &s);
    printf("# n = %i, beta = %g, alpha = %g, s = %g\n", n, beta, alpha, s);

    double lae_cv = ar1_cv_fixed_params(beta, alpha, s, n, cv_reps);
    double cvm_cv = cvm_cv_fixed_params(beta, alpha, s, n, cv_reps);
    double as_cv = as_cv_fixed_params(beta, alpha, s, n, cv_reps);

    printf("lae_cv = %g\n", lae_cv);
    printf("cvm_cv = %g\n", cvm_cv);
    printf("as_cv = %g\n", as_cv);


    // Specify alternative
    double kappa1 = 0.89218;
    double b1 = 0.090495;
    double sigma1 = sqrt(0.032742);
    double gamma = 0.5;
    double rho = 0.5;
    double d = 0.1;
    //double params[5] = {kappa1, b1, sigma1, delta, gamma};
    double params[7] = {kappa1, b1, sigma1, delta, gamma, rho, d};

    double X[n];
    double lae_T[power_reps];
    double cvm_T[power_reps];
    double as_T[power_reps];
    int i;
    double lae_ts, cvm_ts, as_ts;
    srand(time(0));
    int k = rand() % 1000000;
    for (i = 0; i < power_reps; i++) 
    {
        //level_effects_ts(X, params, n, k + i);
        sv_level_effects_ts(X, params, n, k + i);
        lae_ts = ar1_test_stat(beta, alpha, s, X, n);
        cvm_ts = cvm_test_stat(beta, alpha, s, X, n);
        as_ts = as_test_stat(beta, alpha, s, X, n);
        //printf("as_ts = %g\n", as_ts);
        lae_T[i] =  lae_ts > lae_cv;
        cvm_T[i] =  cvm_ts > cvm_cv;
        as_T[i] =  as_ts > as_cv;
    }
    printf("LAE rejection freq: %g\n", mean(lae_T, power_reps));
    printf("CvM rejection freq: %g\n", mean(cvm_T, power_reps));
    printf("AS rejection freq: %g\n", mean(as_T, power_reps));


    return 0;

}

