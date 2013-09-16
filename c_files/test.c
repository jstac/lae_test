/*
 * John Stachurski, Feb 2012
 *
 * A test file for quick calculations
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utilities.h"
#include <gsl/gsl_errno.h>
#include "ar1_functions.h"
#include "ar1_null.h"
#include "gaussian_cvm.h"
#include "alternatives.h"

     

int main(void)
{
    gsl_set_error_handler_off();

    // Quick power calculation
    int power_reps = 100;
    int cv_reps = 100;
    int n = 500;

    double beta = 1;
    double alpha = 0.9;
    double s = 0.1;
    double gamma = 0.5;
    double params[4] = {beta, alpha, s, gamma};
    double p;
    p = cvm_compute_power(&art_ts, params, n, power_reps, cv_reps);
    printf("CvM rejection rate: %g\n", p);
    p = ar1_compute_power(&art_ts, params, n, power_reps, cv_reps);
    printf("LAE rejection rate: %g\n", p);

    /*

    // Calculate a critical value
    int n = 1000;
    int reps = 1000;
    double cvs;
    double mu = 0.0;
    double s = 1.0;
    cvs = cvm_cv_est_params(mu, s, n, reps);
    printf("cvs = %g\n", cvs);
    // End calculating critical value

    
    
    */

    return 0;

}

