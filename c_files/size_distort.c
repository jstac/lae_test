

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utilities.h"
#include <gsl/gsl_errno.h>
#include "ar1_functions.h"
#include "ar1_null.h"
#include "alternatives.h"

     

int main(void)
{
    gsl_set_error_handler_off();

    double kappa = 0.85837;
    double b = 0.089102;
    double sigma = sqrt(0.0021854);
    double delta = 1.0 / 12.0;
    double beta;
    double alpha;
    double s;
    cont_time_to_ar1(kappa, b, sigma, delta, &beta, &alpha, &s);
    printf("True parameters (beta, alpha, s): %g, %g, %g\n", beta, alpha, s);
    // Calculate size distortion
    double sid = ar1_size_distortion(5000, 264, beta, alpha, s);
    printf("Rejection freq = %g\n", sid);

    return 0;

}

