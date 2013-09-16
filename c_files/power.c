

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

    /*

    // ARMA, LSTAR, NLAR, etc. power (print to file)

    int power_reps = 1200;
    int cv_reps = 800;

    int n = 500;
    double beta = 1;
    double alpha = 0.9;
    double s = 0.5;

    FILE *ofp;
    char filename[1024];
    sprintf(filename, "sim_out/art_%g_%g_%g", beta, alpha, s);
    ofp = fopen(filename, "w");

    if (ofp == NULL) {
      fprintf(stderr, "Can't open output file %s!\n", filename);
      exit(1);
    }

    fprintf(ofp, "# n = %i, beta = %g, alpha = %g, s = %g\n", n, beta, alpha, s);
    int num_gammas = 10;
    double gammas[num_gammas];
     linspace(gammas, 0.1, 1, num_gammas);  // for art
    //linspace(gammas, 0.0001, 0.01, num_gammas);  // for rsw
    int i;
    double p1, p2, p3;
    for (i = 0; i < num_gammas; i++) 
    {
        double params[4] = {beta, alpha, s, gammas[i]};
        p1 = ar1_compute_power(&art_ts, params, n, power_reps, cv_reps);
        p2 = cvm_compute_power(&art_ts, params, n, power_reps, cv_reps);
        p3 = cond_m_compute_power(&art_ts, params, n, power_reps);
        fprintf(ofp, "%g, %g, %g, %g\n", gammas[i], p1, p2, p3);
    }
    fclose(ofp);

    */

    // Quick power calculation
    //int n = 264;
    int n = 500;
    int power_reps = 1000;
    int cv_reps = 500;

    // RSW etc baseline
    double beta = 1.0;
    double alpha = 0.9;
    double s = 1;
    //double gamma = 0.005;

    // Vasicek baseline
    //double kappa = 0.858;
    //double b = 0.089;
    //double sigma = sqrt(0.0022);
    
    // CIR baseline
    //double kappa = 0.89218;  
    //double b = 0.0905; // CIR
    //double b = 0.05;
    //double b = 0.025;
    //double sigma = sqrt(0.0327);
    //double delta = 1.0 / 12.0;
    //double gamma = 0.5;
    // With stochastic volatility 
    //double rho = 0.9;
    ////double d = 1;

    int num_gammas = 10;
    double gammas[num_gammas];
    linspace(gammas, 0.0001, 0.01, num_gammas);  // for rsw
    int i;
    for (i = 0; i < num_gammas; i++) 
    {
        double lep;
        double params[4] = {beta, alpha, s, gammas[i]};
        //double params[5] = {kappa, b, sigma, delta, gamma};
        //double params[7] = {kappa, b, sigma, delta, gamma, rho, d};
        lep = cond_m_compute_power(&rsw_ts, params, n, power_reps);
        printf("Rejection rate Cond Moment: %g\n", lep);
        //lep = ar1_compute_power(&t_level_effects_ts, params, n, power_reps, cv_reps);
        //printf("Rejection rate LAE: %g\n", lep);
        //lep = cvm_compute_power(&t_level_effects_ts, params, n, power_reps, cv_reps);
        //printf("Rejection rate CvM: %g\n", lep);
        lep = as_compute_power(&rsw_ts, params, n, power_reps, cv_reps);
        printf("Rejection rate AS: %g\n", lep);
    }


    //int num_gammas = 10;
    //double gammas[num_gammas];
    //linspace(gammas, 0.1, 1, num_gammas);  // for art
    //linspace(gammas, 0.0001, 0.01, num_gammas);  // for rsw
    //int i;
    //double p1;
    //for (i = 0; i < num_gammas; i++) 
    //{
    //    double params[4] = {beta, alpha, s, gammas[i]};
    //    p1 = cond_m_compute_power(&rsw_ts, params, n, power_reps);
    //    printf("%g ", p1);
    //}


    /*

    // Critical value
    int n = 179;
    int num_reps = 10000;
    double beta = 0.589329440425;
    double alpha = 0.292527806261;
    double s = 0.815285122161;
    //double gamma = 0.001;
    //double params[4] = {beta, alpha, s, gamma };
    double lep = ar1_cv_est_params(beta, alpha, s, n, num_reps);
    printf("Critical: %g\n", lep);

    */


    return 0;

}

