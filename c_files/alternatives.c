
/*
John Stachurski, Jan 2012
A collection of functions for generating data from alternatives.
*/


#include <stdio.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_fit.h>

#include "utilities.h"

int arma_ts(double * x, double * params, int n, unsigned long int seed)
{
    /* X' = beta + alpha * X + gamma * Z + Z'  where Z's are N(0,s)
     * 
     */
    double beta = params[0];
    double alpha = params[1];
    double s = params[2];
    double gamma = params[3]; 
    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, seed);

    double current_shock;
    double lagged_shock = gsl_ran_gaussian(r, s);
    int i;
    x[0] = beta / (1 - alpha);  // Start at mean of stationary dist
    for (i = 1; i < n; i++) 
     {
       current_shock = gsl_ran_gaussian(r, s);
       x[i] = beta + alpha * x[i-1] + gamma * lagged_shock + current_shock;
       lagged_shock = current_shock;
     }

    gsl_rng_free (r);
    return 0;
}

int lstar_ts(double * x, double * params, int n, unsigned long int seed)
{
    /* X' = beta + alpha * X + gamma * Z + Z'  where Z's are N(0,s)
     * 
     */
    double beta = params[0];
    double alpha = params[1];
    double s = params[2];
    double gamma = params[3]; 
    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, seed);

    int i;
    x[0] = beta / (1 - alpha);  // Start at mean of stationary dist
    for (i = 1; i < n; i++) 
     {
       x[i] = beta + alpha * x[i-1] + gamma * x[i-1] / (1 + exp(- x[i-1] / 2))
           + gsl_ran_gaussian(r, s);
     }

    gsl_rng_free (r);
    return 0;
}

int nlar_ts(double * x, double * params, int n, unsigned long int seed)
{
    /* X' = beta + alpha * X + gamma * Z + Z'  where Z's are N(0,s)
     * 
     */
    double beta = params[0];
    double alpha = params[1];
    double s = params[2];
    double gamma = params[3]; 

    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, seed);

    int i;
    x[0] = beta / (1 - alpha);  // Start at mean of stationary dist
    for (i = 1; i < n; i++) 
     {
       x[i] = beta + alpha * x[i-1] + gamma * fabs(x[i-1]) / fabs(x[i-1] + 2)
           + gsl_ran_gaussian(r, s);
     }

    gsl_rng_free (r);
    return 0;
}


int sv_ts(double * x, double * params, int n, unsigned long int seed)
{
    /*
     *
     * Stochastic Volatility model
     *
     *   X[t+1] = beta + alpha X[t] + s[t] Z[t+1]
     *   s[t+1] = b * s[t]^0.5 * exp{ gamma W[t+1] }
     *
     * where both shocks are independent N(0,1).  When gamma = 0 we want
     * to recover a certain null, where s is specified and constant at s0.
     * To do so we can set b = sqrt(s0).
     * 
     */
    double beta = params[0];
    double alpha = params[1];
    double s0 = params[2];
    double gamma = params[3]; 
    double b = sqrt(s0);

    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, seed);

    double s[n];
    s[0] = s0;
    x[0] = beta / (1 - alpha);  // Start at mean of stationary dist (gamma=0)
    int i;
    for (i = 1; i < n; i++) 
     {
        s[i] = b * sqrt(s[i-1]) * exp(gamma * gsl_ran_gaussian(r, 1));
        x[i] = beta + alpha * x[i-1] + s[i-1] * gsl_ran_gaussian(r, 1);
     }

    gsl_rng_free (r);
    return 0;
}



int armn_ts(double * x, double * params, int n, unsigned long int seed)
{
    /* 
     * X' = beta + alpha * X + Z'  where Z is mixed normal, with mu1 =
     * -gamma and mu2 = gamma.  The standard deviation is s for both.
     */
    double beta = params[0];
    double alpha = params[1];
    double s = params[2];
    double gamma = params[3]; 
    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, seed);
    double m;

    int i;
    x[0] = beta / (1 - alpha);  // Start at mean of stationary dist
    for (i = 1; i < n; i++) 
     {
         m = (gsl_ran_bernoulli(r, 0.5) == 0) ? - gamma : gamma;
         x[i] = beta + alpha * x[i-1] + gsl_ran_gaussian(r, s) + m;
     }

    gsl_rng_free (r);
    return 0;
}



int rsw_ts(double * x, double * params, int n, unsigned long int seed)
{
    /* 
     * X_t+1 = beta_t + alpha * X_t + s * Z_t+1  where Z is standard normal
     *
     * The process beta_t is Markov, changing state with prob gamma.  The
     * two states are beta and -beta.
     */
    double beta = params[0];
    double alpha = params[1];
    double s = params[2];
    double gamma = params[3]; 
    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, seed);

    int i;
    x[0] = beta / (1 - alpha);  // Start at mean of stationary dist
    for (i = 1; i < n; i++) 
     {
         beta = (gsl_ran_flat(r, 0, 1) < gamma) ? - beta : beta;
         x[i] = beta + alpha * x[i-1] + gsl_ran_gaussian(r, s);
     }

    gsl_rng_free (r);
    return 0;
}


int art_ts(double * x, double * params, int n, unsigned long int seed)
{
    /* 
     * X' = beta + alpha * X + Z'  where Z t-distributed
     *
     */
    double beta = params[0];
    double alpha = params[1];
    double s = params[2];
    double gamma = params[3]; 
    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, seed);

    int i;
    x[0] = beta / (1 - alpha);  // Start at mean of stationary dist
    for (i = 1; i < n; i++) 
     {
       x[i] = beta + alpha * x[i-1] + s * gsl_ran_tdist(r, 1 / gamma);
     }

    gsl_rng_free (r);
    return 0;
}


int level_effects_ts (double * x, double * params, int n, unsigned long int seed)
{
    /* The model is 
     *
     * 
     *  dX = kappa(b - X) dt + sigma X^gamma dW,   W Brownian motion
     *
     * The discretized version of the model has form
     *
     *   X' = X + kappa (b - X) delta + X^gamma sigma sqrt(delta) Z
     *
     * where a prime denotes next period's value, and Z is N(0,1).
     *
     */
    double kappa = params[0];
    double b = params[1];
    double sigma = params[2]; 
    double delta = params[3];
    double gamma = params[4];
    double s = sigma * sqrt(delta);

    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, seed);

    int i;
    double xc;
    x[0] = b;  // Equilibrium value for continuous time model
    for (i = 1; i < n; i++) 
    {
        xc = (x[i-1] > 0) ? x[i-1] : 0;
        x[i] = xc + kappa * (b - xc) * delta + pow(xc, gamma) * gsl_ran_gaussian(r, s);
    }

    gsl_rng_free (r);
    return 0;
}

int sv_level_effects_ts (double * x, double * params, int n, unsigned long int seed)
{
    /* This is a level effects model with stochastic volatility.  The model
     * has form
     *
     *   X' = X + kappa * (b - X) * delta + X^gamma * sigma delta^(1/2) * Z
     *   sigma' = c * sigma^rho * W^d
     *
     * where a prime denotes next period's value, Z is N(0,1) and log W is N(0,1)
     *
     * Note that in the standard form of the model (with constant volatility),
     * the value s is equal to sigma * sqrt(delta).  For consistency, we set c
     * and rho such that the mean c/(1-rho) of the stationary distribution of
     * log(s) is equal to log(sigma * sqrt(delta)).
     *
     */
    double kappa = params[0];
    double b = params[1];
    double sigma0 = params[2]; 
    double delta = params[3];
    double gamma = params[4];
    double rho = params[5];
    double d = params[6];

    double c = pow(sigma0, 1 - rho);
    double sqrtdelta = sqrt(delta);

    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, seed);

    int i;
    double xc; 
    double sigma = sigma0;
    x[0] = b;  // Equilibrium value for continuous time model
    for (i = 1; i < n; i++) 
    {
        sigma = c * pow(sigma, rho) * exp(gsl_ran_gaussian(r, d));
        xc = (x[i-1] > 0) ? x[i-1] : 0;
        x[i] = xc + kappa * (b - xc) * delta + pow(xc, gamma) 
            * gsl_ran_gaussian(r, sigma * sqrtdelta);
    }

    gsl_rng_free (r);
    return 0;
}


int rsw_level_effects_ts (double * x, double * params, int n, unsigned long int seed)
{
    /* The model is 
     *
     * 
     *  dX = kappa(b - X) dt + sigma X^gamma dW,   W Brownian motion
     *
     * The discretized version of the model has form
     *
     *   X' = X + kappa (b - X) delta + X^gamma sigma sqrt(delta) Z
     *
     * where a prime denotes next period's value, and Z is N(0,1).
     *
     * In the regime switching version, gamma switches between gamma1 and
     * gamma2
     *
     */
    double kappa = params[0];
    double b = params[1];
    double sigma = params[2]; 
    double delta = params[3];
    double gamma = params[4];
    double s = sigma * sqrt(delta);
    double gamma1 = gamma;
    double gamma2 = gamma + 1;

    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, seed);

    int i;
    double xc;
    x[0] = b;  // Equilibrium value for continuous time model
    gamma = gamma1;
    for (i = 1; i < n; i++) 
    {
        xc = (x[i-1] > 0) ? x[i-1] : 0;
        if (gsl_ran_flat(r, 0, 1) < 0.5)
        {
            if (gamma == gamma1) 
            {
                gamma = gamma2;
            }
            if (gamma == gamma2) 
            {
                gamma = gamma1;
            }
        }
        x[i] = xc + kappa * (b - xc) * delta + pow(xc, gamma) * gsl_ran_gaussian(r, s);
    }

    gsl_rng_free (r);
    return 0;
}

int t_level_effects_ts (double * x, double * params, int n, unsigned long int seed)
{
     /*
     * Level effects with t-distributed shock
     *
     */
    double kappa = params[0];
    double b = params[1];
    double sigma = params[2]; 
    double delta = params[3];
    double gamma = params[4];
    double s = sigma * sqrt(delta);

    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, seed);

    int i;
    double xc;
    x[0] = b;  // Equilibrium value for continuous time model
    for (i = 1; i < n; i++) 
    {
        xc = (x[i-1] > 0) ? x[i-1] : 0;
        x[i] = xc + kappa * (b - xc) * delta + pow(xc, gamma) * s * gsl_ran_tdist(r, 3);
    }

    gsl_rng_free (r);
    return 0;
}

/*
 
int main(void)
{
    double kappa = 0.85837;
    double b = 0.089102;
    double sigma = sqrt(0.0021854);
    double delta = 1.0 / 12.0;
    int n = 100000;
    double x[n];
    double gamma = 0.5;
    level_effects_ts(x, kappa, b, sigma, delta, gamma, n, time(0));
    printf("Mean, var = %g, %g\n", mean(x, n), var(x, n));
    return 0;
}

*/



