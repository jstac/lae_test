     
double ar1_integrand(double, void *);

double ar1_test_stat(double, double, double, double *, int);

double ar1_cv_fixed_params(double, double, double, int, int);

double ar1_cv_est_params(double, double, double, int, int);

double ar1_size_distortion(int, int, double, double, double);

double ar1_compute_power(int (*generator)(double *, double *, int, unsigned long int), double *, int, int, int);

