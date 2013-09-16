
     
double cvm_test_stat(double, double, double, double *, int);

double cvm_cv_fixed_params(double, double, double, int, int);

double cvm_cv_est_params(double, double, double, int, int);

double cvm_compute_power(int (*generator)(double *, double *, int, unsigned long int), double *, int, int, int);

