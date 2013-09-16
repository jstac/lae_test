
double comp_bandwidth(double *, int);

double npkde(double *, double, double, double);

double as_test_stat(double, double, double, double *, int);

double as_cv_fixed_params(double, double, double, int, int);

double as_cv_est_params(double, double, double, int, int);

double as_compute_power(int (*generator)(double *, double *, int, unsigned long int), double *, int, int, int);

