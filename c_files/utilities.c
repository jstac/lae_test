
void first_diffs(double *x_diffs, double *x, int n) {
    int i;
    for (i = 0; i < n - 1; i++) 
    {
        x_diffs[i] = x[i+1] - x[i];
    }
}


/* A la NumPy linspace.  *ls is a pointer to the target array */
void linspace(double *ls, double lower, double upper, int n)
{
    double step = (upper - lower) / (n - 1);
    int i;
    for (i = 0; i < n; i++) 
    {
        ls[i] = lower;
        lower += step;
    }
}

double mean(double *x, int n)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < n; i++) 
    {
        sum += x[i];
    }
    return sum / n;
}


double var(double *x, int n)
{
    int i;
    double v;
    double xbar = mean(x, n);
    double sum = 0.0;
    for (i = 0; i < n; i++) 
    {
        v = x[i] - xbar;
        sum += v * v;
    }
    return sum / n;
}
