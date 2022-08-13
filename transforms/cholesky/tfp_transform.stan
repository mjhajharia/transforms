#include transform_functions.stan
data {
  int<lower=0> K;
}
parameters {
  // y is a vector K-choose-2 unconstrained parameters
  vector[(K * (K - 1)) %/% 2] y;
}
transformed parameters {
  // L is a Cholesky factor of a K x K correlation matrix
  cholesky_factor_corr[K] L = diag_matrix(rep_vector(1, K));
  real log_det_jacobian = 0;
ß
  {
    int counter = 1;
    real s;

      for (i in 2 : K) {
        L[i, 1:i - 1] = y[counter:counter + i - 2]';
        counter += i - 1;
        s = norm2(L[i,  : i]);
        L[i,  : i] = L[i,  : i] / s;
        log_det_jacobian -= (i + 1) * log(s);
      }
å  }
}
model {
  target += log_det_jacobian;
  target += lkj_corr_cholesky_lpdf(L | 2);
}
