#include transform_functions.stan
data {
  int<lower=0> K;
}
parameters {
  // y is a vector K-choose-2 unconstrained parameters
  vector[choose_2(K)] y;
}
transformed parameters {
  // L is a Cholesky factor of a K x K correlation matrix
  cholesky_factor_corr[K] L = identity_matrix(K);
  real log_det_jacobian = 0;
  {
    int counter = 1;
    real s;
    
    for (i in 2 : K) {
      for (j in 1 : (i - 1)) {
        L[i, j] = y[counter];
        counter += 1;
      }
      s = norm2(L[i,  : i]);
      L[i,  : i] = L[i,  : i] / s;
      log_det_jacobian -= (i + 1) * log(s);
    }
  }
}
model {
  target += log_det_jacobian;
  target += lkj_corr_cholesky_lpdf(L | 2);
}
